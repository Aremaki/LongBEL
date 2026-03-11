import os
import re

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import pickle
import shutil
from pathlib import Path

import idr_torch  # type: ignore
import nltk
import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets, interleave_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer  # type: ignore

# Enable TF32 paths everywhere
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Enable flash attention variants
torch.backends.cuda.enable_flash_sdp(True)


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def get_split_marker(
    dataset_name: str,
) -> tuple[str, nltk.tokenize.PunktSentenceTokenizer]:
    # Determine verb based on dataset
    if dataset_name == "MedMentions":
        nlp = nltk.data.load("tokenizers/punkt/english.pickle")
    elif dataset_name == "SPACCC":
        nlp = nltk.data.load("tokenizers/punkt/spanish.pickle")
    elif dataset_name in ["EMEA", "MEDLINE"]:
        nlp = nltk.data.load("tokenizers/punkt/french.pickle")
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return "}", nlp


def sentence_tokenize_safe(text, nlp):
    """
    Split text into sentences while avoiding splits inside entity mentions `[ ... ]`.
    Returns a list of sentences.
    """
    sentences = []

    # split text by newline blocks first
    chunks = re.split(r"\n+", text)

    cursor = 0

    for chunk in chunks:
        if not chunk.strip():
            cursor += len(chunk) + 1
            continue

        spans = nlp.span_tokenize(chunk)

        last_end = 0
        open_brackets = 0  # track open entity brackets

        for _, end in spans:
            # count brackets in the current span
            open_brackets += chunk[last_end:end].count("[")
            open_brackets -= chunk[last_end:end].count("]")

            # only allow a break if no unclosed brackets
            if open_brackets == 0:
                sent = chunk[last_end:end].strip()
                if sent:
                    sentences.append(sent)
                last_end = end

        # leftover
        tail = chunk[last_end:].strip()
        if tail:
            sentences.append(tail)

        cursor += len(chunk) + 1

    return sentences


def add_headers_to_prompt(source: str, target: str, context_format: str):
    if context_format == "long":
        prompt = f"### Context\n{source.rstrip()}\n\n"
        completion = f"### Predictions\n{target}"
    elif context_format == "short":
        target_split = target.split("}")
        if len(target_split) == 2:
            prefix = target_split[0] + "}"
            completion = target_split[1]
        else:
            raise ValueError(f"Unexpected target format: {target}")
        # Add Instruction prefix to source
        prompt = f"### Context\n{source.rstrip()}\n\n### Prediction\n{prefix}"
    elif context_format in ["hybrid_short", "hybrid_long"]:
        split_target = target.split("\n")
        # remove empty string
        split_target = [s for s in split_target if s]
        if len(split_target) >= 2:
            previous_tgt = "\n".join(split_target[:-1]) + "\n"
            current_tgt = split_target[-1]
        elif len(split_target) == 1:
            previous_tgt = "None"
            current_tgt = split_target[0]
        else:
            raise ValueError(f"Unexpected target format: {target}")
        current_tgt_split = current_tgt.split("}")
        if len(current_tgt_split) == 2:
            current_tgt_prefix = current_tgt_split[0] + "}"
            completion = current_tgt_split[1]
        else:
            raise ValueError(f"Unexpected current target format: {current_tgt}")
        # Add Instruction prefix to source
        prompt = f"### Context\n{source.rstrip()}\n\n### Previous Normalizations\n{previous_tgt.rstrip()}\n\n### Prediction\n{current_tgt_prefix}"
    else:
        raise ValueError(f"Unknown context_format: {context_format}")
    return prompt, completion


def create_prompt_completion_dataset(
    dataset,
    nlp,
    complete_mode: bool,
    context_format: str,
    add_headers: bool = True,
):
    def format_example(example):
        prompts = []
        completions = []
        for source, target in zip(example["source"], example["target"]):
            if add_headers:
                prompt, completion = add_headers_to_prompt(
                    source, target, context_format
                )
            else:
                prompt, completion = source, target
            prompts.append(prompt)
            completions.append(completion)
        if complete_mode:
            for source, target in zip(example["source"], example["target"]):
                for split_target in target.split("<SEP>"):
                    if split_target:
                        split_target = split_target + "<SEP>"
                        if add_headers:
                            prompt, completion = add_headers_to_prompt(
                                source, split_target, context_format
                            )
                        else:
                            prompt, completion = source, split_target
                        prompts.append(prompt)
                        completions.append(completion)
        return {"prompt": prompts, "completion": completions}

    return dataset.map(
        format_example,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting prompt/completion dataset",
    )


def get_length(example, tokenizer):
    texts = [p + c for p, c in zip(example["prompt"], example["completion"])]
    enc = tokenizer(texts, add_special_tokens=False)
    return {"length": [len(x) for x in enc["input_ids"]]}


def extract_entity_char_spans(completion: str, split_marker: str):
    spans = []
    offset = 0
    sep_marker = "<SEP>"

    for line in completion.split(sep_marker):
        line += sep_marker

        split_pos = line.find(split_marker)
        if split_pos != -1:
            entity_start = split_pos + len(split_marker)
            while entity_start < len(line) and line[entity_start].isspace():
                entity_start += 1

            entity_end = len(line)
            while entity_end > entity_start and line[entity_end - 1].isspace():
                entity_end -= 1

            if entity_end > entity_start:
                spans.append((offset + entity_start, offset + entity_end))

        offset += len(line)

    return spans


def preprocess_prompt_completion_example(example, tokenizer, split_marker):
    prompt = example["prompt"]
    completion = example["completion"]

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_encoding = tokenizer(
        completion,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    completion_ids = completion_encoding["input_ids"]
    completion_offsets = completion_encoding["offset_mapping"]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(input_ids)

    entity_spans = extract_entity_char_spans(completion, split_marker)
    prompt_len = len(prompt_ids)

    for token_idx, (start, end) in enumerate(completion_offsets):
        if end <= start:
            continue

        overlaps_entity = any(
            not (end <= span_start or start >= span_end)
            for span_start, span_end in entity_spans
        )
        if overlaps_entity:
            labels[prompt_len + token_idx] = completion_ids[token_idx]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# ---------- compute_metrics using generation ----------
def compute_metrics_generate(preds, labels, tokenizer):
    """
    preds: list of generated token ids (padded arrays)
    labels: list of label ids with -100 for masked positions
    We'll decode generation and compare the generated *target* to gold target.
    """
    # If preds is a tuple (e.g. logits, past_key_values), take the first element
    if isinstance(preds, tuple):
        preds = preds[0]

    gold_texts = []
    decoded_preds = []
    for pred, lbl in zip(preds, labels):
        # Build contiguous spans of token indices where label != -100
        spans = []
        span_start = None
        for idx, tok in enumerate(lbl):
            if tok != -100 and span_start is None:
                span_start = idx
            elif tok == -100 and span_start is not None:
                spans.append((span_start, idx))
                span_start = None
        if span_start is not None:
            spans.append((span_start, len(lbl)))

        if len(spans) == 0:
            print("Warning: all label tokens are -100, skipping example.")
            continue

        for start, end in spans:
            filtered_lbl = [lbl[i] for i in range(start, end)]
            # Keep causal shift alignment used previously: prediction at i-1
            filtered_pred = [pred[i - 1] for i in range(start, end) if (i - 1) >= 0]

            if len(filtered_lbl) == 0 or len(filtered_pred) == 0:
                continue

            dec_lbl = tokenizer.decode(
                filtered_lbl,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            dec_pred = tokenizer.decode(
                filtered_pred,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            gold_texts.append(dec_lbl.strip())
            decoded_preds.append(dec_pred.strip())

    # Exact-match recall (target-level)
    matches = [p.strip() == g.strip() for p, g in zip(decoded_preds, gold_texts)]
    recall = float(np.mean(matches)) if len(matches) > 0 else 0.0

    return {
        "recall": round(recall, 4),
        "num_gold": len(gold_texts),
        "num_guess": len(decoded_preds),
    }


# ---------- Preprocess logits for metrics ----------
def preprocess_logits_for_metrics(logits, labels):
    """
    Reduce memory usage by keeping only the argmax of the logits.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def find_latest_resumable_checkpoint(output_dir: Path):
    checkpoint_dirs = []
    for checkpoint_dir in output_dir.glob("checkpoint-*"):
        if not checkpoint_dir.is_dir():
            continue
        try:
            step = int(checkpoint_dir.name.split("-")[-1])
        except ValueError:
            continue
        checkpoint_dirs.append((step, checkpoint_dir))

    checkpoint_dirs.sort(key=lambda item: item[0], reverse=True)

    for step, checkpoint_dir in checkpoint_dirs:
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        if trainer_state_path.exists():
            return str(checkpoint_dir)
        print(
            f"Skipping checkpoint-{step}: missing trainer_state.json, cannot resume safely."
        )

    return None


# ---------- Main function (converted to TRL SFT) ----------
def main(
    model_name: str,
    lr: float,
    dataset_name: str,
    augmented_data: str,
    max_length: int = 16_000,
    selection_method: str = "tfidf",
    context_format: str = "short",
    complete_mode: bool = False,
    add_headers: bool = False,
    start_entity_token: str = "[",
    end_entity_token: str = "]",
    start_group_token: str = "{",
    end_group_token: str = "}",
):
    # init distributed (if needed)
    if idr_torch.rank == 0:
        print(
            ">>> Training on ",
            len(idr_torch.nodelist),
            " nodes and ",
            idr_torch.world_size,
            " processes",
        )
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=idr_torch.world_size,
            rank=idr_torch.rank,
        )

    model_short_name = model_name.split("/")[-1]
    print(
        f"The model {model_short_name} will start SFT with lr={lr} on dataset {augmented_data} {dataset_name} using selection {selection_method} and context_format is {context_format} and complete_mode is {complete_mode} and add_headers is {add_headers}."
    )
    complete_mode_str = "_complete" if complete_mode else ""
    add_headers_str = "_addheaders" if add_headers else ""
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
    except Exception as hub_err:  # pragma: no cover - network / availability branch
        if augmented_data == "human_only_ft":
            local_dir = (
                Path("models")
                / "NED"
                / f"{dataset_name}_synth_only_{selection_method}_{context_format}{complete_mode_str}{add_headers_str}"
                / model_short_name
                / "model_last"
            )
        else:
            local_dir = Path("models") / str(model_name)
        if local_dir.exists():
            print(
                f"⚠️ Hub load failed for '{model_name}' ({hub_err}); falling back to local path {local_dir}."
            )
            tokenizer = AutoTokenizer.from_pretrained(str(local_dir), use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(local_dir),
                attn_implementation="flash_attention_2",
                dtype=torch.bfloat16,
            )
        else:
            raise hub_err

    # Ensure pad_token exists (use eos_token as pad if necessary)
    num_added = 0
    if tokenizer.pad_token is None:
        num_added += tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    # Add SEP special token and resize embeddings if needed
    plus_token_str = "<+>"
    num_added += tokenizer.add_special_tokens({
        "additional_special_tokens": [
            start_entity_token,
            end_entity_token,
            start_group_token,
            end_group_token,
            plus_token_str,
        ],
    })
    if context_format == "long":
        sep_token_str = "<SEP>"
        num_added += tokenizer.add_special_tokens({"sep_token": sep_token_str})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print("Added special tokens to tokenizer and resized model embeddings.")

    # Determine model context length (for info / warning)
    model_context_length = None
    if hasattr(model.config, "max_position_embeddings"):
        model_context_length = model.config.max_position_embeddings
    elif hasattr(tokenizer, "model_max_length"):
        model_context_length = tokenizer.model_max_length

    # Sanity check for very large values (common in some HF tokenizers)
    if model_context_length is None:
        model_context_length = 2048
        print(
            f"⚠️ Could not determine valid model max length. Defaulting to {model_context_length}."
        )
    else:
        print(f"Using detected model context length: {model_context_length}")

    # Make contiguous parameters (your original precaution)
    for _, param in model.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    # ---------- Load validation dataset (same logic) ----------
    data_folder = Path("data/final_data")

    validation_source_path = (
        data_folder
        / dataset_name
        / f"validation_{selection_method}_source_{context_format}.pkl"
    )
    validation_target_path = (
        data_folder
        / dataset_name
        / f"validation_{selection_method}_target_{context_format}.pkl"
    )
    validation_source_data = load_pickle(validation_source_path)
    validation_target_data = load_pickle(validation_target_path)

    validation_data = {
        "source": validation_source_data,
        "target": validation_target_data,
    }
    dev_dataset = Dataset.from_dict(validation_data)

    # Reduce validation dataset to 10%
    indexes = list(range(len(dev_dataset)))
    split = int(len(dev_dataset) * 0.9)
    split_val = indexes[split:]
    validation_dataset = dev_dataset.select(split_val)

    # ---------- Load training datasets according to augmented_data ----------
    human_train_dataset = None
    synth_train_dataset = None

    if not augmented_data == "synth_only":
        human_train_source_data = load_pickle(
            data_folder
            / dataset_name
            / f"train_{selection_method}_source_{context_format}.pkl"
        )
        human_train_target_data = load_pickle(
            data_folder
            / dataset_name
            / f"train_{selection_method}_target_{context_format}.pkl"
        )
        human_train_dataset = Dataset.from_dict({
            "source": human_train_source_data,
            "target": human_train_target_data,
        })
        # if validation present, add rest to training (same logic)
        split_train = indexes[:split]
        human_train_dataset = concatenate_datasets([
            human_train_dataset,
            dev_dataset.select(split_train),
        ])

    if augmented_data not in ["human_only", "human_only_ft"]:
        if dataset_name == "MedMentions":
            synth_train_source_data = load_pickle(
                data_folder / "SynthMM" / f"train_{selection_method}_source_short.pkl"
            )
            synth_train_target_data = load_pickle(
                data_folder / "SynthMM" / f"train_{selection_method}_target_short.pkl"
            )
            synth_train_dataset = Dataset.from_dict({
                "source": synth_train_source_data,
                "target": synth_train_target_data,
            })
        elif dataset_name in ["EMEA", "MEDLINE"]:
            synth_train_source_data = load_pickle(
                data_folder
                / "SynthQUAERO"
                / f"train_{selection_method}_source_short.pkl"
            )
            synth_train_target_data = load_pickle(
                data_folder
                / "SynthQUAERO"
                / f"train_{selection_method}_target_short.pkl"
            )
            synth_train_dataset = Dataset.from_dict({
                "source": synth_train_source_data,
                "target": synth_train_target_data,
            })
        else:  # SPACCC logic
            synth_train_source_data = load_pickle(
                data_folder
                / "SynthSPACCC"
                / f"train_{selection_method}_source_short.pkl"
            )
            synth_train_target_data = load_pickle(
                data_folder
                / "SynthSPACCC"
                / f"train_{selection_method}_target_short.pkl"
            )
            synth_train_dataset = Dataset.from_dict({
                "source": synth_train_source_data,
                "target": synth_train_target_data,
            })

        # ---------- Prepare final train_dataset based on augmented_data ----------
        # augmented data training
        save_strategy = "steps"
        save_steps = 2000
        eval_strategy = "steps"
        eval_steps = 2000
        logging_strategy = "steps"
        logging_steps = 2000
        num_train_epochs = 3
        if augmented_data == "synth_only":
            num_train_epochs = 5
            train_dataset = synth_train_dataset
        elif augmented_data == "full":
            num_train_epochs = 5
            train_dataset = concatenate_datasets([
                human_train_dataset,  # type: ignore
                synth_train_dataset,
            ])
        else:  # full_upsampled
            train_dataset = interleave_datasets(
                [human_train_dataset, synth_train_dataset],  # type: ignore
                stopping_strategy="all_exhausted",
                seed=42,
            )
    else:
        # human only training
        save_strategy = "epoch"
        save_steps = 0
        eval_strategy = "epoch"
        eval_steps = 0
        logging_strategy = "epoch"
        logging_steps = 0
        train_dataset = human_train_dataset
        if augmented_data == "human_only":
            num_train_epochs = 50
            if "hybrid" in context_format and dataset_name == "MedMentions":
                num_train_epochs = num_train_epochs // 2
        else:  # human_only_ft
            lr = lr / 3.0
            if dataset_name in ["EMEA", "MEDLINE"]:
                num_train_epochs = 50
            elif dataset_name == "SPACCC":
                num_train_epochs = 70
            else:
                num_train_epochs = 20
    split_marker, nlp = get_split_marker(dataset_name)

    # Format datasets into prompt/completion format
    train_dataset = create_prompt_completion_dataset(
        train_dataset,
        nlp,
        complete_mode=complete_mode,
        context_format=context_format,
        add_headers=add_headers,
    )
    validation_dataset = create_prompt_completion_dataset(
        validation_dataset,
        nlp,
        complete_mode=complete_mode,
        context_format=context_format,
        add_headers=add_headers,
    )
    num_proc = os.cpu_count() // 2
    if context_format == "long":
        train_dataset = train_dataset.map(
            lambda x: preprocess_prompt_completion_example(x, tokenizer, split_marker),
            remove_columns=train_dataset.column_names,
            num_proc=num_proc,
        )

        validation_dataset = validation_dataset.map(
            lambda x: preprocess_prompt_completion_example(x, tokenizer, split_marker),
            remove_columns=validation_dataset.column_names,
            num_proc=num_proc,
        )
        completion_only_loss = False
        # Sanity check for tokenization and formatting
        example = train_dataset[0]
        decoded = tokenizer.decode([
            t if lab != -100 else 0
            for t, lab in zip(example["input_ids"], example["labels"])
        ])
        print(decoded)

    else:
        completion_only_loss = True
        # Sanity check for tokenization and formatting
        print(train_dataset[10])

    # Compute longest training example
    lengths = train_dataset.map(
        lambda x: get_length(x, tokenizer), batched=True, num_proc=num_proc
    )
    longest_train = max(lengths["length"])

    print(f"Longest training example has {longest_train} tokens.")
    if longest_train > max_length:
        if "8B" in model_name:
            print(
                f"⚠️ Longest training example ({longest_train} tokens) exceeds hard max_length ({max_length}). They will be truncated..."
            )
        else:
            max_length = longest_train + 512
    print(f"Using training-set max_length: {max_length}")
    if max_length > model_context_length:
        print(
            f"⚠️ training-set max_length ({max_length}) is larger than model context length ({model_context_length})."
        )
    # ---------- SFT TrainingArguments ----------

    output_dir = (
        Path("models")
        / "NED"
        / f"{dataset_name}_{augmented_data}_{selection_method}_{context_format}{complete_mode_str}{add_headers_str}"
        / model_short_name
    )
    logging_dir = (
        Path("logs")
        / f"{dataset_name}_{augmented_data}_{selection_method}_{context_format}{complete_mode_str}{add_headers_str}"
        / model_short_name
    )
    model.gradient_checkpointing_enable()
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else 0,
        logging_strategy=logging_strategy,
        logging_steps=logging_steps if logging_strategy == "steps" else 0,
        num_train_epochs=num_train_epochs,
        bf16=True,
        bf16_full_eval=True,
        learning_rate=lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        load_best_model_at_end=True,
        metric_for_best_model="eval_recall",
        greater_is_better=True,
        seed=42,
        report_to="tensorboard",
        save_total_limit=2,
        eval_packing=False,
        packing=True,
        max_length=max_length,
        completion_only_loss=completion_only_loss,
    )

    # ---------- SFTTrainer ----------
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=lambda pred: compute_metrics_generate(
            pred.predictions,
            pred.label_ids,
            tokenizer,
        ),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Resume only from a valid checkpoint containing trainer_state.json
    resume_from_checkpoint = find_latest_resumable_checkpoint(output_dir)
    if resume_from_checkpoint:
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    else:
        print("No valid checkpoint found. Starting training from scratch.")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the best model checkpoint and last model (similar to original script)
    try:
        best_model_chkpt_dir = Path(trainer.state.best_model_checkpoint)  # type: ignore
        model_dir = best_model_chkpt_dir.parent
        best_model_dir = model_dir / "model_best"
        # Remove target if it exists
        if best_model_dir.exists():
            shutil.rmtree(best_model_dir)
        # Rename checkpoint to target
        best_model_chkpt_dir.rename(best_model_dir)
        best_step = str(trainer.state.best_model_checkpoint).split("-")[-1]
        print(f"Best model saved at step {best_step}")
    except Exception as e:
        print("Warning: unable to copy best model checkpoint:", e)

    try:
        last_model_chkpt_dir = model_dir / f"checkpoint-{trainer.state.global_step}"  # type: ignore
        last_model_dir = model_dir / "model_last"  # type: ignore
        # Remove target if it exists
        if last_model_dir.exists():
            shutil.rmtree(last_model_dir)
        # Rename checkpoint to target
        last_model_chkpt_dir.rename(last_model_dir)
        last_step = str(trainer.state.global_step)
        print(f"Last model saved at step {last_step}")
    except Exception as e:
        print("Warning: unable to copy last model checkpoint:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script for SFT training (decoder-only) with TRL SFTTrainer"
    )
    parser.add_argument("--model-name", type=str, required=True, help="The model name")
    parser.add_argument(
        "--lr", type=float, default=3e-05, help="The optimizer learning rate"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="SPACCC",
        help="The name of the dataset to use",
    )
    parser.add_argument(
        "--augmented-data",
        type=str,
        default="human_only",
        choices=[
            "human_only",
            "human_only_ft",
            "synth_only",
            "full",
            "full_upsampled",
        ],
        help="Whether to use augmented data for training",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default="tfidf",
        choices=["embedding", "tfidf", "levenshtein", "title"],
        help="The method to select concept synonyms",
    )
    parser.add_argument(
        "--context-format",
        type=str,
        default="short",
        choices=[
            "short",
            "long",
            "hybrid_short",
            "hybrid_long",
        ],
        help="Whether to use augmented data for training",
    )
    parser.add_argument(
        "--complete-mode",
        action="store_true",
        help="Whether to use complete mode for prompt/completion formatting",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=16_000,
        help="The maximum sequence length for training examples (after tokenization). Examples longer than this will be truncated. Default is 16,000 tokens.",
    )
    parser.add_argument(
        "--start-entity-token",
        type=str,
        default="[",
        help="The token marking the start of the entity",
    )
    parser.add_argument(
        "--end-entity-token",
        type=str,
        default="]",
        help="The token marking the end of the entity",
    )
    parser.add_argument(
        "--start-group-token",
        type=str,
        default="{",
        help="The token marking the start of the entity group",
    )
    parser.add_argument(
        "--end-group-token",
        type=str,
        default="}",
        help="The token marking the end of the entity group",
    )
    parser.add_argument(
        "--add-context",
        action="store_true",
        help="Whether to add context to the prompt even for short format (for ablation)",
    )
    args = parser.parse_args()

    main(
        model_name=args.model_name,
        lr=args.lr,
        dataset_name=args.dataset_name,
        augmented_data=args.augmented_data,
        selection_method=args.selection_method,
        context_format=args.context_format,
        complete_mode=args.complete_mode,
        add_headers=args.add_headers,
        max_length=args.max_length,
        start_entity_token=args.start_entity_token,
        end_entity_token=args.end_entity_token,
        start_group_token=args.start_group_token,
        end_group_token=args.end_group_token,
    )
