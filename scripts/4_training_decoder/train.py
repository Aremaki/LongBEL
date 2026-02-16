import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import glob
import pickle
import shutil
from pathlib import Path

import idr_torch  # type: ignore
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


def get_split_marker(dataset_name: str) -> str:
    # Determine verb based on dataset
    if dataset_name == "MedMentions":
        verb = "is"
    elif dataset_name == "SPACCC":
        verb = "es"
    elif dataset_name in ["EMEA", "MEDLINE"]:
        verb = "est"
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return "} " + verb


def create_prompt_completion_dataset(dataset):
    def format_example(example):
        prompts = []
        completions = []
        for source, target in zip(example["source"], example["target"]):
            prompts.append(source)
            completions.append(target)
        return {"prompt": prompts, "completion": completions}

    return dataset.map(
        format_example,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting prompt/completion dataset",
    )


def extract_entity_char_spans(completion: str, split_marker: str):
    spans = []
    offset = 0

    for raw_line in completion.splitlines(keepends=True):
        line = raw_line[:-1] if raw_line.endswith("\n") else raw_line
        line = line[:-1] if line.endswith("\r") else line

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

        offset += len(raw_line)

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

    # Recover gold target strings from labels by removing -100 tokens
    gold_texts = []
    decoded_preds = []
    for pred, lbl in zip(preds, labels):
        # keep non -100 tokens index values
        indices = [i for i, tok in enumerate(lbl) if tok != -100]
        if len(indices) == 0:
            print("Warning: all label tokens are -100, skipping example.")
            continue
        else:
            filtered_lbl = [lbl[i] for i in indices]
            filtered_pred = [pred[i - 1] for i in indices]
            # decode both
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


# ---------- Main function (converted to TRL SFT) ----------
def main(
    model_name: str,
    lr: float,
    dataset_name: str,
    augmented_data: str,
    selection_method: str = "tfidf",
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
        f"The model {model_short_name} will start SFT with lr={lr} on dataset {augmented_data} {dataset_name} using selection {selection_method}."
    )
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
                Path("/lustre/fsn1/projects/rech/ssq/usk98ia/expe_data_ratio")
                / f"{dataset_name}_synth_only_{selection_method}"
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
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))
    # Add SEP special token and resize embeddings if needed
    sep_token_str = "<SEP>"
    num_added = tokenizer.add_special_tokens({"sep_token": sep_token_str})
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print("Added <SEP> and resized embeddings.")

    # Determine max_length if not provided
    max_length = None
    if hasattr(model.config, "max_position_embeddings"):
        max_length = model.config.max_position_embeddings
    elif hasattr(tokenizer, "model_max_length"):
        max_length = tokenizer.model_max_length

    # Sanity check for very large values (common in some HF tokenizers)
    if max_length is None:
        max_length = 2048
        print(
            f"⚠️ Could not determine valid model max length. Defaulting to {max_length}."
        )
    else:
        print(f"Using detected max_length: {max_length}")

    # Make contiguous parameters (your original precaution)
    for _, param in model.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    # ---------- Load validation dataset (same logic) ----------
    data_folder = Path("data/final_data")

    validation_source_path = (
        data_folder / dataset_name / f"validation_{selection_method}_source.pkl"
    )
    if validation_source_path.exists():
        split_name = "validation"
    else:
        split_name = "test"
        print("Validation file not found, using test file instead.")

    validation_source_data = load_pickle(
        data_folder / dataset_name / f"{split_name}_{selection_method}_source.pkl"
    )
    validation_target_data = load_pickle(
        data_folder / dataset_name / f"{split_name}_{selection_method}_target.pkl"
    )

    validation_data = {
        "source": validation_source_data,
        "target": validation_target_data,
    }
    dev_dataset = Dataset.from_dict(validation_data)

    # Reduce validation dataset to 10% as before
    indexes = list(range(len(dev_dataset)))
    split = int(len(dev_dataset) * 0.9)
    split_val = indexes[split:]
    validation_dataset = dev_dataset.select(split_val)

    # ---------- Load training datasets according to augmented_data ----------
    human_train_dataset = None
    synth_train_dataset = None

    if not augmented_data == "synth_only":
        human_train_source_data = load_pickle(
            data_folder / dataset_name / f"train_{selection_method}_source.pkl"
        )
        human_train_target_data = load_pickle(
            data_folder / dataset_name / f"train_{selection_method}_target.pkl"
        )
        human_train_dataset = Dataset.from_dict({
            "source": human_train_source_data,
            "target": human_train_target_data,
        })
        # if validation present, add rest to training (same logic)
        if split_name == "validation":
            split_train = indexes[:split]
            human_train_dataset = concatenate_datasets([
                human_train_dataset,
                dev_dataset.select(split_train),
            ])

    if augmented_data not in ["human_only", "human_only_ft"]:
        if dataset_name == "MedMentions":
            synth_train_source_data = load_pickle(
                data_folder / "SynthMM" / f"train_{selection_method}_source.pkl"
            )
            synth_train_target_data = load_pickle(
                data_folder / "SynthMM" / f"train_{selection_method}_target.pkl"
            )
            synth_train_dataset = Dataset.from_dict({
                "source": synth_train_source_data,
                "target": synth_train_target_data,
            })
        elif dataset_name in ["EMEA", "MEDLINE"]:
            synth_train_source_data = load_pickle(
                data_folder / "SynthQUAERO" / f"train_{selection_method}_source.pkl"
            )
            synth_train_target_data = load_pickle(
                data_folder / "SynthQUAERO" / f"train_{selection_method}_target.pkl"
            )
            synth_train_dataset = Dataset.from_dict({
                "source": synth_train_source_data,
                "target": synth_train_target_data,
            })
        else:  # SPACCC logic
            if "filtered" not in augmented_data:
                synth_train_source_data = load_pickle(
                    data_folder
                    / "SynthSPACCC_No_Def"
                    / f"train_{selection_method}_source.pkl"
                )
                synth_train_target_data = load_pickle(
                    data_folder
                    / "SynthSPACCC_No_Def"
                    / f"train_{selection_method}_target.pkl"
                )
                synth_train_dataset = Dataset.from_dict({
                    "source": synth_train_source_data,
                    "target": synth_train_target_data,
                })
            else:
                synth_train_source_data = load_pickle(
                    data_folder
                    / "SynthSPACCC_Filtered"
                    / f"train_{selection_method}_source.pkl"
                )
                synth_train_target_data = load_pickle(
                    data_folder
                    / "SynthSPACCC_Filtered"
                    / f"train_{selection_method}_target.pkl"
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
        elif augmented_data in ["full", "full_filtered"]:
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
            num_train_epochs = 200
        else:  # human_only_ft
            lr = lr / 3.0
            if dataset_name in ["EMEA", "MEDLINE"]:
                num_train_epochs = 50
            elif dataset_name == "SPACCC":
                num_train_epochs = 70
            else:
                num_train_epochs = 20
    split_marker = get_split_marker(dataset_name)

    # Format datasets into prompt/completion format
    train_dataset = create_prompt_completion_dataset(train_dataset)
    validation_dataset = create_prompt_completion_dataset(validation_dataset)

    train_dataset = train_dataset.map(
        lambda x: preprocess_prompt_completion_example(x, tokenizer, split_marker),
        remove_columns=train_dataset.column_names,
        num_proc=8,
    )

    validation_dataset = validation_dataset.map(
        lambda x: preprocess_prompt_completion_example(x, tokenizer, split_marker),
        remove_columns=validation_dataset.column_names,
        num_proc=8,
    )

    # Sanity check for tokenization and formatting
    example = train_dataset[0]
    decoded = tokenizer.decode([
        t if lab != -100 else 0
        for t, lab in zip(example["input_ids"], example["labels"])
    ])
    print(decoded)
    # ---------- SFT TrainingArguments ----------

    output_dir = (
        Path("models")
        / "NED"
        / f"{dataset_name}_{augmented_data}_{selection_method}"
        / model_short_name
    )
    logging_dir = (
        Path("logs")
        / f"{dataset_name}_{augmented_data}_{selection_method}"
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
        packing=False,
        max_length=max_length,
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

    # Resume option (identical logic)
    resume_from_checkpoint = len(glob.glob(rf"{output_dir}/checkpoint-*")) > 0
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the best model checkpoint and last model (similar to original script)
    try:
        best_model_chkpt_dir = Path(trainer.state.best_model_checkpoint)  # type: ignore
        model_dir = best_model_chkpt_dir.parent
        best_model_dir = model_dir / "model_best"
        shutil.copytree(best_model_chkpt_dir, best_model_dir, dirs_exist_ok=True)
        best_step = str(trainer.state.best_model_checkpoint).split("-")[-1]
        print(f"Best model saved at step {best_step}")
    except Exception as e:
        print("Warning: unable to copy best model checkpoint:", e)

    try:
        last_model_chkpt_dir = model_dir / f"checkpoint-{trainer.state.global_step}"  # type: ignore
        last_model_dir = model_dir / "model_last"  # type: ignore
        shutil.copytree(last_model_chkpt_dir, last_model_dir, dirs_exist_ok=True)
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
            "full_filtered",
            "full_filtered_upsampled",
        ],
        help="Whether to use augmented data for training",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default="embedding",
        choices=["embedding", "tfidf", "levenshtein", "title"],
        help="The method to select concept synonyms",
    )
    args = parser.parse_args()

    main(
        model_name=args.model_name,
        lr=args.lr,
        dataset_name=args.dataset_name,
        augmented_data=args.augmented_data,
        selection_method=args.selection_method,
    )
