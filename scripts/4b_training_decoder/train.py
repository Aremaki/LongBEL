# train_sft_trl.py
import argparse
import glob
import pickle
import shutil
from pathlib import Path

import idr_torch  # type: ignore
import numpy as np
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets, interleave_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


# ---------- Preprocess for decoder-only SFT ----------
def preprocess_for_sft(
    examples, tokenizer, sep_token="<SEP>", max_length=None, dataset_name=None
):
    """
    Build input sequence:   source_text <SEP> target_text
    Labels: -100 for source and prefix tokens, actual ids for target tokens.
    """
    sources = examples["source"]
    targets = examples["target"]
    texts = [f"{s}{sep_token}{t}" for s, t in zip(sources, targets)]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # let collator pad
        return_attention_mask=True,
        return_offsets_mapping=True,
    )

    all_input_ids = tokenized["input_ids"]
    all_offsets = tokenized["offset_mapping"]

    # Determine verb based on dataset
    verb = "est"
    if dataset_name == "MedMentions":
        verb = "is"
    elif dataset_name and "SPACCC" in dataset_name:
        verb = "es"

    labels = []
    for i, input_ids in enumerate(all_input_ids):
        text = texts[i]
        offsets = all_offsets[i]

        try:
            sep_pos_char = text.index(sep_token)
        except ValueError:
            # If no SEP, we can't mask properly, label the whole sequence
            labels.append(input_ids.copy())
            continue

        # The target part of the text starts after the separator
        target_text_start_char = sep_pos_char + len(sep_token)

        # Find the split marker within the target part of the text
        split_marker = f"] {verb} "
        split_pos_char = text.find(split_marker, target_text_start_char)
        if split_pos_char == -1:
            raise ValueError(
                f"Expected pattern '[X] {verb} Y' missing in target for dataset '{dataset_name}': {text}"
            )
        # The character position where the actual target (Y) begins
        target_y_start_char = split_pos_char + len(split_marker)

        # Create labels: -100 for everything up to target_y_start_char
        label = []
        for token_id, (_, end_char) in zip(input_ids, offsets):
            # Mask if the token is part of the source or the "[X] is" prefix.
            # We mask if the token ends before or at the start of our desired prediction.
            if end_char <= target_y_start_char:
                label.append(-100)
            else:
                label.append(token_id)

        # Ensure label length equals input length
        if len(label) != len(input_ids):
            # This should ideally not happen with this logic, but as a safeguard:
            if len(label) < len(input_ids):
                label.extend([-100] * (len(input_ids) - len(label)))
            else:
                label = label[: len(input_ids)]

        labels.append(label)

    tokenized["labels"] = labels
    # We don't need offset_mapping anymore
    del tokenized["offset_mapping"]
    return tokenized


# ---------- Utility to clean generated text ----------
def strip_special_tokens_from_generated(text, sep_token="<SEP>"):
    # If model outputs "<source> <SEP> <pred>" we return the part after SEP when possible
    if sep_token in text:
        return text.split(sep_token, 1)[1].strip()
    return text.strip()


# ---------- compute_metrics using generation ----------
def compute_metrics_generate(preds, labels, tokenizer, sep_token="<SEP>"):
    """
    preds: list of generated token ids (padded arrays)
    labels: list of label ids with -100 for masked positions
    We'll decode generation and compare the generated *target* (after SEP) to gold target.
    """
    # If preds is a tuple (e.g. logits, past_key_values), take the first element
    if isinstance(preds, tuple):
        preds = preds[0]

    # If preds are logits (Batch, Seq, Vocab), take argmax
    if isinstance(preds, np.ndarray) and preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    decoded_preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    # Extract predicted target portion after SEP
    decoded_preds = [
        strip_special_tokens_from_generated(p, sep_token=sep_token)
        for p in decoded_preds
    ]

    # Recover gold target strings from labels by removing -100 and decoding only the suffix (after SEP)
    gold_texts = []
    for lbl in labels:
        # keep non -100 tokens
        ids = [tok for tok in lbl if tok != -100]
        if len(ids) == 0:
            gold_texts.append("")
        else:
            # decode; if sep present in decoded input, try to split similarly
            dec = tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            gold_texts.append(dec.strip())

    # Exact-match recall (target-level)
    matches = [p.strip() == g.strip() for p, g in zip(decoded_preds, gold_texts)]
    recall = float(np.mean(matches)) if len(matches) > 0 else 0.0

    return {
        "recall": round(recall, 4),
        "num_gold": len(gold_texts),
        "num_guess": len(decoded_preds),
    }


# ---------- Main function (converted to TRL SFT) ----------
def main(
    model_name: str,
    lr: float,
    dataset_name: str,
    augmented_data: str,
    selection_method: str = "tfidf",
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    max_length: int = 512,
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

    print(
        f"The model {model_name} will start SFT with lr={lr} on dataset {augmented_data} {dataset_name} using selection {selection_method}."
    )

    model_short_name = model_name.split("/")[-1]
    # load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

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

    # Make contiguous parameters (your original precaution)
    for _, param in model.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    # set mbart langs if necessary (keep your logic)
    if "mbart" in model_short_name:
        # NOTE: causal LM typically not mbart; keep for compatibility if you somehow use mbart
        if dataset_name == "MedMentions":
            tokenizer.src_lang = "en_XX"
            tokenizer.tgt_lang = "en_XX"
        elif dataset_name == "SPACCC":
            tokenizer.src_lang = "es_XX"
            tokenizer.tgt_lang = "es_XX"
        elif dataset_name == "SPACCC_UMLS":
            tokenizer.src_lang = "es_XX"
            tokenizer.tgt_lang = "es_XX"
        else:
            tokenizer.src_lang = "fr_XX"
            tokenizer.tgt_lang = "fr_XX"

    # ---------- Load validation dataset (same logic) ----------
    data_folder = Path("data/final_data")
    human_dataset_name = "SPACCC" if "SPACCC" in dataset_name else dataset_name

    validation_source_path = (
        data_folder / human_dataset_name / f"validation_{selection_method}_source.pkl"
    )
    if validation_source_path.exists():
        split_name = "validation"
    else:
        split_name = "test"
        print("Validation file not found, using test file instead.")

    validation_source_data = load_pickle(
        data_folder / human_dataset_name / f"{split_name}_{selection_method}_source.pkl"
    )
    validation_target_data = load_pickle(
        data_folder / human_dataset_name / f"{split_name}_{selection_method}_target.pkl"
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
    synth_def_dataset = None
    synth_no_def_dataset = None

    if augmented_data in ["human_only", "full", "human_only_ft", "full_upsampled"]:
        human_train_source_data = load_pickle(
            data_folder / human_dataset_name / f"train_{selection_method}_source.pkl"
        )
        human_train_target_data = load_pickle(
            data_folder / human_dataset_name / f"train_{selection_method}_target.pkl"
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

    if augmented_data in ["synth_only", "full", "full_upsampled"]:
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
        elif dataset_name == "QUAERO":
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
        elif dataset_name == "SPACCC":
            synth_train_source_data = load_pickle(
                data_folder / "SynthSPACCC" / f"train_{selection_method}_source.pkl"
            )
            synth_train_target_data = load_pickle(
                data_folder / "SynthSPACCC" / f"train_{selection_method}_target.pkl"
            )
            synth_train_dataset = Dataset.from_dict({
                "source": synth_train_source_data,
                "target": synth_train_target_data,
            })
        else:  # SPACCC_UMLS logic
            source_no_def = load_pickle(
                data_folder
                / "SynthSPACCC_UMLS_No_Def"
                / f"train_{selection_method}_source.pkl"
            )
            source_def = load_pickle(
                data_folder
                / "SynthSPACCC_UMLS_Def"
                / f"train_{selection_method}_source.pkl"
            )
            synth_train_source_data = [*source_no_def, *source_def]
            target_no_def = load_pickle(
                data_folder
                / "SynthSPACCC_UMLS_No_Def"
                / f"train_{selection_method}_target.pkl"
            )
            target_def = load_pickle(
                data_folder
                / "SynthSPACCC_UMLS_Def"
                / f"train_{selection_method}_target.pkl"
            )
            synth_train_target_data = [*target_no_def, *target_def]
            synth_train_dataset = Dataset.from_dict({
                "source": synth_train_source_data,
                "target": synth_train_target_data,
            })
            synth_def_dataset = Dataset.from_dict({
                "source": source_def,
                "target": target_def,
            })
            synth_no_def_dataset = Dataset.from_dict({
                "source": source_no_def,
                "target": target_no_def,
            })

    # Choose train_dataset accordingly (same logic as before)
    if augmented_data in ["synth_only", "full", "full_upsampled"]:
        save_strategy = "steps"
        save_steps = 20000
        eval_strategy = "steps"
        eval_steps = 20000
        logging_strategy = "steps"
        logging_steps = 20000
        num_train_epochs = 3
        if augmented_data == "synth_only":
            train_dataset = synth_train_dataset
        elif augmented_data == "full":
            num_train_epochs = 5
            train_dataset = concatenate_datasets([
                human_train_dataset,
                synth_train_dataset,
            ])  # type: ignore
        else:  # full_upsampled
            if dataset_name == "SPACCC_UMLS":
                train_dataset = interleave_datasets(
                    [human_train_dataset, synth_def_dataset, synth_no_def_dataset],  # type: ignore
                    stopping_strategy="all_exhausted",
                    seed=42,
                )
            else:
                train_dataset = interleave_datasets(
                    [human_train_dataset, synth_train_dataset],  # type: ignore
                    stopping_strategy="all_exhausted",
                    seed=42,
                )
    else:
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
            num_train_epochs = 70
            lr = lr / 3.0

    # Tokenize datasets with SFT masking
    tokenized_train = train_dataset.map(  # type: ignore
        lambda x: preprocess_for_sft(
            x,
            tokenizer,
            sep_token=sep_token_str,
            max_length=max_length,
            dataset_name=dataset_name,
        ),
        batched=True,
        remove_columns=["source", "target"],
    )
    tokenized_val = validation_dataset.map(
        lambda x: preprocess_for_sft(
            x,
            tokenizer,
            sep_token=sep_token_str,
            max_length=max_length,
            dataset_name=dataset_name,
        ),
        batched=True,
        remove_columns=["source", "target"],
    )

    # Data collator handles causal-LM padding + label alignment
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None,
    )

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

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
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
    )

    # ---------- SFTTrainer ----------
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics_generate(
            pred.predictions,
            pred.label_ids,
            tokenizer,
            sep_token=sep_token_str,
        ),
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
        default="your_dataset_name",
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
        default="embedding",
        choices=["embedding", "tfidf", "levenshtein", "title"],
        help="The method to select concept synonyms",
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        lr=args.lr,
        dataset_name=args.dataset_name,
        augmented_data=args.augmented_data,
        selection_method=args.selection_method,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_length=args.max_length,
    )
