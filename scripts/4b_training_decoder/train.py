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


def create_prompt_completion_dataset(dataset, dataset_name):
    # Determine verb based on dataset
    verb = "est"
    if dataset_name == "MedMentions":
        verb = "is"
    elif dataset_name == "SPACCC":
        verb = "es"

    split_marker = f"] {verb}"

    def format_example(example):
        prompts = []
        completions = []
        for s, t in zip(example["source"], example["target"]):
            try:
                split_pos = t.find(split_marker)
                if split_pos == -1:
                    # If marker is not found, treat the whole target as completion
                    prompt = f"{s}<SEP>{t}"
                    completion = ""
                    print(
                        f"Warning: split_marker '{split_marker}' not found in target: {t}"
                    )
                else:
                    target_prefix = t[: split_pos + len(split_marker)]
                    prompt = f"{s}<SEP>{target_prefix}"
                    completion = t[split_pos + len(split_marker) :]
                prompts.append(prompt)
                completions.append(completion)
            except Exception:
                # Fallback for any unexpected error
                prompts.append(f"{s}<SEP>{t}")
                completions.append("")
        return {"prompt": prompts, "completion": completions}

    return dataset.map(
        format_example,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Formatting {dataset_name} dataset",
    )


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
    human_ratio: float = 1.0,
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
                Path("models")
                / "NED"
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
    if max_length is None or max_length > 100_000:
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
        # Reduce train dataset according to human_ratio
        if human_ratio < 1.0:
            indexes = list(range(len(human_train_dataset)))
            split = int(len(human_train_dataset) * human_ratio)
            human_train_dataset = human_train_dataset.select(indexes[:split])

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

    # Choose train_dataset accordingly (same logic as before)
    if augmented_data in ["synth_only", "full", "full_upsampled"]:
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
                human_train_dataset,
                synth_train_dataset,
            ])  # type: ignore
        else:  # full_upsampled
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

    # Format datasets into prompt/completion format
    train_dataset = create_prompt_completion_dataset(train_dataset, dataset_name)
    validation_dataset = create_prompt_completion_dataset(
        validation_dataset, dataset_name
    )

    # ---------- SFT TrainingArguments ----------
    if human_ratio < 1.0:
        human_ratio_str = (
            "_" + str(round(human_ratio * 100, 0)).replace(".0", "") + "pct"
        )
    else:
        human_ratio_str = ""
    output_dir = (
        Path("/lustre/fsn1/projects/rech/ssq/usk98ia/expe_data_ratio")
        / f"{dataset_name}_{augmented_data}_{selection_method}{human_ratio_str}"
        / model_short_name
    )
    logging_dir = (
        Path("/lustre/fsn1/projects/rech/ssq/usk98ia/logs")
        / f"{dataset_name}_{augmented_data}_{selection_method}{human_ratio_str}"
        / model_short_name
    )
    # output_dir = (
    #     Path("models")
    #     / "NED"
    #     / f"{dataset_name}_{augmented_data}_{selection_method}"
    #     / model_short_name
    # )
    # logging_dir = (
    #     Path("logs")
    #     / f"{dataset_name}_{augmented_data}_{selection_method!}"
    #     / model_short_name
    # )
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
        completion_only_loss=True,
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
        ],
        help="Whether to use augmented data for training",
    )
    parser.add_argument(
        "--human-ratio",
        type=float,
        default=1.0,
        help="The ratio of augmented data to use for training",
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
        human_ratio=args.human_ratio,
        selection_method=args.selection_method,
    )
