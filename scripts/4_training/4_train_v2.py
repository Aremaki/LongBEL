import argparse
import glob
import pickle
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,  # type: ignore
    Seq2SeqTrainer,  # type: ignore
    Seq2SeqTrainingArguments,  # type: ignore
)


# Preprocess function for tokenization
def preprocess_function(examples, tokenizer):
    """
    Tokenizes the source and target texts in the examples dictionary using the provided tokenizer.

    Args:
        examples (dict): A dictionary containing 'source' and 'target' keys with lists of texts.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for encoding the texts.

    Returns:
        dict: A dictionary containing tokenized inputs and labels suitable for seq2seq training.
    """
    inputs = tokenizer(examples["source"], padding="longest", return_tensors="pt")
    targets = tokenizer(examples["target"], padding="longest", return_tensors="pt")

    labels = targets["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


# Define evaluation metrics
def skip_undesired_tokens(outputs, tokenizer):
    tokens_to_remove = tokenizer.all_special_tokens
    cleaned_outputs = []
    for sequence in outputs:
        for token in tokens_to_remove:
            sequence = sequence.replace(token, "")  # Remove unwanted special tokens
        cleaned_outputs.append(sequence.strip())
    return cleaned_outputs


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(
        preds, skip_special_tokens=False, clean_up_tokenization_spaces=True
    )
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=False, clean_up_tokenization_spaces=True
    )

    # Skip undesired tokens
    decoded_preds = skip_undesired_tokens(decoded_preds, tokenizer)
    decoded_labels = skip_undesired_tokens(decoded_labels, tokenizer)

    # Compute recall
    recall = sum(
        pred == gold for pred, gold in zip(decoded_preds, decoded_labels)
    ) / len(decoded_labels)

    return {
        "recall": round(recall, 4),
        "num_gold": len(decoded_labels),
        "num_guess": len(decoded_preds),
    }


def main(
    model_name: str,
    lr: float,
    dataset_name: str,
    augmented_data: bool,
    with_group: bool = False,
    selection_method: str = "embedding",
    start_mention: str = "[",
    end_mention: str = "]",
):
    print(
        f"The model {model_name} will start training with learning rate {lr} on dataset {dataset_name} {'with' if augmented_data else 'without'} augmented data and selection method {selection_method}."
    )
    model_short_name = model_name.split("/")[-1]
    # Initialize Distributed Training
    # The Trainer will handle the distributed training setup automatically.
    # No need to manually initialize the process group.
    if model_short_name == "mt5-xl":
        # Remove the DDP initialization completely
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=False)
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    start_mention,
                    end_mention,
                ]
            },
            replace_additional_special_tokens=False,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as hub_err:  # pragma: no cover - network / availability branch
        local_dir = Path("models") / str(model_name)
        if local_dir.exists():
            print(
                f"⚠️ Hub load failed for '{model_name}' ({hub_err}); falling back to local path {local_dir}."
            )
            tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(local_dir))
        else:
            raise hub_err

    # Make sure all model parameters are contiguous on memory. This is necessary to save model state dict as safetensor
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            print(f"Parameter {name} is not contiguous. Making it contiguous.")
            param.data = param.data.contiguous()

    # Load and preprocess data
    with_group_extension = "_with_group" if with_group else ""
    data_folder = Path("/data/final_data")
    with open(
        data_folder / dataset_name / f"train_source{with_group_extension}.pkl", "rb"
    ) as file:
        train_source_data = pickle.load(file)
    with open(
        data_folder / dataset_name / f"train_target{with_group_extension}.pkl", "rb"
    ) as file:
        train_target_data = pickle.load(file)
    with open(
        data_folder / dataset_name / f"dev_source{with_group_extension}.pkl", "rb"
    ) as file:
        dev_source_data = pickle.load(file)
    with open(
        data_folder / dataset_name / f"dev_target{with_group_extension}.pkl", "rb"
    ) as file:
        dev_target_data = pickle.load(file)

    train_data = {"source": train_source_data, "target": train_target_data}

    validation_data = {"source": dev_source_data, "target": dev_target_data}

    validation_dataset = Dataset.from_dict(validation_data)
    indexes = list(range(len(validation_dataset)))
    split = int(len(validation_dataset) * 0.9)
    split_train = indexes[:split]
    split_val = indexes[split:]
    train_dataset = concatenate_datasets([
        Dataset.from_dict(train_data),
        validation_dataset.select(split_train),
    ])
    validation_dataset = validation_dataset.select(split_val)

    if augmented_data:
        if dataset_name == "MedMentions":
            with open(
                data_folder / "SynthMM" / f"train_source{with_group_extension}.pkl",
                "rb",
            ) as file:
                train_generated_source_data = pickle.load(file)
            with open(
                data_folder / "SynthMM" / f"train_target{with_group_extension}.pkl",
                "rb",
            ) as file:
                train_generated_target_data = pickle.load(file)
        else:
            with open(
                data_folder / "SynthQUAERO" / f"train_source{with_group_extension}.pkl",
                "rb",
            ) as file:
                train_generated_source_data = pickle.load(file)
            with open(
                data_folder / "SynthQUAERO" / f"train_target{with_group_extension}.pkl",
                "rb",
            ) as file:
                train_generated_target_data = pickle.load(file)
        train_generated_data = {
            "source": train_generated_source_data,
            "target": train_generated_target_data,
        }
        train_dataset = concatenate_datasets([
            train_dataset,
            Dataset.from_dict(train_generated_data),
        ])

    tokenized_datasets = {
        "train": train_dataset.map(
            lambda x: preprocess_function(x, tokenizer), batched=True
        ),
        "validation": validation_dataset.map(
            lambda x: preprocess_function(x, tokenizer), batched=True
        ),
    }

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training params
    train_max_batch = 6
    eval_max_batch = 6
    eval_accumulation_steps = None
    gradient_accumulation_steps = 2
    ddp_backend = "nccl"  # Enable Distributed Data Parallel with NCCL backend
    auto_find_batch_size = False
    fsdp = ""
    fsdp_transformer_layer_cls_to_wrap = None
    gradient_checkpointing = False
    if model_short_name == "mt5-xl":
        fsdp = ["full_shard", "auto_wrap"]
        fsdp_transformer_layer_cls_to_wrap = "MT5Block"
        ddp_backend = None  # Enable Distributed Data Parallel with NCCL backend
        train_max_batch = 4
        eval_max_batch = 6
        gradient_accumulation_steps = 2
        gradient_checkpointing = True
        eval_accumulation_steps = (
            int(len(validation_dataset) / (eval_max_batch * 2)) + 2
        )
    output_dir = f"/models/NED/{dataset_name}_{'augmented' if augmented_data else 'original'}_{selection_method}{with_group_extension}/{model_short_name}"
    print(f"BATCH SIZE : {train_max_batch}")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        logging_dir=f"data/logs/{dataset_name}_{'augmented' if augmented_data else 'original'}_{selection_method}{with_group_extension}/{model_short_name}",
        logging_strategy="epoch",
        report_to="tensorboard",
        eval_strategy="epoch",
        save_strategy="epoch",
        auto_find_batch_size=auto_find_batch_size,
        per_device_train_batch_size=train_max_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=eval_max_batch,
        eval_accumulation_steps=eval_accumulation_steps,
        save_total_limit=2,
        num_train_epochs=150,
        predict_with_generate=True,
        fsdp=fsdp,  # type: ignore
        fsdp_transformer_layer_cls_to_wrap=fsdp_transformer_layer_cls_to_wrap,
        bf16=True,
        bf16_full_eval=True,
        label_smoothing_factor=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_recall",
        greater_is_better=True,
        # eval_on_start=True,
        seed=42,
        warmup_steps=500,
        learning_rate=lr,
        lr_scheduler_type="linear",
        gradient_checkpointing=gradient_checkpointing,
        ddp_backend=ddp_backend,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,  # type: ignore
        data_collator=data_collator,
        compute_metrics=partial(
            compute_metrics,
            tokenizer=tokenizer,
        ),
    )

    resume_from_checkpoint = len(glob.glob(rf"{output_dir}/checkpoint-*")) > 0
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the best model
    best_model_chkpt_dir = Path(trainer.state.best_model_checkpoint)  # type: ignore
    model_dir = best_model_chkpt_dir.parent
    best_model_dir = model_dir / "model_best"
    shutil.copytree(best_model_chkpt_dir, best_model_dir, dirs_exist_ok=True)
    best_step = str(trainer.state.best_model_checkpoint).split("-")[-1]
    print(f"Best model saved at step {best_step}")

    # Save the last model
    last_model_chkpt_dir = model_dir / f"checkpoint-{trainer.state.global_step}"
    last_model_dir = model_dir / "model_last"
    shutil.copytree(last_model_chkpt_dir, last_model_dir, dirs_exist_ok=True)
    last_step = str(trainer.state.global_step)
    print(f"Last model saved at step {last_step}")


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="A script for training seq2seq model")
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
        action="store_true",
        help="Whether to use augmented data for training",
    )
    parser.add_argument(
        "--with-group",
        action="store_true",
        help="Whether to use data with group annotations for training",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default="embedding",
        choices=["embedding", "tfidf", "levenshtein"],
        help="The method to select concept synonyms",
    )
    parser.add_argument(
        "--start-mention",
        type=str,
        default="[",
        help="The token to indicate the start of a mention",
    )
    parser.add_argument(
        "--end-mention",
        type=str,
        default="]",
        help="The token to indicate the end of a mention",
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # Pass the parsed argument to the main function
    main(
        args.model_name,
        args.lr,
        args.dataset_name,
        args.augmented_data,
        args.with_group,
        args.selection_method,
        args.start_mention,
        args.end_mention,
    )
