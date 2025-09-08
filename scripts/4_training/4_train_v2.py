import argparse
import glob
import pickle
import shutil
import time
from functools import partial
from pathlib import Path

import idr_torch  # type: ignore
import numpy as np
import pynvml
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,  # type: ignore
    Seq2SeqTrainer,  # type: ignore
    Seq2SeqTrainingArguments,  # type: ignore
    TrainerCallback,  # type: ignore
)


# Custom Callback for GPU and performance metrics
class GpuUsageCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            self.device_count = 0
        self.log_history = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            logs = {}

        # Log GPU utilization
        if self.device_count > 0:
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                logs[f"gpu_{i}_utilization_percent"] = utilization.gpu

        # Calculate and log samples per second
        if "loss" in logs and "learning_rate" in logs:  # Training step
            current_time = time.time()
            if "time" in self.log_history and "step" in self.log_history:
                delta_time = current_time - self.log_history["time"]
                delta_steps = state.global_step - self.log_history["step"]
                if delta_time > 0 and delta_steps > 0:
                    samples_processed = (
                        delta_steps
                        * args.per_device_train_batch_size
                        * args.gradient_accumulation_steps
                        * args.world_size
                    )
                    logs["samples_per_second"] = samples_processed / delta_time

            self.log_history["time"] = current_time
            self.log_history["step"] = state.global_step

    def on_train_end(self, args, state, control, **kwargs):
        if self.device_count > 0:
            pynvml.nvmlShutdown()


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
):
    # Initialize Distributed Training if available
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
        f"The model {model_name} will start training with learning rate {lr} on dataset {dataset_name} {'with' if augmented_data else 'without'} augmented data and selection method {selection_method}."
    )
    model_short_name = model_name.split("/")[-1]
    # The Trainer will handle the distributed training setup automatically.
    # No need to manually initialize the process group.
    if model_short_name == "mt5-xl":
        # Remove the DDP initialization completely
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    data_folder = Path("data/final_data")
    with open(
        data_folder
        / dataset_name
        / f"train_{selection_method}_source{with_group_extension}.pkl",
        "rb",
    ) as file:
        train_source_data = pickle.load(file)
    with open(
        data_folder / dataset_name / f"train_{selection_method}_target.pkl",
        "rb",
    ) as file:
        train_target_data = pickle.load(file)
    with open(
        data_folder
        / dataset_name
        / f"validation_{selection_method}_source{with_group_extension}.pkl",
        "rb",
    ) as file:
        validation_source_data = pickle.load(file)
    with open(
        data_folder / dataset_name / f"validation_{selection_method}_target.pkl",
        "rb",
    ) as file:
        validation_target_data = pickle.load(file)

    train_data = {"source": train_source_data, "target": train_target_data}

    validation_data = {
        "source": validation_source_data,
        "target": validation_target_data,
    }

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
                data_folder
                / "SynthMM"
                / f"train_{selection_method}_source{with_group_extension}.pkl",
                "rb",
            ) as file:
                train_generated_source_data = pickle.load(file)
            with open(
                data_folder / "SynthMM" / f"train_{selection_method}_target.pkl",
                "rb",
            ) as file:
                train_generated_target_data = pickle.load(file)
        else:
            with open(
                data_folder
                / "SynthQUAERO"
                / f"train_{selection_method}_source{with_group_extension}.pkl",
                "rb",
            ) as file:
                train_generated_source_data = pickle.load(file)
            with open(
                data_folder / "SynthQUAERO" / f"train_{selection_method}_target.pkl",
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
    train_max_batch = 64
    eval_max_batch = 64
    eval_accumulation_steps = None
    gradient_accumulation_steps = 1
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
    output_dir = (
        Path("models")
        / "NED"
        / f"{dataset_name}_{'augmented' if augmented_data else 'original'}_{selection_method}{'_with_group' if with_group else ''}"
        / model_short_name
    )
    logging_dir = (
        Path("logs")
        / f"{dataset_name}_{'augmented' if augmented_data else 'original'}_{selection_method}{'_with_group' if with_group else ''}"
        / model_short_name
    )
    print(f"BATCH SIZE : {train_max_batch}")
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
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
        callbacks=[GpuUsageCallback()],
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
    )
