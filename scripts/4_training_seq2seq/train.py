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
from datasets import Dataset, concatenate_datasets, interleave_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,  # type: ignore
    Seq2SeqTrainer,  # type: ignore
    Seq2SeqTrainingArguments,  # type: ignore
    TrainerCallback,  # type: ignore
)


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


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
    Tokenizes the source and target texts for seq2seq training.

    Args:
        examples (dict): Dictionary with 'source' and 'target' keys.
        tokenizer (PreTrainedTokenizer): Tokenizer instance.

    Returns:
        dict: Encoded inputs with labels (labels = target ids, padded and masked with -100).
    """
    # Encode source
    model_inputs = tokenizer(
        examples["source"],
    )

    # Encode targets
    labels = tokenizer(
        text_target=examples["target"],
    )["input_ids"]

    # Replace padding token id's in labels with -100
    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]
    model_inputs["labels"] = labels

    # --- Debug: print max tokenized length for this batch ---
    source_lengths = [len(ids) for ids in model_inputs["input_ids"]]
    target_lengths = [len(ids) for ids in labels]
    print(
        f"Longest source: {max(source_lengths)} tokens | Longest target: {max(target_lengths)} tokens"
    )

    return model_inputs


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
    recall = np.mean([
        pred.strip() == gold.strip()
        for pred, gold in zip(decoded_preds, decoded_labels)
    ])

    return {
        "recall": round(recall, 4),
        "num_gold": len(decoded_labels),
        "num_guess": len(decoded_preds),
    }


def main(
    model_name: str,
    lr: float,
    dataset_name: str,
    augmented_data: str,
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
        f"The model {model_name} will start training with learning rate {lr} on dataset {augmented_data} {dataset_name} and selection method {selection_method}."
    )
    model_short_name = model_name.split("/")[-1]

    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as hub_err:  # pragma: no cover - network / availability branch
        if augmented_data == "human_only_ft":
            local_dir = (
                Path("models")
                / "NED"
                / f"{dataset_name}_synth_only_{selection_method}{'_with_group' if with_group else ''}"
                / model_short_name
                / "model_last"
            )
        else:
            local_dir = Path("models") / str(model_name)
        if local_dir.exists():
            print(
                f"⚠️ Hub load failed for '{model_name}' ({hub_err}); falling back to local path {local_dir}."
            )
            tokenizer = AutoTokenizer.from_pretrained(str(local_dir))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(local_dir))
        else:
            raise hub_err

    # Unconditionally use <SEP> as the sep_token everywhere and resize embeddings if added
    sep_token_str = "<SEP>"
    num_added_tokens = 0
    num_added_tokens = tokenizer.add_special_tokens({"sep_token": sep_token_str})
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        print("Added <SEP> as sep_token and resized token embeddings.")
    else:
        print("Set sep_token to <SEP> (no new tokens added).")

    # Make sure all model parameters are contiguous on memory. This is necessary to save model state dict as safetensor
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            print(f"Parameter {name} is not contiguous. Making it contiguous.")
            param.data = param.data.contiguous()

    # Set tokenizer lang for mBart model
    if "mbart" in model_short_name:
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
    # Load and preprocess data
    with_group_extension = "_with_group" if with_group else ""
    data_folder = Path("data/final_data")
    validation_source_path = (
        data_folder
        / dataset_name
        / f"validation_{selection_method}_source{with_group_extension}.pkl"
    )
    if validation_source_path.exists():
        split_name = "validation"
    else:
        split_name = "test"
        print("Validation file not found, using test file instead.")

    validation_source_data = load_pickle(
        data_folder
        / dataset_name
        / f"{split_name}_{selection_method}_source{with_group_extension}.pkl"
    )
    validation_target_data = load_pickle(
        data_folder / dataset_name / f"{split_name}_{selection_method}_target.pkl"
    )

    validation_data = {
        "source": validation_source_data,
        "target": validation_target_data,
    }

    dev_dataset = Dataset.from_dict(validation_data)

    # Reduce validation dataset to 10% to save time during training
    indexes = list(range(len(dev_dataset)))
    split = int(len(dev_dataset) * 0.9)
    split_val = indexes[split:]
    validation_dataset = dev_dataset.select(split_val)

    if augmented_data in ["human_only", "full", "human_only_ft", "full_upsampled"]:
        human_dataset_name = "SPACCC" if "SPACCC" in dataset_name else dataset_name
        human_train_source_data = load_pickle(
            data_folder
            / human_dataset_name
            / f"train_{selection_method}_source{with_group_extension}.pkl"
        )
        human_train_target_data = load_pickle(
            data_folder / human_dataset_name / f"train_{selection_method}_target.pkl"
        )
        human_train_data = {
            "source": human_train_source_data,
            "target": human_train_target_data,
        }
        human_train_dataset = Dataset.from_dict(human_train_data)
        # if validation add the rest to training dataset
        if split_name == "validation":
            split_train = indexes[:split]
            human_train_dataset = concatenate_datasets([
                human_train_dataset,
                dev_dataset.select(split_train),
            ])

    if augmented_data in ["synth_only", "full", "full_upsampled"]:
        if dataset_name == "MedMentions":
            synth_train_source_data = load_pickle(
                data_folder
                / "SynthMM"
                / f"train_{selection_method}_source{with_group_extension}.pkl"
            )
            synth_train_target_data = load_pickle(
                data_folder / "SynthMM" / f"train_{selection_method}_target.pkl"
            )
        elif dataset_name == "QUAERO":
            synth_train_source_data = load_pickle(
                data_folder
                / "SynthQUAERO"
                / f"train_{selection_method}_source{with_group_extension}.pkl"
            )
            synth_train_target_data = load_pickle(
                data_folder / "SynthQUAERO" / f"train_{selection_method}_target.pkl"
            )
        elif dataset_name == "SPACCC":
            synth_train_source_data = load_pickle(
                data_folder
                / "SynthSPACCC"
                / f"train_{selection_method}_source{with_group_extension}.pkl"
            )
            synth_train_target_data = load_pickle(
                data_folder / "SynthSPACCC" / f"train_{selection_method}_target.pkl"
            )
        else:  # SPACCC_UMLS
            source_no_def = load_pickle(
                data_folder
                / "SynthSPACCC_UMLS_No_Def"
                / f"train_{selection_method}_source{with_group_extension}.pkl"
            )
            source_def = load_pickle(
                data_folder
                / "SynthSPACCC_UMLS_Def"
                / f"train_{selection_method}_source{with_group_extension}.pkl"
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
            synth_def = {
                "source": source_def,
                "target": target_def,
            }
            synth_no_def = {
                "source": source_no_def,
                "target": target_no_def,
            }
            synth_def_dataset = Dataset.from_dict(synth_def)
            synth_no_def_dataset = Dataset.from_dict(synth_no_def)
        synth_train_data = {
            "source": synth_train_source_data,
            "target": synth_train_target_data,
        }
        synth_train_dataset = Dataset.from_dict(synth_train_data)

    if augmented_data in ["synth_only", "full", "full_upsampled"]:
        save_strategy = "steps"
        save_steps = 20000
        eval_strategy = "steps"
        eval_steps = 20000
        logging_strategy = "steps"
        logging_steps = 20000
        num_train_epochs = 3
        if augmented_data == "synth_only":
            train_dataset = synth_train_dataset  # type: ignore
        elif augmented_data == "full":
            num_train_epochs = 5
            train_dataset = concatenate_datasets([
                human_train_dataset,  # type: ignore
                synth_train_dataset,  # type: ignore
            ])
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
        save_steps = 0  # Not used when strategy is epoch
        eval_strategy = "epoch"
        eval_steps = 0
        logging_strategy = "epoch"
        logging_steps = 0
        train_dataset = human_train_dataset  # type: ignore
        if augmented_data == "human_only":
            num_train_epochs = 200
        else:  # human_only_ft
            num_train_epochs = 70
            lr = lr / 3

    tokenized_datasets = {
        "train": train_dataset.map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=["source", "target"],
        ),
        "validation": validation_dataset.map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=["source", "target"],
        ),
    }
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # Training params
    train_max_batch = 16
    eval_max_batch = 16
    eval_accumulation_steps = None
    gradient_accumulation_steps = 1
    ddp_backend = "nccl"  # Enable Distributed Data Parallel with NCCL backend
    auto_find_batch_size = False
    fsdp = ""
    fsdp_transformer_layer_cls_to_wrap = None
    gradient_checkpointing = False

    output_dir = (
        Path("models")
        / "NED"
        / f"{dataset_name}_{augmented_data}_{selection_method}{'_with_group' if with_group else ''}"
        / model_short_name
    )
    logging_dir = (
        Path("logs")
        / f"{dataset_name}_{augmented_data}_{selection_method}{'_with_group' if with_group else ''}"
        / model_short_name
    )

    print(f"BATCH SIZE : {train_max_batch}")
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        report_to="tensorboard",
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        auto_find_batch_size=auto_find_batch_size,
        per_device_train_batch_size=train_max_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=eval_max_batch,
        eval_accumulation_steps=eval_accumulation_steps,
        save_total_limit=2,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        fsdp=fsdp,  # type: ignore
        fsdp_transformer_layer_cls_to_wrap=fsdp_transformer_layer_cls_to_wrap,
        bf16=True,
        bf16_full_eval=True,
        label_smoothing_factor=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_recall",
        greater_is_better=True,
        eval_on_start=True,
        seed=42,
        warmup_ratio=0.03,
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
        "--with-group",
        action="store_true",
        help="Whether to use data with group annotations for training",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default="embedding",
        choices=["embedding", "tfidf", "levenshtein", "title"],
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
