import argparse
import csv
import json
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from longbel.utils import add_headers_to_prompt


def load_pickle(file_path: Path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def dump_pickle(data, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def make_folds(num_examples: int, num_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(num_examples)
    rng.shuffle(indices)
    return list(np.array_split(indices, num_folds))


def iter_batches(items: list[str], batch_size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def split_prefix_and_target(target: str, context_format: str) -> tuple[str, str, str]:
    if context_format in ["short", "hybrid_medium"]:
        parts = target.split("}", 1)
        if len(parts) != 2:
            raise ValueError(f"Unexpected target format for {context_format}: {target}")
        prefix = parts[0] + "}"
        return "", prefix, parts[1]

    if context_format in ["hybrid_short", "hybrid_long"]:
        lines = [line for line in target.split("\n") if line.strip()]
        if not lines:
            raise ValueError(f"Unexpected empty target for {context_format}: {target}")
        previous = "\n".join(lines[:-1])
        if previous:
            previous += "\n"
        current = lines[-1]
        parts = current.split("}", 1)
        if len(parts) != 2:
            raise ValueError(f"Unexpected hybrid target format: {target}")
        prefix = parts[0] + "}"
        return previous, prefix, parts[1]

    # long context: model predicts the full target sequence
    return "", "", target


def normalize_generated_suffix(generated_suffix: str, context_format: str) -> str:
    cleaned = generated_suffix.strip()
    if context_format in ["short", "hybrid_medium", "hybrid_short", "hybrid_long"]:
        if "<SEP>" in cleaned:
            cleaned = cleaned.split("<SEP>", 1)[0] + "<SEP>"
        else:
            cleaned = cleaned.splitlines()[0] if cleaned else ""
            if cleaned:
                cleaned += "<SEP>"
    return cleaned


def build_prompt(
    source: str, target: str, context_format: str, add_headers: bool
) -> tuple[str, str, str]:
    previous, prefix, _ = split_prefix_and_target(target, context_format)

    if add_headers:
        prompt, _ = add_headers_to_prompt(source, target, context_format)
    else:
        prompt = source

    return prompt, previous, prefix


def build_predicted_target(
    source: str,
    gold_target: str,
    generated_suffix: str,
    context_format: str,
    add_headers: bool,
) -> str:
    if not add_headers:
        # When headers are disabled, training used full target completion.
        return generated_suffix.strip()

    if context_format == "long":
        return generated_suffix.strip()

    previous, prefix, _ = split_prefix_and_target(gold_target, context_format)
    suffix = normalize_generated_suffix(generated_suffix, context_format)

    if context_format in ["short", "hybrid_medium"]:
        return prefix + suffix

    if context_format in ["hybrid_short", "hybrid_long"]:
        return previous + prefix + suffix

    raise ValueError(f"Unsupported context format: {context_format}")


def resolve_model_dir(
    model_name: str,
    dataset_name: str,
    augmented_data: str,
    selection_method: str,
    context_format: str,
    complete_mode: bool,
    add_headers: bool,
    run_name_suffix: str,
    checkpoint_kind: str,
    models_root: Path,
) -> Path:
    model_short_name = model_name.split("/")[-1]
    complete_mode_str = "_complete" if complete_mode else ""
    add_headers_str = "_addheaders" if add_headers else ""

    model_dir = (
        models_root
        / f"{dataset_name}_{augmented_data}_{selection_method}_{context_format}{complete_mode_str}{add_headers_str}{run_name_suffix}"
        / model_short_name
        / ("model_best" if checkpoint_kind == "best" else "model_last")
    )
    return model_dir


def generate_predictions(
    model_dir: Path,
    sources: list[str],
    targets: list[str],
    context_format: str,
    add_headers: bool,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
) -> list[str]:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)  # type: ignore
    model.eval()

    predictions: list[str] = []

    prompts = [
        build_prompt(src, tgt, context_format=context_format, add_headers=add_headers)[
            0
        ]
        for src, tgt in zip(sources, targets)
    ]

    for batch_start, prompt_batch in enumerate(iter_batches(prompts, batch_size)):
        batch_gold_targets = targets[
            batch_start * batch_size : batch_start * batch_size + len(prompt_batch)
        ]
        batch_sources = sources[
            batch_start * batch_size : batch_start * batch_size + len(prompt_batch)
        ]

        encoded = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_lens = encoded["attention_mask"].sum(dim=1).tolist()

        for i, input_len in enumerate(input_lens):
            generated_ids = outputs[i][int(input_len) :]
            generated_suffix = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            pred_target = build_predicted_target(
                source=batch_sources[i],
                gold_target=batch_gold_targets[i],
                generated_suffix=generated_suffix,
                context_format=context_format,
                add_headers=add_headers,
            )
            predictions.append(pred_target)

    return predictions


def write_fold_training_commands(
    scripts_dir: Path,
    folds_root: Path,
    num_folds: int,
    fixed_validation_source_path: Path,
    fixed_validation_target_path: Path,
    model_name: str,
    lr: float,
    dataset_name: str,
    augmented_data: str,
    selection_method: str,
    context_format: str,
    complete_mode: bool,
    add_headers: bool,
):
    header_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=training       # name of job",
        "#SBATCH -C h100                     # uncomment for gpu_p6 partition (80GB H100 GPU)",
        "#SBATCH --nodes=1                    # nombre de noeud",
        "#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)",
        "#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p5)",
        "#SBATCH --cpus-per-task=24          # number of cores per task for gpu_p6 (1/4 of 4-GPUs H100 node)",
        "#SBATCH --hint=nomultithread         # hyperthreading is deactivated",
        "#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)",
        "#SBATCH --output=logs/log_out%j.out    # name of output file",
        "#SBATCH --error=logs/log_err%j.out     # name of error file (here, in common with the output file)",
        "",
        "set -euo pipefail",
        "",
        "module load arch/h100",
        "module load pytorch-gpu/py3/2.3.1",
        "",
        "export OMP_NUM_THREADS=1",
        "",
    ]

    scripts_dir.mkdir(parents=True, exist_ok=True)
    fold_script_paths: list[Path] = []

    for fold_idx in range(num_folds):
        fold_dir = folds_root / f"fold_{fold_idx}"
        run_suffix = f"_fold{fold_idx}"

        cmd = [
            "python",
            "scripts/4_training_decoder/train.py",
            "--model-name",
            f'"{model_name}"',
            "--lr",
            str(lr),
            "--dataset-name",
            dataset_name,
            "--augmented-data",
            augmented_data,
            "--selection-method",
            selection_method,
            "--context-format",
            context_format,
            "--train-source-path",
            str(fold_dir / "train_source.pkl"),
            "--train-target-path",
            str(fold_dir / "train_target.pkl"),
            "--validation-source-path",
            str(fixed_validation_source_path),
            "--validation-target-path",
            str(fixed_validation_target_path),
            "--run-name-suffix",
            run_suffix,
            "--disable-validation-merge",
        ]

        if complete_mode:
            cmd.append("--complete-mode")
        if add_headers:
            cmd.append("--add-headers")

        fold_lines = header_lines + [" \\\n+    ".join(cmd), ""]
        fold_script_path = scripts_dir / f"run_fold_{fold_idx}.slurm"
        fold_script_path.write_text("\n".join(fold_lines), encoding="utf-8")
        fold_script_paths.append(fold_script_path)

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for fold_script_path in fold_script_paths:
        submit_lines.append(f"sbatch {fold_script_path}")
    submit_lines.append("")

    submit_script_path = scripts_dir / "submit_all_folds.sh"
    submit_script_path.write_text("\n".join(submit_lines), encoding="utf-8")

    return fold_script_paths, submit_script_path


def main(
    model_name: str,
    lr: float,
    dataset_name: str,
    selection_method: str,
    context_format: str,
    augmented_data: str,
    num_folds: int,
    seed: int,
    complete_mode: bool,
    add_headers: bool,
    generate_oof_predictions: bool,
    checkpoint_kind: str,
    models_root: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
):
    data_root = Path("data/final_data") / dataset_name
    train_source_path = (
        data_root / f"train_{selection_method}_source_{context_format}.pkl"
    )
    train_target_path = (
        data_root / f"train_{selection_method}_target_{context_format}.pkl"
    )
    validation_source_path = (
        data_root / f"validation_{selection_method}_source_{context_format}.pkl"
    )
    validation_target_path = (
        data_root / f"validation_{selection_method}_target_{context_format}.pkl"
    )

    if (
        not train_source_path.exists()
        or not train_target_path.exists()
        or not validation_source_path.exists()
        or not validation_target_path.exists()
    ):
        raise FileNotFoundError(
            "Missing required input files:\n"
            f"- {train_source_path}\n"
            f"- {train_target_path}\n"
            f"- {validation_source_path}\n"
            f"- {validation_target_path}"
        )

    train_sources = load_pickle(train_source_path)
    train_targets = load_pickle(train_target_path)
    validation_sources = load_pickle(validation_source_path)
    validation_targets = load_pickle(validation_target_path)

    if len(train_sources) != len(train_targets):
        raise ValueError(
            f"Train source/target length mismatch: {len(train_sources)} != {len(train_targets)}"
        )

    if len(validation_sources) != len(validation_targets):
        raise ValueError(
            "Validation source/target length mismatch: "
            f"{len(validation_sources)} != {len(validation_targets)}"
        )

    # Match train.py behavior: first 90% of validation is merged into train, last 10% is evaluation.
    validation_indexes = list(range(len(validation_sources)))
    validation_split = int(len(validation_sources) * 0.9)
    validation_train_indexes = validation_indexes[:validation_split]
    validation_eval_indexes = validation_indexes[validation_split:]

    validation_train_sources = [validation_sources[i] for i in validation_train_indexes]
    validation_train_targets = [validation_targets[i] for i in validation_train_indexes]
    validation_eval_sources = [validation_sources[i] for i in validation_eval_indexes]
    validation_eval_targets = [validation_targets[i] for i in validation_eval_indexes]

    merged_sources = train_sources + validation_train_sources
    merged_targets = train_targets + validation_train_targets

    folds_root = (
        data_root
        / "oof"
        / f"{selection_method}_{context_format}_k{num_folds}_seed{seed}"
    )
    folds_root.mkdir(parents=True, exist_ok=True)

    fixed_validation_source_path = folds_root / "validation_10pct_source.pkl"
    fixed_validation_target_path = folds_root / "validation_10pct_target.pkl"
    dump_pickle(validation_eval_sources, fixed_validation_source_path)
    dump_pickle(validation_eval_targets, fixed_validation_target_path)

    folds = make_folds(len(merged_sources), num_folds=num_folds, seed=seed)

    all_indices = np.arange(len(merged_sources))
    fold_sizes = []

    for fold_idx, heldout_idx in enumerate(folds):
        heldout_set = set(heldout_idx.tolist())
        train_idx = np.array([i for i in all_indices if i not in heldout_set])

        train_source = [merged_sources[i] for i in train_idx]
        train_target = [merged_targets[i] for i in train_idx]
        heldout_source = [merged_sources[i] for i in heldout_idx]
        heldout_target = [merged_targets[i] for i in heldout_idx]

        fold_dir = folds_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        dump_pickle(train_source, fold_dir / "train_source.pkl")
        dump_pickle(train_target, fold_dir / "train_target.pkl")
        dump_pickle(heldout_source, fold_dir / "heldout_source.pkl")
        dump_pickle(heldout_target, fold_dir / "heldout_target.pkl")
        dump_pickle(heldout_idx.tolist(), fold_dir / "heldout_indices.pkl")

        fold_sizes.append({
            "fold": fold_idx,
            "train_size": int(len(train_idx)),
            "heldout_size": int(len(heldout_idx)),
        })

    scripts_dir = folds_root / "slurm"
    fold_script_paths, submit_script_path = write_fold_training_commands(
        scripts_dir=scripts_dir,
        folds_root=folds_root,
        num_folds=num_folds,
        fixed_validation_source_path=fixed_validation_source_path,
        fixed_validation_target_path=fixed_validation_target_path,
        model_name=model_name,
        lr=lr,
        dataset_name=dataset_name,
        augmented_data=augmented_data,
        selection_method=selection_method,
        context_format=context_format,
        complete_mode=complete_mode,
        add_headers=add_headers,
    )

    metadata = {
        "dataset_name": dataset_name,
        "selection_method": selection_method,
        "context_format": context_format,
        "augmented_data": augmented_data,
        "num_folds": num_folds,
        "seed": seed,
        "train_size": len(train_sources),
        "validation_size": len(validation_sources),
        "validation_merged_90pct_size": len(validation_train_sources),
        "validation_fixed_10pct_size": len(validation_eval_sources),
        "merged_pool_size": len(merged_sources),
        "fixed_validation_source_path": str(fixed_validation_source_path),
        "fixed_validation_target_path": str(fixed_validation_target_path),
        "fold_sizes": fold_sizes,
        "fold_slurm_scripts": [str(path) for path in fold_script_paths],
        "submit_all_script": str(submit_script_path),
    }

    if generate_oof_predictions:
        models_root_path = Path(models_root)
        predicted_targets: list[Optional[str]] = [None] * len(merged_sources)
        fold_reports = []

        for fold_idx in range(num_folds):
            fold_dir = folds_root / f"fold_{fold_idx}"
            heldout_source = load_pickle(fold_dir / "heldout_source.pkl")
            heldout_target = load_pickle(fold_dir / "heldout_target.pkl")
            heldout_indices = load_pickle(fold_dir / "heldout_indices.pkl")

            run_name_suffix = f"_fold{fold_idx}"
            model_dir = resolve_model_dir(
                model_name=model_name,
                dataset_name=dataset_name,
                augmented_data=augmented_data,
                selection_method=selection_method,
                context_format=context_format,
                complete_mode=complete_mode,
                add_headers=add_headers,
                run_name_suffix=run_name_suffix,
                checkpoint_kind=checkpoint_kind,
                models_root=models_root_path,
            )

            print(f"[Fold {fold_idx}] Loading model from: {model_dir}")
            fold_predictions = generate_predictions(
                model_dir=model_dir,
                sources=heldout_source,
                targets=heldout_target,
                context_format=context_format,
                add_headers=add_headers,
                batch_size=batch_size,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
            )

            if len(fold_predictions) != len(heldout_indices):
                raise RuntimeError(
                    f"Fold {fold_idx} prediction size mismatch: {len(fold_predictions)} vs {len(heldout_indices)}"
                )

            dump_pickle(fold_predictions, fold_dir / "heldout_pred_target.pkl")

            for idx, pred_target in zip(heldout_indices, fold_predictions):
                predicted_targets[idx] = pred_target

            fold_reports.append({
                "fold": fold_idx,
                "heldout_size": len(heldout_indices),
                "model_dir": str(model_dir),
            })

        missing = [i for i, value in enumerate(predicted_targets) if value is None]
        if missing:
            raise RuntimeError(f"Missing OOF predictions for {len(missing)} indices")

        oof_source_path = (
            data_root
            / f"train_{selection_method}_source_{context_format}_oof{num_folds}.pkl"
        )
        oof_target_path = (
            data_root
            / f"train_{selection_method}_target_{context_format}_oof{num_folds}_pred.pkl"
        )
        oof_gold_target_path = (
            data_root
            / f"train_{selection_method}_target_{context_format}_oof{num_folds}_gold.pkl"
        )

        dump_pickle(merged_sources, oof_source_path)
        dump_pickle(predicted_targets, oof_target_path)
        dump_pickle(merged_targets, oof_gold_target_path)

        tsv_path = (
            data_root
            / f"train_{selection_method}_{context_format}_oof{num_folds}_predictions.tsv"
        )
        with open(tsv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["index", "source", "gold_target", "pred_target"],
                delimiter="\t",
            )
            writer.writeheader()
            for idx, (src, gold, pred) in enumerate(
                zip(merged_sources, merged_targets, predicted_targets)
            ):
                writer.writerow({
                    "index": idx,
                    "source": src,
                    "gold_target": gold,
                    "pred_target": pred,
                })

        metadata["oof_prediction"] = {
            "checkpoint_kind": checkpoint_kind,
            "batch_size": batch_size,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
            "models_root": str(models_root_path),
            "fold_reports": fold_reports,
            "oof_source_path": str(oof_source_path),
            "oof_target_path": str(oof_target_path),
            "oof_gold_target_path": str(oof_gold_target_path),
            "tsv_path": str(tsv_path),
        }

    metadata_path = folds_root / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(f"Saved fold files in: {folds_root}")
    print(f"Saved fold SLURM scripts in: {scripts_dir}")
    print(f"Saved submit-all script in: {submit_script_path}")
    if generate_oof_predictions:
        print("OOF prediction dataset created successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create K-fold train splits for OOF memory dataset creation, and optionally "
            "run held-out predictions with fold-specific models to build a full predicted train set."
        )
    )
    parser.add_argument("--model-name", type=str, required=True, help="Base model name")
    parser.add_argument(
        "--lr", type=float, default=3e-5, help="Training LR for generated command file"
    )
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--selection-method",
        type=str,
        default="tfidf",
        choices=["embedding", "tfidf", "levenshtein", "title"],
        help="Selection method used in train pickle filenames",
    )
    parser.add_argument(
        "--context-format",
        type=str,
        default="long",
        choices=["short", "long", "hybrid_short", "hybrid_long", "hybrid_medium"],
        help="Context format used in train pickle filenames",
    )
    parser.add_argument(
        "--augmented-data",
        type=str,
        default="human_only",
        choices=["human_only", "human_only_ft", "synth_only", "full", "full_upsampled"],
        help="Augmented data setting used in fold model output path",
    )
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for fold split"
    )
    parser.add_argument(
        "--complete-mode",
        action="store_true",
        help="Use complete mode in path/commands",
    )
    parser.add_argument(
        "--add-headers", action="store_true", help="Use header mode in path/commands"
    )

    parser.add_argument(
        "--generate-oof-predictions",
        action="store_true",
        help="If set, load fold models and generate held-out predictions to build OOF dataset.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        type=str,
        default="last",
        choices=["best", "last"],
        help="Which fold checkpoint to use for OOF prediction generation.",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="models/NED",
        help="Root folder containing fold model outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size for OOF generation",
    )
    parser.add_argument(
        "--num-beams", type=int, default=1, help="Number of beams for OOF generation"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max generated tokens for each example",
    )

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        lr=args.lr,
        dataset_name=args.dataset_name,
        selection_method=args.selection_method,
        context_format=args.context_format,
        augmented_data=args.augmented_data,
        num_folds=args.num_folds,
        seed=args.seed,
        complete_mode=args.complete_mode,
        add_headers=args.add_headers,
        generate_oof_predictions=args.generate_oof_predictions,
        checkpoint_kind=args.checkpoint_kind,
        models_root=args.models_root,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
