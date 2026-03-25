import argparse
import copy
import gc
import json
import os
import pickle
from pathlib import Path

import nltk
import numpy as np
import polars as pl
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from longbel.longbel import LongBEL
from longbel.parse_data import parse_text_hybrid_long, parse_text_hybrid_short
from longbel.trie import Trie


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


def get_dataset_short_and_lang(dataset_name: str) -> tuple[str, str]:
    if dataset_name == "MedMentions":
        return "MM", "en"
    if dataset_name in ["EMEA", "MEDLINE"]:
        return "QUAERO", "fr"
    if "SPACCC" in dataset_name:
        return "SPACCC", "es"
    return dataset_name, "en"


def ensure_inference_resources(dataset_name: str, model_name: str):
    dataset_short, _ = get_dataset_short_and_lang(dataset_name)

    candidate_tries_folder = Path("data/candidate_tries")
    candidate_tries_folder.mkdir(parents=True, exist_ok=True)
    candidate_trie_path = (
        candidate_tries_folder / f"trie_{dataset_short}_{model_name}.pkl"
    )

    text_to_codes_folder = Path("data/text_to_codes")
    text_to_codes_folder.mkdir(parents=True, exist_ok=True)
    text_to_code_path = text_to_codes_folder / f"text_to_code_{dataset_short}.json"

    if candidate_trie_path.exists() and text_to_code_path.exists():
        return text_to_code_path, candidate_trie_path

    umls_path = (
        Path("data") / "termino_processed" / dataset_short / "all_disambiguated.parquet"
    )
    umls_df = pl.read_parquet(umls_path)
    umls_df = umls_df.with_columns(
        pl.col("Entity")
        .str.replace_all("\xa0", " ", literal=True)
        .str.replace_all(r"[\{\[]", "(", literal=False)
        .str.replace_all(r"[\}\]]", ")", literal=False)
    )

    if not text_to_code_path.exists():
        code_col = "SNOMED_code" if "SNOMED_code" in umls_df.columns else "CUI"
        result = umls_df.group_by("GROUP").agg([
            pl.col("Entity"),
            pl.col(code_col),
        ])

        text_to_code = {
            row["GROUP"]: dict(zip(row["Entity"], row[code_col]))
            for row in result.to_dicts()
        }
        with open(text_to_code_path, "w", encoding="utf-8") as f:
            json.dump(text_to_code, f, ensure_ascii=False, indent=2)

    if not candidate_trie_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        start_idx = 1
        prefix = "} "
        candidate_trie = {}
        for group in umls_df["GROUP"].unique().to_list():  # type: ignore
            group_umls_df = umls_df.filter(pl.col("GROUP") == group)  # type: ignore
            sequences = []
            for entity in group_umls_df["Entity"].to_list():
                sequence = tokenizer.encode(prefix + entity)[start_idx:]
                if sequence[-1] != tokenizer.eos_token_id:
                    sequence.append(tokenizer.eos_token_id)
                sequences.append(sequence)
            candidate_trie[group] = Trie(sequences)

        with open(candidate_trie_path, "wb") as file:
            pickle.dump(candidate_trie, file, protocol=-1)

    return text_to_code_path, candidate_trie_path


def generate_predictions_bigbio(
    model_dir: Path,
    dataset_name: str,
    heldout_pages,
    context_format: str,
    batch_size: int,
    num_beams: int,
) -> tuple[list[dict[str, object]], Dataset]:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_dir}")

    _, lang = get_dataset_short_and_lang(dataset_name)
    text_to_code_path, candidate_trie_path = ensure_inference_resources(
        dataset_name,
        model_dir.parent.name,
    )
    multiple_answers = dataset_name == "SPACCC"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        LongBEL.from_pretrained(
            str(model_dir),
            lang=lang,
            text_to_code_path=text_to_code_path,
            candidate_trie_path=candidate_trie_path,
        )
        .eval()
        .to(device)  # type: ignore
    )

    preds = model.sample(  # type: ignore
        bigbio_pages=heldout_pages,
        batch_size=batch_size,
        num_beams=num_beams,
        constrained=True,
        multiple_answers=multiple_answers,
        context_format=context_format,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    top1_by_mention: dict[str, tuple[str, str]] = {}
    for row in preds:
        rank = row.get("rank", 0)  # type: ignore
        if rank != 1:
            continue
        mention_id = str(row.get("mention_id", ""))  # type: ignore
        pred_name = str(row.get("pred_concept_name", ""))  # type: ignore
        pred_code = str(row.get("pred_concept_code", ""))  # type: ignore
        if mention_id:
            top1_by_mention[mention_id] = (pred_name, pred_code)

    heldout_pages_with_pred = []
    for page in heldout_pages:
        page_copy = copy.deepcopy(page)
        entities = page_copy.get("entities", [])
        for entity in entities:
            mention_id = entity["id"]
            pred_name, pred_code = top1_by_mention.get(mention_id, ("", ""))
            normalized = entity.get("normalized") or [{}]
            normalized[0]["db_pred_match"] = pred_name
            normalized[0]["db_pred_id"] = pred_code
            entity["normalized"] = normalized

        page_copy["entities"] = entities
        heldout_pages_with_pred.append(page_copy)

    return preds, Dataset.from_list(heldout_pages_with_pred)  # type: ignore


def build_hybrid_short_training_pairs_from_predictions(
    heldout_pages: Dataset,
    dataset_name: str,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    if dataset_name == "MedMentions":
        nlp = nltk.data.load("tokenizers/punkt/english.pickle")
    elif dataset_name == "SPACCC":
        nlp = nltk.data.load("tokenizers/punkt/spanish.pickle")
    elif dataset_name in ["EMEA", "MEDLINE"]:
        nlp = nltk.data.load("tokenizers/punkt/french.pickle")
    else:
        nlp = nltk.data.load("tokenizers/punkt/english.pickle")

    target_data: list[str] = []
    source_data: list[str] = []
    tsv_data: list[dict[str, str]] = []
    for page in heldout_pages:
        sources, targets, tsv_lines = parse_text_hybrid_short(  # type: ignore
            data=page,
            start_entity="[",
            end_entity="]",
            start_group="{",
            end_group="}",
            nlp=nlp,
            train_mode=False,
            train_mode_pred=True,
        )
        target_data.extend(targets)
        source_data.extend(sources)
        tsv_data.extend(tsv_lines)

    return source_data, target_data, tsv_data


def build_hybrid_long_training_pairs_from_predictions(
    heldout_pages: Dataset,
    dataset_name: str,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    if dataset_name == "MedMentions":
        nlp = nltk.data.load("tokenizers/punkt/english.pickle")
    elif dataset_name == "SPACCC":
        nlp = nltk.data.load("tokenizers/punkt/spanish.pickle")
    elif dataset_name in ["EMEA", "MEDLINE"]:
        nlp = nltk.data.load("tokenizers/punkt/french.pickle")
    else:
        nlp = nltk.data.load("tokenizers/punkt/english.pickle")

    target_data: list[str] = []
    source_data: list[str] = []
    tsv_data: list[dict[str, str]] = []
    for page in heldout_pages:
        sources, targets, tsv_lines = parse_text_hybrid_long(  # type: ignore
            data=page,
            start_entity="[",
            end_entity="]",
            start_group="{",
            end_group="}",
            nlp=nlp,
            train_mode=False,
            train_mode_pred=True,
        )
        target_data.extend(targets)
        source_data.extend(sources)
        tsv_data.extend(tsv_lines)

    return source_data, target_data, tsv_data


def write_fold_training_commands(
    scripts_dir: Path,
    folds_root: Path,
    num_folds: int,
    fixed_validation_source_path: Path,
    fixed_validation_target_path: Path,
    models_root: str,
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
            "--models-root",
            models_root,
        ]

        if complete_mode:
            cmd.append("--complete-mode")
        if add_headers:
            cmd.append("--add-headers")

        fold_lines = header_lines + [" \\\n    ".join(cmd), ""]
        fold_script_path = scripts_dir / f"run_fold_{fold_idx}.slurm"
        fold_script_path.write_text("\n".join(fold_lines), encoding="utf-8")
        fold_script_paths.append(fold_script_path)

    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for fold_script_path in fold_script_paths:
        submit_lines.append(f"sbatch -A ssq@h100 {fold_script_path}")
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
    validation_predict_fold: int,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
):
    data_root = Path("data/final_data") / dataset_name
    bigbio_processed_root = data_root / "bigbio_dataset"
    if not bigbio_processed_root.exists():
        raise FileNotFoundError(
            f"BigBio processed dataset not found at {bigbio_processed_root}"
        )
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
    bigbio_train_pages = load_dataset(
        str(bigbio_processed_root),
        data_dir="processed_data",
        split="train",
    )
    bigbio_validation_pages = load_dataset(
        str(bigbio_processed_root),
        data_dir="processed_data",
        split="validation",
    )

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
    bigbio_validation_train = bigbio_validation_pages.select(validation_train_indexes)
    bigbio_validation_eval = bigbio_validation_pages.select(validation_eval_indexes)

    merged_sources = train_sources + validation_train_sources
    merged_targets = train_targets + validation_train_targets
    merged_bigbio = concatenate_datasets([bigbio_train_pages, bigbio_validation_train])

    folds_root = (
        data_root
        / "oof"
        / f"{selection_method}_{context_format}_k{num_folds}_seed{seed}"
    )
    folds_root.mkdir(parents=True, exist_ok=True)

    fixed_validation_source_path = folds_root / "validation_10pct_source.pkl"
    fixed_validation_target_path = folds_root / "validation_10pct_target.pkl"
    validation_10pct_bigbio_path = folds_root / "validation_10pct_bigbio.parquet"
    dump_pickle(validation_eval_sources, fixed_validation_source_path)
    dump_pickle(validation_eval_targets, fixed_validation_target_path)
    bigbio_validation_eval.to_parquet(str(validation_10pct_bigbio_path))

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
        heldout_bigbio = merged_bigbio.select(heldout_idx)

        fold_dir = folds_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        dump_pickle(train_source, fold_dir / "train_source.pkl")
        dump_pickle(train_target, fold_dir / "train_target.pkl")
        dump_pickle(heldout_source, fold_dir / "heldout_source.pkl")
        dump_pickle(heldout_target, fold_dir / "heldout_target.pkl")
        dump_pickle(heldout_idx.tolist(), fold_dir / "heldout_indices.pkl")
        heldout_bigbio_path = fold_dir / "heldout_bigbio.parquet"
        heldout_bigbio.to_parquet(str(heldout_bigbio_path))

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
        models_root=models_root,
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
        "validation_10pct_bigbio_path": str(validation_10pct_bigbio_path),
        "models_root": models_root,
        "prediction_models_root": models_root,
        "bigbio_train_pages": len(bigbio_train_pages),
        "bigbio_validation_pages": len(bigbio_validation_pages),
        "fold_sizes": fold_sizes,
        "fold_slurm_scripts": [str(path) for path in fold_script_paths],
        "submit_all_script": str(submit_script_path),
    }

    if generate_oof_predictions:
        models_root_path = Path(models_root)
        all_hybrid_long_sources = []
        all_hybrid_short_sources = []
        all_hybrid_long_targets = []
        all_hybrid_short_targets = []
        all_hybrid_long_tsv_lines = []
        all_hybrid_short_tsv_lines = []
        fold_reports = []

        for fold_idx in range(num_folds):
            fold_dir = folds_root / f"fold_{fold_idx}"
            heldout_bigbio_path = fold_dir / "heldout_bigbio.parquet"
            heldout_bigbio_raw = load_dataset(
                "parquet",
                data_files=str(heldout_bigbio_path),
                split="train",
            )
            if not isinstance(heldout_bigbio_raw, Dataset):
                raise TypeError("Expected a map-style Dataset for heldout BigBio split")
            heldout_bigbio = heldout_bigbio_raw

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
            fold_predictions, heldout_bigbio_with_pred = generate_predictions_bigbio(
                model_dir=model_dir,
                dataset_name=dataset_name,
                heldout_pages=heldout_bigbio,
                context_format=context_format,
                batch_size=batch_size,
                num_beams=num_beams,
            )

            # Compute recall metrics (top-1) to verify inference quality per fold.
            top_fold_df = pl.DataFrame(fold_predictions).filter(pl.col("rank") == 1)
            true_overall = top_fold_df.filter(
                pl.col("gold_concept_code") == pl.col("pred_concept_code")
            ).height
            total_overall = top_fold_df.height
            recall_overall = true_overall / total_overall if total_overall > 0 else 0.0

            recall_by_group: dict[str, str] = {}
            for semantic_group in top_fold_df["semantic_group"].unique().to_list():
                group_df = top_fold_df.filter(
                    pl.col("semantic_group") == semantic_group
                )
                true_group = group_df.filter(
                    pl.col("gold_concept_code") == pl.col("pred_concept_code")
                ).height
                total_group = group_df.height
                recall_group = true_group / total_group if total_group > 0 else 0.0
                recall_by_group[str(semantic_group)] = (
                    f"{recall_group:.4f} ({true_group}/{total_group})"
                )
                print(
                    f"[Fold {fold_idx}] Semantic Group: {semantic_group} - "
                    f"Recall: {recall_group:.4f} ({true_group}/{total_group})"
                )

            fold_recall_metrics = {
                "overall": f"{recall_overall:.4f} ({true_overall}/{total_overall})",
                "by_semantic_group": recall_by_group,
            }
            print(
                f"[Fold {fold_idx}] Overall Recall Metrics: {fold_recall_metrics['overall']}"
            )
            with open(fold_dir / "recall.json", "w", encoding="utf-8") as f:
                json.dump(fold_recall_metrics, f, indent=2, ensure_ascii=False)

            (
                hybrid_long_source_fold,
                hybrid_long_target_fold,
                tsv_lines_fold,
            ) = build_hybrid_long_training_pairs_from_predictions(
                heldout_pages=heldout_bigbio_with_pred,
                dataset_name=dataset_name,
            )
            dump_pickle(
                hybrid_long_source_fold,
                fold_dir / "train_hybrid_long_v2_source.pkl",
            )
            dump_pickle(
                hybrid_long_target_fold,
                fold_dir / "train_hybrid_long_v2_target.pkl",
            )
            # Save to csv tsv lines
            pl.DataFrame(tsv_lines_fold).write_csv(
                file=fold_dir / "train_hybrid_long_v2.tsv",
                separator="\t",
                include_header=True,
            )
            (
                hybrid_short_source_fold,
                hybrid_short_target_fold,
                tsv_lines_short_fold,
            ) = build_hybrid_short_training_pairs_from_predictions(
                heldout_pages=heldout_bigbio_with_pred,
                dataset_name=dataset_name,
            )
            dump_pickle(
                hybrid_short_source_fold,
                fold_dir / "train_hybrid_short_v2_source.pkl",
            )
            dump_pickle(
                hybrid_short_target_fold,
                fold_dir / "train_hybrid_short_v2_target.pkl",
            )
            # Save to csv tsv lines
            pl.DataFrame(tsv_lines_short_fold).write_csv(
                file=fold_dir / "train_hybrid_short_v2.tsv",
                separator="\t",
                include_header=True,
            )
            all_hybrid_long_sources.extend(hybrid_long_source_fold)
            all_hybrid_long_targets.extend(hybrid_long_target_fold)
            all_hybrid_long_tsv_lines.extend(tsv_lines_fold)
            all_hybrid_short_sources.extend(hybrid_short_source_fold)
            all_hybrid_short_targets.extend(hybrid_short_target_fold)
            all_hybrid_short_tsv_lines.extend(tsv_lines_short_fold)

            fold_reports.append({
                "fold": fold_idx,
                "heldout_bigbio_pages": len(heldout_bigbio),
                "predictions": len(fold_predictions),
                "model_dir": str(model_dir),
                "hybrid_long_pairs": len(hybrid_long_source_fold),
                "hybrid_short_pairs": len(hybrid_short_source_fold),
                "recall": fold_recall_metrics,
            })

        hybrid_long_source_path = (
            data_root / f"train_{selection_method}_source_hybrid_long_v2.pkl"
        )
        hybrid_long_target_path = (
            data_root / f"train_{selection_method}_target_hybrid_long_v2.pkl"
        )
        dump_pickle(all_hybrid_long_sources, hybrid_long_source_path)
        dump_pickle(all_hybrid_long_targets, hybrid_long_target_path)
        # Save to csv tsv lines
        pl.DataFrame(all_hybrid_long_tsv_lines).write_csv(
            file=data_root
            / f"train_{selection_method}_annotations_hybrid_long_pred.tsv",
            separator="\t",
            include_header=True,
        )
        hybrid_short_source_path = (
            data_root / f"train_{selection_method}_source_hybrid_short_v2.pkl"
        )
        hybrid_short_target_path = (
            data_root / f"train_{selection_method}_target_hybrid_short_v2.pkl"
        )
        dump_pickle(all_hybrid_short_sources, hybrid_short_source_path)
        dump_pickle(all_hybrid_short_targets, hybrid_short_target_path)
        # Save to csv tsv lines
        pl.DataFrame(all_hybrid_short_tsv_lines).write_csv(
            file=data_root
            / f"train_{selection_method}_annotations_hybrid_short_pred.tsv",
            separator="\t",
            include_header=True,
        )

        if validation_predict_fold < 0 or validation_predict_fold >= num_folds:
            raise ValueError(
                f"validation_predict_fold must be in [0, {num_folds - 1}], got {validation_predict_fold}"
            )

        validation_model_dir = resolve_model_dir(
            model_name=model_name,
            dataset_name=dataset_name,
            augmented_data=augmented_data,
            selection_method=selection_method,
            context_format=context_format,
            complete_mode=complete_mode,
            add_headers=add_headers,
            run_name_suffix=f"_fold{validation_predict_fold}",
            checkpoint_kind=checkpoint_kind,
            models_root=models_root_path,
        )
        print(
            f"[Validation 10%] Loading model from fold {validation_predict_fold}: {validation_model_dir}"
        )
        val_preds, validation_10pct_bigbio_with_pred = generate_predictions_bigbio(
            model_dir=validation_model_dir,
            dataset_name=dataset_name,
            heldout_pages=bigbio_validation_eval,
            context_format=context_format,
            batch_size=batch_size,
            num_beams=num_beams,
        )
        # Validation 10% recall metrics (top-1), same logic as fold recall.
        val_top_df = pl.DataFrame(val_preds).filter(pl.col("rank") == 1)
        val_true_overall = val_top_df.filter(
            pl.col("gold_concept_code") == pl.col("pred_concept_code")
        ).height
        val_total_overall = val_top_df.height
        val_recall_overall = (
            val_true_overall / val_total_overall if val_total_overall > 0 else 0.0
        )

        val_recall_by_group: dict[str, str] = {}
        for semantic_group in val_top_df["semantic_group"].unique().to_list():
            group_df = val_top_df.filter(pl.col("semantic_group") == semantic_group)
            true_group = group_df.filter(
                pl.col("gold_concept_code") == pl.col("pred_concept_code")
            ).height
            total_group = group_df.height
            recall_group = true_group / total_group if total_group > 0 else 0.0
            val_recall_by_group[str(semantic_group)] = (
                f"{recall_group:.4f} ({true_group}/{total_group})"
            )
            print(
                f"[Validation 10%] Semantic Group: {semantic_group} - "
                f"Recall: {recall_group:.4f} ({true_group}/{total_group})"
            )

        validation_recall = {
            "overall": f"{val_recall_overall:.4f} ({val_true_overall}/{val_total_overall})",
            "by_semantic_group": val_recall_by_group,
        }
        validation_recall_path = folds_root / "validation_10pct_recall.json"
        with open(validation_recall_path, "w", encoding="utf-8") as f:
            json.dump(validation_recall, f, indent=2, ensure_ascii=False)
        print(f"[Validation 10%] Overall Recall: {validation_recall['overall']}")

        (
            validation_hybrid_long_source,
            validation_hybrid_long_target,
            validation_hybrid_long_tsv,
        ) = build_hybrid_long_training_pairs_from_predictions(
            heldout_pages=validation_10pct_bigbio_with_pred,
            dataset_name=dataset_name,
        )

        validation_hybrid_long_source_path = (
            data_root / f"validation_{selection_method}_source_hybrid_long_v2.pkl"
        )
        validation_hybrid_long_target_path = (
            data_root / f"validation_{selection_method}_target_hybrid_long_v2.pkl"
        )
        validation_hybrid_long_tsv_path = (
            data_root / f"validation_{selection_method}_annotations_hybrid_long_v2.tsv"
        )
        dump_pickle(validation_hybrid_long_source, validation_hybrid_long_source_path)
        dump_pickle(validation_hybrid_long_target, validation_hybrid_long_target_path)
        pl.DataFrame(validation_hybrid_long_tsv).write_csv(
            file=validation_hybrid_long_tsv_path,
            separator="\t",
            include_header=True,
        )

        (
            validation_hybrid_short_source,
            validation_hybrid_short_target,
            validation_hybrid_short_tsv,
        ) = build_hybrid_short_training_pairs_from_predictions(
            heldout_pages=validation_10pct_bigbio_with_pred,
            dataset_name=dataset_name,
        )
        validation_hybrid_short_source_path = (
            data_root / f"validation_{selection_method}_source_hybrid_short_v2.pkl"
        )
        validation_hybrid_short_target_path = (
            data_root / f"validation_{selection_method}_target_hybrid_short_v2.pkl"
        )
        validation_hybrid_short_tsv_path = (
            data_root / f"validation_{selection_method}_annotations_hybrid_short_v2.tsv"
        )
        dump_pickle(validation_hybrid_short_source, validation_hybrid_short_source_path)
        dump_pickle(validation_hybrid_short_target, validation_hybrid_short_target_path)
        pl.DataFrame(validation_hybrid_short_tsv).write_csv(
            file=validation_hybrid_short_tsv_path,
            separator="\t",
            include_header=True,
        )

        metadata["oof_prediction"] = {
            "mode": "bigbio_longbel_sample_constrained",
            "checkpoint_kind": checkpoint_kind,
            "batch_size": batch_size,
            "num_beams": num_beams,
            "models_root": str(models_root_path),
            "fold_reports": fold_reports,
            "hybrid_long_source_path": str(hybrid_long_source_path),
            "hybrid_long_target_path": str(hybrid_long_target_path),
            "hybrid_short_source_path": str(hybrid_short_source_path),
            "hybrid_short_target_path": str(hybrid_short_target_path),
            "validation_predict_fold": validation_predict_fold,
            "validation_10pct_recall": validation_recall,
            "validation_10pct_recall_path": str(validation_recall_path),
            "validation_hybrid_long_source_path": str(
                validation_hybrid_long_source_path
            ),
            "validation_hybrid_long_target_path": str(
                validation_hybrid_long_target_path
            ),
            "validation_hybrid_long_tsv_path": str(validation_hybrid_long_tsv_path),
            "validation_hybrid_short_source_path": str(
                validation_hybrid_short_source_path
            ),
            "validation_hybrid_short_target_path": str(
                validation_hybrid_short_target_path
            ),
            "validation_hybrid_short_tsv_path": str(validation_hybrid_short_tsv_path),
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
        default="$SCRATCH/expe_longbel_fold_models",
        help="Root folder containing fold model outputs.",
    )
    parser.add_argument(
        "--validation-predict-fold",
        type=int,
        default=0,
        help="Fold index of the trained model used to predict validation_10pct.",
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
        models_root=os.path.expandvars(args.models_root),
        validation_predict_fold=args.validation_predict_fold,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
