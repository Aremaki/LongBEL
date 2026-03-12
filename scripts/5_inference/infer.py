import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import polars as pl
import torch
from datasets import load_dataset

from longbel.longbel import LongBEL
from longbel.trie import Trie

sys.setrecursionlimit(5000)


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def hf_model_gpu_size_mb(model):
    return (
        sum(p.numel() * p.element_size() for p in model.parameters())
        + sum(b.numel() * b.element_size() for b in model.buffers())
    ) / 1024**2


def deep_getsizeof(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)
    if hasattr(obj, "__dict__"):
        size += deep_getsizeof(obj.__dict__, seen)
    elif isinstance(obj, dict):
        size += sum(
            deep_getsizeof(k, seen) + deep_getsizeof(v, seen) for k, v in obj.items()
        )
    elif isinstance(obj, (list, tuple, set)):
        size += sum(deep_getsizeof(i, seen) for i in obj)
    return size


def main(
    model_name: str,
    num_beams: int,
    best: bool,
    dataset_name: str,
    selection_method: str,
    split_name: str = "test",
    augmented_data: str = "human_only",
    context_format: str = "short",
    complete_mode: bool = False,
    add_headers: bool = False,
    batch_size: int = 64,
    output_folder: str = "results/inference_outputs",
):
    # Set device
    torch.cuda.empty_cache()
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Log the inference configuration
    print(f"Model Name: {model_name}")
    print(f"Dataset Name: {dataset_name}")
    print(f"Selection Method: {selection_method}")
    print(f"Split Name: {split_name}")
    print(f"Augmented Data: {augmented_data}")
    print(f"Context Format: {context_format}")
    print(f"Complete Mode: {complete_mode}")
    print(f"Add Headers: {add_headers}")
    print(f"Number of Beams: {num_beams}")
    print(f"Batch Size: {batch_size}")
    print(f"device: {device}")

    # Load model
    complete_mode_str = "_complete" if complete_mode else ""
    add_headers_str = "_addheaders" if add_headers else ""
    model_path = (
        Path("models/NED")
        / f"{dataset_name}_{augmented_data}_{selection_method}_{context_format}{complete_mode_str}{add_headers_str}"
        / model_name
    )
    if best:
        full_path = model_path / "model_best"
    else:
        full_path = model_path / "model_last"

    # Candidate trie path
    if dataset_name == "MedMentions":
        dataset_short = "MM"
    elif dataset_name in ["EMEA", "MEDLINE"]:
        dataset_short = "QUAERO"
    elif "SPACCC" in dataset_name:
        dataset_short = "SPACCC"
    else:
        dataset_short = dataset_name
    candidate_tries_folder = Path("data/candidate_tries")
    candidate_tries_folder.mkdir(parents=True, exist_ok=True)
    candidate_trie_path = (
        candidate_tries_folder / f"trie_{dataset_short}_{model_name}.pkl"
    )

    # Text to code path
    text_to_codes_folder = Path("data"/"text_to_codes")
    text_to_codes_folder.mkdir(parents=True, exist_ok=True)
    text_to_code_path = text_to_codes_folder / f"text_to_code_{dataset_short}.json"

    # Lang
    if dataset_name == "SPACCC":
        lang = "es"
    elif dataset_name == "MedMentions":
        lang = "en"
    elif dataset_name in ["EMEA", "MEDLINE"]:
        lang = "fr"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    model = (
        LongBEL.from_pretrained(
            full_path,
            lang=lang,
            text_to_code_path=text_to_code_path,
            candidate_trie_path=candidate_trie_path,
        )
        .eval()
        .to(device)  # type: ignore
    )

    # Load data
    data_folder = Path("data/final_data")

    test_data = load_dataset(
        str(data_folder / dataset_name / "bigbio_dataset"),
        data_dir="processed_data",
        split="test",
    )

    # Text to codes or candidate trie generation
    if not os.path.exists(candidate_trie_path) or not os.path.exists(text_to_code_path):
        umls_path = (
            Path("data")
            / "termino_processed"
            / dataset_short
            / "all_disambiguated.parquet"
        )
        umls_df = pl.read_parquet(umls_path)
        umls_df = umls_df.with_columns(
            pl.col("Entity")
            .str.replace_all("\xa0", " ", literal=True)
            .str.replace_all(r"[\{\[]", "(", literal=False)
            .str.replace_all(r"[\}\]]", ")", literal=False)
        )

        # Text to code
        if not os.path.exists(text_to_code_path):
            code_col = "SNOMED_code" if "SNOMED_code" in umls_df.columns else "CUI"
            result = umls_df.group_by("GROUP").agg([
                pl.col("Entity"),
                pl.col(code_col),
            ])

            text_to_code = {
                row["GROUP"]: dict(zip(row["Entity"], row[code_col]))
                for row in result.to_dicts()
            }
            os.makedirs(text_to_code_path.parent, exist_ok=True)
            with open(text_to_code_path, "w", encoding="utf-8") as f:
                json.dump(
                    text_to_code,
                    f,
                    ensure_ascii=False,  # important for biomedical terms
                    indent=2,  # human-readable
                )
            model.text_to_code = text_to_code  # type: ignore

        # candidate Trie
        if not os.path.exists(candidate_trie_path):
            # Compute candidate Trie
            start_idx = 1
            prefix = "} "
            candidate_trie = {}
            for group in umls_df["GROUP"].unique().to_list():  # type: ignore
                print(f"processing {group}")
                group_umls_df = umls_df.filter(pl.col("GROUP") == group)  # type: ignore
                sequences = []
                for entity in group_umls_df["Entity"].to_list():
                    sequence = model.tokenizer.encode(prefix + entity)[start_idx:]  # type: ignore
                    if sequence[-1] != model.tokenizer.eos_token_id:  # type: ignore
                        sequence.append(model.tokenizer.eos_token_id)  # type: ignore
                    sequences.append(sequence)
                candidate_trie[group] = Trie(sequences)

            # Save it
            # Create directory if it doesn't exist
            os.makedirs(candidate_trie_path.parent, exist_ok=True)
            with open(candidate_trie_path, "wb") as file:
                pickle.dump(candidate_trie, file, protocol=-1)
            model.candidate_trie = candidate_trie  # type: ignore

    # Init output folder
    result_folder = (
        Path(output_folder)
        / dataset_name
        / f"{augmented_data}_{selection_method}_{context_format}{complete_mode_str}{add_headers_str}"
        / f"{model_name}_{'best' if best else 'last'}"
    )
    result_folder.mkdir(parents=True, exist_ok=True)

    # Multiple answers setting for guided decoding
    if dataset_name == "SPACCC":
        multiple_answers = True
    else:
        multiple_answers = False

    metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "selection_method": selection_method,
        "split_name": split_name,
        "augmented_data": augmented_data,
        "context_format": context_format,
        "add_headers": add_headers,
        "complete_mode": complete_mode,
        "num_beams": num_beams,
        "batch_size": batch_size,
        "multiple_answers": multiple_answers,
    }
    # Log model size
    model_size_mb = hf_model_gpu_size_mb(model)
    metadata["model_size_mb"] = model_size_mb
    print(f"Model size on GPU: {model_size_mb:.2f} MB")

    # Log candidate Trie size
    trie_size = deep_getsizeof(model.candidate_trie) / 1024**2
    metadata["trie_size_mb"] = trie_size
    print(f"Trie candidate object size (deep): {trie_size:.2f} MB")

    # Log text to code size
    text_to_code_size = deep_getsizeof(model.text_to_code) / 1024**2
    metadata["text_to_code_size_mb"] = text_to_code_size
    print(f"Text_to_code object size (deep): {text_to_code_size:.2f} MB")

    # Perform inference without constraint
    metadata["timing_results"] = {}
    tic = time.time()
    no_constraint_pred = model.sample(
        bigbio_pages=test_data,  # type: ignore
        batch_size=batch_size,
        num_beams=num_beams,
        constrained=False,
        multiple_answers=multiple_answers,
        context_format=context_format,
    )  # type: ignore
    tac = time.time()
    elapsed_time = tac - tic

    # Compute speed metrics
    num_batches = (len(test_data) + batch_size - 1) // batch_size  # type: ignore
    num_entities = sum(len(entities) for entities in test_data["entities"])  # type: ignore
    mean_time_per_batch = elapsed_time / num_batches
    mean_time_of_page = elapsed_time / len(test_data)  # type: ignore
    num_entities_per_second = num_entities / elapsed_time

    print("\n=== Speed Metrics ===")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"Mean time per batch: {mean_time_per_batch:.2f} seconds")
    print(f"Mean time of page: {mean_time_of_page:.2f} seconds")
    print(f"Number of entities per second: {num_entities_per_second:.2f}\n")
    metadata["timing_results"]["no_constraint"] = {
        "total_time_seconds": elapsed_time,
        "mean_time_per_batch": mean_time_per_batch,
        "mean_time_of_page": mean_time_of_page,
        "num_entities_per_second": num_entities_per_second,
    }

    no_constraint_df = pl.DataFrame(no_constraint_pred)
    no_constraint_df = no_constraint_df.sort(["mention_id", "rank"])
    # Compute recall per label
    top_no_constraint_df = no_constraint_df.filter(pl.col("rank") == 1)
    for label in top_no_constraint_df["semantic_group"].unique().to_list():
        label_df = top_no_constraint_df.filter(pl.col("semantic_group") == label)
        true_label = label_df.filter(
            pl.col("gold_concept_code") == pl.col("pred_concept_code")
        ).shape[0]
        total_label = label_df.shape[0]
        recall_label = true_label / total_label if total_label > 0 else 0.0
        print(
            f"Semantic Group: {label} - No Constraint Inference Recall: {recall_label:.4f} ({true_label}/{total_label})"
        )
    # Compute recall overall
    true_overall = top_no_constraint_df.filter(
        pl.col("gold_concept_code") == pl.col("pred_concept_code")
    ).shape[0]
    total_overall = top_no_constraint_df.shape[0]
    recall_overall = true_overall / total_overall if total_overall > 0 else 0.0
    print(
        f"Overall - No Constraint Inference Recall: {recall_overall:.4f} ({true_overall}/{total_overall})"
    )
    print(f"Generated {len(no_constraint_df)} sentences without constraint.")

    # Save results
    no_constraint_df.write_csv(
        file=result_folder / f"pred_{split_name}_no_constraint_{num_beams}_beams.tsv",
        separator="\t",
        include_header=True,
    )

    # Perform inference with constraint
    if "timing_results" not in metadata:
        metadata["timing_results"] = {}
    tic = time.time()
    constraint_preds = model.sample(
        bigbio_pages=test_data,  # type: ignore
        batch_size=batch_size,
        num_beams=num_beams,
        constrained=True,
        multiple_answers=multiple_answers,
        context_format=context_format,
    )  # type: ignore
    tac = time.time()
    elapsed_time = tac - tic

    # Compute speed metrics
    num_batches = (len(test_data) + batch_size - 1) // batch_size  # type: ignore
    num_entities = sum(len(entities) for entities in test_data["entities"])  # type: ignore
    mean_time_per_batch = elapsed_time / num_batches
    mean_time_of_page = elapsed_time / len(test_data)  # type: ignore
    num_entities_per_second = num_entities / elapsed_time

    print("\n=== Speed Metrics ===")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"Mean time per batch: {mean_time_per_batch:.2f} seconds")
    print(f"Mean time of page: {mean_time_of_page:.2f} seconds")
    print(f"Number of entities per second: {num_entities_per_second:.2f}\n")
    metadata["timing_results"]["constraint"] = {
        "total_time_seconds": elapsed_time,
        "mean_time_per_batch": mean_time_per_batch,
        "mean_time_of_page": mean_time_of_page,
        "num_entities_per_second": num_entities_per_second,
    }

    # Compute recall per label
    constraint_df = pl.DataFrame(constraint_preds)
    # sort by mention_id and rank
    constraint_df = constraint_df.sort(["mention_id", "rank"])
    top_constraint_df = constraint_df.filter(pl.col("rank") == 1)
    for semantic_group in top_constraint_df["semantic_group"].unique().to_list():
        label_df = top_constraint_df.filter(pl.col("semantic_group") == semantic_group)
        true_label = label_df.filter(
            pl.col("gold_concept_code") == pl.col("pred_concept_code")
        ).shape[0]
        total_label = label_df.shape[0]
        recall_label = true_label / total_label if total_label > 0 else 0.0
        print(
            f"Semantic Group: {semantic_group} - Constraint Inference Recall: {recall_label:.4f} ({true_label}/{total_label})"
        )
    # Compute recall overall
    true_overall = top_constraint_df.filter(
        pl.col("gold_concept_code") == pl.col("pred_concept_code")
    ).shape[0]
    total_overall = top_constraint_df.shape[0]
    recall_overall = true_overall / total_overall if total_overall > 0 else 0.0
    print(
        f"Overall - Constraint Inference Recall: {recall_overall:.4f} ({true_overall}/{total_overall})"
    )
    print(f"Generated {len(constraint_df)} sentences with constraint.")

    # Save results
    constraint_df.write_csv(
        file=result_folder / f"pred_{split_name}_constraint_{num_beams}_beams.tsv",
        separator="\t",
        include_header=True,
    )
    # Save timing results
    with open(
        result_folder / f"metadata_{split_name}_{num_beams}_beams.json", "w"
    ) as f:
        json.dump(metadata, f, indent=2)
    print("Inference completed and results saved.")


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="A script for inference seq2seq model")
    parser.add_argument("--model-name", type=str, required=True, help="The model name")
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="The number of beams",
    )
    parser.add_argument(
        "--best", default=False, action="store_true", help="Use best if True else last"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="MedMentions",
        help="The dataset name",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default="random",
        help="The selection method for training data",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="The data split name for inference",
    )
    parser.add_argument(
        "--augmented-data",
        type=str,
        default="human_only",
        choices=[
            "human_only",
            "synth_only",
            "full",
            "full_upsampled",
            "full_filtered",
            "full_filtered_upsampled",
            "human_only_ft",
        ],
        help="Whether to use augmented data for training",
    )
    parser.add_argument(
        "--context-format",
        type=str,
        default="short",
        choices=["short", "long", "hybrid_long", "hybrid_short"],
        help="The context format for training",
    )
    parser.add_argument(
        "--complete-mode",
        default=False,
        action="store_true",
        help="Whether to use complete mode for training and inference",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="results/inference_outputs",
        help="Folder to save inference outputs",
    )
    parser.add_argument(
        "--add-headers",
        default=False,
        action="store_true",
        help="Whether to add headers to the output",
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # Pass the parsed argument to the main function
    main(
        model_name=args.model_name,
        num_beams=args.num_beams,
        best=args.best,
        dataset_name=args.dataset_name,
        selection_method=args.selection_method,
        split_name=args.split_name,
        augmented_data=args.augmented_data,
        context_format=args.context_format,
        complete_mode=args.complete_mode,
        add_headers=args.add_headers,
        batch_size=args.batch_size,
        output_folder=args.output_folder,
    )
