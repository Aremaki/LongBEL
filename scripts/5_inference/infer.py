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


def main(
    model_name,
    num_beams,
    best,
    dataset_name,
    selection_method,
    split_name="test",
    augmented_data="human_only",
    long_format=True,
    batch_size=64,
    output_folder="results/inference_outputs",
):
    # Set device
    torch.cuda.empty_cache()
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    long_format_str = "_long" if long_format else ""
    model_path = (
        Path("models/NED")
        / f"{dataset_name}_{augmented_data}_{selection_method}{long_format_str}"
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
    text_to_codes_folder = Path("data") / "text_to_codes"
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
            if model.lang == "fr":  # type: ignore
                verb = "est"
            elif model.lang == "en":  # type: ignore
                verb = "is"
            elif model.lang == "es":  # type: ignore
                verb = "es"
            else:
                raise ValueError(f"Unknown language: {model.lang}")  # type: ignore
            # Compute candidate Trie
            start_idx = 1
            prefix = f" {verb} "
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
    output_folder = (
        Path(output_folder)
        / dataset_name
        / f"{augmented_data}_{selection_method}{long_format_str}"
        / f"{model_name}_{'best' if best else 'last'}"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    # Multiple answers setting for guided decoding
    if not split_name == "test_simple" and dataset_name == "SPACCC":
        multiple_answers = True
    else:
        multiple_answers = False

    # Perform inference without constraint
    tic = time.time()
    no_constraint_pred = model.sample(
        bigbio_examples=test_data,
        batch_size=batch_size,
        num_beams=num_beams,
        constrained=False,
        multiple_answers=multiple_answers,
        long_format=long_format,
    )  # type: ignore
    tac = time.time()
    elapsed_time = tac - tic

    # Compute speed metrics
    num_batches = (len(test_data) + batch_size - 1) // batch_size  # type: ignore
    batches_per_second = num_batches / elapsed_time
    examples_per_second = len(test_data) / elapsed_time  # type: ignore

    print("\n=== Speed Metrics ===")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"Batches per second: {batches_per_second:.2f}")
    print(f"Examples per second: {examples_per_second:.2f}\n")

    # Compute recall per label
    no_constraint_df = pl.DataFrame(no_constraint_pred)
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
        file=output_folder / f"pred_{split_name}_no_constraint_{num_beams}_beams.tsv",
        separator="\t",
        include_header=True,
    )

    # Perform inference with constraint
    tic = time.time()
    constraint_preds = model.sample(
        bigbio_examples=test_data,
        batch_size=batch_size,
        num_beams=num_beams,
        constrained=True,
        multiple_answers=multiple_answers,
        long_format=long_format,
    )  # type: ignore
    tac = time.time()
    elapsed_time = tac - tic

    # Compute speed metrics
    num_batches = (len(test_data) + batch_size - 1) // batch_size  # type: ignore
    batches_per_second = num_batches / elapsed_time
    examples_per_second = len(test_data) / elapsed_time  # type: ignore

    print("\n=== Speed Metrics ===")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"Batches per second: {batches_per_second:.2f}")
    print(f"Examples per second: {examples_per_second:.2f}\n")

    # Compute recall per label
    constraint_df = pl.DataFrame(constraint_preds)
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
        file=output_folder / f"pred_{split_name}_constraint_{num_beams}_beams.tsv",
        separator="\t",
        include_header=True,
    )

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
        "--long-format",
        action="store_true",
        help="Whether to use the long format for training",
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
        long_format=args.long_format,
        batch_size=args.batch_size,
        output_folder=args.output_folder,
    )
