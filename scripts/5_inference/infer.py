import argparse
import gc
import os
import pickle
import sys
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm
from transformers import GenerationConfig  # type: ignore

from syncabel.guided_inference import get_prefix_allowed_tokens_fn
from syncabel.models import MT5_GENRE, Bart_GENRE, Llama_GENRE, MBart_GENRE
from syncabel.trie import Trie

sys.setrecursionlimit(5000)


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def load_model(
    dataset_name: str,
    model_name: str,
    full_path: Path,
    device: str,
    best: bool,
):
    if "mt5" in model_name:
        model = MT5_GENRE.from_pretrained(full_path).eval().to(device)  # type: ignore
        model.generation_config = GenerationConfig(
            decoder_start_token_id=0,
            eos_token_id=1,
            forced_eos_token_id=1,
            pad_token_id=0,
        )
    elif "Llama" in model_name:
        model = Llama_GENRE.from_pretrained(full_path).eval().to(device)  # type: ignore
        model.generation_config = GenerationConfig(
            bos_token_id=model.config.bos_token_id,  # type: ignore
            decoder_start_token_id=model.config.bos_token_id,  # type: ignore
            eos_token_id=model.config.eos_token_id,  # type: ignore
            forced_eos_token_id=model.config.eos_token_id,  # type: ignore
            pad_token_id=model.config.pad_token_id,  # type: ignore
        )
        model.tokenizer.padding_side = "left"  # type: ignore
    elif "mbart" in model_name:
        model = MBart_GENRE.from_pretrained(full_path).eval().to(device)  # type: ignore
        model.generation_config = GenerationConfig(
            bos_token_id=0,
            decoder_start_token_id=2,
            eos_token_id=2,
            forced_eos_token_id=2,
            pad_token_id=1,
        )
        model.tokenizer.bos_token_id = None  # type: ignore
        if dataset_name == "MedMentions":
            model.tokenizer.src_lang = "en_XX"  # type: ignore
            model.tokenizer.tgt_lang = "en_XX"  # type: ignore
        elif dataset_name == "SPACCC":
            model.tokenizer.src_lang = "es_XX"  # type: ignore
            model.tokenizer.tgt_lang = "es_XX"  # type: ignore
        else:
            model.tokenizer.src_lang = "fr_XX"  # type: ignore
            model.tokenizer.tgt_lang = "fr_XX"  # type: ignore
    else:
        model = Bart_GENRE.from_pretrained(full_path).eval().to(device)  # type: ignore
        model.generation_config = GenerationConfig(
            bos_token_id=0,
            early_stopping=True,
            decoder_start_token_id=2,
            eos_token_id=2,
            forced_eos_token_id=2,
            pad_token_id=1,
        )
    print(
        f"Model {model_name} {'best' if best else 'last'} checkpoint is loaded to {device}"
    )
    return model


def add_code_column(df: pl.DataFrame, umls_df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a 'code' column to the DataFrame by mapping composite 'Entity' strings to UMLS concepts.

    The function performs the following steps:
    1.  Cleans up special characters in the 'Entity' column of the UMLS DataFrame.
    2.  Splits the 'Entity' column of the input DataFrame by '<SEP>' into individual terms.
    3.  Explodes the DataFrame so that each row corresponds to a single term.
    4.  Performs a left join with the UMLS DataFrame on the term and 'GROUP' to find the code for each term.
    5.  Groups the data back by the original rows.
    6.  Filters out rows where none of the constituent terms mapped to a code.
    7.  Constructs the final code string by sorting the unique codes and joining them with '+'.
    """
    # 1) Check if there are duplicated entries
    if (
        df.shape[0]
        != df.unique(subset=["filename", "label", "start_span", "end_span"]).shape[0]
    ):
        print("There are duplicated entries. Keeping just the first one...")
        df = df.unique(subset=["filename", "label", "start_span", "end_span"])

    # 2) Explode df by splitting 'Entity' into multiple terms
    df_exploded = df.with_columns(pl.col("Prediction").str.split("<SEP>")).explode(
        "Prediction"
    )

    # 3) Join with UMLS data to get CUIs for each term
    df_joined = df_exploded.join(
        umls_df.select(["SNOMED_code", "Entity", "GROUP"]),
        left_on=["Prediction", "label"],
        right_on=["Entity", "GROUP"],
        how="left",
    )

    # 4) Group back, aggregate CUIs, and count nulls to find incomplete matches
    df_grouped = df_joined.group_by([
        "filename",
        "label",
        "start_span",
        "end_span",
    ]).agg(
        pl.col("SNOMED_code").drop_nulls().unique().alias("code_list"),
    )

    # 5) Filter for rows with at least one match and create final code string
    result = (
        df_grouped.filter(pl.col("code_list").list.len() > 0)
        .with_columns(
            pl.col("code_list").list.sort().list.join("+").alias("Predicted_code")
        )
        .select(["filename", "label", "start_span", "end_span", "Predicted_code"])
    )

    return df.join(
        result, on=["filename", "label", "start_span", "end_span"], how="left"
    )


def main(
    model_name,
    num_beams,
    best,
    dataset_name,
    selection_method,
    split_name="test",
    augmented_data="human_only",
    human_ratio=1.0,
    batch_size=64,
    output_folder="results/inference_outputs",
):
    # Set device
    torch.cuda.empty_cache()
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    if human_ratio < 1.0:
        human_ratio_str = (
            "_" + str(round(human_ratio * 100, 0)).replace(".0", "") + "pct"
        )
    else:
        human_ratio_str = ""
    model_path = (
        Path("/lustre/fsn1/projects/rech/ssq/usk98ia/expe_data_ratio")
        / f"{dataset_name}_{augmented_data}_{selection_method}{human_ratio_str}"
        / model_name
    )
    if best:
        full_path = model_path / "model_best"
    else:
        full_path = model_path / "model_last"
    model = load_model(
        dataset_name,
        model_name,
        full_path,
        device,
        best,
    )

    # Load data
    data_folder = Path("data/final_data")

    test_source_data = load_pickle(
        data_folder / dataset_name / f"{split_name}_{selection_method}_source.pkl"
    )
    test_target_data = load_pickle(
        data_folder / dataset_name / f"{split_name}_{selection_method}_target.pkl"
    )
    # Determine verb based on dataset
    verb = "est"
    if dataset_name == "MedMentions":
        verb = "is"
    elif dataset_name == "SPACCC":
        verb = "es"
    prefix_templates = [
        tgt.split(f"] {verb} ")[0] + f"] {verb}" for tgt in test_target_data
    ]
    test_data = pl.read_csv(
        data_folder / dataset_name / f"{split_name}_{selection_method}_annotations.tsv",
        separator="\t",
        has_header=True,
        schema_overrides={"code": str},  # type: ignore
    )

    # Load UMLS data
    if dataset_name == "MedMentions":
        dataset_short = "MM"
    elif dataset_name in ["EMEA", "MEDLINE"]:
        dataset_short = "QUAERO"
    elif "SPACCC" in dataset_name:
        dataset_short = "SPACCC"
    else:
        dataset_short = dataset_name
    umls_path = (
        Path("data") / "UMLS_processed" / dataset_short / "all_disambiguated.parquet"
    )
    umls_df = pl.read_parquet(umls_path)
    umls_df = umls_df.with_columns(
        pl.col("Entity")
        .str.replace_all("\xa0", " ", literal=True)
        .str.replace_all(r"[\{\[]", "(", literal=False)
        .str.replace_all(r"[\}\]]", ")", literal=False)
    )

    # Load candidate Trie
    tries_folder = Path("data/UMLS_tries")
    tries_folder.mkdir(parents=True, exist_ok=True)
    trie_path = tries_folder / f"trie_{dataset_short}_{model_name}.pkl"
    if os.path.exists(trie_path):  # Check if the file exists
        with open(trie_path, "rb") as file:
            trie_legal_tokens = pickle.load(file)
    else:
        # Compute candidate Trie
        if "mbart" in model_name or "mt5" in model_name or "Llama" in model_name:
            start_idx = 0
            prefix = f" {verb} "
        else:
            start_idx = 1
            prefix = ""
        trie_legal_tokens = {}
        for category in umls_df["GROUP"].unique().to_list():
            print(f"processing {category}")
            cat_umls_df = umls_df.filter(pl.col("GROUP") == category)
            sequences = []
            for entity in cat_umls_df["Entity"].to_list():
                sequences.append(
                    model.tokenizer.encode(prefix + entity)[start_idx:]  # type: ignore
                )
            trie_legal_tokens[category] = Trie(sequences)

        # Save it
        # Create directory if it doesn't exist
        os.makedirs(trie_path.parent, exist_ok=True)
        with open(trie_path, "wb") as file:
            pickle.dump(trie_legal_tokens, file, protocol=-1)

    # Init output folder
    output_folder = (
        Path(output_folder)
        / dataset_name
        / f"{augmented_data}_{selection_method}{human_ratio_str}"
        / f"{model_name}_{'best' if best else 'last'}"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    # Perform inference without constraint
    output_sentences = []
    output_scores = []
    output_beam_scores = []
    for i in tqdm(
        range(0, len(test_source_data), batch_size), desc="Processing Test Data"
    ):
        batch_sources = test_source_data[i : i + batch_size]
        batch_prefixes = prefix_templates[i : i + batch_size]
        batch_input = [f"{a}<SEP>{b}" for a, b in zip(batch_sources, batch_prefixes)]
        if "Llama" in model_name:
            with torch.no_grad():
                batch_output_sentences = model.sample(
                    batch_input,
                    num_beams=num_beams,
                )
            output_sentences.extend([
                batch[0]["text"].split(f"] {verb} ")[1]  # type: ignore
                for batch in batch_output_sentences
            ])
        else:
            with torch.no_grad():
                batch_output_sentences = model.sample(
                    batch_sources,
                    num_beams=num_beams,
                    prefix_templates=batch_prefixes,
                )
            output_sentences.extend([
                batch[0]["text"]  # type: ignore
                for batch in batch_output_sentences
            ])
        output_scores.extend([
            batch[0]["score"]  # type: ignore
            for batch in batch_output_sentences
        ])
        output_beam_scores.extend([
            batch[0]["beam_score"]  # type: ignore
            for batch in batch_output_sentences
        ])
    no_constraint_df = test_data.with_columns(
        pl.Series(name="Prediction", values=output_sentences),
        pl.Series(name="Prediction_score", values=output_scores),
        pl.Series(name="Prediction_beam_score", values=output_beam_scores),
    )
    no_constraint_df = add_code_column(no_constraint_df, umls_df=umls_df)

    # Compute recall per label
    for label in no_constraint_df["label"].unique().to_list():
        label_df = no_constraint_df.filter(pl.col("label") == label)
        true_label = label_df.filter(pl.col("code") == pl.col("Predicted_code")).shape[
            0
        ]
        total_label = label_df.shape[0]
        recall_label = true_label / total_label if total_label > 0 else 0.0
        print(
            f"Label: {label} - No Constraint Inference Recall: {recall_label:.4f} ({true_label}/{total_label})"
        )
    # Compute recall overall
    true_overall = no_constraint_df.filter(
        pl.col("code") == pl.col("Predicted_code")
    ).shape[0]
    total_overall = no_constraint_df.shape[0]
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
    output_sentences = []
    output_scores = []
    output_beam_scores = []
    for i in tqdm(
        range(0, len(test_source_data), batch_size), desc="Processing Test Data"
    ):
        batch_sources = test_source_data[i : i + batch_size]
        batch_prefixes = prefix_templates[i : i + batch_size]
        batch_input = [f"{a}<SEP>{b}" for a, b in zip(batch_sources, batch_prefixes)]
        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
            model,
            batch_sources,
            batch_prefixes,
            candidates_trie=trie_legal_tokens,
        )
        if "Llama" in model_name:
            with torch.no_grad():
                batch_output_sentences = model.sample(
                    batch_input,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=num_beams,
                )
            output_sentences.extend([
                batch[0]["text"].split(f"] {verb} ")[1]  # type: ignore
                for batch in batch_output_sentences
            ])
        else:
            with torch.no_grad():
                batch_output_sentences = model.sample(
                    batch_sources,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=num_beams,
                    prefix_templates=batch_prefixes,
                )
            output_sentences.extend([
                batch[0]["text"]  # type: ignore
                for batch in batch_output_sentences
            ])
        output_scores.extend([
            batch[0]["score"]  # type: ignore
            for batch in batch_output_sentences
        ])
        output_beam_scores.extend([
            batch[0]["beam_score"]  # type: ignore
            for batch in batch_output_sentences
        ])
    constraint_df = test_data.with_columns(
        pl.Series(name="Prediction", values=output_sentences),
        pl.Series(name="Prediction_score", values=output_scores),
        pl.Series(name="Prediction_beam_score", values=output_beam_scores),
    )
    constraint_df = add_code_column(constraint_df, umls_df=umls_df)

    # Compute recall per label
    for label in constraint_df["label"].unique().to_list():
        label_df = constraint_df.filter(pl.col("label") == label)
        true_label = label_df.filter(pl.col("code") == pl.col("Predicted_code")).shape[
            0
        ]
        total_label = label_df.shape[0]
        recall_label = true_label / total_label if total_label > 0 else 0.0
        print(
            f"Label: {label} - Constraint Inference Recall: {recall_label:.4f} ({true_label}/{total_label})"
        )
    # Compute recall overall
    true_overall = constraint_df.filter(
        pl.col("code") == pl.col("Predicted_code")
    ).shape[0]
    total_overall = constraint_df.shape[0]
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
        "--with-group",
        default=False,
        action="store_true",
        help="Use group information if True",
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
            "human_only_ft",
        ],
        help="Whether to use augmented data for training",
    )
    parser.add_argument(
        "--human-ratio",
        type=float,
        default=1.0,
        help="Ratio of human data to use",
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
        human_ratio=args.human_ratio,
        batch_size=args.batch_size,
        output_folder=args.output_folder,
    )
