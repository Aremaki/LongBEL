import argparse
import os
import pickle
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm
from transformers import GenerationConfig  # type: ignore

from syncabel.guided_inference import get_prefix_allowed_tokens_fn
from syncabel.models import MT5_GENRE, Bart_GENRE, MBart_GENRE
from syncabel.trie import Trie


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def main(
    model_name,
    num_beams,
    best,
    dataset_name,
    selection_method,
    with_group=False,
    augmented_data=False,
):
    # Set device
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = (
        Path("models")
        / "NED"
        / f"{dataset_name}_{'augmented' if augmented_data else 'original'}_{selection_method}{'_with_group' if with_group else ''}"
        / model_name
    )
    if best:
        full_path = model_path / "model_best"
    else:
        full_path = model_path / "model_last"

    if "mt5" in model_name:
        model = MT5_GENRE.from_pretrained(full_path).eval().to(device)  # type: ignore
        decoder_start_token_id = 0
        model.generation_config = GenerationConfig(
            decoder_start_token_id=decoder_start_token_id,
            eos_token_id=1,
            forced_eos_token_id=1,
            pad_token_id=0,
        )
    elif "mbart" in model_name:
        model = MBart_GENRE.from_pretrained(full_path).eval().to(device)  # type: ignore
        decoder_start_token_id = 2
        model.generation_config = GenerationConfig(
            bos_token_id=0,
            decoder_start_token_id=decoder_start_token_id,
            eos_token_id=2,
            forced_eos_token_id=2,
            pad_token_id=1,
        )
    else:
        model = Bart_GENRE.from_pretrained(full_path).eval().to(device)  # type: ignore
        decoder_start_token_id = 2
        model.generation_config = GenerationConfig(
            bos_token_id=0,
            decoder_start_token_id=decoder_start_token_id,
            eos_token_id=2,
            forced_eos_token_id=2,
            pad_token_id=1,
        )

    # Load data
    # Load and preprocess data
    with_group_extension = "_with_group" if with_group else ""
    data_folder = Path("data/final_data")

    test_source_data = load_pickle(
        data_folder
        / dataset_name
        / f"test_{selection_method}_source{with_group_extension}.pkl"
    )
    test_target_data = load_pickle(
        data_folder / dataset_name / f"test_{selection_method}_target.pkl"
    )

    test_data = {"source": test_source_data, "target": test_target_data}

    # Load candidate Trie
    tries_folder = Path("data/UMLS_tries")
    tries_folder.mkdir(parents=True, exist_ok=True)
    trie_path = tries_folder / f"trie_{dataset_name}_{model_name}.pkl"
    if os.path.exists(trie_path):  # Check if the file exists
        with open(trie_path, "rb") as file:
            trie_legal_tokens = pickle.load(file)
    else:
        # Compute candidate Trie
        start_idx = 1 if "bart" in model_name else 0
        if dataset_name == "MedMentions":
            legal_umls_token = pl.read_parquet(
                Path("data/UMLS_processed/MM/all_disambiguated.parquet")
            )
        else:
            legal_umls_token = pl.read_parquet(
                Path("data/UMLS_processed/QUAERO/all_disambiguated.parquet")
            )
        legal_umls_token = legal_umls_token.with_columns(
            pl.col("Entity")
            .str.replace_all("\xa0", " ", literal=True)
            .str.replace_all("{", "(", literal=True)
            .str.replace_all("}", ")", literal=True)
            .str.replace_all("[", "(", literal=True)
            .str.replace_all("]", ")", literal=True)
        )
        trie_legal_tokens = {}
        for category in legal_umls_token["GROUP"].unique().to_list():
            print(f"processing {category}")
            cat_legal_umls_token = legal_umls_token.filter(pl.col("GROUP") == category)
            trie_legal_tokens[category] = Trie([
                [decoder_start_token_id] + model.tokenizer.encode(entity)[start_idx:-1]  # type: ignore
                for entity in cat_legal_umls_token["Entity"].to_list()
            ])

        # Save it
        # Create directory if it doesn't exist
        os.makedirs(trie_path.parent, exist_ok=True)
        with open(trie_path, "wb") as file:
            pickle.dump(trie_legal_tokens, file, protocol=-1)

    # Perform inference without constraint
    output_sentences = []
    batch_size = 32
    for i in tqdm(
        range(0, len(test_data["source"]), batch_size), desc="Processing Test Data"
    ):
        batch_sources = test_data["source"][i : i + batch_size]
        batch_output_sentences = model.sample(
            batch_sources,
            num_beams=num_beams,
            num_return_sequences=1,
        )
        output_sentences.extend(batch_output_sentences)
    print(f"Generated {len(output_sentences)} sentences without constraint.")

    # Save results
    output_path = f"{full_path}/pred_test_no_constraint_{num_beams}_beams.pkl"
    with open(output_path, "wb") as file:
        pickle.dump(output_sentences, file, protocol=-1)

    # Perform inference with constraint
    output_sentences = []
    batch_size = 32
    for i in tqdm(
        range(0, len(test_data["source"]), batch_size), desc="Processing Test Data"
    ):
        batch_sources = test_data["source"][i : i + batch_size]
        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
            model,
            batch_sources,
            candidates_trie=trie_legal_tokens,
        )
        batch_output_sentences = model.sample(
            batch_sources,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_return_sequences=1,
        )
        output_sentences.extend(batch_output_sentences)
    print(f"Generated {len(output_sentences)} sentences with constraint.")

    # Save results
    output_path = f"{full_path}/pred_test_constraint_{num_beams}_beams_typed.pkl"
    with open(output_path, "wb") as file:
        pickle.dump(output_sentences, file, protocol=-1)

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
        "--with-group",
        default=False,
        action="store_true",
        help="Use group information if True",
    )
    parser.add_argument(
        "--augmented-data",
        default=False,
        action="store_true",
        help="Use augmented data if True",
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # Pass the parsed argument to the main function
    main(
        args.model_name,
        args.num_beams,
        args.best,
        args.dataset_name,
        args.selection_method,
        args.with_group,
        args.augmented_data,
    )
