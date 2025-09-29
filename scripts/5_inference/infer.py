import argparse
import gc
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm
from transformers import GenerationConfig  # type: ignore

from syncabel.guided_inference import get_prefix_allowed_tokens_fn
from syncabel.models import MT5_GENRE, Bart_GENRE, MBart_GENRE
from syncabel.trie import Trie

sys.setrecursionlimit(5000)


def simple_reset_memory():
    """
    Basic cleanup to free up memory after generation.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_memory(tag=""):
    allocated = torch.cuda.memory_allocated() / 1e9  # actively used by tensors
    reserved = torch.cuda.memory_reserved() / 1e9  # reserved by allocator
    free = torch.cuda.get_device_properties(0).total_memory / 1e9 - reserved
    print(
        f"[{tag}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Free: {free:.2f} GB"
    )


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def safe_generation(
    model,
    sources,
    num_beams,
    model_name,
    full_path,
    device,
    best,
    prefix_allowed_tokens_fn=None,
):
    """
    Ultra-conservative generation with multiple recovery strategies
    """
    try:
        print_memory("Before Generation")
        with torch.no_grad():
            results = model.sample(
                sources,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                num_return_sequences=1,
            )
        print_memory("After Generation")
        simple_reset_memory()
        print_memory("After Simple Reset")
        return results
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"OOM error for batch of size {len(sources)}")
            print_memory("OOM Error - Before simple cleanup")
            simple_reset_memory()
            print_memory("OOM Error - After simple cleanup")
            # Delete Traceback to avoid memory leak
            # Break the traceback reference
            tb = sys.exc_info()[2]
            del tb
            del e
            print_memory("OOM Error - After traceback delete")
            print("Processing items individually...")
            results = []
            for i, single_source in enumerate(sources):
                try:
                    with torch.no_grad():
                        print_memory(f"Before item {i} Generation")
                        single_result = model.sample(
                            [single_source],
                            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                            num_beams=num_beams,
                            num_return_sequences=1,
                        )
                        print_memory(f"After item {i} Generation")
                        simple_reset_memory()
                        print_memory(f"After item {i} Simple Reset")
                    results.extend(single_result)
                except RuntimeError as single_e:
                    if "CUDA out of memory" in str(single_e):
                        print(f"Single item {i} failed")
                        print_memory(f"Error on item {i}")
                        print(single_source)
                        raise single_e
                    else:
                        raise single_e
            return results
        else:
            raise e


def load_model(model_name, full_path, device, best):
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
    print(
        f"Model {model_name} {'best' if best else 'last'} checkpoint is loaded to {device}"
    )
    return model, decoder_start_token_id


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
    gc.collect()
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
    model, decoder_start_token_id = load_model(model_name, full_path, device, best)

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
    batch_size = 256
    for i in tqdm(
        range(0, len(test_data["source"]), batch_size), desc="Processing Test Data"
    ):
        batch_sources = test_data["source"][i : i + batch_size]
        batch_output_sentences = safe_generation(
            model=model,
            sources=batch_sources,
            num_beams=num_beams,
            prefix_allowed_tokens_fn=None,
            model_name=model_name,
            full_path=full_path,
            device=device,
            best=best,
        )
        output_sentences.extend(batch_output_sentences)  # type: ignore

    print(f"Generated {len(output_sentences)} sentences without constraint.")

    # Save results
    output_path = f"{full_path}/pred_test_no_constraint_{num_beams}_beams.pkl"
    with open(output_path, "wb") as file:
        pickle.dump(output_sentences, file, protocol=-1)

    # Perform inference with constraint
    output_sentences = []
    batch_size = 64
    for i in tqdm(
        range(0, len(test_data["source"]), batch_size), desc="Processing Test Data"
    ):
        batch_sources = test_data["source"][i : i + batch_size]
        prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
            model,
            batch_sources,
            candidates_trie=trie_legal_tokens,
        )
        batch_output_sentences = safe_generation(
            model=model,
            sources=batch_sources,
            num_beams=num_beams,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            model_name=model_name,
            full_path=full_path,
            device=device,
            best=best,
        )
        output_sentences.extend(batch_output_sentences)  # type: ignore
        del batch_output_sentences
        torch.cuda.empty_cache()
        gc.collect()
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
