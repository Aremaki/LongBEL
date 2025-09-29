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


def print_kv_cache_memory_usage(model, tag=""):
    """Print memory usage of KV cache in the model"""
    print(f"\n{'=' * 60}")
    print(f"KV CACHE MEMORY USAGE - {tag}")
    print(f"{'=' * 60}")

    total_kv_memory = 0
    has_kv_cache = False

    # Check if model is in inference mode with past_key_values
    if hasattr(model, "past_key_values") and model.past_key_values is not None:
        has_kv_cache = True
        print("Model has past_key_values attribute")

    # Iterate through model layers to find KV cache
    for name, module in model.named_modules():
        if hasattr(module, "past_key_value") and module.past_key_value is not None:
            has_kv_cache = True
            layer_kv_memory = 0

            if isinstance(module.past_key_value, (list, tuple)):
                for i, kv_tensor in enumerate(module.past_key_value):
                    if (
                        kv_tensor is not None
                        and torch.is_tensor(kv_tensor)
                        and kv_tensor.is_cuda
                    ):
                        kv_memory = (
                            kv_tensor.element_size() * kv_tensor.nelement() / 1e9
                        )
                        layer_kv_memory += kv_memory
                        print(
                            f"Layer {name} - KV[{i}]: {kv_tensor.size()} | Memory: {kv_memory:.4f} GB"
                        )

            total_kv_memory += layer_kv_memory

    # Check for past_key_values in model's state
    if hasattr(model, "past_key_values") and model.past_key_values is not None:
        model_kv_memory = 0
        if isinstance(model.past_key_values, (list, tuple)):
            for layer_idx, layer_past in enumerate(model.past_key_values):
                if layer_past is not None:
                    for kv_idx, kv_tensor in enumerate(layer_past):
                        if (
                            kv_tensor is not None
                            and torch.is_tensor(kv_tensor)
                            and kv_tensor.is_cuda
                        ):
                            kv_memory = (
                                kv_tensor.element_size() * kv_tensor.nelement() / 1e9
                            )
                            model_kv_memory += kv_memory
                            print(
                                f"Model past_key_values - Layer {layer_idx} KV[{kv_idx}]: {kv_tensor.size()} | Memory: {kv_memory:.4f} GB"
                            )

        total_kv_memory += model_kv_memory

    if not has_kv_cache:
        print(
            "No KV cache found - model might not be using caching or is in training mode"
        )
    else:
        print(f"{'Total KV Cache Memory:':45} {total_kv_memory:.4f} GB")

    print(f"{'=' * 60}")
    return total_kv_memory


def clear_kv_cache(model):
    """Clear the KV cache from the model"""
    # Clear model-level past_key_values
    if hasattr(model, "past_key_values"):
        model.past_key_values = None

    # Clear module-level past_key_value
    for module in model.modules():
        if hasattr(module, "past_key_value"):
            module.past_key_value = None

    # For newer transformer versions, also clear _past_key_values
    if hasattr(model, "_past_key_values"):
        model._past_key_values = None

    print("✅ KV cache cleared")


def analyze_generation_memory_breakdown(model, inputs, generation_outputs=None):
    """Analyze memory breakdown during generation"""
    print(f"\n{'=' * 80}")
    print("GENERATION MEMORY BREAKDOWN")
    print(f"{'=' * 80}")

    # Model parameters memory
    model_memory = (
        sum(p.element_size() * p.nelement() for p in model.parameters() if p.is_cuda)
        / 1e9
    )

    # Input memory
    input_memory = 0
    if isinstance(inputs, dict):
        for key, tensor in inputs.items():
            if torch.is_tensor(tensor) and tensor.is_cuda:
                input_memory += tensor.element_size() * tensor.nelement() / 1e9
    elif torch.is_tensor(inputs) and inputs.is_cuda:
        input_memory = inputs.element_size() * inputs.nelement() / 1e9

    # KV cache memory
    kv_memory = print_kv_cache_memory_usage(model, "During analysis")

    # Output memory (if provided)
    output_memory = 0
    if generation_outputs is not None:
        if hasattr(generation_outputs, "sequences") and torch.is_tensor(
            generation_outputs.sequences
        ):
            output_memory = (
                generation_outputs.sequences.element_size()
                * generation_outputs.sequences.nelement()
                / 1e9
            )

        # Beam search scores memory
        if hasattr(generation_outputs, "sequences_scores") and torch.is_tensor(
            generation_outputs.sequences_scores
        ):
            scores_memory = (
                generation_outputs.sequences_scores.element_size()
                * generation_outputs.sequences_scores.nelement()
                / 1e9
            )
            output_memory += scores_memory

    print(f"{'Model Parameters:':30} {model_memory:8.4f} GB")
    print(f"{'Input Tensors:':30} {input_memory:8.4f} GB")
    print(f"{'KV Cache:':30} {kv_memory:8.4f} GB")
    print(f"{'Output Tensors:':30} {output_memory:8.4f} GB")
    print(
        f"{'Total Estimated:':30} {model_memory + input_memory + kv_memory + output_memory:8.4f} GB"
    )

    # Compare with actual GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"{'Actual GPU Allocated:':30} {allocated:8.4f} GB")
        print(
            f"{'Difference:':30} {allocated - (model_memory + input_memory + kv_memory + output_memory):8.4f} GB"
        )

    print(f"{'=' * 80}")


def get_tensor_memory_usage():
    """Get memory usage of all tensors currently in memory"""
    tensor_memory = defaultdict(list)

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                tensor = obj.data if hasattr(obj, "data") else obj

                if tensor.is_cuda:
                    size_bytes = tensor.element_size() * tensor.nelement()
                    size_gb = size_bytes / 1e9

                    tensor_info = {
                        "size": tensor.size(),
                        "dtype": str(tensor.dtype),
                        "device": str(tensor.device),
                        "memory_gb": size_gb,
                        "type": type(obj).__name__,
                    }

                    # Group by type and size for easier reading
                    key = (
                        f"{type(obj).__name__}_{str(tensor.size())}_{str(tensor.dtype)}"
                    )
                    tensor_memory[key].append(tensor_info)

        except Exception as e:
            continue

    return tensor_memory


def print_tensor_memory_usage(tag=""):
    """Print detailed tensor memory usage"""
    print(f"\n{'=' * 80}")
    print(f"TENSOR MEMORY USAGE - {tag}")
    print(f"{'=' * 80}")

    tensor_memory = get_tensor_memory_usage()
    total_memory = 0

    for key, tensors in sorted(
        tensor_memory.items(),
        key=lambda x: sum(t["memory_gb"] for t in x[1]),
        reverse=True,
    ):
        group_memory = sum(t["memory_gb"] for t in tensors)
        total_memory += group_memory
        count = len(tensors)

        print(f"{key}:")
        print(f"  Count: {count}")
        print(f"  Total Memory: {group_memory:.4f} GB")
        print(f"  Avg per tensor: {group_memory / count:.4f} GB")
        if tensors:
            print(f"  Example: {tensors[0]}")
        print()

    print(f"TOTAL TENSOR MEMORY: {total_memory:.4f} GB")
    print(f"{'=' * 80}")


def get_object_memory_summary():
    """Get memory usage by object type"""
    object_memory = defaultdict(lambda: {"count": 0, "memory_gb": 0})

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                tensor = obj.data if hasattr(obj, "data") else obj

                if tensor.is_cuda:
                    size_bytes = tensor.element_size() * tensor.nelement()
                    size_gb = size_bytes / 1e9

                    obj_type = type(obj).__name__
                    object_memory[obj_type]["count"] += 1
                    object_memory[obj_type]["memory_gb"] += size_gb  # type: ignore

        except Exception as e:
            continue

    return object_memory


def print_object_memory_summary(tag=""):
    """Print memory usage summary by object type"""
    print(f"\n{'=' * 60}")
    print(f"OBJECT MEMORY SUMMARY - {tag}")
    print(f"{'=' * 60}")

    object_memory = get_object_memory_summary()
    total_memory = 0

    for obj_type, info in sorted(
        object_memory.items(), key=lambda x: x[1]["memory_gb"], reverse=True
    ):
        memory_gb = info["memory_gb"]
        count = info["count"]
        total_memory += memory_gb
        print(f"{obj_type:30} | Count: {count:4d} | Memory: {memory_gb:8.4f} GB")

    print(f"{'Total':30} | {' ':15} | Memory: {total_memory:8.4f} GB")
    print(f"{'=' * 60}")


def find_large_tensors(threshold_gb=0.1):
    """Find individual tensors larger than threshold"""
    print(f"\nLARGE TENSORS (> {threshold_gb} GB):")
    print(f"{'=' * 80}")

    large_tensors = []

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                tensor = obj.data if hasattr(obj, "data") else obj

                if tensor.is_cuda:
                    size_bytes = tensor.element_size() * tensor.nelement()
                    size_gb = size_bytes / 1e9

                    if size_gb > threshold_gb:
                        large_tensors.append({
                            "object": obj,
                            "type": type(obj).__name__,
                            "size": tensor.size(),
                            "dtype": str(tensor.dtype),
                            "memory_gb": size_gb,
                            "id": id(obj),
                        })

        except Exception as e:
            continue

    # Sort by size descending
    large_tensors.sort(key=lambda x: x["memory_gb"], reverse=True)

    for tensor_info in large_tensors:
        print(
            f"Type: {tensor_info['type']:20} | Size: {str(tensor_info['size']):20} | "
            f"Memory: {tensor_info['memory_gb']:6.3f} GB | ID: {tensor_info['id']}"
        )

    return large_tensors


def clear_memory_comprehensive():
    """Comprehensive memory cleanup"""
    # Clear any cached gradients
    if hasattr(torch, "grad"):
        torch.grad = None  # type: ignore

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Force Python garbage collection
    gc.collect()

    # Clear CUDA cache again after GC
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def print_model_memory_usage(model, tag=""):
    """Print memory usage of model parameters and buffers"""
    print(f"\n{'=' * 50}")
    print(f"MODEL MEMORY USAGE - {tag}")
    print(f"{'=' * 50}")

    total_param_memory = 0
    total_buffer_memory = 0

    # Model parameters
    for name, param in model.named_parameters():
        if param.is_cuda:
            param_memory = param.element_size() * param.nelement() / 1e9
            total_param_memory += param_memory
            if param_memory > 0.01:  # Only print large parameters
                print(f"Param: {name:50} | Memory: {param_memory:6.3f} GB")

    # Model buffers
    for name, buffer in model.named_buffers():
        if buffer.is_cuda:
            buffer_memory = buffer.element_size() * buffer.nelement() / 1e9
            total_buffer_memory += buffer_memory
            if buffer_memory > 0.01:  # Only print large buffers
                print(f"Buffer: {name:49} | Memory: {buffer_memory:6.3f} GB")

    print(f"{'Total Parameters:':50} | Memory: {total_param_memory:6.3f} GB")
    print(f"{'Total Buffers:':50} | Memory: {total_buffer_memory:6.3f} GB")
    print(
        f"{'Total Model:':50} | Memory: {total_param_memory + total_buffer_memory:6.3f} GB"
    )
    print(f"{'=' * 50}")


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


def safe_generation(model, sources, num_beams, prefix_allowed_tokens_fn=None):
    """
    Ultra-conservative generation with multiple recovery strategies
    """
    try:
        print_memory("Before generation")
        print_kv_cache_memory_usage(model, "Before generation")
        print_object_memory_summary("Before generation")
        print_tensor_memory_usage("Before generation")
        with torch.no_grad():
            results = model.sample(
                sources,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                num_return_sequences=1,
            )
        print_memory("After successful generation")
        print_kv_cache_memory_usage(model, "After successful generation")
        print_object_memory_summary("After successful generation")
        print_tensor_memory_usage("After successful generation")

        # Clear memory immediately after generation
        clear_memory_comprehensive()
        print_memory("After cleanup")
        print_kv_cache_memory_usage(model, "After cleanup")
        print_object_memory_summary("After cleanup")
        print_tensor_memory_usage("After cleanup")
        return results
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"OOM error for batch of size {len(sources)}")
            print_memory("OOM Error - Before cleanup")

            # Detailed memory analysis at error time
            print_object_memory_summary("OOM Error - Object Summary")
            print_tensor_memory_usage("OOM Error - Tensor Details")
            large_tensors = find_large_tensors(0.01)  # Find tensors > 10MB
            print(f"Number of large tensors: {len(large_tensors)}")
            print_model_memory_usage(model, "OOM Error - Model State")
            print_kv_cache_memory_usage(model, "OOM Error - KV Cache")
            # Aggressive cleanup
            clear_memory_comprehensive()
            # Detailed memory analysis at error time
            print_object_memory_summary("OOM Error after cleanup - Object Summary")
            print_tensor_memory_usage("OOM Error after cleanup - Tensor Details")
            large_tensors = find_large_tensors(0.01)  # Find tensors > 10MB
            print(f"Number of large tensors after cleanup: {len(large_tensors)}")
            print_model_memory_usage(model, "OOM Error after cleanup - Model State")
            print_kv_cache_memory_usage(model, "OOM Error after cleanup - KV Cache")
            clear_kv_cache(model)
            print_memory("OOM Error - After cleanup and KV clear")
            print_kv_cache_memory_usage(model, "After KV clear")
            # Try single item processing
            print("Processing items individually...")
            results = []
            for i, single_source in enumerate(sources):
                try:
                    with torch.no_grad():
                        single_result = model.sample(
                            [single_source],
                            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                            num_beams=num_beams,
                            num_return_sequences=1,
                        )
                    results.extend(single_result)
                    # Aggressive cleanup after each item
                    print_object_memory_summary(f"After single item {i}")
                    print_tensor_memory_usage(f"After single item {i}")
                    large_tensors = find_large_tensors(0.01)  # Find tensors > 10MB
                    print(
                        f"Number of large tensors after item {i} : {len(large_tensors)}"
                    )
                    print_model_memory_usage(model, f"After single item {i}")
                    print_memory(f"After single item {i}")
                    clear_memory_comprehensive()
                except RuntimeError as single_e:
                    if "CUDA out of memory" in str(single_e):
                        print(f"Single item {i} failed")
                        print_memory(f"Error on item {i}")
                        print_object_memory_summary(f"After single item {i}")
                        print_tensor_memory_usage(f"After single item {i}")
                        large_tensors = find_large_tensors(0.01)  # Find tensors > 10MB
                        print(
                            f"Number of large tensors after item {i} : {len(large_tensors)}"
                        )
                        print_model_memory_usage(model, f"After single item {i}")
                        print(single_source)
                        raise single_e
                    else:
                        raise single_e
            return results
        else:
            raise e


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
    batch_size = 64
    for i in tqdm(
        range(0, len(test_data["source"]), batch_size), desc="Processing Test Data"
    ):
        batch_sources = test_data["source"][i : i + batch_size]
        batch_output_sentences = safe_generation(model, batch_sources, num_beams)
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
            model,
            batch_sources,
            num_beams,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
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
