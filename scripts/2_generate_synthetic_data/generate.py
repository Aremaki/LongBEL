"""Generate synthetic biomedical context sentences for concepts."""

import re
import time
from pathlib import Path

import polars as pl
import torch
import typer
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,  # type: ignore
)

app = typer.Typer(help="Generate synthetic sentences for MM prompts.")


def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_model_and_tokenizer(model_path):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, padding_side="left"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    # Align generation config with tokenizer defaults
    try:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    except Exception:
        pass
    return model, tokenizer


def apply_chat_template(tokenizer, batch_user_prompts, system_prompt):
    """Apply chat template in batch and return a list of serialized prompts."""
    batch_chat = []
    for user_prompt in batch_user_prompts:
        batch_chat.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
    prompts = tokenizer.apply_chat_template(
        batch_chat, tokenize=False, add_generation_prompt=True
    )
    return prompts


def generate_batches(
    model,
    tokenizer,
    user_prompts_df: pl.DataFrame,
    system_prompt: str,
    n_examples: int = 5,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    max_retries: int = 5,
    pattern: str = r"\[([^]]+)\]",
    lang: str = "en",
):
    if lang == "fr":
        template_answer = "exemple : "
    elif lang == "es":
        template_answer = "ejemplo: "
    else:
        template_answer = "example: "
    user_prompts = user_prompts_df["user_prompt"].to_list()
    cui_codes = user_prompts_df["CUI"].to_list()
    sem_cats = user_prompts_df["CATEGORY"].to_list()
    sem_types = user_prompts_df["SEM_NAME"].to_list()
    all_outputs = []
    timing_data = []

    # Pre-compute constants
    device = getattr(model, "device", torch.device("cuda"))
    compiled_regex = re.compile(pattern)

    # Keep prompts within context window when adding new tokens
    context_len = getattr(model.config, "max_position_embeddings", 8192)
    safe_input_max_len = max(256, context_len - max_new_tokens - 32)

    for batch_start in tqdm(
        range(0, len(user_prompts), batch_size), desc="Generating in batches"
    ):
        batch_user_prompts = user_prompts[batch_start : batch_start + batch_size]
        batch_cui_codes = cui_codes[batch_start : batch_start + batch_size]
        batch_sem_cats = sem_cats[batch_start : batch_start + batch_size]
        batch_sem_types = sem_types[batch_start : batch_start + batch_size]

        # Apply chat template and prepare batch prompts
        batch_inputs = apply_chat_template(tokenizer, batch_user_prompts, system_prompt)

        # Tokenize as batch (truncate to fit context window to avoid kv-cache bloat)
        inputs = tokenizer(
            batch_inputs,
            padding=True,
            truncation=True,
            max_length=safe_input_max_len,
            return_tensors="pt",
        )
        # Move to device efficiently
        inputs = {key: val.to(device, non_blocking=True) for key, val in inputs.items()}

        success_mask = [False] * len(batch_user_prompts)
        batch_final_text = [None] * len(batch_user_prompts)

        for attempt in range(1, max_retries + 1):
            if all(success_mask):
                break

            # Build a sub-batch of only the failed items
            failed_indices = [i for i, ok in enumerate(success_mask) if not ok]

            # Try generating; on OOM, halve the sub-batch and retry
            current_indices = failed_indices
            while True:
                sub_inputs = {
                    k: v[current_indices] if hasattr(v, "__getitem__") else v
                    for k, v in inputs.items()
                }
                try:
                    with torch.inference_mode():
                        gen_start_time = time.time()
                        output_tokens = model.generate(
                            **sub_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=0.6,
                            top_p=0.9,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                            use_cache=True,
                        )
                        gen_time = time.time() - gen_start_time
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    new_len = max(1, len(current_indices) // 2)
                    print(f"OOM! Reducing sub-batch size to {new_len}")
                    if new_len == len(current_indices):
                        # Can't reduce further; re-raise to surface the issue
                        raise
                    current_indices = current_indices[:new_len]

            # Decode only the newly generated tokens to reduce CPU work
            n_sub = int(sub_inputs["input_ids"].shape[0])
            input_lens = [int(sub_inputs["input_ids"][i].size(0)) for i in range(n_sub)]
            # Slice the generated portion for each sequence
            gen_only = [output_tokens[i][input_lens[i] :] for i in range(n_sub)]
            # Use batch_decode on lists for better throughput
            decoded_outputs = tokenizer.batch_decode(
                [t.tolist() for t in gen_only], skip_special_tokens=True
            )

            # Minimal batch-level log
            print(
                f"Batch {batch_start} attempt {attempt}: {gen_time:.2f}s, sub-batch {n_sub}"
            )

            # Post-process each failed item
            for sub_i, decoded_output in enumerate(decoded_outputs):
                i = current_indices[sub_i]
                decoded_text = decoded_output.strip()

                examples = [
                    line.split(template_answer, 1)[1]
                    for line in decoded_text.split("\n")
                    if template_answer in line and bool(compiled_regex.search(line))
                ]

                if isinstance(examples, list) and len(examples) >= n_examples:
                    success_mask[i] = True
                    batch_final_text[i] = "\n".join(examples[:n_examples])  # type: ignore
                else:
                    # Keep best-effort text for debugging; will retry if attempts left
                    batch_final_text[i] = f"FAIL !!\n\n{decoded_text}"  # type: ignore

                # Timing and metrics per sequence
                new_tokens = len(gen_only[sub_i])
                tokens_per_second = (new_tokens / gen_time) if gen_time > 0 else 0.0
                timing_data.append({
                    "total_new_tokens": int(new_tokens),
                    "tokens_per_second": float(tokens_per_second),
                    "time_per_cui": gen_time / max(1, len(failed_indices)),
                })

        # Collect all outputs
        for cui, sem_cat, sem_type, out_text in zip(
            batch_cui_codes, batch_sem_cats, batch_sem_types, batch_final_text
        ):
            all_outputs.append((cui, sem_cat, sem_type, out_text))

    # Summary
    if timing_data:
        avg_tps = sum(t["tokens_per_second"] for t in timing_data) / len(timing_data)
        avg_sec_per_cui = sum(t["time_per_cui"] for t in timing_data) / len(timing_data)
        total_time = sum(t["time_per_cui"] for t in timing_data)
        total_tokens = sum(t["total_new_tokens"] for t in timing_data)

        print("\n=== Summary Statistics ===")
        print(f"Average tokens/second: {avg_tps:.2f}")
        print(f"Average seconds per CUI: {avg_sec_per_cui:.3f}")
        print(f"Total generation time: {total_time:.3f}")
        print(f"Total tokens generated: {total_tokens:,}")

    return pl.DataFrame(
        all_outputs, schema=["CUI", "CATEGORY", "SEM_NAME", "llm_output"], orient="row"
    )


@app.command()
def run(
    chunk: int = typer.Option(..., help="Chunk index used in input/output filenames"),
    user_prompts_dir: Path = typer.Option(
        Path("data/user_prompts_MM"), help="Directory containing sample_{chunk}.parquet"
    ),
    out_dir: Path = typer.Option(
        Path("data/SynthMM"), help="Directory to write synthesized parquet"
    ),
    model_path: Path = typer.Option(
        Path("models/Llama-3.3-70B-Instruct"), help="Model path"
    ),
    system_prompt_path: Path = typer.Option(
        Path("scripts/2_generate_synthetic_data/prompts/system_prompt_mm.txt"),
        help="System prompt text file",
    ),
    max_new_tokens: int = 512,
    batch_size: int = 16,
    max_retries: int = 5,
    n_examples: int = 5,
) -> None:
    """Generate synthetic sentences for a chunk of user prompts."""
    user_prompts_path = user_prompts_dir / f"sample_{chunk}.parquet"
    if not user_prompts_path.exists():
        raise typer.BadParameter(f"Missing input file: {user_prompts_path}")
    user_prompts_df = pl.read_parquet(user_prompts_path)

    model, tokenizer = load_model_and_tokenizer(str(model_path))  # type: ignore
    system_prompt = load_system_prompt(system_prompt_path)
    if "quaero" in system_prompt_path.name:
        lang = "fr"
    elif "spaccc" in system_prompt_path.name:
        lang = "es"
    else:
        lang = "en"
    result_df = generate_batches(
        model,
        tokenizer,
        user_prompts_df,
        system_prompt=system_prompt,
        n_examples=n_examples,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_retries=max_retries,
        lang=lang,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"sample_{chunk}.parquet"
    result_df.write_parquet(result_path)
    typer.echo(f"✅ Parquet written: {result_path}")


if __name__ == "__main__":
    app()
