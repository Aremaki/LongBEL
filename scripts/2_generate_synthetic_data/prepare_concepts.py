"""Prepare concept user prompts for synthetic data generation.

Turns raw UMLS concept + definition parquet files into per-CUI user prompts stored
in chunked parquet files ready for LLM generation (used by generate_synth scripts).

Usage:
    python prepare_concepts.py --mm-path data/MM_2017_all.parquet --mm-def data/UMLS_MM/umls_def.parquet \
         --out-dir data/user_prompts_MM --chunk-size 2500

Outputs: sample_{i}.parquet each containing columns: CUI, user_prompt.
"""

import codecs
import shutil
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from tqdm import tqdm

app = typer.Typer(
    help="Build per-CUI user prompts from concept mentions and definitions."
)


def _load_join(concepts_path: Path, defs_path: Path) -> pl.DataFrame:
    concepts = pl.read_parquet(concepts_path)
    defs = pl.read_parquet(defs_path)
    return concepts.join(defs, on="CUI", how="left")


def _load_train(train_path: Path) -> pl.DataFrame:
    train_df = pl.read_csv(
        train_path,
        separator="\t",
        has_header=True,
        schema_overrides={
            "code": str,  # force as string
        },  # type: ignore
    )
    return train_df


def clean_natural(text: str) -> str:
    return (
        text.replace("\xa0", " ")
        .replace("{", "(")
        .replace("}", ")")
        .replace("[", "(")
        .replace("]", ")")
    )


def _decode_escaped_utf8(text: Optional[str]) -> str:
    """Decode strings that contain escaped UTF-8 byte sequences like '\\xc3\\xb8'.

    This handles cases where UTF-8 bytes were accidentally serialized as escape sequences.
    We safely no-op if decoding fails or the pattern isn't present.
    """
    if text is None:
        return ""
    s = text
    # Fast path: only attempt if we see escaped hex bytes
    if "\\x" in s:
        try:
            # 1) Interpret escape sequences to raw bytes in the Latin-1 range (Ã, ¸, etc.)
            # 2) Convert those to bytes and decode as UTF-8 to recover original characters
            s = (
                codecs.decode(s, "unicode_escape")
                .encode("latin1", errors="ignore")
                .decode("utf-8", errors="ignore")
            )
        except Exception:
            # If anything goes wrong, fall back to original text
            pass
    return s


def build_templates(
    df: pl.DataFrame,
    train_examples: pl.DataFrame,
    include_definitions: bool = True,
) -> pl.DataFrame:
    # ----------------------------------------
    # 1. Build processed_df as before
    # ----------------------------------------
    processed_df = df.group_by("CUI", "CATEGORY", "GROUP", "SEM_NAME").agg(
        pl.col("Title").first().alias("title"),
        pl.col("Entity").unique().alias("mentions"),
        pl.col("DEF").first().alias("definitions"),
        pl.col("Entity").n_unique().alias("mention_count"),
    )

    # ----------------------------------------
    # 2. Prepare train example lookup: label → list of sentences
    # ----------------------------------------
    train_group_sentences = train_examples.group_by("label").agg(
        pl.col("sentence").unique().alias("sentences")
    )

    # ----------------------------------------
    # 3. Join train examples into processed_df
    # ----------------------------------------
    processed_df = processed_df.join(
        train_group_sentences,
        left_on="GROUP",
        right_on="label",
        how="left",
    )

    # ----------------------------------------
    # 4. Function to pick 5 random examples
    # ----------------------------------------
    def pick_examples(sent_list: list[str]) -> list[str]:
        if not sent_list:
            return []
        k = min(5, len(sent_list))
        # Polars-safe random sampling
        return pl.select(
            pl.lit(sent_list).list.sample(n=k, with_replacement=False)
        ).item()

    # ----------------------------------------
    # 5. Add train_example column
    # ----------------------------------------
    processed_df = processed_df.with_columns(
        train_example=pl.col("sentences").map_elements(
            pick_examples,
            return_dtype=pl.List(pl.String),
        )
    )

    # ----------------------------------------
    # 6. definitions/mentions formatting, final template
    # ----------------------------------------
    def format_definitions(defs: list[str]) -> str:
        normalized = [clean_natural(_decode_escaped_utf8(d)) for d in defs]
        return "\n".join(f"* {i + 1}. {d}" for i, d in enumerate(normalized)) + "\n"

    def format_mentions(mentions: list[str]) -> str:
        normalized = [clean_natural(_decode_escaped_utf8(m)) for m in mentions]
        return "'" + "', '".join(normalized) + "'"

    prompt_parts = [
        pl.lit("- **CUI**:\n"),
        pl.col("CUI"),
        pl.lit("\n- **Title**:\n"),
        pl.col("title_processed"),
        pl.lit("\n- **Semantic group**:\n"),
        pl.col("GROUP"),
        pl.lit("\n- **Semantic type**:\n"),
        pl.col("SEM_NAME"),
    ]

    if include_definitions:
        prompt_parts += [
            pl.lit("\n- **Definitions**:\n"),
            pl.col("definitions_processed"),
        ]
    else:
        prompt_parts += [pl.lit("\n")]

    prompt_parts += [
        pl.lit("- **Mentions**:\n"),
        pl.col("mentions_processed"),
        pl.lit("\n"),
    ]

    return (
        processed_df.with_columns(
            title_processed=pl.col("title").map_elements(
                lambda s: clean_natural(_decode_escaped_utf8(s)),
                return_dtype=pl.String,
            ),
            definitions_processed=pl.col("definitions")
            .map_elements(format_definitions, return_dtype=pl.String)
            .fill_null("No definition found\n"),
            mentions_processed=pl.col("mentions").map_elements(
                format_mentions, return_dtype=pl.String
            ),
        )
        .with_columns(user_prompt=pl.concat_str(prompt_parts))
        .select([
            "CUI",
            "CATEGORY",
            "SEM_NAME",
            "user_prompt",
            "train_example",
        ])
    )


def _write_chunks(df: pl.DataFrame, out_dir: Path, chunk_size: int) -> None:
    # delete output dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(0, len(df), chunk_size), desc=f"Writing {out_dir}"):
        chunk = df.slice(i, chunk_size)
        path = out_dir / f"sample_{i}.parquet"
        chunk.write_parquet(path)


@app.command()
def run(
    mm_train_path: Path = typer.Option(None, help="Path to MM training data examples"),
    mm_path: Path = typer.Option(None, help="Path to MM concepts parquet"),
    mm_def: Path = typer.Option(None, help="Path to MM definitions parquet"),
    medline_train_path: Path = typer.Option(
        None, help="Path to Medline training data examples"
    ),
    emea_train_path: Path = typer.Option(
        None, help="Path to EMEA training data examples"
    ),
    quaero_path: Path = typer.Option(None, help="Path to QUAERO concepts parquet"),
    quaero_def: Path = typer.Option(None, help="Path to QUAERO definitions parquet"),
    spaccc_train_path: Path = typer.Option(
        None, help="Path to SPACCC training data examples"
    ),
    spaccc_path: Path = typer.Option(
        None, help="Path to SPACCC concepts parquet (no definitions)"
    ),
    spaccc_umls_path: Path = typer.Option(
        None, help="Path to SPACCC_UMLS concepts parquet"
    ),
    spaccc_umls_def: Path = typer.Option(
        None, help="Path to SPACCC_UMLS definitions parquet"
    ),
    out_mm_def: Path = typer.Option(
        Path("data/synthetic_data/SynthMM/user_prompts_def"),
        help="Output dir for MM prompts with definitions",
    ),
    out_mm_no_def: Path = typer.Option(
        Path("data/synthetic_data/SynthMM/user_prompts_no_def"),
        help="Output dir for MM prompts without definitions",
    ),
    out_quaero_def: Path = typer.Option(
        Path("data/synthetic_data/SynthQUAERO/user_prompts_def"),
        help="Output dir for QUAERO prompts with definitions",
    ),
    out_quaero_no_def: Path = typer.Option(
        Path("data/synthetic_data/SynthQUAERO/user_prompts_no_def"),
        help="Output dir for QUAERO prompts without definitions",
    ),
    out_spaccc_no_def: Path = typer.Option(
        Path("data/synthetic_data/SynthSPACCC/user_prompts_no_def"),
        help="Output dir for SPACCC prompts (no definitions)",
    ),
    out_spaccc_umls_def: Path = typer.Option(
        Path("data/synthetic_data/SynthSPACCC_UMLS/user_prompts_def"),
        help="Output dir for SPACCC prompts (with definitions)",
    ),
    out_spaccc_umls_no_def: Path = typer.Option(
        Path("data/synthetic_data/SynthSPACCC_UMLS/user_prompts_no_def"),
        help="Output dir for SPACCC prompts (no definitions)",
    ),
    shuffle: bool = typer.Option(True, help="Shuffle concepts (sample fraction=1)"),
    chunk_size: int = typer.Option(2500, help="Chunk size for output parquet files"),
) -> None:
    """Generate user prompts for MM, QUAERO, and SPACCC datasets."""
    if not any([
        mm_path and mm_def and mm_train_path,
        quaero_path and quaero_def and medline_train_path and emea_train_path,
        spaccc_path and spaccc_train_path,
        spaccc_umls_path and spaccc_umls_def and spaccc_train_path,
    ]):
        raise typer.BadParameter(
            "Provide at least one dataset (MM, QUAERO, or SPACCC) with its definition file and training data examples."
        )

    # --- MM dataset ---
    if mm_path and mm_def and mm_train_path:
        mm_train = _load_train(mm_train_path)
        mm_joined = _load_join(mm_path, mm_def)
        if "DEF" in mm_joined.columns:
            mm_filtered = mm_joined.filter(pl.col("DEF").is_not_null())
            if shuffle:
                mm_filtered = mm_filtered.sample(fraction=1)
            user_prompt_mm = build_templates(mm_filtered, mm_train)
            _write_chunks(user_prompt_mm, out_mm_def, chunk_size)
            typer.echo(f"MM concepts written to {out_mm_def}")

            mm_no_def = mm_joined.filter(pl.col("DEF").is_null())
            if shuffle:
                mm_no_def = mm_no_def.sample(fraction=1)
            user_prompt_mm_no_def = build_templates(mm_no_def, mm_train)
            _write_chunks(user_prompt_mm_no_def, out_mm_no_def, chunk_size)
            typer.echo(f"MM concepts without definitions written to {out_mm_no_def}")

    # --- QUAERO dataset ---
    if quaero_path and quaero_def and medline_train_path and emea_train_path:
        medline_train = _load_train(medline_train_path)
        emea_train = _load_train(emea_train_path)
        quaero_train = pl.concat([medline_train, emea_train])
        q_joined = _load_join(quaero_path, quaero_def)
        if "DEF" in q_joined.columns:
            q_filtered = q_joined.filter(pl.col("DEF").is_not_null())
            if shuffle:
                q_filtered = q_filtered.sample(fraction=1)
            user_prompt_q = build_templates(q_filtered, quaero_train)
            _write_chunks(user_prompt_q, out_quaero_def, chunk_size)
            typer.echo(f"QUAERO concepts written to {out_quaero_def}")

            q_no_def = q_joined.filter(pl.col("DEF").is_null())
            if shuffle:
                q_no_def = q_no_def.sample(fraction=1)
            user_prompt_q_no_def = build_templates(q_no_def, quaero_train)
            _write_chunks(user_prompt_q_no_def, out_quaero_no_def, chunk_size)
            typer.echo(
                f"QUAERO concepts without definitions written to {out_quaero_no_def}"
            )

    # --- SPACCC dataset (no definitions) ---
    if spaccc_path and spaccc_train_path:
        spaccc_train = _load_train(spaccc_train_path)
        spaccc_df = pl.read_parquet(spaccc_path)
        if shuffle:
            spaccc_df = spaccc_df.sample(fraction=1)
        # Add empty DEF column to maintain compatibility with build_templates
        spaccc_df = spaccc_df.with_columns(pl.lit(None).alias("DEF"))
        # Do not include definitions section for SPACCC prompts
        user_prompt_spaccc = build_templates(
            spaccc_df, spaccc_train, include_definitions=False
        )
        _write_chunks(user_prompt_spaccc, out_spaccc_no_def, chunk_size)
        typer.echo(f"SPACCC concepts written to {out_spaccc_no_def}")

    # --- SPACCC UMLS dataset ---
    if spaccc_umls_path and spaccc_train_path and spaccc_umls_def:
        spaccc_train = _load_train(spaccc_train_path)
        spaccc_joined = _load_join(spaccc_umls_path, spaccc_umls_def)
        if "DEF" in spaccc_joined.columns:
            spaccc_filtered = spaccc_joined.filter(pl.col("DEF").is_not_null())
            if shuffle:
                spaccc_filtered = spaccc_filtered.sample(fraction=1)
            user_prompt_spaccc_umls = build_templates(spaccc_filtered, spaccc_train)
            _write_chunks(user_prompt_spaccc_umls, out_spaccc_umls_def, chunk_size)
            typer.echo(f"SPACCC UMLS concepts written to {out_spaccc_umls_def}")

            spaccc_no_def = spaccc_joined.filter(pl.col("DEF").is_null())
            if shuffle:
                spaccc_no_def = spaccc_no_def.sample(fraction=1)
            user_prompt_spaccc_umls_no_def = build_templates(
                spaccc_no_def, spaccc_train
            )
            _write_chunks(
                user_prompt_spaccc_umls_no_def, out_spaccc_umls_no_def, chunk_size
            )
            typer.echo(
                f"SPACCC UMLS concepts without definitions written to {out_spaccc_umls_no_def}"
            )


if __name__ == "__main__":
    app()
