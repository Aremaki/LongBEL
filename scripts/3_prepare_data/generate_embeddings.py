"""Precompute and store embeddings for UMLS synonyms and mentions using CODER.

Outputs parquet files with columns: text, embedding (list[float]), and optional metadata.
"""

import json
from pathlib import Path

import polars as pl
import typer
from datasets import load_dataset

from syncabel.embeddings import TextEncoder, save_embeddings_parquet  # type: ignore

app = typer.Typer(
    help="Precompute embeddings for synonyms (UMLS) and mentions (datasets)"
)


def _yield_mentions_from_bigbio(ds) -> list[str]:
    mentions: list[str] = []
    for page in ds:
        for ent in page.get("entities", []):
            if ent.get("text"):
                mentions.append(ent["text"][0])
    return mentions


def _yield_mentions_from_synth(json_path: Path) -> list[str]:
    if not json_path.exists():
        return []
    data = json.loads(json_path.read_text(encoding="utf-8"))
    mentions: list[str] = []
    for page in data:
        for ent in page.get("entities", []):
            if ent.get("text"):
                mentions.append(ent["text"][0])
    return mentions


# No local encoder helpers; we reuse TextEncoder from syncabel.embeddings


@app.command()
def run(
    umls_parquet_mm: Path = typer.Option(
        Path("data/UMLS_processed/MM/all_disambiguated.parquet"),
        help="UMLS processed parquet (MedMentions space)",
    ),
    umls_parquet_quaero: Path = typer.Option(
        Path("data/UMLS_processed/QUAERO/all_disambiguated.parquet"),
        help="UMLS processed parquet (QUAERO space)",
    ),
    out_dir: Path = typer.Option(Path("data/embeddings"), help="Output directory"),
    coder_model: str = typer.Option(
        "GanjinZero/coder-large-v3", help="CODER encoder model name"
    ),
    include_datasets: list[str] = typer.Option(
        ["MedMentions", "EMEA", "MEDLINE"], help="Datasets to include"
    ),
    include_synth: bool = typer.Option(True, help="Include synthetic JSONs if present"),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = TextEncoder(model_name=coder_model)

    # UMLS synonyms (MM and QUAERO)
    for name, parquet in {
        "MM": umls_parquet_mm,
        "QUAERO": umls_parquet_quaero,
    }.items():
        if parquet.exists():
            df = pl.read_parquet(parquet)
            syns = (
                df.select(pl.col("Entity")).to_series().drop_nulls().unique().to_list()
            )
            syn_embs = encoder.encode(syns)
            save_embeddings_parquet(
                syns, syn_embs, out_dir / f"umls_synonyms_{name}.parquet"
            )

    # Mentions from datasets
    ds_map = {
        "MedMentions": ("bigbio/medmentions", "medmentions_st21pv_bigbio_kb"),
        "EMEA": ("bigbio/quaero", "quaero_emea_bigbio_kb"),
        "MEDLINE": ("bigbio/quaero", "quaero_medline_bigbio_kb"),
    }
    for ds_name in include_datasets:
        if ds_name not in ds_map:
            continue
        hf_id, cfg = ds_map[ds_name]
        ds = load_dataset(hf_id, name=cfg)
        all_mentions: list[str] = []
        for split in [k for k in ["train", "validation", "test"] if k in ds]:
            all_mentions += _yield_mentions_from_bigbio(ds[split])
        if all_mentions:
            m_embs = encoder.encode(all_mentions)
            save_embeddings_parquet(
                all_mentions,
                m_embs,
                out_dir / f"mentions_{ds_name}.parquet",
                extra_cols={"dataset": [ds_name] * len(all_mentions)},
            )

    # Mentions from synthetic JSONs
    if include_synth:
        synth_map = {
            "SynthMM": Path("data/synthetic_data/SynthMM/SynthMM_bigbio.json"),
            "SynthQUAERO": Path(
                "data/synthetic_data/SynthQUAERO/SynthQUAERO_bigbio.json"
            ),
        }
        for name, path in synth_map.items():
            mentions = _yield_mentions_from_synth(path)
            if mentions:
                m_embs = encoder.encode(mentions)
                save_embeddings_parquet(
                    mentions,
                    m_embs,
                    out_dir / f"mentions_{name}.parquet",
                    extra_cols={"dataset": [name] * len(mentions)},
                )

    typer.echo("✅ Embeddings built and saved.")


if __name__ == "__main__":  # pragma: no cover
    app()
