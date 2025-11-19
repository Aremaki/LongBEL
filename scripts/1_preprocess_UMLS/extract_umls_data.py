"""Typer CLI to extract and prepare UMLS resources for SynCABEL.

Subcommands:
  codes       -> Extract unique CUIs from MRCONSO.RRF
  semantic    -> Extract semantic types from MRSTY.RRF and enrich with category info
  definitions -> Extract definitions from MRDEF.RRF (older QUAERO release used here)
  synonyms    -> Extract preferred titles and synonyms (EN / FR / main)
  all         -> Run the full pipeline

Examples:
  python extract_umls_data.py all --umls-zip /path/UMLS_2025AA.zip
  python extract_umls_data.py codes --umls-zip /path/UMLS_2017AB.zip
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

import polars as pl
import typer
from tqdm import tqdm

app = typer.Typer(help="Extract UMLS data components into parquet files.")


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _iter_rrf(zip_path: Path, inner_path: str) -> Iterable[list[str]]:
    with zipfile.ZipFile(zip_path) as zf:
        folder_name = zip_path.parts[-1].split(".")[0]
        with zf.open(folder_name + "/" + inner_path, mode="r") as fh:
            for raw in fh:  # streaming, not loading entire file
                parts = str(raw)[2:-3].split("|")  # preserve original parsing behavior
                yield parts


@app.command()
def codes(
    umls_zip: Path = typer.Option(
        ...,
        exists=True,
        readable=True,
        help="Path to UMLS release zip containing MRCONSO.RRF inside UMLS/",
    ),
    out_dir: Path = typer.Option(Path("data/UMLS"), help="Output directory"),
) -> None:
    """Extract unique CUIs to umls_codes.parquet."""
    _ensure_out_dir(out_dir)
    cuis: list[str] = []
    for parts in tqdm(_iter_rrf(umls_zip, "MRCONSO.RRF"), desc="CUIs"):
        if parts and parts[0]:
            cuis.append(parts[0])
    pl.DataFrame({"CUI": cuis}).unique().write_parquet(out_dir / "umls_codes.parquet")
    typer.echo(f"Saved {out_dir / 'umls_codes.parquet'}")


@app.command()
def semantic(
    umls_zip: Path = typer.Option(
        ...,
        exists=True,
        readable=True,
        help="Path to UMLS release zip containing MRSTY.RRF",
    ),
    semantic_group_path: Path | None = typer.Option(
        None, exists=True, readable=True, help="Path to semantic_group.csv"
    ),
    out_dir: Path = typer.Option(Path("data/UMLS"), help="Output directory"),
) -> None:
    """Extract semantic types and enrich with coarse category mapping."""
    _ensure_out_dir(out_dir)
    rows = {"CUI": [], "SEM_CODE": [], "TREE_CODE": [], "SEM_NAME": []}
    for parts in tqdm(_iter_rrf(umls_zip, "MRSTY.RRF"), desc="Semantic"):
        if len(parts) >= 4:
            rows["CUI"].append(parts[0])
            rows["SEM_CODE"].append(parts[1])
            rows["TREE_CODE"].append(parts[2])
            rows["SEM_NAME"].append(parts[3])
    df = pl.DataFrame(rows, orient="row").unique()
    if semantic_group_path:
        df_sem_info = pl.read_csv(
            semantic_group_path,
            separator="|",
            has_header=False,
            new_columns=["CATEGORY", "GROUP", "SEM_CODE", "SEM_NAME"],
        )
        df_sem_info.write_parquet(
            out_dir / "semantic_info.parquet"
        )  # save standalone mapping
        df = df.join(df_sem_info, on=["SEM_CODE", "SEM_NAME"], how="left")
    out_file = out_dir / "umls_semantic.parquet"
    df.write_parquet(out_file)
    typer.echo(f"Saved {out_file}")


@app.command()
def definitions(
    umls_zip: Path = typer.Option(
        ..., exists=True, readable=True, help="Path to UMLS zip containing MRDEF.RRF"
    ),
    out_dir: Path = typer.Option(Path("data/UMLS"), help="Output directory"),
) -> None:
    """Extract definitions grouped per CUI to umls_def.parquet."""
    _ensure_out_dir(out_dir)
    rows = {"CUI": [], "DEF": []}
    for parts in tqdm(_iter_rrf(umls_zip, "MRDEF.RRF"), desc="Definitions"):
        if len(parts) > 5:
            rows["CUI"].append(parts[0])
            rows["DEF"].append(parts[5])
    df = pl.DataFrame(rows).unique().group_by("CUI").agg([pl.col("DEF").unique()])
    out_file = out_dir / "umls_def.parquet"
    df.write_parquet(out_file)
    typer.echo(f"Saved {out_file}")


def _decode_term(term: str) -> str:
    # Preserve original multi-step decoding pipeline
    return (
        term.encode("utf-8")
        .decode("unicode_escape")
        .encode("latin1", errors="ignore")
        .decode("utf-8", errors="ignore")
    )


@app.command()
def synonyms(
    umls_zip: Path = typer.Option(
        ..., exists=True, readable=True, help="Path to UMLS zip containing MRCONSO.RRF"
    ),
    out_dir: Path = typer.Option(Path("data/UMLS"), help="Output directory"),
) -> None:
    """Extract preferred titles (EN/FR/Main) and synonyms into umls_title_syn.parquet."""
    _ensure_out_dir(out_dir)
    preferred_title = {"code": [], "CUI": [], "UMLS_Title_preferred": []}
    main_title = {"code": [], "CUI": [], "UMLS_Title_main": []}
    fr_title = {"code": [], "CUI": [], "UMLS_Title_fr": []}
    en_title = {"code": [], "CUI": [], "UMLS_Title_en": []}
    fr_syn = {"code": [], "CUI": [], "UMLS_alias_fr": []}
    en_syn = {"code": [], "CUI": [], "UMLS_alias_en": []}
    for parts in tqdm(_iter_rrf(umls_zip, "MRCONSO.RRF"), desc="Synonyms"):
        if len(parts) < 15:
            continue
        cui = parts[0]
        lat = parts[1]
        ts = parts[2]
        sab = parts[11]
        tty = parts[12]
        term = _decode_term(parts[14])
        if tty == "PN":
            preferred_title["CUI"].append(cui)
            preferred_title["UMLS_Title_preferred"].append(term)
        if sab == "MTH":
            main_title["CUI"].append(cui)
            main_title["UMLS_Title_main"].append(term)
        elif lat == "FRE":
            if ts == "P":
                fr_title["CUI"].append(cui)
                fr_title["UMLS_Title_fr"].append(term)
            else:
                fr_syn["CUI"].append(cui)
                fr_syn["UMLS_alias_fr"].append(term)
        elif lat == "ENG":
            if ts == "P":
                en_title["CUI"].append(cui)
                en_title["UMLS_Title_en"].append(term)
            else:
                en_syn["CUI"].append(cui)
                en_syn["UMLS_alias_en"].append(term)

    fr_syn_df = pl.DataFrame(fr_syn).unique()
    en_syn_df = pl.DataFrame(en_syn).unique()
    fr_title_df = pl.DataFrame(fr_title).unique()
    en_title_df = pl.DataFrame(en_title).unique()
    main_title_df = pl.DataFrame(main_title).unique()
    preferred_title_df = pl.DataFrame(preferred_title).unique()

    title_df = (
        fr_title_df.join(en_title_df, how="full", on=["CUI", "code"], coalesce=True)
        .join(main_title_df, how="full", on=["CUI", "code"], coalesce=True)
        .join(preferred_title_df, how="full", on=["CUI", "code"], coalesce=True)
        .group_by(["CUI", "code"])
        .agg([
            pl.col("UMLS_Title_main").unique(),
            pl.col("UMLS_Title_fr").unique(),
            pl.col("UMLS_Title_en").unique(),
            pl.col("UMLS_Title_preferred").unique(),
        ])
    )
    syn_df = (
        fr_syn_df.join(en_syn_df, how="full", on=["CUI", "code"], coalesce=True)
        .group_by(["CUI", "code"])
        .agg([
            pl.col("UMLS_alias_fr").unique(),
            pl.col("UMLS_alias_en").unique(),
        ])
    )
    title_syn_df = title_df.join(syn_df, how="full", on=["CUI", "code"], coalesce=True)

    # Process best title
    # Prefer the shortest title; if lengths are equal, prefer one starting with uppercase.
    # Implement tie-breaker by adding +1 penalty when first char is lowercase.
    title_syn_df = title_syn_df.with_columns(
        min_idx_main=pl.col("UMLS_Title_main")
        .list.eval(
            pl.when(pl.element().is_null())
            .then(1_000_000_000)
            .otherwise(
                pl.element().str.len_chars()
                - pl.when(
                    (
                        pl.element().str.slice(0, 1).str.to_uppercase()
                        == pl.element().str.slice(0, 1)
                    )  # first letter uppercase
                    & (
                        pl.element().str.slice(1).str.to_lowercase()
                        == pl.element().str.slice(1)
                    )  # rest all lowercase
                )
                .then(0.5)
                .otherwise(0)
            )
        )
        .list.arg_min()
    )
    title_syn_df = title_syn_df.with_columns(
        shortest_title_main=pl.col("UMLS_Title_main").list.get(pl.col("min_idx_main"))
    ).drop("min_idx_main")
    title_syn_df = title_syn_df.with_columns(
        min_idx_en=pl.col("UMLS_Title_en")
        .list.eval(
            pl.when(pl.element().is_null())
            .then(1_000_000_000)
            .otherwise(
                pl.element().str.len_chars()
                - pl.when(
                    (
                        pl.element().str.slice(0, 1).str.to_uppercase()
                        == pl.element().str.slice(0, 1)
                    )  # first letter uppercase
                    & (
                        pl.element().str.slice(1).str.to_lowercase()
                        == pl.element().str.slice(1)
                    )  # rest all lowercase
                )
                .then(0.5)
                .otherwise(0)
            )
        )
        .list.arg_min()
    )
    title_syn_df = title_syn_df.with_columns(
        shortest_title=pl.col("UMLS_Title_en").list.get(pl.col("min_idx_en"))
    ).drop("min_idx_en")
    title_syn_df = title_syn_df.with_columns(
        pl.when(pl.col("UMLS_Title_preferred").list.get(0).is_not_null())
        .then(pl.col("UMLS_Title_preferred").list.get(0))
        .when(
            (pl.col("UMLS_Title_main").list.len() == 1)
            & (pl.col("UMLS_Title_main").list.get(0).is_not_null())
        )
        .then(pl.col("UMLS_Title_main").list.get(0))
        .when(pl.col("shortest_title_main").is_not_null())
        .then(pl.col("shortest_title_main"))
        .when(
            (pl.col("UMLS_Title_en").list.len() == 1)
            & (pl.col("UMLS_Title_en").list.get(0).is_not_null())
        )
        .then(pl.col("UMLS_Title_en").list.get(0))
        .when(pl.col("shortest_title").is_not_null())
        .then(pl.col("shortest_title"))
        .when(
            (pl.col("UMLS_Title_fr").list.len() == 1)
            & (pl.col("UMLS_Title_fr").list.get(0).is_not_null())
        )
        .then(pl.col("UMLS_Title_fr").list.get(0))
        .otherwise(None)
        .alias("Title")
    ).drop("shortest_title_main", "shortest_title", "UMLS_Title_preferred")
    out_file = out_dir / "umls_title_syn.parquet"
    title_syn_df.write_parquet(out_file)
    typer.echo(f"Saved {out_file}")


@app.command()
def all(
    umls_zip: Path = typer.Option(
        ...,
        exists=True,
        readable=True,
        help="Path to UMLS release zip (MRCONSO.RRF & MRSTY.RRF)",
    ),
    semantic_group_path: Path | None = typer.Option(
        None, exists=True, readable=True, help="Path to semantic_group.csv"
    ),
    out_dir: Path = typer.Option(Path("data/UMLS"), help="Output directory"),
) -> None:
    """Run full extraction pipeline."""
    codes(umls_zip=umls_zip, out_dir=out_dir)
    semantic(
        umls_zip=umls_zip, out_dir=out_dir, semantic_group_path=semantic_group_path
    )
    definitions(umls_zip=umls_zip, out_dir=out_dir)
    synonyms(umls_zip=umls_zip, out_dir=out_dir)


if __name__ == "__main__":  # pragma: no cover
    app()
