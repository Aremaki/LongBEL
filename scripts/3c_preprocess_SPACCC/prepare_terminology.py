import logging
import random
from pathlib import Path

import polars as pl
import typer

app = typer.Typer(help="Preprocess SPACCC terminology to resolve ambiguous entities.")


def resolve_entity_ambiguity(
    df: pl.DataFrame, priority_pairs: set[tuple[str, str]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Resolve ambiguous entities by selecting one CUI per (Entity, GROUP) combination.

    Strategy:
    1. Keep entities with only one CUI
    2. For ambiguous entities, prefer priority CUIs
    3. Maximize CUI coverage by avoiding duplicates

    Args:
        df: Terminology dataframe with columns [CUI, Entity, CATEGORY, GROUP, is_main]
        priority_pairs: Set of (CUI, GROUP) pairs to prioritize

    Returns:
        df_clean: DataFrame with one CUI per entity
        mapping_df: Mapping from discarded CUIs to chosen CUIs
    """

    # Step 1: Count CUIs per entity and identify unambiguous cases
    entity_cui_counts = (
        df.group_by(["Entity", "GROUP"])
        .agg(pl.col("CUI").n_unique().alias("n_CUI"))
        .join(df, on=["Entity", "GROUP"])
    )

    # Stage 1: Directly unambiguous entities (only one CUI)
    unambiguous_entities = (
        entity_cui_counts.filter(pl.col("n_CUI") == 1)
        .with_columns(ambiguous_level=pl.lit(0))
        .select(["CUI", "Entity", "CATEGORY", "GROUP", "is_main", "ambiguous_level"])
    )

    # Track which (CUI, GROUP) pairs we've already assigned
    assigned_pairs = set(unambiguous_entities.select(["CUI", "GROUP"]).iter_rows())

    # Stage 2: Entities that become unambiguous after removing already assigned CUIs
    ambiguous_entities = entity_cui_counts.filter(pl.col("n_CUI") > 1)

    # Filter out rows where CUI+GROUP is already assigned to another entity
    filtered_ambiguous = ambiguous_entities.filter(
        ~pl.struct(["CUI", "GROUP"]).map_elements(
            lambda x: (x["CUI"], x["GROUP"]) in assigned_pairs,
            return_dtype=pl.Boolean,
        )
    )

    # Now some entities might have only one CUI left
    became_unambiguous = (
        filtered_ambiguous.group_by(["Entity", "GROUP"])
        .agg(pl.col("CUI").n_unique().alias("n_CUI"))
        .join(filtered_ambiguous, on=["Entity", "GROUP"])
        .filter(pl.col("n_CUI") == 1)
        .with_columns(ambiguous_level=pl.lit(1))
        .select(["CUI", "Entity", "CATEGORY", "GROUP", "is_main", "ambiguous_level"])
    )

    # Update assigned pairs
    assigned_pairs.update(set(became_unambiguous.select(["CUI", "GROUP"]).iter_rows()))

    # Combine unambiguous entities
    all_unambiguous = pl.concat([unambiguous_entities, became_unambiguous])

    # Stage 3: Handle remaining truly ambiguous entities
    remaining_ambiguous = (
        filtered_ambiguous.group_by(["Entity", "GROUP"])
        .agg(pl.col("CUI").n_unique().alias("n_CUI"))
        .join(filtered_ambiguous, on=["Entity", "GROUP"])
        .filter(pl.col("n_CUI") > 1)
    )

    # Mark priority pairs and sort
    prioritized_ambiguous = remaining_ambiguous.with_columns(
        is_priority=pl.struct(["CUI", "GROUP"]).map_elements(
            lambda x: (x["CUI"], x["GROUP"]) in priority_pairs, return_dtype=pl.Boolean
        )
    ).sort(
        ["Entity", "GROUP", "is_priority", "CUI"],
        descending=[False, False, True, False],
    )

    # Greedy selection: pick first available CUI for each entity
    final_selections = []

    for (_, _), group_df in prioritized_ambiguous.group_by([
        "Entity",
        "GROUP",
    ]):
        group_rows = group_df.to_dicts()

        # Try to find an unassigned CUI
        selected_row = None
        for row in group_rows:
            if (row["CUI"], row["GROUP"]) not in assigned_pairs:
                selected_row = row
                assigned_pairs.add((row["CUI"], row["GROUP"]))
                break

        # Fallback: use first row (highest priority)
        if selected_row is None:
            selected_row = group_rows[0]

        final_selections.append(selected_row)

    # Create final ambiguous selections dataframe
    ambiguous_final = (
        pl.DataFrame(final_selections)
        .with_columns(ambiguous_level=pl.lit(2))
        .select(["CUI", "Entity", "CATEGORY", "GROUP", "is_main", "ambiguous_level"])
    )

    # Combine all results
    df_clean = pl.concat([all_unambiguous, ambiguous_final])

    # Create mapping from discarded CUIs to chosen CUIs
    original_triplets = df.select(["CUI", "Entity", "GROUP"])
    chosen_triplets = df_clean.select(
        pl.col("CUI").alias("chosen_CUI"), "Entity", "GROUP"
    )

    mapping_df = (
        original_triplets.join(chosen_triplets, on=["Entity", "GROUP"])
        .filter(pl.col("CUI") != pl.col("chosen_CUI"))
        .select(["CUI", "chosen_CUI", "GROUP"])
        .unique()
    )

    return df_clean, mapping_df


def load_priority_pairs(train_file: Path, test_file: Path) -> set[tuple[str, str]]:
    """
    Load and expand priority (CUI, GROUP) pairs from train and test files.

    Handles composite CUIs like '123+456+NO_CODE' by splitting them.
    """

    def extract_pairs_from_file(file_path: Path) -> set[tuple[str, str]]:
        df = pl.read_csv(file_path, separator="\t", quote_char=None)
        pairs = set()

        for code, label in zip(df["code"], df["label"]):
            code = str(code).strip()
            label = str(label).strip()

            # Split composite codes like "397974008+32696007"
            for sub_cui in code.split("+"):
                sub_cui = sub_cui.strip()
                if sub_cui and sub_cui != "NO_CODE":
                    pairs.add((sub_cui, label))

        return pairs

    train_pairs = extract_pairs_from_file(train_file)
    test_pairs = extract_pairs_from_file(test_file)

    all_pairs = train_pairs | test_pairs  # Union of sets

    logging.info(f"Loaded {len(all_pairs)} unique priority (CUI, GROUP) pairs")
    return all_pairs


def load_terminology(terminology_file: Path) -> pl.DataFrame:
    """Load and preprocess the terminology file."""

    df = pl.read_csv(
        terminology_file,
        separator="\t",
        null_values=["NO_CODE"],
        schema_overrides={
            "code": pl.Utf8,
            "term": pl.Utf8,
            "semantic_tag": pl.Utf8,
        },
    )

    df_clean = (
        df.filter(pl.col("code").is_not_null())
        .with_columns([
            pl.col("code").alias("CUI").cast(pl.Utf8),
            pl.col("term").alias("Entity"),
            pl.col("semantic_tag").alias("GROUP"),
            pl.col("semantic_tag").str.slice(0, 4).alias("CATEGORY"),
            pl.col("mainterm").cast(pl.Boolean).alias("is_main"),
        ])
        .select(["CUI", "Entity", "CATEGORY", "GROUP", "is_main"])
    )

    logging.info(f"Loaded {df_clean.height} terminology entries")
    return df_clean


def validate_coverage(
    df_final: pl.DataFrame, priority_pairs: set[tuple[str, str]]
) -> set[tuple[str, str]]:
    """
    Validate that priority pairs are covered in the final terminology.

    Returns missing pairs for mapping generation.
    """
    final_pairs = set(df_final.select(["CUI", "GROUP"]).iter_rows())
    missing_pairs = priority_pairs - final_pairs

    if not missing_pairs:
        logging.info("✅ All priority pairs covered in final terminology")
        return set()

    # Log missing pairs by group
    missing_by_group = (
        pl.DataFrame([{"CUI": cui, "GROUP": group} for cui, group in missing_pairs])
        .group_by("GROUP")
        .agg(pl.len().alias("missing_count"))
    )

    for row in missing_by_group.iter_rows():
        logging.warning(f"⚠️ {row[1]} priority pairs missing for GROUP '{row[0]}'")

    # Log sample missing pairs
    sample_missing = random.sample(list(missing_pairs), min(5, len(missing_pairs)))
    for cui, group in sample_missing:
        logging.debug(f"Missing: (CUI: {cui}, GROUP: {group})")

    return missing_pairs


@app.command()
def main(
    spaccc_dir: Path = typer.Option(
        Path("data/SPACCC"),
        "--spaccc-dir",
        "-i",
        help="Path to SPACCC data directory with terminology.tsv, train.tsv, test.tsv",
    ),
    corrected_dir: Path = typer.Option(
        Path("data/corrected_cui"),
        "--corrected-dir",
        "-c",
        help="Directory for corrected CUI mapping file",
    ),
    terminology_dir: Path = typer.Option(
        Path("data/UMLS_processed/SPACCC"),
        "--terminology-dir",
        "-t",
        help="Directory for filtered terminology file",
    ),
):
    """Preprocess SPACCC terminology to resolve ambiguous entities."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Define file paths
    terminology_file = spaccc_dir / "terminology.tsv"
    train_file = spaccc_dir / "train.tsv"
    test_file = spaccc_dir / "test.tsv"
    corrected_cui_file = corrected_dir / "SPACCC_adapted.csv"
    clean_terminology_file = terminology_dir / "all_disambiguated.parquet"

    # Create output directory
    corrected_dir.mkdir(exist_ok=True)
    terminology_dir.mkdir(exist_ok=True)

    logging.info(f"Processing data from: {spaccc_dir}")

    try:
        # Load priority pairs
        priority_pairs = load_priority_pairs(train_file, test_file)

        # Load terminology
        terminology_df = load_terminology(terminology_file)

        # Resolve ambiguity
        clean_terminology, mapping_df = resolve_entity_ambiguity(
            terminology_df, priority_pairs
        )

        # Validate coverage
        missing_pairs = validate_coverage(clean_terminology, priority_pairs)

        _ = validate_coverage(terminology_df, priority_pairs)

        # Filter mapping to only include missing priority pairs
        if missing_pairs:
            mapping_df = (
                mapping_df.filter(
                    pl.struct(["CUI", "GROUP"]).map_elements(
                        lambda x: (x["CUI"], x["GROUP"]) in missing_pairs,
                        return_dtype=pl.Boolean,
                    )
                )
                .unique(subset=["CUI", "GROUP"])
                .select(["CUI", "chosen_CUI"])
            )

        # Save results
        logging.info(
            f"💾 Saving {mapping_df.height} mapping entries to {corrected_cui_file}"
        )
        mapping_df.write_csv(corrected_cui_file)

        logging.info(
            f"💾 Saving {clean_terminology.height} clean terminology entries to {clean_terminology_file}"
        )
        clean_terminology.write_parquet(clean_terminology_file)

        logging.info("📊 Final statistics:")
        logging.info(f"   - Unique entities: {clean_terminology['Entity'].n_unique()}")
        logging.info(f"   - Unique CUIs: {clean_terminology['CUI'].n_unique()}")
        logging.info(
            f"   - Ambiguity levels: {clean_terminology['ambiguous_level'].value_counts()}"
        )

        logging.info("✅ Processing completed successfully")

    except Exception as e:
        logging.error(f"❌ Processing failed: {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
