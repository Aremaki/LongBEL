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
    Resolve ambiguous entities by selecting one SNOMED code per (Entity, GROUP) combination.

    Strategy:
    1. Keep entities with only one SNOMED code
    2. For ambiguous entities, prefer priority SNOMED codes
    3. Maximize SNOMED code coverage by avoiding duplicates

    Args:
        df: Terminology dataframe with columns [SNOMED_code, Entity, CATEGORY, GROUP, is_main]
        priority_pairs: Set of (SNOMED_code, GROUP) pairs to prioritize
    Returns:
        df_clean: DataFrame with one SNOMED code per entity
        mapping_df: Mapping from discarded SNOMED codes to chosen SNOMED codes
    """

    # Step 1: Count SNOMED codes per entity and identify unambiguous cases
    entity_code_counts = (
        df.group_by(["Entity", "GROUP"])
        .agg(pl.col("SNOMED_code").n_unique().alias("n_codes"))
        .join(df, on=["Entity", "GROUP"])
    )

    # Stage 1: Directly unambiguous entities (only one SNOMED code)
    unambiguous_entities = (
        entity_code_counts.filter(pl.col("n_codes") == 1)
        .with_columns(ambiguous_level=pl.lit(0))
        .select([
            "SNOMED_code",
            "Entity",
            "CATEGORY",
            "GROUP",
            "is_main",
            "ambiguous_level",
            "UMLS_CUI",
        ])
    )

    # Track which (SNOMED_code, GROUP) pairs we've already assigned
    assigned_pairs = set(
        unambiguous_entities.select(["SNOMED_code", "GROUP"]).iter_rows()
    )

    # Stage 2: Entities that become unambiguous after removing already assigned SNOMED codes
    ambiguous_entities = entity_code_counts.filter(pl.col("n_codes") > 1)

    # Filter out rows where SNOMED_code+GROUP is already assigned to another entity
    filtered_ambiguous = ambiguous_entities.filter(
        ~pl.struct(["SNOMED_code", "GROUP"]).map_elements(
            lambda x: (x["SNOMED_code"], x["GROUP"]) in assigned_pairs,
            return_dtype=pl.Boolean,
        )
    )

    # Now some entities might have only one SNOMED code left
    became_unambiguous = (
        filtered_ambiguous.group_by(["Entity", "GROUP"])
        .agg(pl.col("SNOMED_code").n_unique().alias("n_codes"))
        .join(filtered_ambiguous, on=["Entity", "GROUP"])
        .filter(pl.col("n_codes") == 1)
        .with_columns(ambiguous_level=pl.lit(1))
        .select([
            "SNOMED_code",
            "Entity",
            "CATEGORY",
            "GROUP",
            "is_main",
            "ambiguous_level",
            "UMLS_CUI",
        ])
    )

    # Update assigned pairs
    assigned_pairs.update(
        set(became_unambiguous.select(["SNOMED_code", "GROUP"]).iter_rows())
    )

    # Combine unambiguous entities
    all_unambiguous = pl.concat([unambiguous_entities, became_unambiguous])

    assigned_pairs = set(all_unambiguous.select(["SNOMED_code", "GROUP"]).iter_rows())
    assigned_pairs_entity = set(all_unambiguous.select(["Entity", "GROUP"]).iter_rows())

    # Stage 3: Handle remaining truly ambiguous entities
    remaining_ambiguous = df.join(
        all_unambiguous.select(["SNOMED_code", "GROUP", "Entity"]),
        on=["SNOMED_code", "GROUP", "Entity"],
        how="anti",
    )

    # Mark priority pairs and sort
    prioritized_ambiguous = remaining_ambiguous.with_columns(
        is_priority=pl.struct(["SNOMED_code", "GROUP"]).map_elements(
            lambda x: (x["SNOMED_code"], x["GROUP"]) in priority_pairs,
            return_dtype=pl.Boolean,
        )
    ).sort(
        ["Entity", "GROUP", "is_priority", "SNOMED_code"],
        descending=[False, False, True, False],
    )

    # Greedy selection: pick first available SNOMED code for each entity
    final_selections = []

    for (_, _), group_df in prioritized_ambiguous.group_by(
        ["Entity", "GROUP"], maintain_order=True
    ):
        group_rows = group_df.to_dicts()

        # Try to find an unassigned SNOMED code
        selected_row = None
        for row in group_rows:
            if (row["SNOMED_code"], row["GROUP"]) not in assigned_pairs:
                selected_row = row
                assigned_pairs.add((row["SNOMED_code"], row["GROUP"]))
                break
            if (row["Entity"], row["GROUP"]) not in assigned_pairs_entity:
                selected_row = row
                assigned_pairs_entity.add((row["Entity"], row["GROUP"]))
                break

        # Fallback: use first row (highest priority)
        if selected_row is not None:
            final_selections.append(selected_row)

    # Create final ambiguous selections dataframe
    if len(final_selections) > 0:
        ambiguous_final = (
            pl.DataFrame(final_selections)
            .with_columns(ambiguous_level=pl.lit(2))
            .select([
                "SNOMED_code",
                "Entity",
                "CATEGORY",
                "GROUP",
                "is_main",
                "ambiguous_level",
                "UMLS_CUI",
            ])
        )
    else:
        # explicit empty dataframe with expected schema so later selects/concats won't fail
        ambiguous_final = pl.DataFrame(
            schema=[
                ("SNOMED_code", pl.Utf8),
                ("Entity", pl.Utf8),
                ("CATEGORY", pl.Utf8),
                ("GROUP", pl.Utf8),
                ("is_main", pl.Boolean),
                ("ambiguous_level", pl.Int32),
                ("UMLS_CUI", pl.Utf8),
            ]
        )

    # Combine all results
    # Ensure all_unambiguous has same columns as ambiguous_final (if empty ensure schema)
    if all_unambiguous.height == 0:
        all_unambiguous = pl.DataFrame(
            schema=[
                ("SNOMED_code", pl.Utf8),
                ("Entity", pl.Utf8),
                ("CATEGORY", pl.Utf8),
                ("GROUP", pl.Utf8),
                ("is_main", pl.Boolean),
                ("ambiguous_level", pl.Int32),
                ("UMLS_CUI", pl.Utf8),
            ]
        )
    # Combine all results
    df_clean = pl.concat([all_unambiguous, ambiguous_final])

    # Create mapping from discarded SNOMED codes to chosen SNOMED codes
    original_triplets = df.select(["SNOMED_code", "Entity", "GROUP"])
    chosen_triplets = df_clean.select(
        pl.col("SNOMED_code").alias("chosen_SNOMED_code"), "Entity", "GROUP"
    )

    mapping_df = (
        original_triplets.join(chosen_triplets, on=["Entity", "GROUP"])
        .filter(pl.col("SNOMED_code") != pl.col("chosen_SNOMED_code"))
        .select(["SNOMED_code", "chosen_SNOMED_code", "GROUP"])
        .unique()
    )

    return df_clean, mapping_df


def load_priority_pairs(train_file: Path, test_file: Path) -> set[tuple[str, str]]:
    """
    Load and expand priority (SNOMED_code, GROUP) pairs from train and test files.

    Handles composite SNOMED codes like '123+456+NO_CODE' by splitting them.
    """

    def extract_pairs_from_file(file_path: Path) -> set[tuple[str, str]]:
        df = pl.read_csv(
            file_path,
            separator="\t",
            quote_char=None,
            schema_overrides={"code": str, "mention_id": str, "filename": str},  # type: ignore
        )
        pairs = set()

        for code, label in zip(df["code"], df["label"]):
            code = str(code).strip()
            label = str(label).strip()

            # Split composite codes like "397974008+32696007"
            for sub_snomed_code in code.split("+"):
                sub_snomed_code = sub_snomed_code.strip()
                if sub_snomed_code and sub_snomed_code != "NO_CODE":
                    pairs.add((sub_snomed_code, label))

        return pairs

    train_pairs = extract_pairs_from_file(train_file)
    test_pairs = extract_pairs_from_file(test_file)

    all_pairs = train_pairs | test_pairs  # Union of sets

    logging.info(f"Loaded {len(all_pairs)} unique priority (SNOMED_code, GROUP) pairs")
    return all_pairs


def load_terminology(terminology_file: Path) -> pl.DataFrame:
    """Load and preprocess the terminology file."""

    df = pl.read_csv(
        terminology_file,
        separator="\t",
        null_values=["NO_CODE"],
        schema_overrides={
            "code": pl.Utf8,
            "SNOMED_code": pl.Utf8,
            "term": pl.Utf8,
            "semantic_tag": pl.Utf8,
        },
    )

    df_clean = (
        df.filter(pl.col("code").is_not_null())
        .with_columns([
            pl.col("code").alias("SNOMED_code").cast(pl.Utf8),
            pl.col("CUI").alias("UMLS_CUI").cast(pl.Utf8),
            pl.col("term").alias("Entity"),
            pl.col("semantic_tag").alias("GROUP"),
            pl.col("semantic_tag").str.slice(0, 4).alias("CATEGORY"),
            pl.col("mainterm").cast(pl.Boolean).alias("is_main"),
        ])
        .select(["SNOMED_code", "Entity", "CATEGORY", "GROUP", "is_main", "UMLS_CUI"])
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
    final_pairs = set(df_final.select(["SNOMED_code", "GROUP"]).iter_rows())
    missing_pairs = priority_pairs - final_pairs

    if not missing_pairs:
        logging.info("✅ All priority pairs covered in final terminology")
        return set()

    # Log missing pairs by group
    missing_by_group = (
        pl.DataFrame([
            {"SNOMED_code": snomed_code, "GROUP": group}
            for snomed_code, group in missing_pairs
        ])
        .group_by("GROUP")
        .agg(pl.len().alias("missing_count"))
    )

    for row in missing_by_group.iter_rows():
        logging.warning(f"⚠️ {row[1]} priority pairs missing for GROUP '{row[0]}'")

    # Log sample missing pairs
    sample_missing = random.sample(list(missing_pairs), min(5, len(missing_pairs)))
    for snomed_code, group in sample_missing:
        logging.debug(f"Missing: (SNOMED_code: {snomed_code}, GROUP: {group})")

    return missing_pairs


def augment_with_missing_pairs(
    df_final: pl.DataFrame, missing_pairs: set[tuple[str, str]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Duplicate SNOMED codes for missing (SNOMED_code, GROUP) pairs using a single canonical row per SNOMED_code.
    CATEGORY is derived from GROUP.
    """

    if not missing_pairs:
        return df_final, pl.DataFrame({})

    # Build a tiny frame of missing pairs with derived CATEGORY
    missing_df = (
        pl.DataFrame([
            {"SNOMED_code": snomed_code, "GROUP": group}
            for snomed_code, group in missing_pairs
        ])
        .with_columns([
            pl.col("SNOMED_code").cast(pl.Utf8),
            pl.col("GROUP").cast(pl.Utf8),
            pl.col("GROUP").str.slice(0, 4).alias("CATEGORY"),
        ])
        .sort(["SNOMED_code", "GROUP"])
    )

    # Pick one canonical row per SNOMED_code (prefer is_main=True)
    base_per_snomed_code = (
        df_final.with_columns([
            pl.col("is_main").fill_null(False),
        ])
        .sort(
            ["SNOMED_code", "is_main", "Entity"],
            descending=[False, True, False],
        )
        .unique(subset=["SNOMED_code"], keep="first")
        .select(["SNOMED_code", "Entity", "is_main", "UMLS_CUI"])
    )

    # Create duplicates by joining missing pairs with canonical rows
    dup_df = missing_df.join(base_per_snomed_code, on="SNOMED_code", how="left").select([
        "SNOMED_code",
        "Entity",
        "CATEGORY",
        "GROUP",
        "is_main",
        "UMLS_CUI",
    ])

    # If some SNOMED_code truly doesn't exist in df_final (edge case), keep but Entity will be null
    # Filter those out to avoid invalid rows
    dup_df = dup_df.filter(pl.col("Entity").is_not_null())

    if dup_df.is_empty():
        return df_final, pl.DataFrame({})

    augmented = pl.concat([df_final, dup_df], how="align")
    logging.info(
        f"➕ Added {dup_df.height} duplicated rows to cover missing priority pairs"
    )
    return augmented, dup_df


def _complete_snomed_entity_group_combinations(df: pl.DataFrame) -> pl.DataFrame:
    """
    For each SNOMED code, ensure all its associated entities appear for all its associated groups.
    """
    logging.info("Completing SNOMED-Entity-GROUP combinations...")

    # Create a frame with all combinations of SNOMED, Entity, and GROUP
    all_combinations = df.select(["SNOMED_code", "Entity", "GROUP"]).unique()

    # For each SNOMED_code, get all its unique entities and groups
    snomed_to_entities = all_combinations.group_by("SNOMED_code").agg(
        pl.col("Entity").unique().alias("all_entities")
    )
    snomed_to_groups = all_combinations.group_by("SNOMED_code").agg(
        pl.col("GROUP").unique().alias("all_groups")
    )

    # Create the full cartesian product of entities and groups for each SNOMED_code
    full_product = (
        snomed_to_entities.join(snomed_to_groups, on="SNOMED_code")
        .explode("all_entities")
        .explode("all_groups")
        .rename({"all_entities": "Entity", "all_groups": "GROUP"})
        .sort(["SNOMED_code", "Entity", "GROUP"])
    )

    # Use the original data as the source of truth for other columns
    # and join it with the full product.
    # This will put nulls where a combination did not originally exist.
    completed_df = full_product.join(
        df.unique(subset=["SNOMED_code", "Entity", "GROUP"]),
        on=["SNOMED_code", "Entity", "GROUP"],
        how="left",
    )

    # Forward fill missing values within each SNOMED-Entity group.
    # This assumes that for a given SNOMED-Entity pair, other attributes are consistent.
    completed_df = completed_df.with_columns(
        pl.all().forward_fill().over(["SNOMED_code", "Entity"])
    )

    # For any remaining nulls (e.g., a SNOMED-Entity pair only existed with nulls), backfill.
    completed_df = completed_df.with_columns(
        pl.all().backward_fill().over(["SNOMED_code", "Entity"])
    )

    # Ensure CATEGORY is consistent with GROUP
    completed_df = completed_df.with_columns(CATEGORY=pl.col("GROUP").str.slice(0, 4))

    logging.info(
        f"DF size before completion: {df.height}, after: {completed_df.height}"
    )
    return completed_df.unique()


@app.command()
def main(
    spaccc_dir: Path = typer.Option(
        Path("data/termino_raw/SPACCC"),
        "--spaccc-dir",
        "-i",
        help="Path to Terminology data directory with terminology.tsv and FSN files",
    ),
    corrected_dir: Path = typer.Option(
        Path("data/corrected_code"),
        "--corrected-dir",
        "-c",
        help="Directory for corrected code mapping file",
    ),
    terminology_dir: Path = typer.Option(
        Path("data/termino_processed/SPACCC"),
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
    corrected_code_file = corrected_dir / "SPACCC_adapted.csv"
    clean_terminology_file = terminology_dir / "all_disambiguated.parquet"
    semantic_info_file = terminology_dir / "semantic_info.parquet"
    # Create output directory
    corrected_dir.mkdir(exist_ok=True)
    terminology_dir.mkdir(exist_ok=True)

    logging.info(f"Processing data from: {spaccc_dir}")

    # Load priority pairs
    priority_pairs = load_priority_pairs(train_file, test_file)

    # Load terminology
    terminology_df = load_terminology(terminology_file)

    # Validate coverage (pre-augmentation)
    missing_pairs_pre = validate_coverage(terminology_df, priority_pairs)

    # If some (SNOMED_code, GROUP) exist in train/test and the SNOMED_code is in the terminology but not with the corresponding GROUP
    # duplicate the SNOMED_code under the requested GROUP to ensure coverage
    if missing_pairs_pre:
        terminology_df, added_rows_df = augment_with_missing_pairs(
            terminology_df, missing_pairs_pre
        )
    else:
        added_rows_df = pl.DataFrame({})

    # Complete SNOMED-Entity-GROUP combinations
    terminology_df = _complete_snomed_entity_group_combinations(terminology_df)

    # Resolve ambiguity
    clean_terminology, mapping_df = resolve_entity_ambiguity(
        terminology_df, priority_pairs
    )

    # Validate coverage (pre-augmentation)
    missing_pairs = validate_coverage(clean_terminology, priority_pairs)

    # Filter mapping to only include missing priority pairs
    if missing_pairs:
        mapping_df = (
            mapping_df.filter(
                pl.struct(["SNOMED_code", "GROUP"]).map_elements(
                    lambda x: (x["SNOMED_code"], x["GROUP"]) in missing_pairs,
                    return_dtype=pl.Boolean,
                )
            )
            .unique(subset=["SNOMED_code", "GROUP"])
            .select(["SNOMED_code", "chosen_SNOMED_code"])
        )

    # --- Load SNOMED_FSN.tsv ---
    snomed_fsn_file = spaccc_dir / "SNOMED_FSN.tsv"
    if snomed_fsn_file.exists():
        snomed_df = pl.read_csv(
            snomed_fsn_file,
            separator="\t",
            has_header=True,
            quote_char=None,
            schema_overrides=[pl.Utf8, pl.Utf8, pl.Utf8],
        )
        logging.info(f"Loaded {snomed_df.height} SNOMED FSN entries")

        # Join with clean_terminology on SNOMED_code to add FSN information
        clean_terminology = clean_terminology.join(
            snomed_df, on="SNOMED_code", how="left"
        ).unique()

        logging.info(
            f"After joining with SNOMED_FSN: {clean_terminology.height} rows, "
            f"columns: {clean_terminology.columns}"
        )

    else:
        logging.warning(
            f"SNOMED_FSN.tsv file not found at {snomed_fsn_file}, skipping join."
        )
    # Semantic Info file
    semantic_info = clean_terminology.group_by("SNOMED_code").agg([
        pl.col("GROUP").first(),
        pl.col("CATEGORY").first().alias("SEM_CODE"),
        pl.col("CATEGORY").first(),
    ])

    # Add lang column
    clean_terminology = clean_terminology.with_columns(pl.lit("es").alias("lang"))

    # Save results
    logging.info(
        f"💾 Saving {mapping_df.height} mapping entries to {corrected_code_file}"
    )
    mapping_df.write_csv(corrected_code_file)

    logging.info(
        f"💾 Saving {clean_terminology.height} clean terminology entries to {clean_terminology_file}"
    )
    clean_terminology.write_parquet(clean_terminology_file)

    logging.info(
        f"💾 Saving {semantic_info.height} semantic info entries to {semantic_info_file}"
    )
    semantic_info.write_parquet(semantic_info_file)

    logging.info("📊 Final statistics:")
    logging.info(f"   - Unique entities: {clean_terminology['Entity'].n_unique()}")
    logging.info(
        f"   - Unique SNOMED codes: {clean_terminology['SNOMED_code'].n_unique()}"
    )
    if added_rows_df.height if hasattr(added_rows_df, "height") else 0:
        logging.info(
            f"   - Added duplicates for coverage: {getattr(added_rows_df, 'height', 0)}"
        )

    logging.info("✅ Processing completed successfully")


if __name__ == "__main__":
    app()
