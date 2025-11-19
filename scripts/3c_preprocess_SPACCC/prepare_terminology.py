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

    assigned_pairs = set(all_unambiguous.select(["CUI", "GROUP"]).iter_rows())
    assigned_pairs_entity = set(all_unambiguous.select(["Entity", "GROUP"]).iter_rows())

    # Stage 3: Handle remaining truly ambiguous entities
    remaining_ambiguous = df.join(
        all_unambiguous.select(["CUI", "GROUP", "Entity"]),
        on=["CUI", "GROUP", "Entity"],
        how="anti",
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
                "CUI",
                "Entity",
                "CATEGORY",
                "GROUP",
                "is_main",
                "ambiguous_level",
            ])
        )
    else:
        # explicit empty dataframe with expected schema so later selects/concats won't fail
        ambiguous_final = pl.DataFrame(
            schema=[
                ("CUI", pl.Utf8),
                ("Entity", pl.Utf8),
                ("CATEGORY", pl.Utf8),
                ("GROUP", pl.Utf8),
                ("is_main", pl.Boolean),
                ("ambiguous_level", pl.Int32),
            ]
        )

    # Combine all results
    # Ensure all_unambiguous has same columns as ambiguous_final (if empty ensure schema)
    if all_unambiguous.height == 0:
        all_unambiguous = pl.DataFrame(
            schema=[
                ("CUI", pl.Utf8),
                ("Entity", pl.Utf8),
                ("CATEGORY", pl.Utf8),
                ("GROUP", pl.Utf8),
                ("is_main", pl.Boolean),
                ("ambiguous_level", pl.Int32),
            ]
        )
    # Combine all results
    df_clean = pl.concat([all_unambiguous, ambiguous_final]).drop("ambiguous_level")

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


def load_terminology_umls(terminology_file: Path) -> pl.DataFrame:
    """Load and preprocess the terminology UMLS file."""

    df = pl.read_csv(
        terminology_file,
        separator="\t",
        null_values=["NO_CODE"],
        schema_overrides={
            "code": pl.Utf8,
            "CUI": pl.Utf8,
            "term": pl.Utf8,
            "semantic_tag": pl.Utf8,
        },
    )

    df_clean = (
        df.filter(pl.col("code").is_not_null())
        .with_columns([
            pl.col("code").alias("CUI").cast(pl.Utf8),
            pl.col("CUI").alias("CUI_UMLS").cast(pl.Utf8),
            pl.col("term").alias("Entity"),
            pl.col("semantic_tag").alias("GROUP"),
            pl.col("semantic_tag").str.slice(0, 4).alias("CATEGORY"),
            pl.col("mainterm").cast(pl.Boolean).alias("is_main"),
        ])
        .select(["CUI", "Entity", "CATEGORY", "GROUP", "is_main", "CUI_UMLS"])
    )

    logging.info(f"Loaded {df_clean.height} terminology entries")
    return df_clean


def augment_with_umls_cui(terminology_umls_df, umls_synonyms_file):
    """Augment terminology with UMLS CUIs using synonyms file."""
    terminology_umls_df = terminology_umls_df.with_columns(lang=pl.lit("es"))
    umls_syn = pl.read_parquet(umls_synonyms_file).rename({"CUI": "CUI_UMLS"})
    filtered_syn = umls_syn.join(
        terminology_umls_df.select(["CUI_UMLS", "CATEGORY", "GROUP", "CUI"]).unique(),
        on="CUI_UMLS",
        how="inner",
    )
    exploded_syn = (
        _explode_language_frames(filtered_syn)
        .rename({"Syn": "Entity"})
        .unique(subset=["CUI_UMLS", "Entity"])
    )

    additional_terms = exploded_syn.join(
        terminology_umls_df.select(["CUI_UMLS", "CATEGORY", "GROUP", "CUI"]).unique(),
        on="CUI_UMLS",
    ).unique(subset=["CUI_UMLS", "CUI", "Entity"])
    return pl.concat([terminology_umls_df, additional_terms], how="align")


def _explode_language_frames(base: pl.DataFrame) -> pl.DataFrame:
    # Split language columns and explode synonyms / titles
    fr = (
        base.select([
            "CUI_UMLS",
            "UMLS_Title_fr",
            "UMLS_alias_fr",
        ])
        .with_columns(pl.lit("fr").alias("lang"))
        .explode("UMLS_Title_fr")
        .explode("UMLS_alias_fr")
    )

    en = base.select([
        "CUI_UMLS",
        "UMLS_Title_main",
        "UMLS_Title_en",
        "UMLS_alias_en",
    ]).with_columns(pl.lit("en").alias("lang"))
    en = en.explode("UMLS_Title_main").explode("UMLS_Title_en").explode("UMLS_alias_en")

    es = base.select([
        "CUI_UMLS",
        "UMLS_Title_es",
        "UMLS_alias_es",
    ]).with_columns(pl.lit("es").alias("lang"))
    es = es.explode("UMLS_Title_es").explode("UMLS_alias_es")

    # Build unified rows for each type source (mark main True appropriately)
    parts = []

    def _mk(df: pl.DataFrame, col: str, is_main: bool) -> pl.DataFrame:
        return (
            df.select([
                "CUI_UMLS",
                "lang",
                pl.col(col).alias("Syn"),
            ])
            .filter((pl.col("Syn") != "") & (pl.col("Syn").is_not_null()))
            .with_columns(is_main=pl.lit(is_main))
        )

    # Titles and aliases
    if "UMLS_Title_main" in en.columns:  # main (English main title)
        parts.append(_mk(en, "UMLS_Title_main", is_main=True))
    if "UMLS_Title_fr" in fr.columns:
        parts.append(_mk(fr, "UMLS_Title_fr", is_main=False))
    if "UMLS_Title_en" in en.columns:
        parts.append(_mk(en, "UMLS_Title_en", is_main=False))
    if "UMLS_alias_fr" in fr.columns:
        parts.append(_mk(fr, "UMLS_alias_fr", is_main=False))
    if "UMLS_alias_en" in en.columns:
        parts.append(_mk(en, "UMLS_alias_en", is_main=False))
    if "UMLS_Title_es" in es.columns:
        parts.append(_mk(es, "UMLS_Title_es", is_main=False))
    if "UMLS_alias_es" in es.columns:
        parts.append(_mk(es, "UMLS_alias_es", is_main=False))

    return pl.concat(parts)


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


def augment_with_missing_pairs(
    df_final: pl.DataFrame, missing_pairs: set[tuple[str, str]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Duplicate CUIs for missing (CUI, GROUP) pairs using a single canonical row per CUI.
    CATEGORY is derived from GROUP.
    """

    if not missing_pairs:
        return df_final, pl.DataFrame({})

    # Build a tiny frame of missing pairs with derived CATEGORY
    missing_df = pl.DataFrame([
        {"CUI": cui, "GROUP": group} for cui, group in missing_pairs
    ]).with_columns([
        pl.col("CUI").cast(pl.Utf8),
        pl.col("GROUP").cast(pl.Utf8),
        pl.col("GROUP").str.slice(0, 4).alias("CATEGORY"),
    ])

    # Pick one canonical row per CUI (prefer is_main=True)
    base_per_cui = (
        df_final.with_columns([
            pl.col("is_main").fill_null(False),
        ])
        .sort(
            ["CUI", "is_main", "Entity"],
            descending=[False, True, False],
        )
        .unique(subset=["CUI"], keep="first")
        .select(["CUI", "Entity", "is_main"])
    )

    # Create duplicates by joining missing pairs with canonical rows
    dup_df = missing_df.join(base_per_cui, on="CUI", how="left").select([
        "CUI",
        "Entity",
        "CATEGORY",
        "GROUP",
        "is_main",
    ])

    # If some CUI truly doesn't exist in df_final (edge case), keep but Entity will be null
    # Filter those out to avoid invalid rows
    dup_df = dup_df.filter(pl.col("Entity").is_not_null())

    if dup_df.is_empty():
        return df_final, pl.DataFrame({})

    augmented = pl.concat([df_final, dup_df])
    logging.info(
        f"➕ Added {dup_df.height} duplicated rows to cover missing priority pairs"
    )
    return augmented, dup_df


def _complete_cui_entity_group_combinations(df: pl.DataFrame) -> pl.DataFrame:
    """
    For each CUI, ensure all its associated entities appear for all its associated groups.
    """
    logging.info("Completing CUI-Entity-GROUP combinations...")

    # Create a frame with all combinations of CUI, Entity, and GROUP
    all_combinations = df.select(["CUI", "Entity", "GROUP"]).unique()

    # For each CUI, get all its unique entities and groups
    cui_to_entities = all_combinations.group_by("CUI").agg(
        pl.col("Entity").unique().alias("all_entities")
    )
    cui_to_groups = all_combinations.group_by("CUI").agg(
        pl.col("GROUP").unique().alias("all_groups")
    )

    # Create the full cartesian product of entities and groups for each CUI
    full_product = (
        cui_to_entities.join(cui_to_groups, on="CUI")
        .explode("all_entities")
        .explode("all_groups")
        .rename({"all_entities": "Entity", "all_groups": "GROUP"})
    )

    # Use the original data as the source of truth for other columns
    # and join it with the full product.
    # This will put nulls where a combination did not originally exist.
    completed_df = full_product.join(
        df.unique(subset=["CUI", "Entity", "GROUP"]),
        on=["CUI", "Entity", "GROUP"],
        how="left",
    )

    # Forward fill missing values within each CUI-Entity group.
    # This assumes that for a given CUI-Entity pair, other attributes are consistent.
    completed_df = completed_df.with_columns(
        pl.all().forward_fill().over(["CUI", "Entity"])
    )

    # For any remaining nulls (e.g., a CUI-Entity pair only existed with nulls), backfill.
    completed_df = completed_df.with_columns(
        pl.all().backward_fill().over(["CUI", "Entity"])
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
        Path("data/SPACCC/Normalization"),
        "--spaccc-dir",
        "-i",
        help="Path to SPACCC data directory with terminology.tsv, train.tsv, test.tsv",
    ),
    umls_dir: Path = typer.Option(
        Path("data/UMLS_processed/SPACCC_UMLS"),
        "--umls-dir",
        "-u",
        help="Path to preprocessed UMLS data for SPACCC",
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
    terminology_umls_dir: Path = typer.Option(
        Path("data/UMLS_processed/SPACCC_UMLS"),
        "--terminology-umls-dir",
        "-m",
        help="Directory for filtered UMLS-augmented terminology file",
    ),
):
    """Preprocess SPACCC terminology to resolve ambiguous entities."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Define file paths
    terminology_file = spaccc_dir / "terminology.tsv"
    terminology_umls_file = spaccc_dir / "terminology_umls.tsv"
    umls_synonyms_file = umls_dir / "umls_title_syn.parquet"
    train_file = spaccc_dir / "train.tsv"
    test_file = spaccc_dir / "test.tsv"
    corrected_cui_file = corrected_dir / "SPACCC_adapted.csv"
    corrected_umls_cui_file = corrected_dir / "SPACCC_adapted_umls.csv"
    clean_terminology_file = terminology_dir / "all_disambiguated.parquet"
    clean_terminology_umls_file = terminology_umls_dir / "all_disambiguated.parquet"
    # Create output directory
    corrected_dir.mkdir(exist_ok=True)
    terminology_dir.mkdir(exist_ok=True)

    logging.info(f"Processing data from: {spaccc_dir}")

    try:
        # Load priority pairs
        priority_pairs = load_priority_pairs(train_file, test_file)

        # Load terminology
        terminology_df = load_terminology(terminology_file)
        terminology_umls_df = load_terminology_umls(terminology_umls_file)

        # Augment terminology with UMLS CUIs
        augmented_umls_df = augment_with_umls_cui(
            terminology_umls_df, umls_synonyms_file
        )

        # Complete CUI-Entity-GROUP combinations
        terminology_df = _complete_cui_entity_group_combinations(terminology_df)
        augmented_umls_df = _complete_cui_entity_group_combinations(augmented_umls_df)

        # Resolve ambiguity
        clean_terminology, _ = resolve_entity_ambiguity(terminology_df, priority_pairs)
        clean_terminology_umls, _ = resolve_entity_ambiguity(
            augmented_umls_df, priority_pairs
        )

        # Validate coverage (pre-augmentation)
        missing_pairs_pre = validate_coverage(clean_terminology, priority_pairs)
        missing_pairs_pre_umls = validate_coverage(
            clean_terminology_umls, priority_pairs
        )

        # If some (CUI, GROUP) exist in train/test but are missing in the final set,
        # duplicate the CUI under the requested GROUP to ensure coverage
        if missing_pairs_pre:
            clean_terminology, added_rows_df = augment_with_missing_pairs(
                clean_terminology, missing_pairs_pre
            )
        else:
            added_rows_df = pl.DataFrame({})
        if missing_pairs_pre_umls:
            clean_terminology_umls, added_rows_df_umls = augment_with_missing_pairs(
                clean_terminology_umls, missing_pairs_pre_umls
            )
        else:
            added_rows_df_umls = pl.DataFrame({})

        _ = validate_coverage(clean_terminology, priority_pairs)
        _ = validate_coverage(clean_terminology_umls, priority_pairs)

        # Resolve ambiguity after augmentation
        clean_terminology, mapping_df = resolve_entity_ambiguity(
            clean_terminology, priority_pairs
        )
        clean_terminology_umls, mapping_umls_df = resolve_entity_ambiguity(
            clean_terminology_umls, priority_pairs
        )

        missing_pairs = validate_coverage(clean_terminology, priority_pairs)
        missing_pairs_umls = validate_coverage(clean_terminology_umls, priority_pairs)

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

        if missing_pairs_umls:
            mapping_umls_df = (
                mapping_umls_df.filter(
                    pl.struct(["CUI", "GROUP"]).map_elements(
                        lambda x: (x["CUI"], x["GROUP"]) in missing_pairs_umls,
                        return_dtype=pl.Boolean,
                    )
                )
                .unique(subset=["CUI", "GROUP"])
                .select(["CUI", "chosen_CUI"])
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

            # Join with clean_terminology on CUI
            clean_terminology = clean_terminology.join(
                snomed_df, on="CUI", how="left"
            ).unique()

            clean_terminology_umls = clean_terminology_umls.join(
                snomed_df, on="CUI", how="left"
            ).unique()

            logging.info(
                f"After joining with SNOMED_FSN: {clean_terminology.height} rows, "
                f"columns: {clean_terminology.columns}"
            )
            logging.info(
                f"After joining with SNOMED_FSN (UMLS): {clean_terminology_umls.height} rows, "
                f"columns: {clean_terminology_umls.columns}"
            )
        else:
            logging.warning(
                f"SNOMED_FSN.tsv file not found at {snomed_fsn_file}, skipping join."
            )
        # Save results
        logging.info(
            f"💾 Saving {mapping_df.height} mapping entries to {corrected_cui_file}"
        )
        mapping_df.write_csv(corrected_cui_file)

        logging.info(
            f"💾 Saving {mapping_umls_df.height} UMLS mapping entries to {corrected_umls_cui_file}"
        )
        mapping_umls_df.write_csv(corrected_umls_cui_file)

        logging.info(
            f"💾 Saving {clean_terminology.height} clean terminology entries to {clean_terminology_file}"
        )
        clean_terminology.write_parquet(clean_terminology_file)
        logging.info(
            f"💾 Saving {clean_terminology_umls.height} clean UMLS terminology entries to {clean_terminology_umls_file}"
        )
        clean_terminology_umls.write_parquet(clean_terminology_umls_file)

        logging.info("📊 Final statistics:")
        logging.info(f"   - Unique entities: {clean_terminology['Entity'].n_unique()}")
        logging.info(f"   - Unique CUIs: {clean_terminology['CUI'].n_unique()}")
        if added_rows_df.height if hasattr(added_rows_df, "height") else 0:
            logging.info(
                f"   - Added duplicates for coverage: {getattr(added_rows_df, 'height', 0)}"
            )
        logging.info("📊 Final UMLS statistics:")
        logging.info(
            f"   - Unique entities: {clean_terminology_umls['Entity'].n_unique()}"
        )
        logging.info(f"   - Unique CUIs: {clean_terminology_umls['CUI'].n_unique()}")
        if added_rows_df_umls.height if hasattr(added_rows_df_umls, "height") else 0:
            logging.info(
                f"   - Added UMLS duplicates for coverage: {getattr(added_rows_df_umls, 'height', 0)}"
            )

        logging.info("✅ Processing completed successfully")

    except Exception as e:
        logging.error(f"❌ Processing failed: {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
