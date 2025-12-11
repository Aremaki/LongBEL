from pathlib import Path
from typing import Any, Optional

import polars as pl


def normalize_codes(col: pl.Expr) -> pl.Expr:
    """
    Normalize composite CUIs (e.g. "A+B") by:
    - splitting
    - trimming
    - sorting
    - joining deterministically
    """
    return (
        col.str.split("+")
        .list.eval(pl.element().str.strip_chars())
        .list.sort()
        .list.join("+")
    )


def get_concepts_from_codes(
    codes: list[str],
    labels: list[str],
    umls_df: pl.DataFrame,
    mentions: Optional[list[str]] = None,
    context_sentences: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    """Provide a list of dictionaries with 'mentions', 'synonym', 'title', 'semantic_group', and 'code' for each code in the input list."""
    concepts = []
    if mentions is not None and context_sentences is not None:
        for code, label, mention, context in zip(
            codes, labels, mentions, context_sentences
        ):
            concept_info = umls_df.filter(pl.col("CUI") == code)
            if concept_info.height == 0:
                continue
            synonyms = concept_info["Entity"].unique().to_list()
            title = concept_info["title"].first()
            semantic_group = concept_info["GROUP"].first()
            concept_dict = {
                "text": label,
                "synonyms": synonyms,
                "title": title,
                "semantic_group": semantic_group,
                "code": code,
                "mention": mention,
                "context_sentence": context,
            }
            concepts.append(concept_dict)
        return concepts
    for code, label in zip(codes, labels):
        concept_info = umls_df.filter(pl.col("CUI") == code)
        if concept_info.height == 0:
            continue
        synonyms = concept_info["Entity"].unique().to_list()
        title = concept_info["title"].first()
        semantic_group = concept_info["GROUP"].first()
        concept_dict = {
            "text": label,
            "synonyms": synonyms,
            "title": title,
            "semantic_group": semantic_group,
            "code": code,
        }
        concepts.append(concept_dict)
    return concepts


def load_predictions(
    prediction_path: Path,
    dataset: str,
    umls_df: pl.DataFrame = pl.DataFrame(),
    relextractor=None,
) -> pl.DataFrame:
    """
    Add semantic relation column to the dataframe using KeyCare's RelationExtractor.
    """
    df = pl.read_csv(
        prediction_path,
        separator="\t",
        has_header=True,
        schema_overrides={
            "code": str,  # force as string
            "Predicted_code": str,
            "mention_id": str,
            "filename": str,  # force as string
        },  # type: ignore
    ).unique(
        subset=[
            "filename",
            "label",
            "start_span",
            "end_span",
        ]
    )
    # Filter NO_CODE annotations
    df = df.filter(~(pl.col("code").str.contains("NO_CODE")))
    df = df.with_columns(
        normalize_codes(pl.col("code")), normalize_codes(pl.col("Predicted_code"))
    )
    df = df.with_columns(
        pl.when(pl.col("Predicted_code").is_null())
        .then(pl.lit(""))  # replace Prediction with empty string
        .otherwise(pl.col("Prediction"))
        .alias("Prediction")
    )
    if "semantic_rel_pred" in df.columns or relextractor is None:
        return df  # already processed
    else:
        source_codes = df["code"].to_list()
        source_labels = df["annotations"].to_list()
        target_codes = df["Prediction_code"].to_list()
        target_labels = df["Prediction"].to_list()
        mentions = df["span"].to_list()
        context_sentences = df["sentences"].to_list()
        source_concepts = get_concepts_from_codes(
            source_codes, source_labels, umls_df, mentions, context_sentences
        )
        target_concepts = get_concepts_from_codes(target_codes, target_labels, umls_df)
        relextractor(source_concepts, target_concepts)
        relations = []
        for rel in relextractor.relations:
            if len(rel.rel_type) == 0:
                relations.append("NO_RELATION")
            else:
                relations.append(rel.rel_type[0])
        df = df.with_columns(semantic_rel_pred=pl.Series(relations))
        df.write_csv(file=prediction_path, separator="\t")
        return df


def compute_simple_recall(df):
    recall = {}
    labels = sorted(df["label"].unique())
    for label in labels:
        df_label = df.filter(pl.col("label") == label)
        total_label = df_label.height
        true_label = df_label.filter(pl.col("code") == pl.col("Predicted_code")).height
        if total_label == 0:
            recall[label] = 0.0
            continue
        recall[label] = round(true_label / total_label * 100, 1)

    # compute overall
    total_df = df.height
    true = df.filter(pl.col("code") == pl.col("Predicted_code")).height
    recall["overall"] = round(true / total_df * 100, 1)
    return recall


# ---------------------------
# Core: compute_recall_ratios
# ---------------------------
def compute_recall_ratios(
    df: pl.DataFrame,
    df_full: pl.DataFrame,
    index: str = "overall",
    compute_all: bool = True,
) -> tuple[Any, ...]:
    """
    Compute recall_strict (+ optional exact/narrow/broad) and ratios for each label.
    Returns:
      - if compute_all: (recall_strict, recall_exact, recall_narrow, recall_broad, ratios)
      - else: (recall_strict, ratios)
    Structure:
      recall_strict[label_name][IndexKey] = float
      ratios[label_name][IndexKey] = "X% (a/b)"
    """
    index_key = index.capitalize()

    # outputs
    recall_strict: dict[str, dict[str, float]] = {}
    recall_exact: dict[str, dict[str, float]] = {}
    recall_narrow: dict[str, dict[str, float]] = {}
    recall_broad: dict[str, dict[str, float]] = {}
    ratios: dict[str, dict[str, str]] = {}

    # Overall row (aggregated across labels)
    total_full_overall = df_full.height
    total_pred_overall = df.height
    true_strict_overall = df.filter(pl.col("code") == pl.col("Predicted_code")).height

    recall_strict["overall"] = {
        index_key: round(true_strict_overall / total_pred_overall * 100, 1)
    }
    ratios["overall"] = {
        index_key: f"{round(total_pred_overall / total_full_overall * 100, 1)}% ({total_pred_overall}/{total_full_overall})"
    }

    if compute_all:
        true_exact_overall = df.filter(pl.col("semantic_rel_pred") == "EXACT").height
        true_narrow_overall = df.filter(pl.col("semantic_rel_pred") == "NARROW").height
        true_broad_overall = df.filter(pl.col("semantic_rel_pred") == "BROAD").height

        recall_exact["overall"] = {
            index_key: round(true_exact_overall / total_pred_overall * 100, 1)
        }
        recall_narrow["overall"] = {
            index_key: round(
                (true_exact_overall + true_narrow_overall) / total_pred_overall * 100, 1
            )
        }
        recall_broad["overall"] = {
            index_key: round(
                (true_exact_overall + true_narrow_overall + true_broad_overall)
                / total_pred_overall
                * 100,
                1,
            )
        }

    # deterministic label list from df_full
    unique_labels = sorted(df_full["label"].unique())

    for label in unique_labels:
        # precompute label-specific frames and totals (vectorized-ish)
        df_label_full = df_full.filter(pl.col("label") == label)
        df_label_pred = df.filter(pl.col("label") == label)

        total_full = df_label_full.height
        total_pred = df_label_pred.height

        # counts for strict / semantic relations
        true_strict = df_label_pred.filter(
            pl.col("code") == pl.col("Predicted_code")
        ).height

        # Only compute semantic-rel counts if compute_all requested (avoid cost otherwise)
        if compute_all:
            true_exact = df_label_pred.filter(
                pl.col("semantic_rel_pred") == "EXACT"
            ).height
            true_narrow = df_label_pred.filter(
                pl.col("semantic_rel_pred") == "NARROW"
            ).height
            true_broad = df_label_pred.filter(
                pl.col("semantic_rel_pred") == "BROAD"
            ).height
        else:
            true_exact = true_narrow = true_broad = 0

        # init containers
        recall_strict[label] = {}
        ratios[label] = {}
        if compute_all:
            recall_exact[label] = {}
            recall_narrow[label] = {}
            recall_broad[label] = {}

        if total_pred == 0:
            # no predictions for this label in current df partition
            recall_strict[label][index_key] = 0.0
            ratios[label][index_key] = f"0% (0/{total_full})"
            if compute_all:
                recall_exact[label][index_key] = 0.0
                recall_narrow[label][index_key] = 0.0
                recall_broad[label][index_key] = 0.0
            continue

        # strict recall
        recall_strict[label][index_key] = round(true_strict / total_pred * 100, 1)
        # ratios: how much of the full set is present in this partition
        if index_key == "All":
            ratios[label][index_key] = (
                f"{round(total_pred / total_pred_overall * 100, 1)}% ({total_pred}/{total_pred_overall})"
            )
        else:
            ratios[label][index_key] = (
                f"{round(total_pred / total_full * 100, 1)}% ({total_pred}/{total_full})"
            )

        if compute_all:
            recall_exact[label][index_key] = round(true_exact / total_pred * 100, 1)
            recall_narrow[label][index_key] = round(
                (true_exact + true_narrow) / total_pred * 100, 1
            )
            recall_broad[label][index_key] = round(
                (true_exact + true_narrow + true_broad) / total_pred * 100, 1
            )

    if compute_all:
        return recall_strict, recall_exact, recall_narrow, recall_broad, ratios

    return recall_strict, ratios


# ---------------------------
# Driver: compute_metrics
# ---------------------------
def compute_metrics(
    pred_df: pl.DataFrame,
    train_mentions: set[str],
    train_cuis: set[str],
    top_100_cuis: set[str],
    top_100_mentions: set[str],
    unique_pairs: set[tuple[str, str]],
    compute_all_recalls: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Compute full suite of metrics across many partitions.
    - pred_df: predicted dataframe (contains columns: label, code, Predicted_code, semantic_rel_pred, span, annotation, filename, start_span, etc.)
    - train_mentions / train_cuis / top_100_cuis / top_100_mentions / unique_pairs: sets used to build partitions
    - compute_all_recalls: if False, only strict recall + ratios are computed (faster)
    """

    # ensure codes normalized if you use normalize_codes; if you have it available do it here
    # pred_df = pred_df.with_columns(normalize_codes(pl.col("code")), normalize_codes(pl.col("Predicted_code")))

    # We'll call compute_recall_ratios for each partition and then aggregate
    # Helper to call and return a consistent tuple for easier aggregation
    def _call_partition(df_partition: pl.DataFrame, index_name: str):
        if compute_all_recalls:
            return compute_recall_ratios(
                df_partition, pred_df, index=index_name, compute_all=True
            )
        else:
            return compute_recall_ratios(
                df_partition, pred_df, index=index_name, compute_all=False
            )

    # All
    if compute_all_recalls:
        (
            recall_strict_all,
            recall_exact_all,
            recall_narrow_all,
            recall_broad_all,
            ratios_all,
        ) = _call_partition(pred_df, "All")
    else:
        recall_strict_all, ratios_all = _call_partition(pred_df, "All")

    # seen / unseen CUIs
    pred_df_seen_cuis = pred_df.filter(pl.col("code").is_in(list(train_cuis)))
    pred_df_unseen_cuis = pred_df.filter(~pl.col("code").is_in(list(train_cuis)))
    if compute_all_recalls:
        (
            recall_strict_seen_cuis,
            recall_exact_seen_cuis,
            recall_narrow_seen_cuis,
            recall_broad_seen_cuis,
            ratios_seen_cuis,
        ) = _call_partition(pred_df_seen_cuis, "seen_cuis")
        (
            recall_strict_unseen_cuis,
            recall_exact_unseen_cuis,
            recall_narrow_unseen_cuis,
            recall_broad_unseen_cuis,
            ratios_unseen_cuis,
        ) = _call_partition(pred_df_unseen_cuis, "unseen_cuis")
    else:
        recall_strict_seen_cuis, ratios_seen_cuis = _call_partition(
            pred_df_seen_cuis, "seen_cuis"
        )
        recall_strict_unseen_cuis, ratios_unseen_cuis = _call_partition(
            pred_df_unseen_cuis, "unseen_cuis"
        )

    # seen / unseen mentions
    pred_df_seen_mentions = pred_df.filter(pl.col("span").is_in(list(train_mentions)))
    pred_df_unseen_mentions = pred_df.filter(
        ~pl.col("span").is_in(list(train_mentions))
    )
    if compute_all_recalls:
        (
            recall_strict_seen_mentions,
            recall_exact_seen_mentions,
            recall_narrow_seen_mentions,
            recall_broad_seen_mentions,
            ratios_seen_mentions,
        ) = _call_partition(pred_df_seen_mentions, "seen_mentions")
        (
            recall_strict_unseen_mentions,
            recall_exact_unseen_mentions,
            recall_narrow_unseen_mentions,
            recall_broad_unseen_mentions,
            ratios_unseen_mentions,
        ) = _call_partition(pred_df_unseen_mentions, "unseen_mentions")
    else:
        recall_strict_seen_mentions, ratios_seen_mentions = _call_partition(
            pred_df_seen_mentions, "seen_mentions"
        )
        recall_strict_unseen_mentions, ratios_unseen_mentions = _call_partition(
            pred_df_unseen_mentions, "unseen_mentions"
        )

    # seen / unseen top 100 CUIs
    pred_df_seen_top_100_cuis = pred_df.filter(pl.col("code").is_in(list(top_100_cuis)))
    pred_df_unseen_top_100_cuis = pred_df.filter(
        ~pl.col("code").is_in(list(top_100_cuis))
    )
    if compute_all_recalls:
        (
            recall_strict_seen_top_100_cuis,
            recall_exact_seen_top_100_cuis,
            recall_narrow_seen_top_100_cuis,
            recall_broad_seen_top_100_cuis,
            ratios_seen_top_100_cuis,
        ) = _call_partition(pred_df_seen_top_100_cuis, "in_top_100_cuis")
        (
            recall_strict_unseen_top_100_cuis,
            recall_exact_unseen_top_100_cuis,
            recall_narrow_unseen_top_100_cuis,
            recall_broad_unseen_top_100_cuis,
            ratios_unseen_top_100_cuis,
        ) = _call_partition(pred_df_unseen_top_100_cuis, "not_in_top_100_cuis")
    else:
        recall_strict_seen_top_100_cuis, ratios_seen_top_100_cuis = _call_partition(
            pred_df_seen_top_100_cuis, "in_top_100_cuis"
        )
        recall_strict_unseen_top_100_cuis, ratios_unseen_top_100_cuis = _call_partition(
            pred_df_unseen_top_100_cuis, "not_in_top_100_cuis"
        )

    # seen / unseen top 100 mentions
    pred_df_seen_top_100_mentions = pred_df.filter(
        pl.col("span").is_in(list(top_100_mentions))
    )
    pred_df_unseen_top_100_mentions = pred_df.filter(
        ~pl.col("span").is_in(list(top_100_mentions))
    )
    if compute_all_recalls:
        (
            recall_strict_seen_top_100_mentions,
            recall_exact_seen_top_100_mentions,
            recall_narrow_seen_top_100_mentions,
            recall_broad_seen_top_100_mentions,
            ratios_seen_top_100_mentions,
        ) = _call_partition(pred_df_seen_top_100_mentions, "in_top_100_mentions")
        (
            recall_strict_unseen_top_100_mentions,
            recall_exact_unseen_top_100_mentions,
            recall_narrow_unseen_top_100_mentions,
            recall_broad_unseen_top_100_mentions,
            ratios_unseen_top_100_mentions,
        ) = _call_partition(pred_df_unseen_top_100_mentions, "not_in_top_100_mentions")
    else:
        recall_strict_seen_top_100_mentions, ratios_seen_top_100_mentions = (
            _call_partition(pred_df_seen_top_100_mentions, "in_top_100_mentions")
        )
        recall_strict_unseen_top_100_mentions, ratios_unseen_top_100_mentions = (
            _call_partition(pred_df_unseen_top_100_mentions, "not_in_top_100_mentions")
        )

    # seen / unseen unique pairs
    unique_df = pl.DataFrame(list(unique_pairs), schema=["span", "code"], orient="row")
    pred_df_seen_unique_pairs = pred_df.join(
        unique_df, on=["span", "code"], how="inner"
    )
    pred_df_unseen_unique_pairs = pred_df.join(
        unique_df, on=["span", "code"], how="anti"
    )
    if compute_all_recalls:
        (
            recall_strict_seen_unique_pairs,
            recall_exact_seen_unique_pairs,
            recall_narrow_seen_unique_pairs,
            recall_broad_seen_unique_pairs,
            ratios_seen_unique_pairs,
        ) = _call_partition(pred_df_seen_unique_pairs, "seen_unique_pairs")
        (
            recall_strict_unseen_unique_pairs,
            recall_exact_unseen_unique_pairs,
            recall_narrow_unseen_unique_pairs,
            recall_broad_unseen_unique_pairs,
            ratios_unseen_unique_pairs,
        ) = _call_partition(pred_df_unseen_unique_pairs, "unseen_unique_pairs")
    else:
        recall_strict_seen_unique_pairs, ratios_seen_unique_pairs = _call_partition(
            pred_df_seen_unique_pairs, "seen_unique_pairs"
        )
        recall_strict_unseen_unique_pairs, ratios_unseen_unique_pairs = _call_partition(
            pred_df_unseen_unique_pairs, "unseen_unique_pairs"
        )

    # identical / not identical (span == annotation)
    pred_df_is_identical = pred_df.filter(pl.col("span") == pl.col("annotation"))
    pred_df_is_not_identical = pred_df.filter(pl.col("span") != pl.col("annotation"))
    if compute_all_recalls:
        (
            recall_strict_identical,
            recall_exact_identical,
            recall_narrow_identical,
            recall_broad_identical,
            ratios_identical,
        ) = _call_partition(pred_df_is_identical, "identical")
        (
            recall_strict_not_identical,
            recall_exact_not_identical,
            recall_narrow_not_identical,
            recall_broad_not_identical,
            ratios_not_identical,
        ) = _call_partition(pred_df_is_not_identical, "not_identical")
    else:
        recall_strict_identical, ratios_identical = _call_partition(
            pred_df_is_identical, "identical"
        )
        recall_strict_not_identical, ratios_not_identical = _call_partition(
            pred_df_is_not_identical, "not_identical"
        )

    # word-count buckets
    pred_df_one_word = pred_df.filter(pl.col("span").str.count_matches(" ") == 0)
    pred_df_two_words = pred_df.filter(pl.col("span").str.count_matches(" ") == 1)
    pred_df_three_words = pred_df.filter(pl.col("span").str.count_matches(" ") == 2)
    pred_df_more_than_three_words = pred_df.filter(
        pl.col("span").str.count_matches(" ") >= 3
    )

    if compute_all_recalls:
        (
            recall_strict_one_word,
            recall_exact_one_word,
            recall_narrow_one_word,
            recall_broad_one_word,
            ratios_one_word,
        ) = _call_partition(pred_df_one_word, "one_word")
        (
            recall_strict_two_words,
            recall_exact_two_words,
            recall_narrow_two_words,
            recall_broad_two_words,
            ratios_two_words,
        ) = _call_partition(pred_df_two_words, "two_words")
        (
            recall_strict_three_words,
            recall_exact_three_words,
            recall_narrow_three_words,
            recall_broad_three_words,
            ratios_three_words,
        ) = _call_partition(pred_df_three_words, "three_words")
        (
            recall_strict_more_than_three_words,
            recall_exact_more_than_three_words,
            recall_narrow_more_than_three_words,
            recall_broad_more_than_three_words,
            ratios_more_than_three_words,
        ) = _call_partition(pred_df_more_than_three_words, "more_than_three_words")
    else:
        recall_strict_one_word, ratios_one_word = _call_partition(
            pred_df_one_word, "one_word"
        )
        recall_strict_two_words, ratios_two_words = _call_partition(
            pred_df_two_words, "two_words"
        )
        recall_strict_three_words, ratios_three_words = _call_partition(
            pred_df_three_words, "three_words"
        )
        recall_strict_more_than_three_words, ratios_more_than_three_words = (
            _call_partition(pred_df_more_than_three_words, "more_than_three_words")
        )

    # Abbreviation buckets
    pred_df_abbrev_only = pred_df.filter(
        pl.col("span").str.strip_chars().str.contains(r"^[A-Z0-9\-]{2,}$")
    )
    pred_df_abbrev_or_contains = pred_df.filter(
        pl.col("span").str.contains(r"\b[A-Z0-9\-]{2,}\b")
    )
    pred_df_not_abbrev = pred_df.filter(
        ~pl.col("span").str.contains(r"\b[A-Z0-9\-]{2,}\b")
    )

    if compute_all_recalls:
        (
            recall_strict_abbrev_only,
            recall_exact_abbrev_only,
            recall_narrow_abbrev_only,
            recall_broad_abbrev_only,
            ratios_abbrev_only,
        ) = _call_partition(pred_df_abbrev_only, "abbrev_only")
        (
            recall_strict_abbrev_or_contains,
            recall_exact_abbrev_or_contains,
            recall_narrow_abbrev_or_contains,
            recall_broad_abbrev_or_contains,
            ratios_abbrev_or_contains,
        ) = _call_partition(pred_df_abbrev_or_contains, "abbrev_or_contains")
        (
            recall_strict_not_abbrev,
            recall_exact_not_abbrev,
            recall_narrow_not_abbrev,
            recall_broad_not_abbrev,
            ratios_not_abbrev,
        ) = _call_partition(pred_df_not_abbrev, "not_abbrev")
    else:
        recall_strict_abbrev_only, ratios_abbrev_only = _call_partition(
            pred_df_abbrev_only, "abbrev_only"
        )
        recall_strict_abbrev_or_contains, ratios_abbrev_or_contains = _call_partition(
            pred_df_abbrev_or_contains, "abbrev_or_contains"
        )
        recall_strict_not_abbrev, ratios_not_abbrev = _call_partition(
            pred_df_not_abbrev, "not_abbrev"
        )

    # Inconsistency (repeated filename+code)
    repeated = (
        pred_df.group_by(["filename", "code"])
        .agg(pl.count().alias("count"))
        .filter(pl.col("count") >= 2)
        .select(["filename", "code"])
    )
    pred_df_repeated = pred_df.join(repeated, on=["filename", "code"], how="inner")
    pred_df_not_repeated = pred_df.join(repeated, on=["filename", "code"], how="anti")

    if compute_all_recalls:
        (
            recall_strict_repeated,
            recall_exact_repeated,
            recall_narrow_repeated,
            recall_broad_repeated,
            ratios_repeated,
        ) = _call_partition(pred_df_repeated, "repeated")
        (
            recall_strict_not_repeated,
            recall_exact_not_repeated,
            recall_narrow_not_repeated,
            recall_broad_not_repeated,
            ratios_not_repeated,
        ) = _call_partition(pred_df_not_repeated, "not_repeated")
    else:
        recall_strict_repeated, ratios_repeated = _call_partition(
            pred_df_repeated, "repeated"
        )
        recall_strict_not_repeated, ratios_not_repeated = _call_partition(
            pred_df_not_repeated, "not_repeated"
        )

    # -------------------------
    # Aggregation into final_results
    # -------------------------
    final_results: dict[str, dict[str, list[Any]]] = {}

    # initializer factory
    def _init_label_entry(lbl: str):
        if compute_all_recalls:
            return {
                "index": [],
                "recall_strict": [],
                "recall_exact": [],
                "recall_narrow": [],
                "recall_broad": [],
                "ratios": [],
            }
        else:
            return {
                "index": [],
                "recall_strict": [],
                "ratios": [],
            }

    # helper aggregator for each label and partition outputs
    def _aggregate(
        label: str,
        recall_strict_map: dict[str, dict[str, float]],
        ratios_map: dict[str, dict[str, str]],
        recall_exact_map=None,
        recall_narrow_map=None,
        recall_broad_map=None,
    ):
        if label not in final_results:
            final_results[label] = _init_label_entry(label)

        # safe-get first (and only) kv from maps
        strict_val = list(recall_strict_map.get(label, {"": 0.0}).values())[0]
        ratio_val = list(ratios_map.get(label, {"": "0% (0/0)"}).values())[0]

        final_results[label]["index"].append(list(ratios_map.get(label, {}).keys())[0])
        final_results[label]["recall_strict"].append(strict_val)
        final_results[label]["ratios"].append(ratio_val)

        if compute_all_recalls:
            exact_val = list(recall_exact_map.get(label, {"": 0.0}).values())[0]  # type: ignore
            narrow_val = list(recall_narrow_map.get(label, {"": 0.0}).values())[0]  # type: ignore
            broad_val = list(recall_broad_map.get(label, {"": 0.0}).values())[0]  # type: ignore
            final_results[label]["recall_exact"].append(exact_val)
            final_results[label]["recall_narrow"].append(narrow_val)
            final_results[label]["recall_broad"].append(broad_val)

    # Now aggregate in the same order you had originally:
    partitions = []

    # For each partition, gather the maps returned above and append a tuple to partitions:
    # (recall_strict_map, recall_exact_map_or_None, recall_narrow_map_or_None, recall_broad_map_or_None, ratios_map)
    # "All"
    if compute_all_recalls:
        partitions.append((
            recall_strict_all,
            recall_exact_all,  # type: ignore
            recall_narrow_all,  # type: ignore
            recall_broad_all,  # type: ignore
            ratios_all,
        ))
    else:
        partitions.append((recall_strict_all, None, None, None, ratios_all))

    # seen/unseen cuis
    if compute_all_recalls:
        partitions.append((
            recall_strict_seen_cuis,
            recall_exact_seen_cuis,  # type: ignore
            recall_narrow_seen_cuis,  # type: ignore
            recall_broad_seen_cuis,  # type: ignore
            ratios_seen_cuis,
        ))  # type: ignore
        partitions.append((
            recall_strict_unseen_cuis,
            recall_exact_unseen_cuis,  # type: ignore
            recall_narrow_unseen_cuis,  # type: ignore
            recall_broad_unseen_cuis,  # type: ignore
            ratios_unseen_cuis,
        ))
    else:
        partitions.append((recall_strict_seen_cuis, None, None, None, ratios_seen_cuis))
        partitions.append((
            recall_strict_unseen_cuis,
            None,
            None,
            None,
            ratios_unseen_cuis,
        ))

    # seen/unseen mentions
    if compute_all_recalls:
        partitions.append((
            recall_strict_seen_mentions,
            recall_exact_seen_mentions,  # type: ignore
            recall_narrow_seen_mentions,  # type: ignore
            recall_broad_seen_mentions,  # type: ignore
            ratios_seen_mentions,
        ))
        partitions.append((
            recall_strict_unseen_mentions,
            recall_exact_unseen_mentions,  # type: ignore
            recall_narrow_unseen_mentions,  # type: ignore
            recall_broad_unseen_mentions,  # type: ignore
            ratios_unseen_mentions,
        ))
    else:
        partitions.append((
            recall_strict_seen_mentions,
            None,
            None,
            None,
            ratios_seen_mentions,
        ))
        partitions.append((
            recall_strict_unseen_mentions,
            None,
            None,
            None,
            ratios_unseen_mentions,
        ))

    # top 100 cuis
    if compute_all_recalls:
        partitions.append((
            recall_strict_seen_top_100_cuis,
            recall_exact_seen_top_100_cuis,  # type: ignore
            recall_narrow_seen_top_100_cuis,  # type: ignore
            recall_broad_seen_top_100_cuis,  # type: ignore
            ratios_seen_top_100_cuis,
        ))
        partitions.append((
            recall_strict_unseen_top_100_cuis,
            recall_exact_unseen_top_100_cuis,  # type: ignore
            recall_narrow_unseen_top_100_cuis,  # type: ignore
            recall_broad_unseen_top_100_cuis,  # type: ignore
            ratios_unseen_top_100_cuis,
        ))
    else:
        partitions.append((
            recall_strict_seen_top_100_cuis,
            None,
            None,
            None,
            ratios_seen_top_100_cuis,
        ))
        partitions.append((
            recall_strict_unseen_top_100_cuis,
            None,
            None,
            None,
            ratios_unseen_top_100_cuis,
        ))

    # top 100 mentions
    if compute_all_recalls:
        partitions.append((
            recall_strict_seen_top_100_mentions,
            recall_exact_seen_top_100_mentions,  # type: ignore
            recall_narrow_seen_top_100_mentions,  # type: ignore
            recall_broad_seen_top_100_mentions,  # type: ignore
            ratios_seen_top_100_mentions,
        ))
        partitions.append((
            recall_strict_unseen_top_100_mentions,
            recall_exact_unseen_top_100_mentions,  # type: ignore
            recall_narrow_unseen_top_100_mentions,  # type: ignore
            recall_broad_unseen_top_100_mentions,  # type: ignore
            ratios_unseen_top_100_mentions,
        ))
    else:
        partitions.append((
            recall_strict_seen_top_100_mentions,
            None,
            None,
            None,
            ratios_seen_top_100_mentions,
        ))
        partitions.append((
            recall_strict_unseen_top_100_mentions,
            None,
            None,
            None,
            ratios_unseen_top_100_mentions,
        ))

    # unique pairs
    if compute_all_recalls:
        partitions.append((
            recall_strict_seen_unique_pairs,
            recall_exact_seen_unique_pairs,  # type: ignore
            recall_narrow_seen_unique_pairs,  # type: ignore
            recall_broad_seen_unique_pairs,  # type: ignore
            ratios_seen_unique_pairs,
        ))
        partitions.append((
            recall_strict_unseen_unique_pairs,
            recall_exact_unseen_unique_pairs,  # type: ignore
            recall_narrow_unseen_unique_pairs,  # type: ignore
            recall_broad_unseen_unique_pairs,  # type: ignore
            ratios_unseen_unique_pairs,
        ))
    else:
        partitions.append((
            recall_strict_seen_unique_pairs,
            None,
            None,
            None,
            ratios_seen_unique_pairs,
        ))
        partitions.append((
            recall_strict_unseen_unique_pairs,
            None,
            None,
            None,
            ratios_unseen_unique_pairs,
        ))

    # identical / not identical
    if compute_all_recalls:
        partitions.append((
            recall_strict_identical,
            recall_exact_identical,  # type: ignore
            recall_narrow_identical,  # type: ignore
            recall_broad_identical,  # type: ignore
            ratios_identical,
        ))
        partitions.append((
            recall_strict_not_identical,
            recall_exact_not_identical,  # type: ignore
            recall_narrow_not_identical,  # type: ignore
            recall_broad_not_identical,  # type: ignore
            ratios_not_identical,
        ))
    else:
        partitions.append((recall_strict_identical, None, None, None, ratios_identical))
        partitions.append((
            recall_strict_not_identical,
            None,
            None,
            None,
            ratios_not_identical,
        ))

    # one / two / three / more_than_three words
    if compute_all_recalls:
        partitions.append((
            recall_strict_one_word,
            recall_exact_one_word,  # type: ignore
            recall_narrow_one_word,  # type: ignore
            recall_broad_one_word,  # type: ignore
            ratios_one_word,
        ))
        partitions.append((
            recall_strict_two_words,
            recall_exact_two_words,  # type: ignore
            recall_narrow_two_words,  # type: ignore
            recall_broad_two_words,  # type: ignore
            ratios_two_words,
        ))
        partitions.append((
            recall_strict_three_words,
            recall_exact_three_words,  # type: ignore
            recall_narrow_three_words,  # type: ignore
            recall_broad_three_words,  # type: ignore
            ratios_three_words,
        ))
        partitions.append((
            recall_strict_more_than_three_words,
            recall_exact_more_than_three_words,  # type: ignore
            recall_narrow_more_than_three_words,  # type: ignore
            recall_broad_more_than_three_words,  # type: ignore
            ratios_more_than_three_words,
        ))
    else:
        partitions.append((recall_strict_one_word, None, None, None, ratios_one_word))
        partitions.append((recall_strict_two_words, None, None, None, ratios_two_words))
        partitions.append((
            recall_strict_three_words,
            None,
            None,
            None,
            ratios_three_words,
        ))
        partitions.append((
            recall_strict_more_than_three_words,
            None,
            None,
            None,
            ratios_more_than_three_words,
        ))

    # abbreviation buckets
    if compute_all_recalls:
        partitions.append((
            recall_strict_abbrev_only,
            recall_exact_abbrev_only,  # type: ignore
            recall_narrow_abbrev_only,  # type: ignore
            recall_broad_abbrev_only,  # type: ignore
            ratios_abbrev_only,
        ))
        partitions.append((
            recall_strict_abbrev_or_contains,
            recall_exact_abbrev_or_contains,  # type: ignore
            recall_narrow_abbrev_or_contains,  # type: ignore
            recall_broad_abbrev_or_contains,  # type: ignore
            ratios_abbrev_or_contains,
        ))
        partitions.append((
            recall_strict_not_abbrev,
            recall_exact_not_abbrev,  # type: ignore
            recall_narrow_not_abbrev,  # type: ignore
            recall_broad_not_abbrev,  # type: ignore
            ratios_not_abbrev,
        ))
    else:
        partitions.append((
            recall_strict_abbrev_only,
            None,
            None,
            None,
            ratios_abbrev_only,
        ))
        partitions.append((
            recall_strict_abbrev_or_contains,
            None,
            None,
            None,
            ratios_abbrev_or_contains,
        ))
        partitions.append((
            recall_strict_not_abbrev,
            None,
            None,
            None,
            ratios_not_abbrev,
        ))

    # repeated / not repeated
    if compute_all_recalls:
        partitions.append((
            recall_strict_repeated,
            recall_exact_repeated,  # type: ignore
            recall_narrow_repeated,  # type: ignore
            recall_broad_repeated,  # type: ignore
            ratios_repeated,
        ))
        partitions.append((
            recall_strict_not_repeated,
            recall_exact_not_repeated,  # type: ignore
            recall_narrow_not_repeated,  # type: ignore
            recall_broad_not_repeated,  # type: ignore
            ratios_not_repeated,
        ))
    else:
        partitions.append((recall_strict_repeated, None, None, None, ratios_repeated))
        partitions.append((
            recall_strict_not_repeated,
            None,
            None,
            None,
            ratios_not_repeated,
        ))

    # iterate partitions in order and aggregate for each label
    for rs_map, re_map, rn_map, rb_map, ratios_map in partitions:
        # rs_map and ratios_map are always present
        for label in sorted(rs_map.keys()):
            _aggregate(label, rs_map, ratios_map, re_map, rn_map, rb_map)

    return final_results
