from pathlib import Path

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


def load_predictions(
    prediction_path: Path,
    dataset: str,
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
            "Predicted_code": str,  # force as string
        },  # type: ignore
    ).unique(
        subset=[
            "filename",
            "label",
            "start_span",
            "end_span",
        ]
    )
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
        source = df["annotation"].to_list()
        target = df["Prediction"].to_list()
        relextractor(source, target)
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


def compute_recall_ratios(df, df_full, index="overall"):
    recall_strict = {}
    recall_exact = {}
    recall_narrow = {}
    recall_broad = {}
    ratios = {}
    labels = sorted(df_full["label"].unique())
    for label in labels:
        label_names = [label]
        if label == "Procedures":
            label_names = ["PROCEDIMIENTO"]
        if label == "Disorders":
            label_names = ["ENFERMEDAD", "SINTOMA"]
        for label_name in label_names:
            ratios[label_name] = {}
            recall_strict[label_name] = {}
            recall_exact[label_name] = {}
            recall_narrow[label_name] = {}
            recall_broad[label_name] = {}
            df_label = df.filter(pl.col("label") == label)
            total_full = df_full.filter(pl.col("label") == label).height
            total_df = df_label.height
            true = df_label.filter(pl.col("code") == pl.col("Predicted_code")).height
            true_exact = df_label.filter(pl.col("semantic_rel_pred") == "EXACT").height
            true_narrow = df_label.filter(
                pl.col("semantic_rel_pred") == "NARROW"
            ).height
            true_broad = df_label.filter(pl.col("semantic_rel_pred") == "BROAD").height
            if total_df == 0:
                recall_strict[label_name][index.capitalize()] = 0.0
                recall_exact[label_name][index.capitalize()] = 0.0
                recall_narrow[label_name][index.capitalize()] = 0.0
                recall_broad[label_name][index.capitalize()] = 0.0
                ratios[label_name][index.capitalize()] = f"0% (0/{total_full})"
                continue
            recall_strict[label_name][index.capitalize()] = round(
                true / total_df * 100, 1
            )
            recall_exact[label_name][index.capitalize()] = round(
                true_exact / total_df * 100, 1
            )
            recall_narrow[label_name][index.capitalize()] = round(
                (true_exact + true_narrow) / total_df * 100, 1
            )
            recall_broad[label_name][index.capitalize()] = round(
                (true_exact + true_narrow + true_broad) / total_df * 100, 1
            )
            ratios[label_name][index.capitalize()] = (
                f"{round(total_df / total_full * 100, 1)}% ({total_df}/{total_full})"
            )
    # compute overall
    total_full = df_full.height
    total_df = df.height
    true = df.filter(pl.col("code") == pl.col("Predicted_code")).height
    true_exact = df.filter(pl.col("semantic_rel_pred") == "EXACT").height
    true_narrow = df.filter(pl.col("semantic_rel_pred") == "NARROW").height
    true_broad = df.filter(pl.col("semantic_rel_pred") == "BROAD").height
    recall_strict["overall"] = {}
    recall_exact["overall"] = {}
    recall_narrow["overall"] = {}
    recall_broad["overall"] = {}
    recall_strict["overall"][index.capitalize()] = round(true / total_df * 100, 1)
    recall_exact["overall"][index.capitalize()] = round(true_exact / total_df * 100, 1)
    recall_narrow["overall"][index.capitalize()] = round(
        (true_exact + true_narrow) / total_df * 100, 1
    )
    recall_broad["overall"][index.capitalize()] = round(
        (true_exact + true_narrow + true_broad) / total_df * 100, 1
    )
    ratios["overall"] = {}
    ratios["overall"][index.capitalize()] = (
        f"{round(total_df / total_full * 100, 1)}% ({total_df}/{total_full})"
    )
    return recall_strict, recall_exact, recall_narrow, recall_broad, ratios


def compute_metrics(
    pred_df, train_mentions, train_cuis, top_100_cuis, top_100_mentions, unique_pairs
):
    pred_df = pred_df.with_columns(
        normalize_codes(pl.col("code")), normalize_codes(pl.col("Predicted_code"))
    )
    (
        recall_strict_all,
        recall_exact_all,
        recall_narrow_all,
        recall_broad_all,
        ratios_all,
    ) = compute_recall_ratios(pred_df, pred_df, index="All")

    # seen / unseen CUIs
    pred_df_seen_cuis = pred_df.filter(pl.col("code").is_in(train_cuis))
    pred_df_unseen_cuis = pred_df.filter(~pl.col("code").is_in(train_cuis))

    (
        recall_strict_seen_cuis,
        recall_exact_seen_cuis,
        recall_narrow_seen_cuis,
        recall_broad_seen_cuis,
        ratios_seen_cuis,
    ) = compute_recall_ratios(pred_df_seen_cuis, pred_df, index="seen_cuis")
    (
        recall_strict_unseen_cuis,
        recall_exact_unseen_cuis,
        recall_narrow_unseen_cuis,
        recall_broad_unseen_cuis,
        ratios_unseen_cuis,
    ) = compute_recall_ratios(pred_df_unseen_cuis, pred_df, index="unseen_cuis")

    # seen / unseen mentions
    pred_df_seen_mentions = pred_df.filter(pl.col("span").is_in(train_mentions))
    pred_df_unseen_mentions = pred_df.filter(~pl.col("span").is_in(train_mentions))
    (
        recall_strict_seen_mentions,
        recall_exact_seen_mentions,
        recall_narrow_seen_mentions,
        recall_broad_seen_mentions,
        ratios_seen_mentions,
    ) = compute_recall_ratios(pred_df_seen_mentions, pred_df, index="seen_mentions")
    (
        recall_strict_unseen_mentions,
        recall_exact_unseen_mentions,
        recall_narrow_unseen_mentions,
        recall_broad_unseen_mentions,
        ratios_unseen_mentions,
    ) = compute_recall_ratios(pred_df_unseen_mentions, pred_df, index="unseen_mentions")

    # seen / unseen top 100 CUIs
    pred_df_seen_top_100_cuis = pred_df.filter(pl.col("code").is_in(top_100_cuis))
    pred_df_unseen_top_100_cuis = pred_df.filter(~pl.col("code").is_in(top_100_cuis))
    (
        recall_strict_seen_top_100_cuis,
        recall_exact_seen_top_100_cuis,
        recall_narrow_seen_top_100_cuis,
        recall_broad_seen_top_100_cuis,
        ratios_seen_top_100_cuis,
    ) = compute_recall_ratios(
        pred_df_seen_top_100_cuis, pred_df, index="in_top_100_cuis"
    )
    (
        recall_strict_unseen_top_100_cuis,
        recall_exact_unseen_top_100_cuis,
        recall_narrow_unseen_top_100_cuis,
        recall_broad_unseen_top_100_cuis,
        ratios_unseen_top_100_cuis,
    ) = compute_recall_ratios(
        pred_df_unseen_top_100_cuis, pred_df, index="not_in_top_100_cuis"
    )

    # seen / unseen top 100 mentions
    pred_df_seen_top_100_mentions = pred_df.filter(
        pl.col("span").is_in(top_100_mentions)
    )
    pred_df_unseen_top_100_mentions = pred_df.filter(
        ~pl.col("span").is_in(top_100_mentions)
    )
    (
        recall_strict_seen_top_100_mentions,
        recall_exact_seen_top_100_mentions,
        recall_narrow_seen_top_100_mentions,
        recall_broad_seen_top_100_mentions,
        ratios_seen_top_100_mentions,
    ) = compute_recall_ratios(
        pred_df_seen_top_100_mentions, pred_df, index="in_top_100_mentions"
    )
    (
        recall_strict_unseen_top_100_mentions,
        recall_exact_unseen_top_100_mentions,
        recall_narrow_unseen_top_100_mentions,
        recall_broad_unseen_top_100_mentions,
        ratios_unseen_top_100_mentions,
    ) = compute_recall_ratios(
        pred_df_unseen_top_100_mentions, pred_df, index="not_in_top_100_mentions"
    )

    # seen / unseen unique pairs
    unique_df = pl.DataFrame(list(unique_pairs), schema=["span", "code"], orient="row")
    pred_df_seen_unique_pairs = pred_df.join(
        unique_df, on=["span", "code"], how="inner"
    )
    pred_df_unseen_unique_pairs = pred_df.join(
        unique_df, on=["span", "code"], how="anti"
    )
    (
        recall_strict_seen_unique_pairs,
        recall_exact_seen_unique_pairs,
        recall_narrow_seen_unique_pairs,
        recall_broad_seen_unique_pairs,
        ratios_seen_unique_pairs,
    ) = compute_recall_ratios(
        pred_df_seen_unique_pairs, pred_df, index="seen_unique_pairs"
    )
    (
        recall_strict_unseen_unique_pairs,
        recall_exact_unseen_unique_pairs,
        recall_narrow_unseen_unique_pairs,
        recall_broad_unseen_unique_pairs,
        ratios_unseen_unique_pairs,
    ) = compute_recall_ratios(
        pred_df_unseen_unique_pairs, pred_df, index="unseen_unique_pairs"
    )

    # identical span and annotation
    pred_df_is_identical = pred_df.filter(pl.col("span") == pl.col("annotation"))
    pred_df_is_not_identical = pred_df.filter(pl.col("span") != pl.col("annotation"))
    (
        recall_strict_identical,
        recall_exact_identical,
        recall_narrow_identical,
        recall_broad_identical,
        ratios_identical,
    ) = compute_recall_ratios(pred_df_is_identical, pred_df, index="identical")
    (
        recall_strict_not_identical,
        recall_exact_not_identical,
        recall_narrow_not_identical,
        recall_broad_not_identical,
        ratios_not_identical,
    ) = compute_recall_ratios(pred_df_is_not_identical, pred_df, index="not_identical")

    # one word mentions
    pred_df_one_word = pred_df.filter(pl.col("span").str.count_matches(" ") == 0)
    pred_df_two_words = pred_df.filter(pl.col("span").str.count_matches(" ") == 1)
    pred_df_three_words = pred_df.filter(pl.col("span").str.count_matches(" ") == 2)
    pred_df_more_than_three_words = pred_df.filter(
        pl.col("span").str.count_matches(" ") >= 3
    )
    (
        recall_strict_one_word,
        recall_exact_one_word,
        recall_narrow_one_word,
        recall_broad_one_word,
        ratios_one_word,
    ) = compute_recall_ratios(pred_df_one_word, pred_df, index="one_word")
    (
        recall_strict_two_words,
        recall_exact_two_words,
        recall_narrow_two_words,
        recall_broad_two_words,
        ratios_two_words,
    ) = compute_recall_ratios(pred_df_two_words, pred_df, index="two_words")
    (
        recall_strict_three_words,
        recall_exact_three_words,
        recall_narrow_three_words,
        recall_broad_three_words,
        ratios_three_words,
    ) = compute_recall_ratios(pred_df_three_words, pred_df, index="three_words")
    (
        recall_strict_more_than_three_words,
        recall_exact_more_than_three_words,
        recall_narrow_more_than_three_words,
        recall_broad_more_than_three_words,
        ratios_more_than_three_words,
    ) = compute_recall_ratios(
        pred_df_more_than_three_words, pred_df, index="more_than_three_words"
    )

    # Abbreviation mentions
    pred_df_abbrev_only = pred_df.filter(
        pl.col("span").str.strip_chars().str.contains(r"^[A-Z0-9\-]{2,}$")
    )
    pred_df_abbrev_or_contains = pred_df.filter(
        pl.col("span").str.contains(r"\b[A-Z0-9\-]{2,}\b")
    )
    pred_df_not_abbrev = pred_df.filter(
        ~pl.col("span").str.contains(r"\b[A-Z0-9\-]{2,}\b")
    )
    (
        recall_strict_abbrev_only,
        recall_exact_abbrev_only,
        recall_narrow_abbrev_only,
        recall_broad_abbrev_only,
        ratios_abbrev_only,
    ) = compute_recall_ratios(pred_df_abbrev_only, pred_df, index="abbrev_only")
    (
        recall_strict_abbrev_or_contains,
        recall_exact_abbrev_or_contains,
        recall_narrow_abbrev_or_contains,
        recall_broad_abbrev_or_contains,
        ratios_abbrev_or_contains,
    ) = compute_recall_ratios(
        pred_df_abbrev_or_contains, pred_df, index="abbrev_or_contains"
    )
    (
        recall_strict_not_abbrev,
        recall_exact_not_abbrev,
        recall_narrow_not_abbrev,
        recall_broad_not_abbrev,
        ratios_not_abbrev,
    ) = compute_recall_ratios(pred_df_not_abbrev, pred_df, index="not_abbrev")

    # Inconsistency
    repeated = (
        pred_df.group_by(["filename", "code"])
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= 2)
        .select("filename", "code")
    )
    pred_df_repeated = pred_df.join(repeated, on=["filename", "code"], how="inner")
    pred_df_not_repeated = pred_df.join(repeated, on=["filename", "code"], how="anti")
    (
        recall_strict_repeated,
        recall_exact_repeated,
        recall_narrow_repeated,
        recall_broad_repeated,
        ratios_repeated,
    ) = compute_recall_ratios(pred_df_repeated, pred_df, index="repeated")
    (
        recall_strict_not_repeated,
        recall_exact_not_repeated,
        recall_narrow_not_repeated,
        recall_broad_not_repeated,
        ratios_not_repeated,
    ) = compute_recall_ratios(pred_df_not_repeated, pred_df, index="not_repeated")

    # aggregate scores and ratios into a compact structure used by main()
    final_results = {}
    labels = recall_strict_all.keys()

    def aggregate_results(
        final_results,
        label,
        recall_strict,
        recall_exact,
        recall_narrow,
        recall_broad,
        ratios,
    ):
        index_key = list(ratios.get(label, {}).keys())[0]
        final_results[label]["index"].append(index_key)
        final_results[label]["recall_strict"].append(
            list(recall_strict.get(label, {}).values())[0]
        )
        final_results[label]["recall_exact"].append(
            list(recall_exact.get(label, {}).values())[0]
        )
        final_results[label]["recall_narrow"].append(
            list(recall_narrow.get(label, {}).values())[0]
        )
        final_results[label]["recall_broad"].append(
            list(recall_broad.get(label, {}).values())[0]
        )
        final_results[label]["ratios"].append(list(ratios.get(label, {}).values())[0])

    for label in labels:
        final_results[label] = {
            "index": [],
            "recall_strict": [],
            "recall_exact": [],
            "recall_narrow": [],
            "recall_broad": [],
            "ratios": [],
        }
        aggregate_results(
            final_results,
            label,
            recall_strict_all,
            recall_exact_all,
            recall_narrow_all,
            recall_broad_all,
            ratios_all,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_seen_cuis,
            recall_exact_seen_cuis,
            recall_narrow_seen_cuis,
            recall_broad_seen_cuis,
            ratios_seen_cuis,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_unseen_cuis,
            recall_exact_unseen_cuis,
            recall_narrow_unseen_cuis,
            recall_broad_unseen_cuis,
            ratios_unseen_cuis,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_seen_mentions,
            recall_exact_seen_mentions,
            recall_narrow_seen_mentions,
            recall_broad_seen_mentions,
            ratios_seen_mentions,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_unseen_mentions,
            recall_exact_unseen_mentions,
            recall_narrow_unseen_mentions,
            recall_broad_unseen_mentions,
            ratios_unseen_mentions,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_seen_top_100_cuis,
            recall_exact_seen_top_100_cuis,
            recall_narrow_seen_top_100_cuis,
            recall_broad_seen_top_100_cuis,
            ratios_seen_top_100_cuis,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_unseen_top_100_cuis,
            recall_exact_unseen_top_100_cuis,
            recall_narrow_unseen_top_100_cuis,
            recall_broad_unseen_top_100_cuis,
            ratios_unseen_top_100_cuis,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_seen_top_100_mentions,
            recall_exact_seen_top_100_mentions,
            recall_narrow_seen_top_100_mentions,
            recall_broad_seen_top_100_mentions,
            ratios_seen_top_100_mentions,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_unseen_top_100_mentions,
            recall_exact_unseen_top_100_mentions,
            recall_narrow_unseen_top_100_mentions,
            recall_broad_unseen_top_100_mentions,
            ratios_unseen_top_100_mentions,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_seen_unique_pairs,
            recall_exact_seen_unique_pairs,
            recall_narrow_seen_unique_pairs,
            recall_broad_seen_unique_pairs,
            ratios_seen_unique_pairs,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_unseen_unique_pairs,
            recall_exact_unseen_unique_pairs,
            recall_narrow_unseen_unique_pairs,
            recall_broad_unseen_unique_pairs,
            ratios_unseen_unique_pairs,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_identical,
            recall_exact_identical,
            recall_narrow_identical,
            recall_broad_identical,
            ratios_identical,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_not_identical,
            recall_exact_not_identical,
            recall_narrow_not_identical,
            recall_broad_not_identical,
            ratios_not_identical,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_one_word,
            recall_exact_one_word,
            recall_narrow_one_word,
            recall_broad_one_word,
            ratios_one_word,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_two_words,
            recall_exact_two_words,
            recall_narrow_two_words,
            recall_broad_two_words,
            ratios_two_words,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_three_words,
            recall_exact_three_words,
            recall_narrow_three_words,
            recall_broad_three_words,
            ratios_three_words,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_more_than_three_words,
            recall_exact_more_than_three_words,
            recall_narrow_more_than_three_words,
            recall_broad_more_than_three_words,
            ratios_more_than_three_words,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_abbrev_only,
            recall_exact_abbrev_only,
            recall_narrow_abbrev_only,
            recall_broad_abbrev_only,
            ratios_abbrev_only,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_abbrev_or_contains,
            recall_exact_abbrev_or_contains,
            recall_narrow_abbrev_or_contains,
            recall_broad_abbrev_or_contains,
            ratios_abbrev_or_contains,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_not_abbrev,
            recall_exact_not_abbrev,
            recall_narrow_not_abbrev,
            recall_broad_not_abbrev,
            ratios_not_abbrev,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_repeated,
            recall_exact_repeated,
            recall_narrow_repeated,
            recall_broad_repeated,
            ratios_repeated,
        )
        aggregate_results(
            final_results,
            label,
            recall_strict_not_repeated,
            recall_exact_not_repeated,
            recall_narrow_not_repeated,
            recall_broad_not_repeated,
            ratios_not_repeated,
        )

    return final_results
