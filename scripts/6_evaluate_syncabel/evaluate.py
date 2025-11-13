import argparse
import logging
import pickle
import re
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def add_cui_column(df: pl.DataFrame, umls_df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a 'CUI' column to the DataFrame by mapping composite 'Entity' strings to UMLS concepts.

    The function performs the following steps:
    1.  Cleans up special characters in the 'Entity' column of the UMLS DataFrame.
    2.  Splits the 'Entity' column of the input DataFrame by '<SEP>' into individual terms.
    3.  Explodes the DataFrame so that each row corresponds to a single term.
    4.  Performs a left join with the UMLS DataFrame on the term and 'GROUP' to find the CUI for each term.
    5.  Groups the data back by the original rows.
    6.  Filters out rows where none of the constituent terms mapped to a CUI.
    7.  Constructs the final CUI string by sorting the unique CUIs and joining them with '+'.
    """
    # 1) Prepare umls_df: clean Entity strings
    umls_df_clean = umls_df.select(["CUI", "Entity", "GROUP"]).with_columns(
        pl.col("Entity")
        .str.replace_all("\xa0", " ", literal=True)
        .str.replace_all(r"[\{\[]", "(", literal=False)
        .str.replace_all(r"[\}\]]", ")", literal=False)
    )

    # 2) Explode df by splitting 'Entity' into multiple terms
    df = df.with_row_index()
    df_exploded = (
        df.select(["row_index", "Entity", "GROUP"])
        .with_columns(pl.col("Entity").str.split("<SEP>"))
        .explode("Entity")
    )

    # 3) Join with UMLS data to get CUIs for each term
    df_joined = df_exploded.join(
        umls_df_clean,
        on=["Entity", "GROUP"],
        how="left",
    )

    # 4) Group back, aggregate CUIs, and count nulls to find incomplete matches
    df_grouped = df_joined.group_by("row_idx").agg(
        pl.col("CUI").drop_nulls().unique().alias("cui_list"),
    )

    # 5) Filter for rows with at least one match and create final CUI string
    result = (
        df_grouped.filter(pl.col("cui_list").list.len() > 0)
        .with_columns(pl.col("cui_list").list.sort().list.join("+").alias("CUI"))
        .select(["CUI", "row_idx"])
    )

    return df.join(result, on="row_idx", how="left")


def _extract_mention_and_type(text: str, transition_verb: str) -> tuple[str, str]:
    """Extract the mention and type from a string"""
    mention = text.split("<SEP>")[-1].rsplit(f" {transition_verb}", 1)[0].strip()
    mention = text.split("</s>")[-1].rsplit(" is", 1)[0].strip()
    type_match = re.search(r"\{(.*?)\}", text)
    ent_type = type_match.group(1).strip() if type_match else "Unknown"
    return mention, ent_type


def structure_data(
    umls_df: pl.DataFrame,
    source: list[str],
    target: list[str],
    pred: list[str],
    transition_verb: str,
) -> pl.DataFrame:
    mentions = []
    ent_types = []
    for sentence in source:
        mention, ent_type = _extract_mention_and_type(sentence, transition_verb)
        mentions.append(mention)
        ent_types.append(ent_type)
    df_mentions = pl.DataFrame({"Mention": mentions}).with_row_index("row_idx")
    df_source = pl.DataFrame({
        "GROUP": ent_types,
        "Entity": target,
    })
    df_pred = pl.DataFrame({"GROUP": ent_types, "Entity": pred})

    # add CUI columns
    df_source = add_cui_column(df_source, umls_df).rename({
        "CUI": "CUI_gold",
        "Entity": "Entity_gold",
    })
    df_pred = add_cui_column(df_pred, umls_df).drop("GROUP")

    # concat pred and source dataframes
    result = pl.concat(
        [
            df_pred,
            df_source,
            df_mentions,
        ],
        how="align",
    )
    return result


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def compute_metrics(
    result_df, train_mentions, train_cuis, top_100_cuis, per_group=False
):
    if per_group:
        groups = result_df["GROUP"].unique().to_list()
        all_metrics = []
        for group in groups:
            if group is None:
                continue
            group_df = result_df.filter(pl.col("GROUP") == group)
            metrics = compute_metrics(
                group_df, train_mentions, train_cuis, top_100_cuis, per_group=False
            )
            if isinstance(metrics, dict):
                metrics["GROUP"] = group
                all_metrics.append(metrics)
        return all_metrics

    correct = result_df.filter(pl.col("CUI") == pl.col("CUI_gold"))

    recall = len(correct) / len(result_df)

    seen_mentions_df = result_df.filter(pl.col("Mention").is_in(train_mentions))
    recall_seen_mention = (
        len(seen_mentions_df.filter(pl.col("CUI") == pl.col("CUI_gold")))
        / len(seen_mentions_df)
        if len(seen_mentions_df) > 0
        else 0
    )

    unseen_mentions_df = result_df.filter(~pl.col("Mention").is_in(train_mentions))
    recall_unseen_mention = (
        len(unseen_mentions_df.filter(pl.col("CUI") == pl.col("CUI_gold")))
        / len(unseen_mentions_df)
        if len(unseen_mentions_df) > 0
        else 0
    )

    seen_cui_df = result_df.filter(pl.col("CUI_gold").is_in(train_cuis))
    recall_seen_cui = (
        len(seen_cui_df.filter(pl.col("CUI") == pl.col("CUI_gold"))) / len(seen_cui_df)
        if len(seen_cui_df) > 0
        else 0
    )

    unseen_cui_df = result_df.filter(~pl.col("CUI_gold").is_in(train_cuis))
    recall_unseen_cui = (
        len(unseen_cui_df.filter(pl.col("CUI") == pl.col("CUI_gold")))
        / len(unseen_cui_df)
        if len(unseen_cui_df) > 0
        else 0
    )

    top_100_cui_df = result_df.filter(pl.col("CUI_gold").is_in(top_100_cuis))
    recall_top_100_cui = (
        len(top_100_cui_df.filter(pl.col("CUI") == pl.col("CUI_gold")))
        / len(top_100_cui_df)
        if len(top_100_cui_df) > 0
        else 0
    )

    return {
        "Recall": recall,
        "Recall Seen Mention": recall_seen_mention,
        "Recall Unseen Mention": recall_unseen_mention,
        "Recall Seen CUI": recall_seen_cui,
        "Recall Unseen CUI": recall_unseen_cui,
        "Recall Top 100 CUI": recall_top_100_cui,
    }


def main(args) -> None:
    datasets = args.datasets
    model_names = args.model_names
    selection_methods = args.selection_methods
    num_beams = args.num_beams
    with_group = args.with_group
    best = args.best
    augmented_data = args.augmented_data
    constraints = args.constraints
    add_group_column = args.add_group_column

    all_results = []

    for dataset in datasets:
        if dataset == "MedMentions":
            dataset_short = "MM"
            transition_verb = "is"
        elif dataset == "QUAERO":
            transition_verb = "est"
            dataset_short = "QUAERO"
        elif dataset == "SPACCC":
            transition_verb = "es"
            dataset_short = "SPACCC"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        umls_path = (
            Path("data")
            / "UMLS_processed"
            / dataset_short
            / "all_disambiguated.parquet"
        )
        umls_df = pl.read_parquet(umls_path)

        for selection_method in selection_methods:
            # Load train data for seen/unseen evaluation
            train_source_path = (
                Path("data")
                / "final_data"
                / dataset
                / f"train_{selection_method}_source_with_group.pkl"
            )
            train_target_path = (
                Path("data")
                / "final_data"
                / dataset
                / f"train_{selection_method}_target.pkl"
            )
            train_source_list = load_pickle(train_source_path)
            train_target_list = load_pickle(train_target_path)

            train_mentions = set()
            train_ent_types = []
            for sentence in train_source_list:
                mention, ent_type = _extract_mention_and_type(sentence, transition_verb)
                train_mentions.add(mention)
                train_ent_types.append(ent_type)

            train_df = pl.DataFrame({
                "Entity": train_target_list,
                "GROUP": train_ent_types,
            })
            train_df = add_cui_column(train_df, umls_df)
            train_cuis = set(train_df["CUI"].drop_nulls())
            top_100_cuis = set(train_df["CUI"].value_counts().head(100)["CUI"])

            source_path = (
                Path("data")
                / "final_data"
                / dataset
                / f"test_{selection_method}_source_with_group.pkl"
            )
            target_path = (
                Path("data")
                / "final_data"
                / dataset
                / f"test_{selection_method}_target.pkl"
            )
            source = load_pickle(source_path)
            target = load_pickle(target_path)
            for model_name in model_names:
                for n_beam in num_beams:
                    for group in with_group:
                        for is_best in best:
                            for aug_data in augmented_data:
                                for constraint in constraints:
                                    logging.info(
                                        f"Evaluating {model_name} on {dataset} with {selection_method}, num_beams={n_beam}, with_group={group}, best={is_best}, augmented_data={aug_data}, constraints={constraint}"
                                    )
                                    model_path = (
                                        Path("models")
                                        / "NED"
                                        / f"{dataset}_{aug_data}_{selection_method}{'_with_group' if group else ''}"
                                        / model_name
                                        / f"model_{'best' if is_best else 'last'}"
                                    )
                                    pred_path = (
                                        model_path
                                        / f"pred_test_{'no_constraint' if not constraint else 'constraint'}_{n_beam}_beams{'_typed' if constraint else ''}.pkl"
                                    )
                                    if not pred_path.exists():
                                        logging.warning(
                                            f"Prediction file not found, skipping: {pred_path}"
                                        )
                                        continue
                                    pred = load_pickle(pred_path)
                                    result_df = structure_data(
                                        umls_df=umls_df,
                                        source=source,
                                        target=target,
                                        pred=pred,
                                        transition_verb=transition_verb,
                                    )
                                    print(result_df.head())
                                    print(result_df.shape)

                                    metrics = compute_metrics(
                                        result_df,
                                        train_mentions,
                                        train_cuis,
                                        top_100_cuis,
                                        add_group_column,
                                    )

                                    # Support both overall metrics (dict) and per-group metrics (list of dicts)
                                    common_fields = {
                                        "model_name": model_name,
                                        "dataset": dataset,
                                        "selection_method": selection_method,
                                        "num_beams": n_beam,
                                        "with_group": group,
                                        "best": is_best,
                                        "augmented_data": aug_data,
                                        "constraints": constraint,
                                    }

                                    if isinstance(metrics, list):
                                        for m in metrics:
                                            all_results.append({**common_fields, **m})
                                    else:
                                        all_results.append({**common_fields, **metrics})

    results_df = pl.DataFrame(all_results)
    # Ensure the parent directory exists before writing
    output_folder = Path(args.output)
    if output_folder.parent != Path(""):
        output_folder.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_csv(str(output_folder))
    logging.info(f"Evaluation finished. Results saved to {output_folder}")


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(
        description="A script for evaluating seq2seq model"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["MedMentions", "EMEA", "MEDLINE", "SPACCC"],
        required=True,
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=["mt5-large"],
        help="List of model names",
    )
    parser.add_argument(
        "--selection_methods",
        nargs="+",
        default=["embedding", "levenshtein", "tfidf", "title"],
        help="List of selection methods",
    )
    parser.add_argument(
        "--num_beams",
        nargs="+",
        type=int,
        default=[5],
        help="List of number of beams",
    )
    parser.add_argument(
        "--with_group",
        nargs="+",
        type=lambda x: (str(x).lower() == "true"),
        default=[True, False],
        help="List of with_group flags",
    )
    parser.add_argument(
        "--best",
        nargs="+",
        type=lambda x: (str(x).lower() == "true"),
        default=[True, False],
        help="List of best flags",
    )
    parser.add_argument(
        "--augmented_data",
        nargs="+",
        default=["human_only"],
        help="List of augmented_data flags",
    )
    parser.add_argument(
        "--constraints",
        nargs="+",
        type=lambda x: (str(x).lower() == "true"),
        default=[True, False],
        help="List of constraints flags",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Output CSV filename or path. Example: result_spaccc.csv",
    )
    parser.add_argument(
        "--add_group_column",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Add GROUP column to the result table and compute metrics per group.",
    )
    args = parser.parse_args()
    main(args)
