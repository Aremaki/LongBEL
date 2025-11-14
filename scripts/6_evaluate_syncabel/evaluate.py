import argparse
import logging
import pickle
import re
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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
    split_names = args.split_names

    all_results = []

    for dataset in datasets:
        for selection_method in selection_methods:
            # Load train data for seen/unseen evaluation
            train_path = (
                Path("data")
                / "final_data"
                / dataset
                / f"train_{selection_method}_annotations.tsv"
            )
            train_df = pl.read_csv(
                train_path,
                separator="\t",
                has_header=True,
            )

            train_cuis = set(train_df["code"].drop_nulls())
            train_mentions = set(train_df["span"].drop_nulls())
            top_100_cuis = set(train_df["code"].value_counts().head(100)["code"])

            test_path = (
                Path("data")
                / "final_data"
                / dataset
                / f"test_{selection_method}_annotations.tsv"
            )
            test_df = pl.read_csv(
                test_path,
                separator="\t",
                has_header=True,
            )
            for split_name in split_names:
                for model_name in model_names:
                    for n_beam in num_beams:
                        for group in with_group:
                            for is_best in best:
                                for aug_data in augmented_data:
                                    for constraint in constraints:
                                        logging.info(
                                            f"Evaluating {model_name} on {dataset} with {selection_method}, num_beams={n_beam}, with_group={group}, best={is_best}, augmented_data={aug_data}, constraints={constraint}"
                                        )
                                        result_path = (
                                            Path("results")
                                            / "inference_outputs"
                                            / dataset
                                            / f"{aug_data}_{selection_method}{'_with_group' if group else ''}"
                                            / f"{model_name}_{'best' if is_best else 'last'}"
                                            / f"pred_{split_name}_{'no_constraint' if not constraint else 'constraint'}_{n_beam}_beams.tsv"
                                        )
                                        if not result_path.exists():
                                            logging.warning(
                                                f"Prediction file not found, skipping: {result_path}"
                                            )
                                            continue
                                        result_df = pl.read_csv(
                                            result_path,
                                            separator="\t",
                                            has_header=True,
                                        )
                                        print(result_df.head())
                                        print(result_df.shape)
                                        # Il faut le NER, NED, many score thresholds, all in one -> par label puis overall
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
                                                all_results.append({
                                                    **common_fields,
                                                    **m,
                                                })
                                        else:
                                            all_results.append({
                                                **common_fields,
                                                **metrics,
                                            })

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
    parser.add_argument(
        "--split_names",
        nargs="+",
        default=["test"],
        help="List of data splits to use (e.g., test, test_ner).",
    )
    args = parser.parse_args()
    main(args)
