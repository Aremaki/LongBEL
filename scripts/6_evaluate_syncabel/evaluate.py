import argparse
import itertools
import logging
from pathlib import Path

import polars as pl
from tqdm import tqdm

from syncabel.evaluation import calculate_ner_per_label, calculate_norm_per_label

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def compute_metrics(pred_df, test_df, train_mentions, train_cuis, top_100_cuis):
    scores_all = calculate_norm_per_label(test_df.to_pandas(), pred_df.to_pandas())
    test_seen_mentions = test_df.filter(pl.col("span").is_in(train_mentions))
    pred_seen_mentions = pred_df.filter(pl.col("span").is_in(train_mentions))
    scores_seen_mention = calculate_norm_per_label(
        test_seen_mentions.to_pandas(), pred_seen_mentions.to_pandas()
    )
    test_unseen_mentions = test_df.filter(~pl.col("span").is_in(train_mentions))
    pred_unseen_mentions = pred_df.filter(~pl.col("span").is_in(train_mentions))
    scores_unseen_mention = calculate_norm_per_label(
        test_unseen_mentions.to_pandas(), pred_unseen_mentions.to_pandas()
    )
    test_seen_code = test_df.filter(pl.col("code").is_in(train_cuis))
    pred_seen_code = pred_df.filter(pl.col("code").is_in(train_cuis))
    scores_seen_code = calculate_norm_per_label(
        test_seen_code.to_pandas(), pred_seen_code.to_pandas()
    )
    test_unseen_code = test_df.filter(~pl.col("code").is_in(train_cuis))
    pred_unseen_code = pred_df.filter(~pl.col("code").is_in(train_cuis))
    scores_unseen_code = calculate_norm_per_label(
        test_unseen_code.to_pandas(), pred_unseen_code.to_pandas()
    )
    test_top_100_code = test_df.filter(pl.col("code").is_in(top_100_cuis))
    pred_top_100_code = pred_df.filter(pl.col("code").is_in(top_100_cuis))
    scores_top_100_code = calculate_norm_per_label(
        test_top_100_code.to_pandas(), pred_top_100_code.to_pandas()
    )
    # Combine all scores
    combined_scores = {}
    for label in scores_all.keys():
        combined_scores[label] = {
            "precision": scores_all[label]["total"]["precision"],
            "recall": scores_all[label]["total"]["recall"],
            "f_score": scores_all[label]["total"]["f_score"],
            "seen_mention_precision": scores_seen_mention[label]["total"]["precision"],
            "seen_mention_recall": scores_seen_mention[label]["total"]["recall"],
            "seen_mention_f_score": scores_seen_mention[label]["total"]["f_score"],
            "unseen_mention_precision": scores_unseen_mention[label]["total"][
                "precision"
            ],
            "unseen_mention_recall": scores_unseen_mention[label]["total"]["recall"],
            "unseen_mention_f_score": scores_unseen_mention[label]["total"]["f_score"],
            "seen_code_precision": scores_seen_code[label]["total"]["precision"],
            "seen_code_recall": scores_seen_code[label]["total"]["recall"],
            "seen_code_f_score": scores_seen_code[label]["total"]["f_score"],
            "unseen_code_precision": scores_unseen_code[label]["total"]["precision"],
            "unseen_code_recall": scores_unseen_code[label]["total"]["recall"],
            "unseen_code_f_score": scores_unseen_code[label]["total"]["f_score"],
            "top_100_code_precision": scores_top_100_code[label]["total"]["precision"],
            "top_100_code_recall": scores_top_100_code[label]["total"]["recall"],
            "top_100_code_f_score": scores_top_100_code[label]["total"]["f_score"],
        }
    return combined_scores


def main(args) -> None:
    datasets = args.datasets
    model_names = args.model_names
    selection_methods = args.selection_methods
    num_beams = args.num_beams
    with_group = args.with_group
    best = args.best
    augmented_data = args.augmented_data
    constraints = args.constraints
    tasks = args.tasks

    if "ner" in tasks:
        # NER results
        ner_results = []

        ner_models = [
            "bert_es",
            "bsc_ehr",
            "camembertav2",
            "cardioberta",
            "xlm_roberta",
        ]
        ner_product = list(itertools.product(datasets, ner_models))
        for dataset, ner_model in tqdm(ner_product, desc="NER Evaluation"):
            test_ner_df = pl.read_csv(
                Path("data") / dataset / "NER" / "gold" / "test.tsv",
                separator="\t",
                has_header=True,
            ).to_pandas()
            pred_ner_df = pl.read_csv(
                Path("data") / dataset / "NER" / "pred" / f"pred_{ner_model}.tsv",
                separator="\t",
                has_header=True,
            ).to_pandas()
            ner_score = calculate_ner_per_label(test_ner_df, pred_ner_df)
            for label in ner_score.keys():
                ner_results.append({
                    "dataset": dataset,
                    "ner_model": ner_model,
                    "label": label,
                    **ner_score[label]["total"],
                })

        ner_results_df = pl.DataFrame(ner_results)
        # Ensure the parent directory exists before writing
        output_folder = Path(args.output)
        output_folder.mkdir(parents=True, exist_ok=True)
        ner_results_df.write_csv(output_folder / "ner_evaluation_results.csv")
        logging.info(
            f"Evaluation finished. Results saved to {output_folder / 'ner_evaluation_results.csv'}"
        )

    if "norm" in tasks:
        # NORM results
        norm_results = []

        norm_product = list(
            itertools.product(
                datasets,
                selection_methods,
                model_names,
                num_beams,
                with_group,
                best,
                augmented_data,
                constraints,
            )
        )
        for (
            dataset,
            selection_method,
            model_name,
            n_beam,
            group,
            is_best,
            aug_data,
            constraint,
        ) in tqdm(norm_product, desc="NORM Evaluation"):
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
            ).unique(
                subset=[
                    "filename",
                    "label",
                    "start_span",
                    "end_span",
                ]
            )
            logging.info(
                f"Evaluating {model_name} on {dataset} with {selection_method}, num_beams={n_beam}, with_group={group}, best={is_best}, augmented_data={aug_data}, constraints={constraint}"
            )
            result_path = (
                Path("results")
                / "inference_outputs"
                / dataset
                / f"{aug_data}_{selection_method}{'_with_group' if group else ''}"
                / f"{model_name}_{'best' if is_best else 'last'}"
                / f"pred_test_{'no_constraint' if not constraint else 'constraint'}_{n_beam}_beams.tsv"
            )
            if not result_path.exists():
                logging.warning(f"Prediction file not found, skipping: {result_path}")
                continue
            pred_df = (
                pl.read_csv(
                    result_path,
                    separator="\t",
                    has_header=True,
                )
                .with_columns(pl.col("Predicted_CUI").cast(str))
                .unique(
                    subset=[
                        "filename",
                        "label",
                        "start_span",
                        "end_span",
                    ]
                )
            )
            scores = compute_metrics(
                pred_df,
                test_df,
                train_mentions,
                train_cuis,
                top_100_cuis,
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
            for label in scores.keys():
                norm_results.append({
                    **common_fields,
                    "threshold": "All data",
                    "score_label": "NA",
                    "label": label,
                    **scores[label],
                })
            for score_label in [
                "Prediction_score",
                "Prediction_beam_score",
            ]:
                for i in range(10):
                    threshold = i * 0.1
                    logging.info(f"Computing metrics for threshold: {threshold}")
                    pred_df_thresholded = pred_df.filter(
                        pl.col("Predicted_CUI").is_not_null()
                    ).filter(pl.col(score_label) >= threshold)
                    scores_thresholded = compute_metrics(
                        pred_df_thresholded,
                        test_df,
                        train_mentions,
                        train_cuis,
                        top_100_cuis,
                    )
                    for label in scores_thresholded.keys():
                        norm_results.append({
                            **common_fields,
                            "threshold": str(round(threshold, 1)),
                            "score_label": score_label,
                            "label": label,
                            **scores_thresholded[label],
                        })

        norm_results_df = pl.DataFrame(norm_results)
        # Ensure the parent directory exists before writing
        output_folder = Path(args.output)
        output_folder.mkdir(parents=True, exist_ok=True)
        norm_results_df.write_csv(output_folder / "norm_evaluation_results.csv")
        logging.info(
            f"Evaluation finished. Results saved to {output_folder / 'norm_evaluation_results.csv'}"
        )

    if "ner+norm" in tasks:
        # NER + NORM results
        bel_results = []

        bel_product = list(
            itertools.product(
                datasets,
                selection_methods,
                model_names,
                num_beams,
                with_group,
                best,
                augmented_data,
                constraints,
            )
        )
        for (
            dataset,
            selection_method,
            model_name,
            n_beam,
            group,
            is_best,
            aug_data,
            constraint,
        ) in tqdm(bel_product, desc="NER+NORM Evaluation"):
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
            ).unique(
                subset=[
                    "filename",
                    "label",
                    "start_span",
                    "end_span",
                ]
            )
            logging.info(
                f"Evaluating {model_name} on {dataset} with {selection_method}, num_beams={n_beam}, with_group={group}, best={is_best}, augmented_data={aug_data}, constraints={constraint}"
            )
            result_path = (
                Path("results")
                / "inference_outputs"
                / dataset
                / f"{aug_data}_{selection_method}{'_with_group' if group else ''}"
                / f"{model_name}_{'best' if is_best else 'last'}"
                / f"pred_test_ner_{'no_constraint' if not constraint else 'constraint'}_{n_beam}_beams.tsv"
            )
            if not result_path.exists():
                logging.warning(f"Prediction file not found, skipping: {result_path}")
                continue
            pred_df = (
                pl.read_csv(
                    result_path,
                    separator="\t",
                    has_header=True,
                )
                .with_columns(pl.col("Predicted_CUI").cast(str))
                .unique(
                    subset=[
                        "filename",
                        "label",
                        "start_span",
                        "end_span",
                    ]
                )
            )
            scores = compute_metrics(
                pred_df,
                test_df,
                train_mentions,
                train_cuis,
                top_100_cuis,
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
            for label in scores.keys():
                bel_results.append({
                    **common_fields,
                    "threshold": "All data",
                    "score_label": "NA",
                    "label": label,
                    **scores[label],
                })
            for score_label in [
                "Prediction_score",
                "Prediction_beam_score",
            ]:
                for i in range(11):
                    threshold = i * 0.1
                    logging.info(f"Computing metrics for threshold: {threshold}")
                    pred_df_thresholded = pred_df.filter(
                        pl.col("Predicted_CUI").is_not_null()
                    ).filter(pl.col(score_label) >= threshold)
                    scores_thresholded = compute_metrics(
                        pred_df_thresholded,
                        test_df,
                        train_mentions,
                        train_cuis,
                        top_100_cuis,
                    )
                    for label in scores_thresholded.keys():
                        bel_results.append({
                            **common_fields,
                            "threshold": threshold,
                            "score_label": score_label,
                            "label": label,
                            **scores_thresholded[label],
                        })

        bel_results_df = pl.DataFrame(bel_results)
        # Ensure the parent directory exists before writing
        output_folder = Path(args.output)
        output_folder.mkdir(parents=True, exist_ok=True)
        if len(args.model_names) == 1 and len(args.augmented_data) == 1:
            bel_results_df.write_csv(
                output_folder
                / f"bel_evaluation_results_{args.model_names[0]}_{args.augmented_data[0]}.csv"
            )
            logging.info(
                f"Evaluation finished. Results saved to {output_folder / f'bel_evaluation_results_{args.model_names[0]}_{args.augmented_data[0]}.csv'}"
            )
        else:
            bel_results_df.write_csv(output_folder / "bel_evaluation_results.csv")
            logging.info(
                f"Evaluation finished. Results saved to {output_folder / 'bel_evaluation_results.csv'}"
            )


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
        "--tasks",
        nargs="+",
        default=["ner", "norm", "ner+norm"],
        help="List of tasks to evaluate on",
    )
    args = parser.parse_args()
    main(args)
