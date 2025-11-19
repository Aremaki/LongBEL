import argparse
import itertools
import logging
from pathlib import Path

import polars as pl
from tqdm import tqdm

from syncabel.evaluation import evaluate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def compute_metrics(pred_df, test_df, train_mentions, train_cuis, top_100_cuis):
    scores_all = evaluate(test_df, pred_df, task="norm")
    test_seen_mentions = test_df.filter(pl.col("span").is_in(train_mentions))
    pred_seen_mentions = pred_df.join(
        test_seen_mentions.select(["filename", "label", "start_span", "end_span"]),
        on=["filename", "label", "start_span", "end_span"],
        how="inner",
    )
    scores_seen_mention = evaluate(test_seen_mentions, pred_seen_mentions, task="norm")
    test_unseen_mentions = test_df.filter(~pl.col("span").is_in(train_mentions))
    pred_unseen_mentions = pred_df.join(
        test_unseen_mentions.select(["filename", "label", "start_span", "end_span"]),
        on=["filename", "label", "start_span", "end_span"],
        how="inner",
    )
    scores_unseen_mention = evaluate(
        test_unseen_mentions, pred_unseen_mentions, task="norm"
    )
    test_seen_code = test_df.filter(pl.col("code").is_in(train_cuis))
    pred_seen_code = pred_df.join(
        test_seen_code.select(["filename", "label", "start_span", "end_span"]),
        on=["filename", "label", "start_span", "end_span"],
        how="inner",
    )
    scores_seen_code = evaluate(test_seen_code, pred_seen_code, task="norm")
    test_unseen_code = test_df.filter(~pl.col("code").is_in(train_cuis))
    pred_unseen_code = pred_df.join(
        test_unseen_code.select(["filename", "label", "start_span", "end_span"]),
        on=["filename", "label", "start_span", "end_span"],
        how="inner",
    )
    scores_unseen_code = evaluate(test_unseen_code, pred_unseen_code, task="norm")
    test_top_100_code = test_df.filter(pl.col("code").is_in(top_100_cuis))
    pred_top_100_code = pred_df.join(
        test_top_100_code.select(["filename", "label", "start_span", "end_span"]),
        on=["filename", "label", "start_span", "end_span"],
        how="inner",
    )
    scores_top_100_code = evaluate(test_top_100_code, pred_top_100_code, task="norm")
    test_identical = test_df.filter(
        pl.col("span").str.to_lowercase() == pl.col("annotation").str.to_lowercase()
    )
    pred_identical = pred_df.join(
        test_identical.select(["filename", "label", "start_span", "end_span"]),
        on=["filename", "label", "start_span", "end_span"],
        how="inner",
    )
    scores_identical = evaluate(test_identical, pred_identical, task="norm")
    test_not_identical = test_df.filter(
        pl.col("span").str.to_lowercase() != pl.col("annotation").str.to_lowercase()
    )
    pred_not_identical = pred_df.join(
        test_not_identical.select(["filename", "label", "start_span", "end_span"]),
        on=["filename", "label", "start_span", "end_span"],
        how="inner",
    )
    scores_not_identical = evaluate(test_not_identical, pred_not_identical, task="norm")
    test_one_word = test_df.filter(pl.col("span").str.count_matches(" ") == 0)
    pred_one_word = pred_df.join(
        test_one_word.select(["filename", "label", "start_span", "end_span"]),
        on=["filename", "label", "start_span", "end_span"],
        how="inner",
    )
    scores_one_word = evaluate(test_one_word, pred_one_word, task="norm")
    test_multi_word = test_df.filter(pl.col("span").str.count_matches(" ") >= 1)
    pred_multi_word = pred_df.join(
        test_multi_word.select(["filename", "label", "start_span", "end_span"]),
        on=["filename", "label", "start_span", "end_span"],
        how="inner",
    )
    scores_multi_word = evaluate(test_multi_word, pred_multi_word, task="norm")
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
            "identical_precision": scores_identical[label]["total"]["precision"],
            "identical_recall": scores_identical[label]["total"]["recall"],
            "identical_f_score": scores_identical[label]["total"]["f_score"],
            "not_identical_precision": scores_not_identical[label]["total"][
                "precision"
            ],
            "not_identical_recall": scores_not_identical[label]["total"]["recall"],
            "not_identical_f_score": scores_not_identical[label]["total"]["f_score"],
            "one_word_precision": scores_one_word[label]["total"]["precision"],
            "one_word_recall": scores_one_word[label]["total"]["recall"],
            "one_word_f_score": scores_one_word[label]["total"]["f_score"],
            "multi_word_precision": scores_multi_word[label]["total"]["precision"],
            "multi_word_recall": scores_multi_word[label]["total"]["recall"],
            "multi_word_f_score": scores_multi_word[label]["total"]["f_score"],
        }
    return combined_scores


def evaluate_ner(args) -> None:
    datasets = args.datasets
    output_folder = Path(args.output)

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
        )
        pred_ner_df = pl.read_csv(
            Path("data") / dataset / "NER" / "pred" / f"pred_{ner_model}.tsv",
            separator="\t",
            has_header=True,
        )
        ner_score = evaluate(test_ner_df, pred_ner_df, task="ner")
        for label in ner_score.keys():
            ner_results.append({
                "dataset": dataset,
                "ner_model": ner_model,
                "label": label,
                **ner_score[label]["total"],
            })

    ner_results_df = pl.DataFrame(ner_results)

    # Ensure the parent directory exists before writing
    output_folder.mkdir(parents=True, exist_ok=True)
    ner_results_df.write_csv(output_folder / "ner_evaluation_results.csv")
    logging.info(
        f"Evaluation finished. Results saved to {output_folder / 'ner_evaluation_results.csv'}"
    )


def evaluate_ned(args, task) -> None:
    datasets = args.datasets
    model_names = args.model_names
    selection_methods = args.selection_methods
    num_beams = args.num_beams
    with_group = args.with_group
    best = args.best
    augmented_data = args.augmented_data
    constraints = args.constraints
    output_folder = Path(args.output)
    split_name = "test" if task == "norm" else "test_ner"

    ned_results = []

    var_product = list(
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
    ) in tqdm(var_product, desc=f"{task.upper()} Evaluation"):
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
            schema_overrides={
                "code": str,  # force as string
            },  # type: ignore
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
            schema_overrides={
                "code": str,  # force as string
            },  # type: ignore
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
            / f"pred_{split_name}_{'no_constraint' if not constraint else 'constraint'}_{n_beam}_beams.tsv"
        )
        if not result_path.exists():
            logging.warning(f"Prediction file not found, skipping: {result_path}")
            continue
        pred_df = pl.read_csv(
            result_path,
            separator="\t",
            has_header=True,
            schema_overrides={
                "code": str,  # force as string
                "Predicted_CUI": str,  # force as string
            },  # type: ignore
        ).unique(
            subset=[
                "filename",
                "label",
                "start_span",
                "end_span",
            ]
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
            ned_results.append({
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
                threshold = round(i * 0.1, 1)
                logging.debug(
                    f"Computing metrics for threshold: {threshold} on score {score_label}"
                )
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
                    ned_results.append({
                        **common_fields,
                        "threshold": str(threshold),
                        "score_label": score_label,
                        "label": label,
                        **scores_thresholded[label],
                    })

    ned_results_df = pl.DataFrame(ned_results)
    # Ensure the parent directory exists before writing
    output_folder.mkdir(parents=True, exist_ok=True)

    if len(model_names) == 1 and len(augmented_data) == 1:
        parquet_folder = output_folder / f"all_{task}_parquets"
        parquet_folder.mkdir(parents=True, exist_ok=True)
        ned_results_df.write_parquet(
            parquet_folder
            / f"{task}_evaluation_results_{model_names[0]}_{augmented_data[0]}.parquet"
        )  # save standalone mapping
        output_folder = output_folder / f"all_{task}_csv"
        output_folder.mkdir(parents=True, exist_ok=True)
        ned_results_df.write_csv(
            output_folder
            / f"{task}_evaluation_results_{model_names[0]}_{augmented_data[0]}.csv"
        )
        logging.info(
            f"Evaluation finished. Results saved to {output_folder / f'{task}_evaluation_results_{model_names[0]}_{augmented_data[0]}.csv'}"
        )
    else:
        ned_results_df.write_parquet(
            output_folder / f"{task}_evaluation_results.parquet"
        )
        ned_results_df.write_csv(output_folder / f"{task}_evaluation_results.csv")
        logging.info(
            f"Evaluation finished. Results saved to {output_folder / f'{task}_evaluation_results.csv'}"
        )


def main(args) -> None:
    tasks = args.tasks
    if "ner" in tasks:
        evaluate_ner(args)
    if "norm" in tasks:
        evaluate_ned(args, task="norm")
    if "ner+norm" in tasks:
        evaluate_ned(args, task="ner+norm")


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
