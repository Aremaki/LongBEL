import logging
from pathlib import Path

import polars as pl

# from keycare.RelExtractor import RelExtractor
from tqdm import tqdm

from syncabel.error_analysis import compute_metrics, load_predictions


def main(datasets: list[str]):
    all_scores = {}
    all_ratios = {}
    selection_method = "tfidf"
    model_name = "Meta-Llama-3-8B-Instruct"
    # relextractor = RelExtractor(
    #     relation_method="llm",
    #     batch_size=8,
    #     model="gpt-4.1-mini",
    #     temperature=0.0,
    # )
    for dataset in tqdm(datasets, desc="Evaluation"):
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
                "code": str,
                "mention_id": str,
                "filename": str,  # force as string
            },  # type: ignore
        )
        validation_path = (
            Path("data")
            / "final_data"
            / dataset
            / f"validation_{selection_method}_annotations.tsv"
        )
        if validation_path.exists():
            val_df = pl.read_csv(
                validation_path,
                separator="\t",
                has_header=True,
                schema_overrides={
                    "code": str,
                    "mention_id": str,
                    "filename": str,  # force as string
                },  # type: ignore
            )
            # Reduce validation dataset to 10% as before
            split = int(len(val_df) * 0.9)
            val_df = val_df[:split]
            train_df = pl.concat([train_df, val_df])
        train_cuis = set(train_df["code"].drop_nulls())
        train_mentions = set(train_df["span"].drop_nulls())
        unique_pairs = (
            train_df.select(["span", "code"]).drop_nulls().unique().iter_rows()
        )
        unique_pairs = set(unique_pairs)
        top_100_cuis = set(train_df["code"].value_counts().head(100)["code"])
        top_100_mentions = set(train_df["span"].value_counts().head(100)["span"])
        for aug_data in ["human_only", "full_upsampled"]:
            for constraint in [True, False]:
                for ckpt in ["last", "best"]:
                    for n_beams in [5, 10]:
                        preditction_path = (
                            Path("results")
                            / "inference_outputs"
                            / dataset
                            / f"{aug_data}_{selection_method}"
                            / f"{model_name}_{ckpt}"
                            / f"pred_test_{'no_constraint' if not constraint else 'constraint'}_{n_beams}_beams.tsv"
                        )
                        if not preditction_path.exists():
                            logging.warning(
                                f"Prediction file not found, skipping: {preditction_path}"
                            )
                            continue
                        pred_df = load_predictions(
                            preditction_path,
                            dataset=dataset,
                            # relextractor=relextractor,
                        )
                        scores = compute_metrics(
                            pred_df=pred_df,
                            train_mentions=train_mentions,
                            train_cuis=train_cuis,
                            top_100_cuis=top_100_cuis,
                            top_100_mentions=top_100_mentions,
                            unique_pairs=unique_pairs,
                        )
                        aug_label = (
                            "normal" if aug_data == "human_only" else "augmented"
                        )
                        for label in scores.keys():
                            if label not in all_ratios:
                                all_ratios[label] = {}
                            if dataset not in all_ratios[label]:
                                all_ratios[label][dataset] = {
                                    "index": scores[label]["index"]
                                }
                            if label not in all_scores:
                                all_scores[label] = {}
                            if (
                                f"{dataset}_{aug_label}_{ckpt}_{n_beams}"
                                not in all_scores[label]
                            ):
                                all_scores[label][
                                    f"{dataset}_{aug_label}_{ckpt}_{n_beams}"
                                ] = {
                                    "constraint": {"index": scores[label]["index"]},
                                    "no_constraint": {"index": scores[label]["index"]},
                                }
                            all_scores[label][
                                f"{dataset}_{aug_label}_{ckpt}_{n_beams}"
                            ]["no_constraint" if not constraint else "constraint"][
                                "recall_strict"
                            ] = scores[label]["recall_strict"]
                            all_scores[label][
                                f"{dataset}_{aug_label}_{ckpt}_{n_beams}"
                            ]["no_constraint" if not constraint else "constraint"][
                                "recall_exact"
                            ] = scores[label]["recall_exact"]
                            all_scores[label][
                                f"{dataset}_{aug_label}_{ckpt}_{n_beams}"
                            ]["no_constraint" if not constraint else "constraint"][
                                "recall_narrow"
                            ] = scores[label]["recall_narrow"]
                            all_scores[label][
                                f"{dataset}_{aug_label}_{ckpt}_{n_beams}"
                            ]["no_constraint" if not constraint else "constraint"][
                                "recall_broad"
                            ] = scores[label]["recall_broad"]
                            all_ratios[label][dataset]["ratios"] = scores[label][
                                "ratios"
                            ]
    # Write results
    for label in all_scores.keys():
        for dataset in datasets:
            for aug_data in ["normal", "augmented"]:
                for ckpt in ["last", "best"]:
                    for n_beams in [5, 10]:
                        if (
                            f"{dataset}_{aug_data}_{ckpt}_{n_beams}"
                            not in all_scores[label]
                        ):
                            continue
                        for key in all_scores[label][
                            f"{dataset}_{aug_data}_{ckpt}_{n_beams}"
                        ].keys():
                            df_score = pl.DataFrame(
                                all_scores[label][
                                    f"{dataset}_{aug_data}_{ckpt}_{n_beams}"
                                ][key]
                            )
                            score_path = (
                                Path("results")
                                / "error_analysis"
                                / label
                                / dataset
                                / f"score_{key}_{aug_data}_{ckpt}_{n_beams}.csv"
                            )
                            score_path.parent.mkdir(parents=True, exist_ok=True)
                            df_score.write_csv(score_path)
            if dataset not in all_ratios[label]:
                continue
            df_ratio = pl.DataFrame(all_ratios[label][dataset])
            ratio_path = (
                Path("results") / "error_analysis" / label / dataset / "ratio.csv"
            )
            ratio_path.parent.mkdir(parents=True, exist_ok=True)
            df_ratio.write_csv(ratio_path)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    datasets_to_evaluate = ["SPACCC", "MedMentions", "EMEA", "MEDLINE"]
    main(datasets=datasets_to_evaluate)
