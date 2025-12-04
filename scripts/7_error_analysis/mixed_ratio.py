import logging
from pathlib import Path

import polars as pl
from tqdm import tqdm

from syncabel.error_analysis import compute_simple_recall, load_predictions


def main(datasets: list[str]):
    for ckpt in ["best", "last"]:
        for num_beams in [5, 10]:
            all_scores = {
                "dataset": [],
                "data_augmentation": [],
                "human_ratio": [],
                "recall": [],
                "label": [],
            }
            selection_method = "tfidf"
            model_name = "Meta-Llama-3-8B-Instruct"
            for dataset in tqdm(datasets, desc="Evaluation"):
                for aug_data in ["human_only", "full_upsampled"]:
                    for human_ratio in [0.2, 0.4, 0.6, 0.8, 1.0]:
                        if human_ratio < 1.0:
                            human_ratio_str = (
                                "_"
                                + str(round(human_ratio * 100, 0)).replace(".0", "")
                                + "pct"
                            )
                        else:
                            human_ratio_str = ""
                        if human_ratio == 0.0 and aug_data == "full_upsampled":
                            preditction_path = (
                                Path("results")
                                / "inference_outputs"
                                / dataset
                                / f"synth_only_{selection_method}"
                                / f"{model_name}_{ckpt}"
                                / f"pred_test_constraint_{num_beams}_beams.tsv"
                            )
                        else:
                            preditction_path = (
                                Path("results")
                                / "inference_outputs"
                                / dataset
                                / f"{aug_data}_{selection_method}{human_ratio_str}"
                                / f"{model_name}_{ckpt}"
                                / f"pred_test_constraint_{num_beams}_beams.tsv"
                            )
                        if not preditction_path.exists():
                            logging.warning(
                                f"Prediction file not found, skipping: {preditction_path}"
                            )
                            continue
                        pred_df = load_predictions(
                            preditction_path,
                            dataset=dataset,
                        )
                        score = compute_simple_recall(pred_df)
                        for label in score.keys():
                            all_scores["dataset"].append(dataset)
                            all_scores["data_augmentation"].append(aug_data)
                            all_scores["human_ratio"].append(human_ratio)
                            all_scores["label"].append(label)
                            all_scores["recall"].append(score[label])

            results = pl.DataFrame(all_scores)
            results_path = (
                Path("results") / "mixed_ratio" / f"results_{ckpt}_{num_beams}.csv"
            )
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results.write_csv(results_path)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    datasets_to_evaluate = ["SPACCC", "MedMentions", "EMEA", "MEDLINE"]
    main(datasets=datasets_to_evaluate)
