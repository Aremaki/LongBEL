from pathlib import Path
from typing import Optional

import polars as pl
import typer
from tqdm import tqdm

from syncabel.error_analysis import load_predictions
from syncabel.llm_as_a_judge import LLMRelator


def get_concepts_from_codes(
    codes: list[str],
    representative_names: list[str],
    semantic_group: str,
    umls_df: pl.DataFrame,
    mention: Optional[str] = None,
    context_sentence: Optional[str] = None,
) -> list[dict[str, str]]:
    """Provide a list of dictionaries with 'mentions', 'synonym', 'title', 'semantic_group', and 'code' for each code in the input list."""
    concepts = []
    code_col = "SNOMED_code" if "SNOMED_code" in umls_df.columns else "CUI"
    for code, representative_name in zip(codes, representative_names):
        concept_info = umls_df.filter(pl.col(code_col) == code)
        if concept_info.height == 0:
            print("Code not found in UMLS df:", code)
            continue
        synonyms = concept_info["Entity"].unique().to_list()
        title = concept_info["Title"].first()
        concept_dict = {
            "text": representative_name,
            "synonyms": synonyms,
            "title": title,
            "semantic_group": semantic_group,
            "code": code,
        }
        if mention is not None and context_sentence is not None:
            concept_dict["mention"] = mention
            concept_dict["context_sentence"] = context_sentence
        concepts.append(concept_dict)
    return concepts


app = typer.Typer()


@app.command()
def main(
    datasets: list[str] = typer.Option(
        ["SPACCC", "MEDLINE", "EMEA", "MedMentions"],
        help="Dataset name (e.g., MEDLINE)",
    ),
    model_name: str = typer.Option("Meta-Llama-3-8B-Instruct", help="Model name"),
    aug_data_list: list[str] = typer.Option(
        ["human_only", "full_upsampled"], help="Augmented data type"
    ),
    data_splits: list[str] = typer.Option(
        ["test"], help="Data split (e.g., test, val, train)"
    ),
):
    """Evaluate LLM predictions using LLMRelator."""

    for dataset in datasets:
        if dataset in ["SPACCC", "MEDLINE"]:
            ckpt = "last"
        else:
            ckpt = "best"
        # Load UMLS data
        if dataset == "MedMentions":
            dataset_short = "MM"
        elif dataset in ["EMEA", "MEDLINE"]:
            dataset_short = "QUAERO"
        elif "SPACCC" in dataset:
            dataset_short = "SPACCC"
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        umls_path = Path(
            f"data/UMLS_processed/{dataset_short}/all_disambiguated.parquet"
        )
        umls_df = pl.read_parquet(umls_path)
        for aug_data in aug_data_list:
            for data_split in data_splits:
                print(
                    f"Evaluating dataset: {dataset}, aug_data: {aug_data}, split: {data_split}"
                )
                example_path = Path(
                    f"results/final_outputs/{dataset}/{aug_data}_tfidf/{model_name}_{ckpt}/pred_{data_split}_constraint_5_beams.tsv"
                )
                example_results = load_predictions(example_path)
                # loop over the rows
                llm_evaluation = []
                relator = LLMRelator()

                for row in tqdm(
                    example_results.iter_rows(named=True), total=example_results.height
                ):
                    gold_codes = row["code"]
                    pred_codes = row["Predicted_code"]
                    if gold_codes == pred_codes:
                        llm_evaluation.append("STRICT")
                        continue
                    if not pred_codes:
                        llm_evaluation.append("NO_RELATION")
                        continue
                    gold_codes = gold_codes.split("+")
                    pred_codes = pred_codes.split("+")
                    # Check if all predicted codes are in gold codes
                    if all(code in gold_codes for code in pred_codes):
                        llm_evaluation.append("PARTIAL")
                        continue
                    semantic_group = row["label"]
                    gold_representative_names = row["annotation"].split("<SEP>")
                    mention = row["span"]
                    context_sentence = row["sentence"]
                    gold_concepts = get_concepts_from_codes(
                        gold_codes,
                        gold_representative_names,
                        semantic_group,
                        umls_df,
                        mention,
                        context_sentence,
                    )
                    pred_representative_names = row["Prediction"].split("<SEP>")
                    pred_concepts = get_concepts_from_codes(
                        pred_codes, pred_representative_names, semantic_group, umls_df
                    )
                    label = relator.compute_relation(gold_concepts, pred_concepts)
                    llm_evaluation.append(label)
                    # print("Human label:", row["Human_Evaluation"])
                    print("Gold codes:", gold_codes)
                    print("Pred codes:", pred_codes)
                    print("LLM relation:", label)
                    print("-----")

                example_results = example_results.with_columns(
                    pl.Series("LLM_Evaluation_v2", llm_evaluation)
                )
                example_results.write_csv(
                    example_path,
                    separator="\t",
                    include_header=True,
                )
                print("Saved LLM evaluation results to:", example_path)


if __name__ == "__main__":
    app()
