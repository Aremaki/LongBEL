from pathlib import Path

import polars as pl
from owlready2 import default_world, get_ontology
from tqdm import tqdm

from longbel.error_analysis import load_predictions

# Load UMLS into a local SQLite database
default_world.set_backend(filename="pym_2023AA.sqlite3")
PYM = get_ontology("http://PYM/").load()


model_name = "Meta-Llama-3-8B-Instruct"
aug_data = "full_upsampled"
dataset = "SPACCC"
data_split = "test"
example_path = Path(
    f"results/inference_outputs/{dataset}/{aug_data}_tfidf/{model_name}_best/pred_{data_split}_constraint_5_beams.tsv"
)
example_results = load_predictions(example_path)
# relator = LLMRelator()

# loop over the rows
termino_evaluation = []
SNOMEDCT_US = PYM["SNOMEDCT_US"]

for row in tqdm(example_results.iter_rows(named=True), total=example_results.height):
    eval_codes = []
    gold_codes = row["code"].split("+")
    pred_codes = row["Predicted_code"].split("+")
    if len(gold_codes) < len(pred_codes):
        gold_codes += [gold_codes[-1]] * (len(pred_codes) - len(gold_codes))
    elif len(pred_codes) < len(gold_codes):
        pred_codes += [pred_codes[-1]] * (len(gold_codes) - len(pred_codes))
    for gold_code, pred_code in zip(gold_codes, pred_codes):
        if gold_code == pred_code:
            eval_codes.append("STRICT")
        else:
            gold_code = int(gold_code)
            pred_code = int(pred_code)
            if SNOMEDCT_US.has_concept(gold_code) and SNOMEDCT_US.has_concept(
                pred_code
            ):
                gold_concept = SNOMEDCT_US[gold_code]
                pred_concept = SNOMEDCT_US[pred_code]
                if hasattr(gold_concept, "children") and hasattr(
                    pred_concept, "children"
                ):
                    if issubclass(gold_concept, pred_concept):
                        eval_codes.append("BROAD")
                    elif issubclass(pred_concept, gold_concept):
                        eval_codes.append("NARROW")
                    else:
                        eval_codes.append("NO_RELATION")
                else:
                    eval_codes.append("NO_RELATION")
            else:
                print("Code not found in SNOMEDCT_US:", gold_code, pred_code)
                eval_codes.append("NO_RELATION")

    # aggregate eval_codes
    if all(code == "STRICT" for code in eval_codes):
        termino_evaluation.append("STRICT")
    elif "NO_RELATION" in eval_codes:
        termino_evaluation.append("NO_RELATION")
    elif "NARROW" in eval_codes:
        termino_evaluation.append("NARROW")
    elif "BROAD" in eval_codes:
        termino_evaluation.append("BROAD")


example_results = example_results.with_columns(
    pl.Series("Termino_Evaluation", termino_evaluation)
)
example_results.write_csv(
    example_path,
    separator="\t",
    include_header=True,
)
