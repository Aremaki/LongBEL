"""
MedProcNER evaluation library evaluation and util functions.
Partially based on the DisTEMIST and MEDDOPLACE evaluation scripts.
@author: salva
"""


# METRICS
def calculate_fscore(gold_standard, predictions, task):
    """
    Calculate micro-averaged precision, recall and f-score from two pandas dataframe
    Depending on the task, do some different pre-processing to the data
    """
    # Cumulative true positives, false positives, false negatives
    total_tp, total_fp, total_fn = 0, 0, 0
    # Dictionary to store files in gold and prediction data.
    gs_files = {}
    pred_files = {}
    for document in gold_standard:
        document_id = document[0][0]
        gs_files[document_id] = document
    for document in predictions:
        document_id = document[0][0]
        pred_files[document_id] = document

    # Dictionary to store scores
    scores = {}

    # Iterate through documents in the Gold Standard
    for document_id in gs_files.keys():
        doc_tp, doc_fp, doc_fn = 0, 0, 0
        gold_doc = gs_files[document_id]
        #  Check if there are predictions for the current document, default to empty document if false
        if document_id not in pred_files.keys():
            predicted_doc = []
        else:
            predicted_doc = pred_files[document_id]
        # Iterate through a copy of our gold mentions
        for gold_annotation in gold_doc[:]:
            # Iterate through predictions looking for a match
            for prediction in predicted_doc[:]:
                # Separate possible composite normalizations
                if task == "norm":
                    separate_prediction = prediction[:-1] + [
                        code.rstrip() for code in sorted(str(prediction[-1]).split("+"))
                    ]  # Need to sort
                    separate_gold_annotation = gold_annotation[:-1] + [
                        code.rstrip() for code in str(gold_annotation[-1]).split("+")
                    ]
                    if set(separate_gold_annotation) == set(separate_prediction):
                        # Add a true positive
                        doc_tp += 1
                        # Remove elements from list to calculate later false positives and false negatives
                        predicted_doc.remove(prediction)
                        gold_doc.remove(gold_annotation)
                        break
                if set(gold_annotation) == set(prediction):
                    # Add a true positive
                    doc_tp += 1
                    # Remove elements from list to calculate later false positives and false negatives
                    predicted_doc.remove(prediction)
                    gold_doc.remove(gold_annotation)
                    break
        # Get the number of false positives and false negatives from the items remaining in our lists
        doc_fp += len(predicted_doc)
        doc_fn += len(gold_doc)
        # Calculate document score
        try:
            precision = doc_tp / (doc_tp + doc_fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = doc_tp / (doc_tp + doc_fn)
        except ZeroDivisionError:
            recall = 0
        if precision == 0 or recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)
        # Add to dictionary
        scores[document_id] = {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f_score": round(f_score, 4),
        }
        # Update totals
        total_tp += doc_tp
        total_fn += doc_fn
        total_fp += doc_fp

    # Now let's calculate the micro-averaged score using the cumulative TP, FP, FN
    try:
        precision = total_tp / (total_tp + total_fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = total_tp / (total_tp + total_fn)
    except ZeroDivisionError:
        recall = 0
    if precision == 0 or recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    scores["total"] = {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f_score": round(f_score, 4),
    }

    return scores


# HELPER
def write_results(task, scores, output_path, verbose):
    """
    Helper function to write the results for each of the tasks
    """
    headers_dict = {
        "ner": "MedProcNER Shared Task: Subtask 1 (Named Entity Recognition) Results",
        "norm": "MedProcNER Shared Task: Subtask 2 (Entity Linking) Results",
    }

    with open(output_path, "w") as f_out:
        # This looks super ugly, but if we keep the indentation it will also appear in the output file
        f_out.write(
            f"""-------------------------------------------------------------------
{headers_dict[task]}
-------------------------------------------------------------------
"""
        )
        if verbose:
            for k in scores.keys():
                if k != "total":
                    f_out.write(
                        """-------------------------------------------------------------------
Results for document: {}
-------------------------------------------------------------------
Precision: {}
Recall: {}
F-score: {}
""".format(k, scores[k]["precision"], scores[k]["recall"], scores[k]["f_score"])
                    )

        f_out.write(
            """-------------------------------------------------------------------
Overall results:
-------------------------------------------------------------------
Micro-average precision: {}
Micro-average recall: {}
Micro-average F-score: {}
""".format(
                scores["total"]["precision"],
                scores["total"]["recall"],
                scores["total"]["f_score"],
            )
        )
    print(f"Written MedProcNER {task} scores to {output_path}")


# def main(argv=None):
#     """
#     Parse options and call the appropriate evaluation scripts
#     """
#     # Parse options
#     parser = ArgumentParser()
#     parser.add_argument(
#         "-r",
#         "--reference",
#         dest="reference",
#         help=".TSV file with Gold Standard or reference annotations",
#         required=True,
#     )
#     parser.add_argument(
#         "-p",
#         "--prediction",
#         dest="prediction",
#         help=".TSV file with your predictions",
#         required=True,
#     )
#     parser.add_argument(
#         "-t",
#         "--task",
#         dest="task",
#         choices=["ner", "norm", "index"],
#         help="Task that you want to evaluate (ner, norm or index)",
#         required=True,
#     )
#     parser.add_argument(
#         "-o",
#         "--output",
#         dest="output",
#         help="Path to save the scoring results",
#         required=True,
#     )
#     parser.add_argument(
#         "-v",
#         "--verbose",
#         dest="verbose",
#         action="store_true",
#         help="Set to True to print the results for each individual file instead of just the final score",
#     )
#     args = parser.parse_args(argv)

#     # Set output file name
#     timedate = datetime.now().strftime("%Y%m%d_%H%M%S")
#     out_file = os.path.join(
#         args.output, "medprocner_results_{}_{}.txt".format(args.task, timedate)
#     )

#     # Read gold_standard and predictions
#     print("Reading reference and prediction .tsv files")
#     df_gs = pd.read_csv(args.reference, sep="\t")
#     df_preds = pd.read_csv(args.prediction, sep="\t")
#     if args.task in ["ner", "norm"]:
#         df_preds = df_preds.drop_duplicates(
#             subset=["filename", "label", "start_span", "end_span"]
#         ).reset_index(drop=True)  # Remove any duplicate predictions

#     if args.task == "ner":
#         calculate_ner(df_gs, df_preds, out_file, args.verbose)
#     elif args.task == "norm":
#         calculate_norm(df_gs, df_preds, out_file, args.verbose)
#     else:
#         print("Please choose a valid task (ner, norm, index)")


def calculate_ner(df_gs, df_preds, output_path, verbose=False):
    print("Computing evaluation scores for Task 1 (ner)")
    # Group annotations by filename
    list_gs_per_doc = (
        df_gs.groupby("filename")
        .apply(
            lambda x: x[
                ["filename", "start_span", "end_span", "text", "label"]
            ].values.tolist()
        )
        .to_list()
    )
    list_preds_per_doc = (
        df_preds.groupby("filename")
        .apply(
            lambda x: x[
                ["filename", "start_span", "end_span", "text", "label"]
            ].values.tolist()
        )
        .to_list()
    )
    scores = calculate_fscore(list_gs_per_doc, list_preds_per_doc, "ner")
    write_results("ner", scores, output_path, verbose)


# Ajouter train mentions...Etc pour avoir les autres recalls


def calculate_norm(df_gs, df_preds, output_path, verbose=False):
    print("Computing evaluation scores for Task 2 (norm)")
    # Group annotations by filename
    list_gs_per_doc = (
        df_gs.groupby("filename")
        .apply(
            lambda x: x[
                ["filename", "start_span", "end_span", "text", "label", "code"]
            ].values.tolist()
        )
        .to_list()
    )
    list_preds_per_doc = (
        df_preds.groupby("filename")
        .apply(
            lambda x: x[
                ["filename", "start_span", "end_span", "text", "label", "code"]
            ].values.tolist()
        )
        .to_list()
    )
    scores = calculate_fscore(list_gs_per_doc, list_preds_per_doc, "norm")
    write_results("norm", scores, output_path, verbose)
