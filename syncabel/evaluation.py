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


def calculate_ner_per_label(df_gs, df_preds):
    print("Computing evaluation scores for Task 1 (ner) per label")

    # All labels present in GS
    labels = sorted(df_gs["label"].unique())

    scores_per_label = {}

    for label in labels:
        print(f" → Computing scores for label: {label}")

        # Filter GS and predictions for the current label
        df_gs_label = df_gs[df_gs["label"] == label]
        df_preds_label = df_preds[df_preds["label"] == label]

        # Group annotations by filename
        list_gs_per_doc = (
            df_gs_label.groupby("filename")
            .apply(
                lambda x: x[
                    ["filename", "start_span", "end_span", "span", "label"]
                ].values.tolist()
            )
            .to_list()
        )

        list_preds_per_doc = (
            df_preds_label.groupby("filename")
            .apply(
                lambda x: x[
                    ["filename", "start_span", "end_span", "span", "label"]
                ].values.tolist()
            )
            .to_list()
        )

        # Call your existing scoring function
        score = calculate_fscore(list_gs_per_doc, list_preds_per_doc, "ner")
        scores_per_label[label] = score

    return scores_per_label

    # write_results("ner", scores, output_path, verbose)


# Ajouter train mentions...Etc pour avoir les autres recalls


def calculate_norm_per_label(df_gs, df_preds):
    print("Computing evaluation scores for Task 2 (norm) per label")

    # All labels present in GS
    labels = sorted(df_gs["label"].unique())

    scores_per_label = {}

    for label in labels:
        print(f" → Computing scores for label: {label}")

        # Filter GS and predictions for this label
        df_gs_label = df_gs[df_gs["label"] == label]
        df_preds_label = df_preds[df_preds["label"] == label]

        # Group annotations by filename
        list_gs_per_doc = (
            df_gs_label.groupby("filename")
            .apply(
                lambda x: x[
                    ["filename", "start_span", "end_span", "span", "label", "code"]
                ].values.tolist()
            )
            .to_list()
        )

        list_preds_per_doc = (
            df_preds_label.groupby("filename")
            .apply(
                lambda x: x[
                    [
                        "filename",
                        "start_span",
                        "end_span",
                        "span",
                        "label",
                        "Predicted_CUI",
                    ]
                ].values.tolist()
            )
            .to_list()
        )

        # Compute score using your existing function
        score = calculate_fscore(list_gs_per_doc, list_preds_per_doc, "norm")

        # Store score
        scores_per_label[label] = score

    return scores_per_label
