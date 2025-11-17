# fast_medproc_eval.py
# Ultra-fast MedProcNER evaluation using Polars
# Equivalent results to the original pandas + Python-loop implementation
# but MUCH faster and without per-document results.

import polars as pl

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def normalize_codes(col: pl.Expr) -> pl.Expr:
    """
    Normalize composite CUIs (e.g. "A+B") by:
    - splitting
    - trimming
    - sorting
    - joining deterministically
    """
    return (
        col.str.split("+")
        .list.eval(pl.element().str.strip_chars())
        .list.sort()
        .list.join("+")
    )


# ---------------------------------------------------------
# Core TP matching
# ---------------------------------------------------------


def match_true_positives(
    df_gs: pl.DataFrame, df_pred: pl.DataFrame, task: str
) -> pl.DataFrame:
    """
    Return a DF of true positives via fast Polars join.
    """

    if task == "ner":
        # Matching keys for NER
        join_cols = ["filename", "start_span", "end_span", "label"]

        gs_df = df_gs.with_columns([pl.col("span").alias("span_norm")])

        pred_df = df_pred.with_columns([pl.col("text").alias("span_norm")])

    elif task == "norm":
        # Matching keys for NORM
        join_cols = ["filename", "start_span", "end_span", "label"]

        gs_df = df_gs.with_columns([
            pl.col("span").alias("span_norm"),
            normalize_codes(pl.col("code")).alias("code_norm"),
        ])

        pred_df = df_pred.with_columns([
            pl.col("span").alias("span_norm"),
            normalize_codes(pl.col("Predicted_CUI")).alias("code_norm"),
        ])

        join_cols.append("code_norm")

    else:
        raise ValueError("task must be 'ner' or 'norm'")

    # Inner join = true positives
    tp = gs_df.join(pred_df, on=join_cols + ["span_norm"], how="inner")

    return tp


# ---------------------------------------------------------
# Compute TP / FP / FN (no per-document breakdown)
# ---------------------------------------------------------


def compute_micro_metrics(
    df_gs: pl.DataFrame, df_pred: pl.DataFrame, task: str
) -> dict:
    """
    Compute micro-averaged TP/FP/FN + precision, recall, f1.
    """

    # True Positives
    tp = match_true_positives(df_gs, df_pred, task)
    tp_count = tp.height

    gold_count = df_gs.height
    pred_count = df_pred.height

    fp_count = pred_count - tp_count
    fn_count = gold_count - tp_count

    # Precision, Recall, F1
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
    f1 = (
        0.0
        if (precision == 0 or recall == 0)
        else (2 * precision * recall / (precision + recall))
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f_score": round(f1, 4),
    }


# ---------------------------------------------------------
# Per-label + overall evaluation
# ---------------------------------------------------------


def evaluate(df_gs: pl.DataFrame, df_pred: pl.DataFrame, task: str) -> dict:
    """
    Compute:
        - per-label micro-averaged scores
        - overall micro-averaged scores
    """

    labels = sorted(df_gs["label"].unique())
    scores = {}

    for label in labels:
        gs_lb = df_gs.filter(pl.col("label") == label)
        pred_lb = df_pred.filter(pl.col("label") == label)

        if pred_lb.height == 0:
            scores[label] = {"total": {"precision": 0.0, "recall": 0.0, "f_score": 0.0}}
            continue

        scores[label] = {"total": compute_micro_metrics(gs_lb, pred_lb, task)}

    # Overall micro score
    scores["overall"] = {"total": compute_micro_metrics(df_gs, df_pred, task)}

    return scores
