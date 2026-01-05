from pathlib import Path
from typing import Any, Optional

import numpy as np
import polars as pl
from tqdm import tqdm


def _run_vectorized_bootstrap(
    doc_agg: pl.DataFrame,
    unique_docs: np.ndarray,
    unique_labels: np.ndarray,
    n_bootstrap: int,
    ci: float,
    rng: np.random.Generator,
    num_col: str,
    denom_col: str,
) -> dict[str, tuple[float, float]]:
    """
    Helper to run vectorized bootstrap for a specific metric (num/denom).
    """
    n_docs = len(unique_docs)

    # Generate Weights (B x D)
    # We use multinomial to simulate sampling with replacement
    weights = rng.multinomial(n_docs, [1.0 / n_docs] * n_docs, size=n_bootstrap)

    results = {}
    alpha = (1 - ci) / 2

    # --- Overall ---
    doc_overall = doc_agg.group_by("filename").agg([
        pl.col(num_col).sum().alias("num"),
        pl.col(denom_col).sum().alias("denom"),
    ])
    docs_df = pl.DataFrame({"filename": unique_docs})
    doc_overall = docs_df.join(doc_overall, on="filename", how="left").fill_null(0)

    num_vec = doc_overall["num"].to_numpy()
    denom_vec = doc_overall["denom"].to_numpy()

    boot_num = weights @ num_vec
    boot_denom = weights @ denom_vec

    with np.errstate(divide="ignore", invalid="ignore"):
        boot_metric = boot_num / boot_denom
        boot_metric = np.nan_to_num(boot_metric, nan=0.0)

    results["overall"] = (
        float(np.quantile(boot_metric, alpha)),
        float(np.quantile(boot_metric, 1 - alpha)),
    )

    # --- Per Label (Chunked) ---
    # Map labels to indices for filtering
    # label_map = {l: i for i, l in enumerate(unique_labels)}

    # Add indices to doc_agg
    label_map_df = pl.DataFrame({
        "label": unique_labels,
        "label_idx": np.arange(len(unique_labels), dtype=np.int32),
    })

    # Filter doc_agg to only labels we care about (in unique_labels)
    doc_agg_idx = doc_agg.join(label_map_df, on="label", how="inner")

    # Add doc indices
    doc_map_df = pl.DataFrame({
        "filename": unique_docs,
        "doc_idx": np.arange(len(unique_docs), dtype=np.int32),
    })
    doc_agg_idx = doc_agg_idx.join(doc_map_df, on="filename", how="inner")

    # Extract arrays
    rows = doc_agg_idx["doc_idx"].to_numpy()
    cols = doc_agg_idx["label_idx"].to_numpy()
    vals_num = doc_agg_idx[num_col].to_numpy()
    vals_denom = doc_agg_idx[denom_col].to_numpy()

    chunk_size = 1000
    n_labels = len(unique_labels)

    for i in range(0, n_labels, chunk_size):
        end = min(i + chunk_size, n_labels)
        size = end - i

        # Identify entries in this chunk
        mask = (cols >= i) & (cols < end)
        if not np.any(mask):
            # No data for any label in this chunk
            for idx in range(i, end):
                results[unique_labels[idx]] = (0.0, 0.0)
            continue

        rows_chunk = rows[mask]
        cols_chunk = cols[mask] - i  # relative index
        num_chunk = vals_num[mask]
        denom_chunk = vals_denom[mask]

        # Build dense matrices for chunk
        C_num = np.zeros((n_docs, size), dtype=np.float64)
        C_denom = np.zeros((n_docs, size), dtype=np.float64)

        # Use np.add.at for unbuffered summation
        np.add.at(C_num, (rows_chunk, cols_chunk), num_chunk)
        np.add.at(C_denom, (rows_chunk, cols_chunk), denom_chunk)

        # Matrix Multiply
        B_num = weights @ C_num
        B_denom = weights @ C_denom

        # Compute Metrics
        with np.errstate(divide="ignore", invalid="ignore"):
            B_metric = B_num / B_denom
            B_metric = np.nan_to_num(B_metric, nan=0.0)

        # Quantiles
        q_low = np.quantile(B_metric, alpha, axis=0)
        q_high = np.quantile(B_metric, 1 - alpha, axis=0)

        for idx in range(size):
            results[unique_labels[i + idx]] = (float(q_low[idx]), float(q_high[idx]))

    return results


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


def load_predictions(
    prediction_path: Path,
) -> pl.DataFrame:
    """
    Add semantic relation column to the dataframe using KeyCare's RelationExtractor.
    """
    df = pl.read_csv(
        prediction_path,
        separator="\t",
        has_header=True,
        schema_overrides={
            "code": str,  # force as string
            "Predicted_code": str,
            "mention_id": str,
            "filename": str,  # force as string
        },  # type: ignore
    ).unique(
        subset=[
            "filename",
            "label",
            "start_span",
            "end_span",
        ]
    )
    # Filter NO_CODE annotations
    df = df.filter(~(pl.col("code").str.contains("NO_CODE")))
    df = df.with_columns(
        normalize_codes(pl.col("code")), normalize_codes(pl.col("Predicted_code"))
    )
    df = df.with_columns(
        pl.when(pl.col("Predicted_code").is_null())
        .then(pl.lit(""))  # replace Prediction with empty string
        .otherwise(pl.col("Prediction"))
        .alias("Prediction")
    )
    return df  # already processed


def compute_metrics_simple(
    df: pl.DataFrame,
    df_full: pl.DataFrame,
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    threshold: Optional[float] = None,
    score_column: str = "Prediction_score",
    include_ratios: bool = True,
    index_key: str = "Overall",
) -> dict[str, dict[str, Any]]:  # type: ignore
    """
    Core metric computation used by downstream aggregations.

    - Computes recall (and precision/F1 when `threshold` is provided) per label and overall.
    - Supports optional document-level bootstrap (vectorized).
    - Adds ratio coverage information versus `df_full` (or within-partition shares when `index_key == "All"`).
    """

    rng = np.random.default_rng(seed)

    # Encode dataframe as numpy arrays; guard empty frames to keep shapes consistent
    filenames = df["filename"].to_numpy() if df.height else np.array([], dtype=object)

    # Use labels from the full set to keep deterministic ordering and include missing labels
    unique_labels = np.array(sorted(df_full["label"].unique()))
    unique_docs = np.unique(filenames)

    # Precompute counts for ratios
    total_full_overall = df_full.height
    total_pred_overall = df.height

    full_counts = (
        df_full.group_by("label")
        .count()
        .select(["label", "count"])
        .to_dict(as_series=False)
    )
    full_counts_map = dict(zip(full_counts["label"], full_counts["count"]))

    pred_counts = (
        df.group_by("label").count().select(["label", "count"]).to_dict(as_series=False)
        if df.height
        else {"label": [], "count": []}
    )
    pred_counts_map = dict(zip(pred_counts["label"], pred_counts["count"]))

    def _format_ratio(label: str, pred_count: int) -> str:
        if not include_ratios:
            return "N/A"
        if index_key == "All":
            denom = total_pred_overall
        else:
            denom = full_counts_map.get(label, 0)
        if denom == 0:
            return "0% (0/0)"
        return f"{round(pred_count / denom * 100, 1)}% ({pred_count}/{denom})"

    # Build ratios once
    ratios = {
        lbl: _format_ratio(lbl, pred_counts_map.get(lbl, 0))
        for lbl in (*unique_labels, "overall")
    }
    ratios["overall"] = (
        f"{round(total_pred_overall / total_full_overall * 100, 1)}% ({total_pred_overall}/{total_full_overall})"
        if include_ratios and total_full_overall
        else ("0% (0/0)" if include_ratios else "N/A")
    )

    if threshold is None:
        # -------- Strict Recall --------

        # 1. Point Estimate (Polars)
        label_stats = df.group_by("label").agg([
            pl.count().alias("total"),
            (pl.col("code") == pl.col("Predicted_code")).sum().alias("correct"),
        ])

        all_labels_df = pl.DataFrame({"label": unique_labels})
        stats = all_labels_df.join(label_stats, on="label", how="left").fill_null(0)

        stats = stats.with_columns(
            (pl.col("correct") / pl.col("total")).fill_nan(0.0).alias("recall")
        )

        point = dict(zip(stats["label"], stats["recall"]))

        overall_correct = df.filter(pl.col("code") == pl.col("Predicted_code")).height
        overall_total = df.height
        point["overall"] = overall_correct / overall_total if overall_total > 0 else 0.0

        # 2. Bootstrap (Vectorized)
        ci_data = {}
        if bootstrap and len(unique_docs) > 0:
            doc_agg = df.group_by(["filename", "label"]).agg([
                pl.len().alias("denom"),
                (pl.col("code") == pl.col("Predicted_code")).sum().alias("num"),
            ])
            ci_data = _run_vectorized_bootstrap(
                doc_agg,
                unique_docs,
                unique_labels,
                n_bootstrap,
                ci,
                rng,
                "num",
                "denom",
            )

        return {
            k: {
                "recall": round(v * 100, 1),
                "ci_low": round(ci_data[k][0] * 100, 1) if k in ci_data else None,
                "ci_high": round(ci_data[k][1] * 100, 1) if k in ci_data else None,
                "ratio": ratios.get(k, "N/A"),
            }
            for k, v in point.items()
        }

    # -------- Thresholded precision / recall / F1 --------
    if score_column not in df.columns:
        raise ValueError(
            "threshold was provided but the requested score column is missing;"
            f" got '{score_column}' but expected a column named exactly that"
        )

    # 1. Point Estimate (Polars)
    agg_exprs = [
        pl.count().alias("total"),
        (pl.col(score_column) >= threshold).sum().alias("selected"),
        (
            (pl.col(score_column) >= threshold)
            & (pl.col("code") == pl.col("Predicted_code"))
        )
        .sum()
        .alias("tp"),
    ]

    label_stats = df.group_by("label").agg(agg_exprs)

    all_labels_df = pl.DataFrame({"label": unique_labels})
    stats = all_labels_df.join(label_stats, on="label", how="left").fill_null(0)

    stats = stats.with_columns([
        (pl.col("tp") / pl.col("total")).fill_nan(0.0).alias("recall"),
        (pl.col("tp") / pl.col("selected")).fill_nan(0.0).alias("precision"),
    ])

    stats = stats.with_columns(
        (
            2
            * pl.col("precision")
            * pl.col("recall")
            / (pl.col("precision") + pl.col("recall"))
        )
        .fill_nan(0.0)
        .alias("f1")
    )

    point = {
        lbl: {
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
        }
        for lbl, row in zip(stats["label"], stats.to_dicts())
    }

    # Overall
    total_all = df.height
    selected_all = df.filter(pl.col(score_column) >= threshold).height
    tp_all = df.filter(
        (pl.col(score_column) >= threshold)
        & (pl.col("code") == pl.col("Predicted_code"))
    ).height

    recall_all = tp_all / total_all if total_all > 0 else 0.0
    precision_all = tp_all / selected_all if selected_all > 0 else 0.0
    f1_all = (
        2 * precision_all * recall_all / (precision_all + recall_all)
        if (precision_all + recall_all) > 0
        else 0.0
    )

    point["overall"] = {
        "precision": precision_all,
        "recall": recall_all,
        "f1": f1_all,
    }

    # 2. Bootstrap (Vectorized)
    ci_recall = {}
    ci_precision = {}
    ci_f1 = {}

    if bootstrap and len(unique_docs) > 0:
        doc_agg = df.group_by(["filename", "label"]).agg([
            pl.count().alias("total"),
            (pl.col(score_column) >= threshold).sum().alias("selected"),
            (
                (pl.col(score_column) >= threshold)
                & (pl.col("code") == pl.col("Predicted_code"))
            )
            .sum()
            .alias("tp"),
        ])

        doc_agg = doc_agg.with_columns([
            (pl.col("tp") * 2).alias("f1_num"),
            (pl.col("total") + pl.col("selected")).alias("f1_denom"),
        ])

        ci_recall = _run_vectorized_bootstrap(
            doc_agg, unique_docs, unique_labels, n_bootstrap, ci, rng, "tp", "total"
        )
        ci_precision = _run_vectorized_bootstrap(
            doc_agg,
            unique_docs,
            unique_labels,
            n_bootstrap,
            ci,
            rng,
            "tp",
            "selected",
        )
        ci_f1 = _run_vectorized_bootstrap(
            doc_agg,
            unique_docs,
            unique_labels,
            n_bootstrap,
            ci,
            rng,
            "f1_num",
            "f1_denom",
        )

    return {
        k: {
            "precision": round(v["precision"] * 100, 1),
            "precision_ci_low": round(ci_precision[k][0] * 100, 1)
            if k in ci_precision
            else None,
            "precision_ci_high": round(ci_precision[k][1] * 100, 1)
            if k in ci_precision
            else None,
            "recall": round(v["recall"] * 100, 1),
            "recall_ci_low": round(ci_recall[k][0] * 100, 1)
            if k in ci_recall
            else None,
            "recall_ci_high": round(ci_recall[k][1] * 100, 1)
            if k in ci_recall
            else None,
            "f1": round(v["f1"] * 100, 1),
            "f1_ci_low": round(ci_f1[k][0] * 100, 1) if k in ci_f1 else None,
            "f1_ci_high": round(ci_f1[k][1] * 100, 1) if k in ci_f1 else None,
            "ratio": ratios.get(k, "N/A"),
        }
        for k, v in point.items()
    }


# -------------------------------
# Core: compute_partition_metrics
# -------------------------------
def compute_partition_metrics(
    df: pl.DataFrame,
    df_full: pl.DataFrame,
    full_counts_map: dict[str, int],
    unique_labels: list[str],
    total_full_count: int,
    index: str = "overall",
    compute_all: bool = True,
    include_ratios: bool = True,
    threshold: Optional[float] = None,
    score_column: str = "Prediction_score",
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Compute strict (and optional semantic) recalls plus ratios for each label.
    Optimized to use single-pass Polars aggregation when bootstrap=False.
    """

    index_key = index.capitalize()

    # ---------------------------------------------------------
    # FAST PATH: No bootstrap -> Single pass aggregation
    # ---------------------------------------------------------
    if not bootstrap:
        # 1. Define Aggregations
        aggs = [
            pl.len().alias("count"),
            (pl.col("code") == pl.col("Predicted_code")).sum().alias("strict_correct"),
        ]

        if compute_all:
            aggs.extend([
                (pl.col("LLM_Evaluation") == "EXACT").sum().alias("exact_correct"),
                (pl.col("LLM_Evaluation") == "PARTIAL").sum().alias("partial_correct"),
                (pl.col("LLM_Evaluation") == "NARROW").sum().alias("narrow_correct"),
                (pl.col("LLM_Evaluation") == "BROAD").sum().alias("broad_correct"),
            ])

        if threshold is not None:
            # Thresholded metrics
            is_selected = pl.col(score_column) >= threshold
            is_correct = pl.col("code") == pl.col("Predicted_code")
            aggs.extend([
                is_selected.sum().alias("thresh_selected"),
                (is_selected & is_correct).sum().alias("thresh_tp"),
            ])

        # 2. Perform GroupBy Aggregation
        # We use a left join on unique_labels to ensure all labels are present
        grouped = df.group_by("label").agg(aggs)

        # Create base dataframe with all labels
        # Note: unique_labels is a list of strings
        base_df = pl.DataFrame({"label": unique_labels}, schema={"label": pl.String})

        # Join and fill nulls with 0
        stats = base_df.join(grouped, on="label", how="left").fill_null(0)

        # 3. Calculate Metrics (Vectorized)
        # Strict Recall
        stats = stats.with_columns(
            (pl.col("strict_correct") / pl.col("count"))
            .fill_nan(0.0)
            .alias("recall_strict")
        )

        # Semantic Recalls
        if compute_all:
            stats = stats.with_columns([
                (pl.col("exact_correct") / pl.col("count"))
                .fill_nan(0.0)
                .alias("recall_exact"),
                (pl.col("partial_correct") / pl.col("count"))
                .fill_nan(0.0)
                .alias("recall_partial"),
                (pl.col("narrow_correct") / pl.col("count"))
                .fill_nan(0.0)
                .alias("recall_narrow"),
                (pl.col("broad_correct") / pl.col("count"))
                .fill_nan(0.0)
                .alias("recall_broad"),
            ])

        # Thresholded Metrics
        if threshold is not None:
            stats = stats.with_columns([
                (pl.col("thresh_tp") / pl.col("count")).fill_nan(0.0).alias("recall"),
                (pl.col("thresh_tp") / pl.col("thresh_selected"))
                .fill_nan(0.0)
                .alias("precision"),
            ])
            stats = stats.with_columns(
                (
                    2
                    * pl.col("precision")
                    * pl.col("recall")
                    / (pl.col("precision") + pl.col("recall"))
                )
                .fill_nan(0.0)
                .alias("f1")
            )

        # 4. Construct Result Dictionary
        results: dict[str, Any] = {
            "recall_strict": {},
            "ratios": {},
        }
        if compute_all:
            results.update({
                "recall_exact": {},
                "recall_partial": {},
                "recall_narrow": {},
                "recall_broad": {},
            })
        if threshold is not None:
            results.update({"recall": {}, "precision": {}, "f1": {}})

        # Helper for ratios
        total_pred_overall = df.height

        def _get_ratio_str(lbl, pred_cnt):
            if not include_ratios:
                return "N/A"
            if index_key == "All":
                denom = total_pred_overall
            else:
                denom = full_counts_map.get(lbl, 0)
            if denom == 0:
                return "0% (0/0)"
            return f"{round(pred_cnt / denom * 100, 1)}% ({pred_cnt}/{denom})"

        # Iterate rows to populate dict (fast enough for ~thousands of labels)
        # Using iter_rows(named=True)
        for row in stats.iter_rows(named=True):
            lbl = row["label"]
            cnt = row["count"]

            results["recall_strict"][lbl] = {
                index_key: round(row["recall_strict"] * 100, 1)
            }
            results["ratios"][lbl] = {index_key: _get_ratio_str(lbl, cnt)}

            if compute_all:
                results["recall_exact"][lbl] = {
                    index_key: round(row["recall_exact"] * 100, 1)
                }
                results["recall_partial"][lbl] = {
                    index_key: round(row["recall_partial"] * 100, 1)
                }
                results["recall_narrow"][lbl] = {
                    index_key: round(row["recall_narrow"] * 100, 1)
                }
                results["recall_broad"][lbl] = {
                    index_key: round(row["recall_broad"] * 100, 1)
                }

            if threshold is not None:
                results["recall"][lbl] = {index_key: round(row["recall"] * 100, 1)}
                results["precision"][lbl] = {
                    index_key: round(row["precision"] * 100, 1)
                }
                results["f1"][lbl] = {index_key: round(row["f1"] * 100, 1)}

        # 5. Overall Metrics
        # We need to calculate overall stats separately
        # (Polars sum of columns)

        # If df is empty, handle gracefully
        if df.height == 0:
            # ... (defaults are 0)
            pass
        else:
            # We can sum the counts from the grouped dataframe or the original df
            # Using original df is safer/easier for overall
            strict_correct_all = df.filter(
                pl.col("code") == pl.col("Predicted_code")
            ).height

            results["recall_strict"]["overall"] = {
                index_key: round(strict_correct_all / total_pred_overall * 100, 1)
                if total_pred_overall
                else 0.0
            }

            # Ratio overall
            if include_ratios:
                if total_full_count > 0:
                    ratio_str = f"{round(total_pred_overall / total_full_count * 100, 1)}% ({total_pred_overall}/{total_full_count})"
                else:
                    ratio_str = "0% (0/0)"
                results["ratios"]["overall"] = {index_key: ratio_str}
            else:
                results["ratios"]["overall"] = {index_key: "N/A"}

            if compute_all:
                exact_all = df.filter(pl.col("LLM_Evaluation") == "EXACT").height
                partial_all = df.filter(pl.col("LLM_Evaluation") == "PARTIAL").height
                narrow_all = df.filter(pl.col("LLM_Evaluation") == "NARROW").height
                broad_all = df.filter(pl.col("LLM_Evaluation") == "BROAD").height

                denom = total_pred_overall if total_pred_overall else 1
                results["recall_exact"]["overall"] = {
                    index_key: round(exact_all / denom * 100, 1)
                }
                results["recall_partial"]["overall"] = {
                    index_key: round(partial_all / denom * 100, 1)
                }
                results["recall_narrow"]["overall"] = {
                    index_key: round(narrow_all / denom * 100, 1)
                }
                results["recall_broad"]["overall"] = {
                    index_key: round(broad_all / denom * 100, 1)
                }

            if threshold is not None:
                # Overall thresholded
                sel_all = df.filter(pl.col(score_column) >= threshold).height
                tp_all = df.filter(
                    (pl.col(score_column) >= threshold)
                    & (pl.col("code") == pl.col("Predicted_code"))
                ).height

                rec_all = tp_all / total_pred_overall if total_pred_overall else 0.0
                prec_all = tp_all / sel_all if sel_all else 0.0
                f1_all = (
                    2 * prec_all * rec_all / (prec_all + rec_all)
                    if (prec_all + rec_all)
                    else 0.0
                )

                results["recall"]["overall"] = {index_key: round(rec_all * 100, 1)}
                results["precision"]["overall"] = {index_key: round(prec_all * 100, 1)}
                results["f1"]["overall"] = {index_key: round(f1_all * 100, 1)}

        return results

    # ---------------------------------------------------------
    # SLOW PATH: Bootstrap enabled (fallback to original logic)
    # ---------------------------------------------------------

    # Base strict metrics (and ratios) computed without threshold
    base_metrics = compute_metrics_simple(
        df,
        df_full,
        bootstrap=bootstrap,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        threshold=None,  # never use threshold for strict/semantic recalls
        score_column=score_column,
        include_ratios=include_ratios,
        index_key=index_key,
    )

    # Thresholded metrics (recall/precision/F1) computed only if threshold is provided
    threshold_metrics = (
        compute_metrics_simple(
            df,
            df_full,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            threshold=threshold,
            score_column=score_column,
            include_ratios=include_ratios,
            index_key=index_key,
        )
        if threshold is not None
        else None
    )

    results: dict[str, Any] = {
        "recall_strict": {},
        "ratios": {},
    }
    if bootstrap:
        results.update({
            "recall_strict_ci_low": {},
            "recall_strict_ci_high": {},
        })

    if compute_all:
        results.update({
            "recall_exact": {},
            "recall_partial": {},
            "recall_narrow": {},
            "recall_broad": {},
        })
    if threshold is not None:
        results.update({"recall": {}, "precision": {}, "f1": {}})
        if bootstrap:
            results.update({
                "recall_ci_low": {},
                "recall_ci_high": {},
                "precision_ci_low": {},
                "precision_ci_high": {},
                "f1_ci_low": {},
                "f1_ci_high": {},
            })

    # overall entry
    overall_metrics = base_metrics.get("overall", {})
    results["recall_strict"]["overall"] = {
        index_key: overall_metrics.get("recall", 0.0)
    }
    results["ratios"]["overall"] = {index_key: overall_metrics.get("ratio", "N/A")}
    if bootstrap:
        results["recall_strict_ci_low"]["overall"] = {
            index_key: overall_metrics.get("ci_low")
        }
        results["recall_strict_ci_high"]["overall"] = {
            index_key: overall_metrics.get("ci_high")
        }

    if threshold_metrics is not None:
        tm_overall = threshold_metrics.get("overall", {})
        results["recall"]["overall"] = {index_key: tm_overall.get("recall", 0.0)}
        results["precision"]["overall"] = {index_key: tm_overall.get("precision", 0.0)}
        results["f1"]["overall"] = {index_key: tm_overall.get("f1", 0.0)}
        if bootstrap:
            results["recall_ci_low"]["overall"] = {
                index_key: tm_overall.get("recall_ci_low")
            }
            results["recall_ci_high"]["overall"] = {
                index_key: tm_overall.get("recall_ci_high")
            }
            results["precision_ci_low"]["overall"] = {
                index_key: tm_overall.get("precision_ci_low")
            }
            results["precision_ci_high"]["overall"] = {
                index_key: tm_overall.get("precision_ci_high")
            }
            results["f1_ci_low"]["overall"] = {index_key: tm_overall.get("f1_ci_low")}
            results["f1_ci_high"]["overall"] = {index_key: tm_overall.get("f1_ci_high")}

    if compute_all:
        total_pred_overall = df.height
        true_exact_overall = df.filter(pl.col("LLM_Evaluation") == "EXACT").height
        true_partial_overall = df.filter(pl.col("LLM_Evaluation") == "PARTIAL").height
        true_narrow_overall = df.filter(pl.col("LLM_Evaluation") == "NARROW").height
        true_broad_overall = df.filter(pl.col("LLM_Evaluation") == "BROAD").height

        denom_overall = total_pred_overall if total_pred_overall else 1
        results["recall_exact"]["overall"] = {
            index_key: round(true_exact_overall / denom_overall * 100, 1)
        }
        results["recall_broad"]["overall"] = {
            index_key: round(true_broad_overall / denom_overall * 100, 1)
        }
        results["recall_partial"]["overall"] = {
            index_key: round(true_partial_overall / denom_overall * 100, 1)
        }
        results["recall_narrow"]["overall"] = {
            index_key: round(true_narrow_overall / denom_overall * 100, 1)
        }

    for label in unique_labels:
        metrics = base_metrics.get(label, {})
        results["recall_strict"][label] = {index_key: metrics.get("recall", 0.0)}
        results["ratios"][label] = {index_key: metrics.get("ratio", "N/A")}
        if bootstrap:
            results["recall_strict_ci_low"][label] = {index_key: metrics.get("ci_low")}
            results["recall_strict_ci_high"][label] = {
                index_key: metrics.get("ci_high")
            }

        if threshold_metrics is not None:
            tm_label = threshold_metrics.get(label, {})
            results["recall"][label] = {index_key: tm_label.get("recall", 0.0)}
            results["precision"][label] = {index_key: tm_label.get("precision", 0.0)}
            results["f1"][label] = {index_key: tm_label.get("f1", 0.0)}
            if bootstrap:
                results["recall_ci_low"][label] = {
                    index_key: tm_label.get("recall_ci_low")
                }
                results["recall_ci_high"][label] = {
                    index_key: tm_label.get("recall_ci_high")
                }
                results["precision_ci_low"][label] = {
                    index_key: tm_label.get("precision_ci_low")
                }
                results["precision_ci_high"][label] = {
                    index_key: tm_label.get("precision_ci_high")
                }
                results["f1_ci_low"][label] = {index_key: tm_label.get("f1_ci_low")}
                results["f1_ci_high"][label] = {index_key: tm_label.get("f1_ci_high")}

        if not compute_all:
            continue

        df_label_pred = df.filter(pl.col("label") == label)
        total_pred = df_label_pred.height
        denom = total_pred if total_pred else 1

        true_exact = df_label_pred.filter(pl.col("LLM_Evaluation") == "EXACT").height
        true_partial = df_label_pred.filter(
            pl.col("LLM_Evaluation") == "PARTIAL"
        ).height
        true_narrow = df_label_pred.filter(pl.col("LLM_Evaluation") == "NARROW").height
        true_broad = df_label_pred.filter(pl.col("LLM_Evaluation") == "BROAD").height

        results["recall_exact"][label] = {index_key: round(true_exact / denom * 100, 1)}
        results["recall_partial"][label] = {
            index_key: round(true_partial / denom * 100, 1)
        }
        results["recall_narrow"][label] = {
            index_key: round(true_narrow / denom * 100, 1)
        }
        results["recall_broad"][label] = {index_key: round(true_broad / denom * 100, 1)}

    return results


# ---------------------------
# Driver: compute_metrics
# ---------------------------
def compute_metrics(
    pred_df: pl.DataFrame,
    train_mentions: set[str],
    train_cuis: set[str],
    top_100_cuis: set[str],
    top_100_mentions: set[str],
    unique_pairs: set[tuple[str, str]],
    compute_all_recalls: bool = True,
    include_ratios: bool = True,
    threshold: Optional[float] = None,
    score_column: str = "Prediction_score",
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    Compute full suite of metrics across many partitions.
    - pred_df: predicted dataframe (contains columns: label, code, Predicted_code, semantic_rel_pred, span, annotation, filename, start_span, etc.)
    - train_mentions / train_cuis / top_100_cuis / top_100_mentions / unique_pairs: sets used to build partitions
    - compute_all_recalls: if False, only strict recall + ratios are computed (faster)
    - include_ratios / threshold / score_column / bootstrap / n_bootstrap / ci / seed: forwarded to compute_partition_metrics
    """

    # ensure codes normalized if you use normalize_codes; if you have it available do it here
    # pred_df = pred_df.with_columns(normalize_codes(pl.col("code")), normalize_codes(pl.col("Predicted_code")))

    # Pre-calculate full counts and unique labels once
    # This avoids re-calculating them for every partition
    full_counts = (
        pred_df.group_by(
            "label"
        )  # Note: Using pred_df as proxy for df_full if df_full is not available in this scope?
        # Wait, compute_metrics does NOT take df_full as argument!
        # It takes pred_df.
        # In the original code, compute_metrics_simple took df_full.
        # But compute_metrics calls compute_partition_metrics passing pred_df as df_full?
        # Let's check the original call:
        # _call_partition(df_partition, pred_df, ...)
        # Yes, pred_df is treated as df_full in the context of compute_metrics.
        .count()
        .select(["label", "count"])
        .to_dict(as_series=False)
    )
    full_counts_map = dict(zip(full_counts["label"], full_counts["count"]))
    unique_labels = sorted(pred_df["label"].unique().to_list())
    total_full_count = pred_df.height

    # We'll call compute_partition_metrics for each partition and then aggregate
    # Helper to call and return a consistent tuple for easier aggregation

    def _call_partition(df_partition: pl.DataFrame, index_name: str):
        return compute_partition_metrics(
            df_partition,
            pred_df,  # df_full
            full_counts_map,
            unique_labels,
            total_full_count,
            index=index_name,
            compute_all=compute_all_recalls,
            include_ratios=include_ratios,
            threshold=threshold,
            score_column=score_column,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )

    has_threshold = threshold is not None

    # Prepare partitions
    # We define them as (name, dataframe) tuples

    # Pre-calculate some filters to avoid repetition
    unique_df = pl.DataFrame(list(unique_pairs), schema=["span", "code"], orient="row")

    repeated = (
        pred_df.group_by(["filename", "code"])
        .agg(pl.count().alias("count"))
        .filter(pl.col("count") >= 2)
        .select(["filename", "code"])
    )

    partition_specs = [
        ("All", pred_df),
        ("seen_cuis", pred_df.filter(pl.col("code").is_in(list(train_cuis)))),
        ("unseen_cuis", pred_df.filter(~pl.col("code").is_in(list(train_cuis)))),
        ("seen_mentions", pred_df.filter(pl.col("span").is_in(list(train_mentions)))),
        (
            "unseen_mentions",
            pred_df.filter(~pl.col("span").is_in(list(train_mentions))),
        ),
        ("in_top_100_cuis", pred_df.filter(pl.col("code").is_in(list(top_100_cuis)))),
        (
            "not_in_top_100_cuis",
            pred_df.filter(~pl.col("code").is_in(list(top_100_cuis))),
        ),
        (
            "in_top_100_mentions",
            pred_df.filter(pl.col("span").is_in(list(top_100_mentions))),
        ),
        (
            "not_in_top_100_mentions",
            pred_df.filter(~pl.col("span").is_in(list(top_100_mentions))),
        ),
        (
            "seen_unique_pairs",
            pred_df.join(unique_df, on=["span", "code"], how="inner"),
        ),
        (
            "unseen_unique_pairs",
            pred_df.join(unique_df, on=["span", "code"], how="anti"),
        ),
        ("identical", pred_df.filter(pl.col("span") == pl.col("annotation"))),
        ("not_identical", pred_df.filter(pl.col("span") != pl.col("annotation"))),
        ("one_word", pred_df.filter(pl.col("span").str.count_matches(" ") == 0)),
        ("two_words", pred_df.filter(pl.col("span").str.count_matches(" ") == 1)),
        ("three_words", pred_df.filter(pl.col("span").str.count_matches(" ") == 2)),
        (
            "more_than_three_words",
            pred_df.filter(pl.col("span").str.count_matches(" ") >= 3),
        ),
        (
            "abbrev_only",
            pred_df.filter(
                pl.col("span").str.strip_chars().str.contains(r"^[A-Z0-9\-]{2,}$")
            ),
        ),
        (
            "abbrev_or_contains",
            pred_df.filter(pl.col("span").str.contains(r"\b[A-Z0-9\-]{2,}\b")),
        ),
        (
            "not_abbrev",
            pred_df.filter(~pl.col("span").str.contains(r"\b[A-Z0-9\-]{2,}\b")),
        ),
        ("repeated", pred_df.join(repeated, on=["filename", "code"], how="inner")),
        ("not_repeated", pred_df.join(repeated, on=["filename", "code"], how="anti")),
    ]

    # -------------------------
    # Aggregation into final_results
    # -------------------------
    final_results: dict[str, dict[str, list[Any]]] = {}

    def _init_label_entry(lbl: str):
        entry = {
            "index": [],
            "recall_strict": [],
            "ratios": [],
        }
        if bootstrap:
            entry.update({
                "recall_strict_ci_low": [],
                "recall_strict_ci_high": [],
            })

        if compute_all_recalls:
            entry.update({
                "recall_exact": [],
                "recall_partial": [],
                "recall_narrow": [],
                "recall_broad": [],
            })
        if has_threshold:
            entry.update({"recall": [], "precision": [], "f1": []})
            if bootstrap:
                entry.update({
                    "recall_ci_low": [],
                    "recall_ci_high": [],
                    "precision_ci_low": [],
                    "precision_ci_high": [],
                    "f1_ci_low": [],
                    "f1_ci_high": [],
                })
        return entry

    def _aggregate(label: str, part: dict[str, Any]):
        if label not in final_results:
            final_results[label] = _init_label_entry(label)

        rs_map = part.get("recall_strict", {})
        ratio_map = part.get("ratios", {})
        exact_map = part.get("recall_exact", {})
        partial_map = part.get("recall_partial", {})
        narrow_map = part.get("recall_narrow", {})
        broad_map = part.get("recall_broad", {})
        recall_thresh_map = part.get("recall", {})
        precision_map = part.get("precision", {})
        f1_map = part.get("f1", {})

        strict_val = list(rs_map.get(label, {"": 0.0}).values())[0]
        ratio_val = list(ratio_map.get(label, {"": "N/A"}).values())[0]
        final_results[label]["index"].append(list(ratio_map.get(label, {}).keys())[0])
        final_results[label]["recall_strict"].append(strict_val)
        final_results[label]["ratios"].append(ratio_val)

        if bootstrap:
            rs_ci_low_map = part.get("recall_strict_ci_low", {})
            rs_ci_high_map = part.get("recall_strict_ci_high", {})
            final_results[label]["recall_strict_ci_low"].append(
                list(rs_ci_low_map.get(label, {"": None}).values())[0]
            )
            final_results[label]["recall_strict_ci_high"].append(
                list(rs_ci_high_map.get(label, {"": None}).values())[0]
            )

        if compute_all_recalls:
            exact_val = list(exact_map.get(label, {"": 0.0}).values())[0]
            partial_val = list(partial_map.get(label, {"": 0.0}).values())[0]
            narrow_val = list(narrow_map.get(label, {"": 0.0}).values())[0]
            broad_val = list(broad_map.get(label, {"": 0.0}).values())[0]
            final_results[label]["recall_exact"].append(exact_val)
            final_results[label]["recall_partial"].append(partial_val)
            final_results[label]["recall_narrow"].append(narrow_val)
            final_results[label]["recall_broad"].append(broad_val)

        if has_threshold:
            recall_thresh_val = list(recall_thresh_map.get(label, {"": 0.0}).values())[
                0
            ]
            precision_val = list(precision_map.get(label, {"": 0.0}).values())[0]
            f1_val = list(f1_map.get(label, {"": 0.0}).values())[0]
            final_results[label]["recall"].append(recall_thresh_val)
            final_results[label]["precision"].append(precision_val)
            final_results[label]["f1"].append(f1_val)

            if bootstrap:
                r_ci_low_map = part.get("recall_ci_low", {})
                r_ci_high_map = part.get("recall_ci_high", {})
                p_ci_low_map = part.get("precision_ci_low", {})
                p_ci_high_map = part.get("precision_ci_high", {})
                f1_ci_low_map = part.get("f1_ci_low", {})
                f1_ci_high_map = part.get("f1_ci_high", {})

                final_results[label]["recall_ci_low"].append(
                    list(r_ci_low_map.get(label, {"": None}).values())[0]
                )
                final_results[label]["recall_ci_high"].append(
                    list(r_ci_high_map.get(label, {"": None}).values())[0]
                )
                final_results[label]["precision_ci_low"].append(
                    list(p_ci_low_map.get(label, {"": None}).values())[0]
                )
                final_results[label]["precision_ci_high"].append(
                    list(p_ci_high_map.get(label, {"": None}).values())[0]
                )
                final_results[label]["f1_ci_low"].append(
                    list(f1_ci_low_map.get(label, {"": None}).values())[0]
                )
                final_results[label]["f1_ci_high"].append(
                    list(f1_ci_high_map.get(label, {"": None}).values())[0]
                )

    # Iterate with tqdm
    for name, df_part in tqdm(partition_specs, desc="Computing partitions"):
        part_res = _call_partition(df_part, name)

        # Aggregate immediately
        rs_map = part_res.get("recall_strict", {})
        for label in sorted(rs_map.keys()):
            _aggregate(label, part_res)

    return final_results
