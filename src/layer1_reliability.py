"""Layer 1: test-retest reliability screening using ICC."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from . import config
from .icc_utils import compute_icc_a_1, compute_icc_c_1

try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False


def _compute_model_icc(
    model_id: int,
    round1_scores: np.ndarray,
    round2_scores: np.ndarray,
    icc_type: str = "ICC(2,1)",
) -> Dict[str, object]:
    """Compute ICC and distribution diagnostics for a single model.

    Args:
        model_id: Model identifier.
        round1_scores: Round-1 score vector with NaN allowed.
        round2_scores: Round-2 score vector with NaN allowed.

    Returns:
        Dictionary containing ICC, CI, skewness, kurtosis and missingness stats.
    """
    r1 = np.asarray(round1_scores, dtype=float)
    r2 = np.asarray(round2_scores, dtype=float)

    valid_scores = np.concatenate([r1[~np.isnan(r1)], r2[~np.isnan(r2)]])
    skewness = (
        float(skew(valid_scores, bias=False)) if valid_scores.size > 2 else np.nan
    )
    kurt = (
        float(kurtosis(valid_scores, fisher=True, bias=False))
        if valid_scores.size > 3
        else np.nan
    )

    paired_mask = (~np.isnan(r1)) & (~np.isnan(r2))
    n_pairs = int(paired_mask.sum())

    result: Dict[str, object] = {
        "model_id": int(model_id),
        "n_pairs": n_pairs,
        "icc_type": str(icc_type),
        "icc_value": np.nan,
        # Keep legacy column name for backward compatibility.
        "icc_2_1": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "skewness": skewness,
        "kurtosis": kurt,
        "missing_round1": float(np.isnan(r1).mean()),
        "missing_round2": float(np.isnan(r2).mean()),
        "missing_overall": float(np.isnan(np.concatenate([r1, r2])).mean()),
        "error": None,
    }

    if n_pairs < 5:
        return result

    try:
        if icc_type == "ICC(2,1)":
            icc_val, ci_low, ci_high, n_pairs = compute_icc_a_1(r1, r2)
        elif icc_type == "ICC(3,1)":
            icc_val, ci_low, ci_high, n_pairs = compute_icc_c_1(r1, r2)
        else:
            raise ValueError(f"Unsupported icc_type: {icc_type}")

        result["n_pairs"] = n_pairs
        result["icc_value"] = icc_val
        result["icc_2_1"] = icc_val
        result["ci95_low"] = ci_low
        result["ci95_high"] = ci_high
    except Exception as exc:
        result["error"] = str(exc)

    return result


def _resolve_workers(n_jobs: int, task_count: int) -> int:
    """Resolve actual number of workers from n_jobs setting."""
    cpu = os.cpu_count() or 1
    if n_jobs == -1:
        return max(1, min(cpu, task_count))
    return max(1, min(int(n_jobs), task_count))


def _run_parallel_icc(
    payloads: List[Tuple[int, np.ndarray, np.ndarray]],
    n_jobs: int,
    icc_type: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Run per-model ICC computations in parallel with backend fallback."""
    task_count = len(payloads)
    workers = _resolve_workers(n_jobs=n_jobs, task_count=task_count)

    benchmark_count = min(3, task_count)
    serial_benchmark_start = time.perf_counter()
    for model_id, round1, round2 in payloads[:benchmark_count]:
        _compute_model_icc(model_id, round1, round2, icc_type=icc_type)
    serial_benchmark_elapsed = time.perf_counter() - serial_benchmark_start
    estimated_serial = (
        (serial_benchmark_elapsed / benchmark_count) * task_count
        if benchmark_count
        else 0.0
    )

    start = time.perf_counter()
    backend = "joblib" if HAS_JOBLIB else "processpool"

    if HAS_JOBLIB:
        try:
            results = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(_compute_model_icc)(model_id, round1, round2, icc_type=icc_type)
                for model_id, round1, round2 in payloads
            )
        except Exception:
            backend = "processpool_fallback"
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _compute_model_icc,
                        model_id,
                        round1,
                        round2,
                        icc_type,
                    )
                    for model_id, round1, round2 in payloads
                ]
                results = [f.result() for f in futures]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _compute_model_icc,
                    model_id,
                    round1,
                    round2,
                    icc_type,
                )
                for model_id, round1, round2 in payloads
            ]
            results = [f.result() for f in futures]

    elapsed = time.perf_counter() - start
    speedup = (estimated_serial / elapsed) if elapsed > 0 else float("nan")

    parallel_log: Dict[str, object] = {
        "model_count": task_count,
        "backend": backend,
        "workers": workers,
        "parallel_elapsed_sec": elapsed,
        "estimated_serial_sec": estimated_serial,
        "serial_benchmark_sec": serial_benchmark_elapsed,
        "speedup": speedup,
    }
    return results, parallel_log


def filter_reliable_models(
    scores_df: pd.DataFrame,
    icc_type: str = "ICC(2,1)",
) -> Tuple[List[int], pd.DataFrame, Dict[str, object]]:
    """Filter reliable models by selected ICC type, returning diagnostics.

    Args:
        scores_df: Long table with columns book_id, model_id, round, score.
        icc_type: Supported options are 'ICC(2,1)' and 'ICC(3,1)'.

    Returns:
        Tuple[List[int], pd.DataFrame, Dict[str, object]]:
            - retained model ids
            - ICC summary table
            - diagnostics dictionary
    """
    if icc_type not in {"ICC(2,1)", "ICC(3,1)"}:
        raise ValueError(
            f"Unsupported icc_type={icc_type}. Use 'ICC(2,1)' or 'ICC(3,1)'."
        )

    icc_records: List[Dict[str, object]] = []
    dist_records: List[Dict[str, object]] = []
    status_records: List[Dict[str, object]] = []
    missing_records: List[Dict[str, object]] = []
    reliable_models: List[int] = []
    boundary_models: List[int] = []
    excluded_models: List[int] = []
    excluded_no_icc: List[int] = []
    warnings: List[str] = []

    all_models = sorted(scores_df["model_id"].astype(int).unique().tolist())
    all_books = sorted(scores_df["book_id"].astype(int).unique().tolist())

    round1 = scores_df[scores_df["round"] == 1].pivot(
        index="book_id", columns="model_id", values="score"
    )
    round2 = scores_df[scores_df["round"] == 2].pivot(
        index="book_id", columns="model_id", values="score"
    )

    round1 = round1.reindex(index=all_books, columns=all_models)
    round2 = round2.reindex(index=all_books, columns=all_models)

    payloads: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for model_id in all_models:
        payloads.append(
            (
                int(model_id),
                round1[model_id].to_numpy(dtype=float),
                round2[model_id].to_numpy(dtype=float),
            )
        )

    worker_results, parallel_log = _run_parallel_icc(
        payloads,
        n_jobs=config.N_JOBS,
        icc_type=icc_type,
    )
    worker_results = sorted(worker_results, key=lambda x: int(x["model_id"]))

    for res in worker_results:
        model_id = int(res["model_id"])
        icc_val = (
            float(res["icc_value"])
            if not pd.isna(res.get("icc_value", np.nan))
            else np.nan
        )
        ci_low = float(res["ci95_low"]) if not pd.isna(res["ci95_low"]) else np.nan
        ci_high = float(res["ci95_high"]) if not pd.isna(res["ci95_high"]) else np.nan
        n_pairs = int(res["n_pairs"])

        missing_records.append(
            {
                "model_id": model_id,
                "group": config.model_group(model_id),
                "missing_round1": float(res["missing_round1"]),
                "missing_round2": float(res["missing_round2"]),
                "missing_overall": float(res["missing_overall"]),
            }
        )

        if pd.isna(icc_val):
            status = "excluded_no_icc"
            reason = "insufficient paired rounds"
            excluded_no_icc.append(model_id)
            excluded_models.append(model_id)
        elif icc_val >= config.THETA_REL:
            status = "retained"
            reason = "icc_above_threshold"
            reliable_models.append(model_id)
        elif (not pd.isna(ci_high)) and ci_high >= config.THETA_REL:
            status = "boundary_retained"
            reason = "ci_reaches_threshold"
            reliable_models.append(model_id)
            boundary_models.append(model_id)
            warnings.append(
                (
                    f"Model {model_id} ICC={icc_val:.3f} below {config.THETA_REL:.2f}, "
                    f"but CI upper={ci_high:.3f} >= threshold. Kept as boundary model."
                )
            )
        else:
            status = "excluded"
            reason = "icc_below_threshold"
            excluded_models.append(model_id)

        icc_records.append(
            {
                "model_id": model_id,
                "group": config.model_group(model_id),
                "icc_type": icc_type,
                "n_pairs": n_pairs,
                "icc_value": icc_val,
                # Legacy alias kept for old charts and scripts.
                "icc_2_1": icc_val,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "status": status,
            }
        )
        status_records.append(
            {"model_id": model_id, "status": status, "reason": reason}
        )

        sk = float(res["skewness"]) if not pd.isna(res["skewness"]) else np.nan
        ku = float(res["kurtosis"]) if not pd.isna(res["kurtosis"]) else np.nan
        dist_flag = "normal"
        if not np.isnan(sk) and abs(sk) > 1.0:
            dist_flag = "skewed"
        if not np.isnan(ku) and abs(ku) > 3.0:
            dist_flag = "heavy_tail"

        dist_records.append(
            {
                "model_id": model_id,
                "group": config.model_group(model_id),
                "skewness": sk,
                "kurtosis": ku,
                "dist_flag": dist_flag,
            }
        )

    icc_df = pd.DataFrame(icc_records).sort_values("model_id").reset_index(drop=True)
    dist_df = pd.DataFrame(dist_records).sort_values("model_id").reset_index(drop=True)
    status_df = (
        pd.DataFrame(status_records).sort_values("model_id").reset_index(drop=True)
    )
    missing_df = (
        pd.DataFrame(missing_records).sort_values("model_id").reset_index(drop=True)
    )

    diagnostics: Dict[str, object] = {
        "icc_type": icc_type,
        "distribution_table": dist_df,
        "status_table": status_df,
        "missing_profile": missing_df,
        "boundary_models": boundary_models,
        "excluded_models": excluded_models,
        "excluded_no_icc": excluded_no_icc,
        "warnings": warnings,
        "parallel_log": parallel_log,
    }

    return reliable_models, icc_df, diagnostics
