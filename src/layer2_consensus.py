"""Layer 2: group consensus screening and subset search."""

from __future__ import annotations

import time
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from . import config


def _kendalls_w(rank_df: pd.DataFrame) -> Tuple[float, float]:
    """Compute Kendall's W and p-value based on rank matrix."""
    rank_df = rank_df.copy()
    rank_df = rank_df.dropna(how="all", axis=0)
    n_items = rank_df.shape[0]
    n_raters = rank_df.shape[1]
    if n_items < 2 or n_raters < 2:
        return 0.0, 1.0

    rank_df = rank_df.apply(lambda col: col.fillna(col.mean()), axis=0)
    rank_df = rank_df.dropna(how="any", axis=1)
    if rank_df.shape[1] < 2:
        return 0.0, 1.0
    n_raters = rank_df.shape[1]

    rank_sums = rank_df.sum(axis=1)
    s_term = ((rank_sums - rank_sums.mean()) ** 2).sum()
    denom = (n_raters**2) * (n_items**3 - n_items)
    w = float(12.0 * s_term / denom) if denom > 0 else 0.0
    w = float(np.clip(w, 0.0, 1.0))

    chi_sq = n_raters * (n_items - 1) * w
    p_value = float(chi2.sf(chi_sq, df=n_items - 1))
    return w, p_value


def _mean_and_var_rho(corr: pd.DataFrame, model_id: int) -> Tuple[float, float]:
    """Get mean and variance of model's pairwise Spearman correlations."""
    vals = corr.loc[model_id, corr.columns != model_id]
    if vals.dropna().empty:
        return float(np.nan), float(np.nan)
    return float(vals.mean()), float(vals.var(ddof=0))


def _subset_stats(score_wide: pd.DataFrame, subset: Sequence[int]) -> Dict[str, float]:
    """Calculate consensus metrics for a model subset."""
    sub_scores = score_wide[list(subset)]
    sub_ranks = sub_scores.rank(
        axis=0, ascending=False, method="average", na_option="keep"
    )
    w, p_value = _kendalls_w(sub_ranks)
    corr = sub_scores.corr(method="spearman")
    if len(subset) > 1:
        upper = corr.where(~np.tril(np.ones(corr.shape), k=0).astype(bool)).stack()
        mean_rho = float(upper.mean()) if not upper.empty else float("nan")
    else:
        mean_rho = 0.0

    return {"w": w, "p_value": p_value, "mean_rho": mean_rho}


def _build_mds(score_wide: pd.DataFrame, model_ids: List[int]) -> pd.DataFrame:
    """Create MDS 2D coordinates from Spearman distance matrix."""
    if len(model_ids) == 0:
        return pd.DataFrame(columns=["model_id", "mds_x", "mds_y", "cluster"])

    if len(model_ids) == 1:
        return pd.DataFrame(
            [{"model_id": model_ids[0], "mds_x": 0.0, "mds_y": 0.0, "cluster": 0}]
        )

    corr = score_wide[model_ids].corr(method="spearman")
    corr = corr.fillna(0.0)
    dist = (1.0 - corr).to_numpy(copy=True)
    np.fill_diagonal(dist, 0.0)

    try:
        mds = MDS(
            n_components=2,
            metric="precomputed",
            random_state=config.RANDOM_SEED,
            n_init=4,
            init="random",
        )
        coords = mds.fit_transform(dist)
        algo = "mds"
    except Exception:
        pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
        coords = pca.fit_transform(corr.to_numpy(copy=True))
        algo = "pca_fallback"

    n_clusters = min(3, len(model_ids))
    clusters = KMeans(
        n_clusters=n_clusters, random_state=config.RANDOM_SEED, n_init=10
    ).fit_predict(coords)

    return pd.DataFrame(
        {
            "model_id": model_ids,
            "mds_x": coords[:, 0],
            "mds_y": coords[:, 1],
            "cluster": clusters,
            "group": [config.model_group(m) for m in model_ids],
            "algo": algo,
        }
    )


def _local_search_best(
    score_wide: pd.DataFrame,
    greedy_models: List[int],
    available_models: List[int],
    theta_con: float,
    time_budget_sec: float = 30.0,
) -> Dict[str, object]:
    """Local exact search by trying to drop 0-3 models within a bounded candidate set."""
    start = time.perf_counter()
    sample_n = min(2500, len(score_wide))
    score_for_search = (
        score_wide.sample(n=sample_n, random_state=config.RANDOM_SEED)
        if sample_n < len(score_wide)
        else score_wide
    )
    corr = score_for_search[available_models].corr(method="spearman")

    mean_rho = corr.mean(axis=1, skipna=True)
    candidate_pool = mean_rho.sort_values().head(min(12, len(mean_rho))).index.tolist()

    for m in available_models:
        if m not in candidate_pool and np.isnan(mean_rho.loc[m]):
            candidate_pool.append(m)

    best_models = greedy_models.copy()
    best_stats = (
        _subset_stats(score_for_search, best_models)
        if len(best_models) >= 3
        else {"w": 0.0, "mean_rho": -np.inf}
    )

    evaluated = 0
    for drop_k in range(0, min(3, len(candidate_pool)) + 1):
        for dropped in combinations(candidate_pool, drop_k):
            if time.perf_counter() - start > time_budget_sec:
                return {
                    "models": best_models,
                    "size": len(best_models),
                    "w": best_stats["w"],
                    "mean_rho": best_stats["mean_rho"],
                    "method": "local_search_time_limited",
                    "evaluated": evaluated,
                }

            candidate = [m for m in available_models if m not in dropped]
            if len(candidate) < 3:
                continue
            stats = _subset_stats(score_for_search, candidate)
            evaluated += 1
            if stats["w"] < theta_con:
                continue

            better = False
            if len(candidate) > len(best_models):
                better = True
            elif (
                len(candidate) == len(best_models)
                and stats["mean_rho"] > best_stats["mean_rho"]
            ):
                better = True

            if better:
                best_models = candidate
                best_stats = stats

    return {
        "models": best_models,
        "size": len(best_models),
        "w": best_stats["w"],
        "mean_rho": best_stats["mean_rho"],
        "method": "local_search_drop_0_to_3",
        "evaluated": evaluated,
    }


def find_consensus_subset(
    scores_df: pd.DataFrame,
    reliable_models: List[int],
    theta_con: float | None = None,
    include_mds: bool = True,
    run_local_search: bool = True,
    selection_sample_n: int | None = None,
) -> Dict[str, object]:
    """Find high-consensus model subset via greedy and local bounded search."""
    threshold = float(config.THETA_CON if theta_con is None else theta_con)
    round1 = scores_df[scores_df["round"] == 1]
    score_wide = round1.pivot(
        index="book_id", columns="model_id", values="score"
    ).sort_index()
    selection_wide = score_wide
    if selection_sample_n is not None and selection_sample_n < len(score_wide):
        selection_wide = score_wide.sample(
            n=max(3, int(selection_sample_n)), random_state=config.RANDOM_SEED
        )

    available_models = [m for m in reliable_models if m in score_wide.columns]
    if len(available_models) < 3:
        return {
            "consensus_models": [],
            "final_w": 0.0,
            "final_p": 1.0,
            "iteration_log": [],
            "mds_coords": _build_mds(score_wide, available_models)
            if include_mds
            else pd.DataFrame(),
            "exhaustive_best": {
                "models": [],
                "size": 0,
                "w": 0.0,
                "mean_rho": -np.inf,
                "method": "not_run",
            },
        }

    current = list(sorted(available_models))
    iteration_log: List[Dict[str, object]] = []
    step = 0

    while True:
        step += 1
        stats = _subset_stats(selection_wide, current)
        w_before = stats["w"]
        p_before = stats["p_value"]

        if w_before >= threshold or len(current) < 3:
            iteration_log.append(
                {
                    "step": step,
                    "remaining_models": current.copy(),
                    "removed_model": None,
                    "w_before": w_before,
                    "w_after": w_before,
                    "p_value_after": p_before,
                    "avg_rho_removed": None,
                    "var_rho_removed": None,
                    "note": "stop",
                }
            )
            break

        corr = selection_wide[current].corr(method="spearman")
        model_metrics = []
        for model_id in current:
            mean_rho, var_rho = _mean_and_var_rho(corr, model_id)
            model_metrics.append((model_id, mean_rho, var_rho))

        to_remove = sorted(
            model_metrics,
            key=lambda x: (
                0 if np.isnan(x[1]) else 1,
                np.inf if np.isnan(x[1]) else x[1],
                -np.inf if np.isnan(x[2]) else -x[2],
            ),
        )[0]
        removed_model, avg_rho, var_rho = to_remove
        current.remove(removed_model)

        new_stats = _subset_stats(selection_wide, current)
        iteration_log.append(
            {
                "step": step,
                "remaining_models": current.copy(),
                "removed_model": int(removed_model),
                "w_before": w_before,
                "w_after": new_stats["w"],
                "p_value_after": new_stats["p_value"],
                "avg_rho_removed": avg_rho,
                "var_rho_removed": var_rho,
                "note": "greedy_remove",
            }
        )

    final_stats = (
        _subset_stats(selection_wide, current)
        if len(current) >= 3
        else {"w": 0.0, "p_value": 1.0}
    )
    consensus_models = current if final_stats["w"] >= threshold else []

    if run_local_search:
        exhaustive_best = _local_search_best(
            score_wide=score_wide,
            greedy_models=current if current else available_models,
            available_models=available_models,
            theta_con=threshold,
            time_budget_sec=30.0,
        )
    else:
        exhaustive_best = {
            "models": current.copy(),
            "size": len(current),
            "w": final_stats["w"],
            "mean_rho": final_stats.get("mean_rho", float("nan")),
            "method": "skipped",
            "evaluated": 0,
        }

    mds_coords = (
        _build_mds(score_wide, available_models) if include_mds else pd.DataFrame()
    )
    return {
        "consensus_models": consensus_models,
        "final_w": final_stats["w"],
        "final_p": final_stats["p_value"],
        "iteration_log": iteration_log,
        "mds_coords": mds_coords,
        "exhaustive_best": exhaustive_best,
        "theta_con": threshold,
    }
