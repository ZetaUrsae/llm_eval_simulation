"""Density-based alternative nine-quadrant partitioning for layer3."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from . import config
from . import layer3_decision as layer3


def _density_level(value: float, low_cutoff: float, high_cutoff: float) -> str:
    """Map KDE density to consensus-like levels."""
    if value >= high_cutoff:
        return "高共识"
    if value <= low_cutoff:
        return "低共识"
    return "中共识"


def generate_density_decision_matrix(
    scores_df: pd.DataFrame,
    consensus_models: List[int],
    scheme_name: str | None = None,
) -> pd.DataFrame:
    """Generate a KDE-driven alternative decision matrix in score-IQR space."""
    decision_space = layer3.prepare_decision_space(scores_df, consensus_models)
    if decision_space.empty:
        return decision_space

    scheme_label, low_fraction, high_fraction = layer3.get_scheme_fractions(scheme_name)
    density_df = decision_space.copy()
    valid_df = density_df[~density_df["data_insufficient"]].copy()
    if valid_df.empty:
        density_df["score_level"] = "数据不足"
        density_df["density_level"] = "数据不足"
        density_df["consensus_level"] = "数据不足"
        density_df["quadrant"] = "数据不足"
        density_df["action_label"] = "数据不足，需人工评估"
        density_df["decision_method"] = "density_kde"
        density_df["scheme_name"] = scheme_label
        return density_df

    score_q_low = float(valid_df["median_score"].quantile(low_fraction))
    score_q_high = float(valid_df["median_score"].quantile(1.0 - high_fraction))
    density_df["score_level"] = density_df["median_score"].apply(
        lambda x: layer3._score_level(x, score_q_low, score_q_high)
    )
    valid_df["score_level"] = valid_df["median_score"].apply(
        lambda x: layer3._score_level(x, score_q_low, score_q_high)
    )

    x = valid_df["median_score"].to_numpy(dtype=float)
    y = valid_df["rank_iqr"].to_numpy(dtype=float)
    x_scale = float(np.std(x, ddof=0) or 1.0)
    y_scale = float(np.std(y, ddof=0) or 1.0)
    scaled_points = np.vstack([(x - x.mean()) / x_scale, (y - y.mean()) / y_scale])

    if scaled_points.shape[1] > config.DENSITY_KDE_SAMPLE_N:
        rng = np.random.default_rng(config.SEED)
        sample_idx = rng.choice(
            scaled_points.shape[1], size=config.DENSITY_KDE_SAMPLE_N, replace=False
        )
        kde_fit_points = scaled_points[:, sample_idx]
    else:
        kde_fit_points = scaled_points

    kde = gaussian_kde(kde_fit_points)
    density_values = kde(scaled_points)
    low_cutoff = float(np.quantile(density_values, 0.20))
    high_cutoff = float(np.quantile(density_values, 0.80))

    valid_df["density"] = density_values
    valid_df["density_level"] = valid_df["density"].apply(
        lambda x: _density_level(x, low_cutoff, high_cutoff)
    )
    valid_df["consensus_level"] = valid_df["density_level"]
    valid_df["quadrant"] = valid_df["score_level"] + "_" + valid_df["consensus_level"]
    valid_df["action_label"] = valid_df["quadrant"].map(layer3.ACTION_MAP)

    density_df = density_df.merge(
        valid_df[
            [
                "book_id",
                "density",
                "density_level",
                "consensus_level",
                "quadrant",
                "action_label",
            ]
        ],
        on="book_id",
        how="left",
    )
    insuff_mask = density_df["data_insufficient"]
    density_df.loc[insuff_mask, "density_level"] = "数据不足"
    density_df.loc[insuff_mask, "consensus_level"] = "数据不足"
    density_df.loc[insuff_mask, "quadrant"] = "数据不足"
    density_df.loc[insuff_mask, "action_label"] = "数据不足，需人工评估"
    density_df["decision_method"] = "density_kde"
    density_df["scheme_name"] = scheme_label
    density_df["density_q20"] = low_cutoff
    density_df["density_q80"] = high_cutoff
    return density_df.sort_values("book_id").reset_index(drop=True)


def summarize_density_overlap(
    percentile_df: pd.DataFrame,
    density_df: pd.DataFrame,
    low_stability_books: List[int],
) -> Dict[str, float | int]:
    """Compute overlap metrics between percentile and density decision variants."""
    percentile_accept = set(
        percentile_df.loc[percentile_df["quadrant"] == "高分_高共识", "book_id"].astype(
            int
        )
    )
    density_accept = set(
        density_df.loc[density_df["quadrant"] == "高分_高共识", "book_id"].astype(int)
    )
    percentile_reject = set(
        percentile_df.loc[percentile_df["quadrant"] == "低分_高共识", "book_id"].astype(
            int
        )
    )
    density_reject = set(
        density_df.loc[density_df["quadrant"] == "低分_高共识", "book_id"].astype(int)
    )
    density_controversy = set(
        density_df.loc[density_df["density_level"] == "低共识", "book_id"].astype(int)
    )
    low_stability_set = set(int(book_id) for book_id in low_stability_books)

    def _jaccard(left: set[int], right: set[int]) -> float:
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    controversy_overlap = density_controversy & low_stability_set
    return {
        "accept_overlap_jaccard": _jaccard(percentile_accept, density_accept),
        "reject_overlap_jaccard": _jaccard(percentile_reject, density_reject),
        "controversy_low_stability_jaccard": _jaccard(
            density_controversy, low_stability_set
        ),
        "controversy_low_stability_recall": (
            len(controversy_overlap) / len(low_stability_set)
            if low_stability_set
            else 0.0
        ),
        "controversy_overlap_count": len(controversy_overlap),
    }
