"""Layer 3: nine-quadrant decision matrix generation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config

ACTION_MAP: Dict[str, str] = {
    "高分_高共识": "直接接受",
    "高分_中共识": "建议复核",
    "高分_低共识": "重点评估/专家介入",
    "中分_高共识": "储备/待定",
    "中分_中共识": "酌情考虑",
    "中分_低共识": "暂缓决策，补充信息",
    "低分_高共识": "明确不推荐",
    "低分_中共识": "建议弃用",
    "低分_低共识": "搁置/需决策者裁定",
}

PERCENTILE_SCHEMES: Dict[str, Tuple[float, float]] = {
    "15/35/50": (0.50, 0.15),
    "20/40/40": (0.40, 0.20),
    "25/50/25": (0.25, 0.25),
}


def _score_level(value: float, q_low: float, q_high: float) -> str:
    """Map score to high/mid/low tier."""
    if value >= q_high:
        return "高分"
    if value <= q_low:
        return "低分"
    return "中分"


def _consensus_level(rank_iqr: float, q_low: float, q_high: float) -> str:
    """Map rank IQR to high/mid/low consensus tier."""
    if rank_iqr <= q_low:
        return "高共识"
    if rank_iqr >= q_high:
        return "低共识"
    return "中共识"


def _consensus_metric_col(df: pd.DataFrame) -> str:
    """Resolve consensus dispersion column with backward-compatible fallback."""
    if "rank_iqr" in df.columns:
        return "rank_iqr"
    return "rank_range"


def get_scheme_fractions(scheme_name: str | None = None) -> Tuple[str, float, float]:
    """Resolve percentile scheme name into low/high quantile fractions."""
    resolved = scheme_name or config.DEFAULT_PERCENTILE_SCHEME
    if resolved in PERCENTILE_SCHEMES:
        low_fraction, high_fraction = PERCENTILE_SCHEMES[resolved]
        return resolved, low_fraction, high_fraction

    low_fraction = config.to_quantile_fraction(config.P_LOW)
    high_fraction = config.to_quantile_fraction(config.P_HIGH)
    return resolved, low_fraction, high_fraction


def prepare_decision_space(
    scores_df: pd.DataFrame, consensus_models: List[int]
) -> pd.DataFrame:
    """Prepare per-book decision-space metrics used by multiple layer3 analyses."""
    if len(consensus_models) == 0:
        return pd.DataFrame(
            columns=[
                "book_id",
                "median_score",
                "rank_iqr",
                "missing_fraction",
                "data_insufficient",
            ]
        )

    round1 = scores_df[scores_df["round"] == 1]
    wide = round1.pivot(index="book_id", columns="model_id", values="score")
    wide = wide[consensus_models]

    rank_df = wide.rank(axis=0, ascending=False, method="average")
    missing_fraction = wide.isna().mean(axis=1)
    rank_q25 = rank_df.quantile(0.25, axis=1, interpolation="linear")
    rank_q75 = rank_df.quantile(0.75, axis=1, interpolation="linear")
    decision_space = pd.DataFrame(
        {
            "book_id": wide.index.astype(int),
            "median_score": wide.median(axis=1, skipna=True).values,
            "rank_iqr": (rank_q75 - rank_q25).values,
            "missing_fraction": missing_fraction.values,
        }
    )
    # Keep legacy alias for compatibility with old artifacts/scripts.
    decision_space["rank_range"] = decision_space["rank_iqr"]
    decision_space["data_insufficient"] = decision_space["missing_fraction"] > 0.5
    return decision_space.sort_values("book_id").reset_index(drop=True)


def classify_prepared_space(
    decision_space: pd.DataFrame,
    low_fraction: float,
    high_fraction: float,
    scheme_name: str = "custom",
    threshold_source: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify prepared book metrics into the percentile-based decision matrix."""
    if decision_space.empty:
        empty = decision_space.copy()
        for column in [
            "score_level",
            "consensus_level",
            "quadrant",
            "action_label",
            "spearman_rho",
            "conditional_consensus",
            "scheme_name",
            "decision_method",
        ]:
            empty[column] = []
        return empty

    decision_df = decision_space.copy()
    source_df = (
        threshold_source.copy() if threshold_source is not None else decision_df.copy()
    )

    valid_df = decision_df[~decision_df["data_insufficient"]].copy()
    valid_source = source_df[~source_df["data_insufficient"]].copy()
    if valid_df.empty or valid_source.empty:
        decision_df["score_level"] = "数据不足"
        decision_df["consensus_level"] = "数据不足"
        decision_df["quadrant"] = "数据不足"
        decision_df["action_label"] = "数据不足，需人工评估"
        decision_df["spearman_rho"] = np.nan
        decision_df["conditional_consensus"] = False
        decision_df["scheme_name"] = scheme_name
        decision_df["decision_method"] = "percentile"
        return decision_df.sort_values("book_id").reset_index(drop=True)

    score_q_low = float(valid_source["median_score"].quantile(low_fraction))
    score_q_high = float(valid_source["median_score"].quantile(1.0 - high_fraction))

    decision_df["score_level"] = decision_df["median_score"].apply(
        lambda x: _score_level(x, score_q_low, score_q_high)
    )
    valid_source["score_level"] = valid_source["median_score"].apply(
        lambda x: _score_level(x, score_q_low, score_q_high)
    )

    source_consensus_col = _consensus_metric_col(valid_source)
    target_consensus_col = _consensus_metric_col(decision_df)
    rho, _ = spearmanr(
        valid_source["median_score"],
        valid_source[source_consensus_col],
        nan_policy="omit",
    )
    rho = float(0.0 if np.isnan(rho) else rho)
    conditional = abs(rho) > 0.3

    if conditional:
        consensus_levels = []
        for level in ["低分", "中分", "高分"]:
            source_part = valid_source[valid_source["score_level"] == level]
            target_part = decision_df[
                (~decision_df["data_insufficient"])
                & (decision_df["score_level"] == level)
            ]
            if source_part.empty or target_part.empty:
                continue
            part_q_low = float(source_part[source_consensus_col].quantile(low_fraction))
            part_q_high = float(
                source_part[source_consensus_col].quantile(1.0 - high_fraction)
            )
            consensus_levels.append(
                pd.DataFrame(
                    {
                        "book_id": target_part["book_id"].astype(int).to_numpy(),
                        "consensus_level": target_part[target_consensus_col].apply(
                            lambda x: _consensus_level(x, part_q_low, part_q_high)
                        ),
                    }
                )
            )
        if consensus_levels:
            level_df = pd.concat(consensus_levels, ignore_index=True)
            decision_df = decision_df.merge(level_df, on="book_id", how="left")
        else:
            decision_df["consensus_level"] = "中共识"
    else:
        range_q_low = float(valid_source[source_consensus_col].quantile(low_fraction))
        range_q_high = float(
            valid_source[source_consensus_col].quantile(1.0 - high_fraction)
        )
        decision_df["consensus_level"] = decision_df[target_consensus_col].apply(
            lambda x: _consensus_level(x, range_q_low, range_q_high)
        )

    decision_df["quadrant"] = (
        decision_df["score_level"] + "_" + decision_df["consensus_level"]
    )
    decision_df["action_label"] = decision_df["quadrant"].map(ACTION_MAP)

    insuff_mask = decision_df["data_insufficient"]
    decision_df.loc[insuff_mask, "score_level"] = "数据不足"
    decision_df.loc[insuff_mask, "consensus_level"] = "数据不足"
    decision_df.loc[insuff_mask, "quadrant"] = "数据不足"
    decision_df.loc[insuff_mask, "action_label"] = "数据不足，需人工评估"

    decision_df["spearman_rho"] = rho
    decision_df["conditional_consensus"] = conditional
    decision_df["scheme_name"] = scheme_name
    decision_df["decision_method"] = "percentile"

    return decision_df.sort_values("book_id").reset_index(drop=True)


def generate_decision_matrix(
    scores_df: pd.DataFrame,
    consensus_models: List[int],
    scheme_name: str | None = None,
) -> pd.DataFrame:
    """Generate nine-quadrant decision matrix for books.

    Args:
        scores_df: Long table with columns book_id, model_id, round, score.
        consensus_models: Retained high-consensus model ids.

    Returns:
        DataFrame with one row per book and decision annotations.
    """
    decision_space = prepare_decision_space(scores_df, consensus_models)
    scheme_label, low_fraction, high_fraction = get_scheme_fractions(scheme_name)
    return classify_prepared_space(
        decision_space=decision_space,
        low_fraction=low_fraction,
        high_fraction=high_fraction,
        scheme_name=scheme_label,
    )
