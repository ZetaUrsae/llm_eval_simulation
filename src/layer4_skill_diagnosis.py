"""Layer 4: optional skill-dimension diagnosis for disputed books."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def skill_diagnosis(
    disputed_books: pd.DataFrame,
    skill_df: pd.DataFrame,
    consensus_models: List[int],
    results_dir: Path,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Diagnose which skill dimensions drive disagreement for disputed books.

    Args:
        disputed_books: Decision rows for books in disputed quadrants.
        skill_df: Long skill score table: book_id, model_id, skill, score.
        consensus_models: High-consensus model ids.
        results_dir: Output directory for radar plots.

    Returns:
        Tuple of (diagnosis dataframe, text reports, radar file paths).
    """
    from . import visualizations

    if disputed_books.empty or len(consensus_models) == 0:
        return pd.DataFrame(), [], []

    target_ids = disputed_books["book_id"].astype(int).unique().tolist()
    filtered = skill_df[
        (skill_df["book_id"].isin(target_ids))
        & (skill_df["model_id"].isin(consensus_models))
    ].copy()

    if filtered.empty:
        return pd.DataFrame(), [], []

    diag_rows = []
    text_reports: List[str] = []
    radar_paths: List[str] = []

    for book_id, part in filtered.groupby("book_id"):
        summary = (
            part.groupby("skill")["score"]
            .agg(mean_score="mean", min_score="min", max_score="max", std_score="std")
            .reset_index()
        )
        summary["range"] = summary["max_score"] - summary["min_score"]

        top_dispute = summary.sort_values("range", ascending=False).iloc[0]
        top_skill = str(top_dispute["skill"])
        dist_part = part[part["skill"] == top_skill]
        p10 = float(dist_part["score"].quantile(0.10))
        p90 = float(dist_part["score"].quantile(0.90))
        text = (
            f"Book {int(book_id)}: highest disagreement on '{top_skill}' "
            f"(range={top_dispute['range']:.2f}, mean={top_dispute['mean_score']:.2f}, "
            f"p10={p10:.2f}, p90={p90:.2f})."
        )
        text_reports.append(text)

        for _, row in summary.iterrows():
            diag_rows.append(
                {
                    "book_id": int(book_id),
                    "skill": row["skill"],
                    "mean_score": float(row["mean_score"]),
                    "range": float(row["range"]),
                    "std_score": float(
                        0.0 if pd.isna(row["std_score"]) else row["std_score"]
                    ),
                    "most_disputed_skill": row["skill"] == top_skill,
                }
            )

        radar_path = visualizations.plot_skill_radar(
            int(book_id),
            summary,
            output_dir=results_dir,
            highlight_skill=top_skill,
        )
        radar_paths.append(radar_path)

    diagnosis_df = (
        pd.DataFrame(diag_rows).sort_values(["book_id", "skill"]).reset_index(drop=True)
    )
    return diagnosis_df, text_reports, radar_paths
