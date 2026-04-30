"""Bootstrap-based stability analysis for layer3 percentile partitions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config, visualizations
from . import layer3_decision as layer3

QUADRANT_ORDER: List[str] = [
    "低分_高共识",
    "低分_中共识",
    "低分_低共识",
    "中分_高共识",
    "中分_中共识",
    "中分_低共识",
    "高分_高共识",
    "高分_中共识",
    "高分_低共识",
]
QUADRANT_TO_CODE: Dict[str, int] = {
    name: idx for idx, name in enumerate(QUADRANT_ORDER)
}
CODE_TO_QUADRANT: Dict[int, str] = {idx: name for name, idx in QUADRANT_TO_CODE.items()}
SCHEME_LABELS: Dict[str, str] = {
    "15/35/50": "conservative",
    "20/40/40": "moderate",
    "25/50/25": "aggressive",
}


def _score_codes(values: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    """Map score values to low/mid/high codes."""
    codes = np.ones(values.shape[0], dtype=np.int8)
    codes[values <= q_low] = 0
    codes[values >= q_high] = 2
    return codes


def _consensus_codes(values: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    """Map rank-range values to high/mid/low consensus codes."""
    codes = np.ones(values.shape[0], dtype=np.int8)
    codes[values <= q_low] = 0
    codes[values >= q_high] = 2
    return codes


def _classify_codes(
    target_scores: np.ndarray,
    target_ranges: np.ndarray,
    source_scores: np.ndarray,
    source_ranges: np.ndarray,
    low_fraction: float,
    high_fraction: float,
) -> Tuple[np.ndarray, float, bool]:
    """Classify books into quadrant codes using bootstrap-derived percentile cutoffs."""
    score_q_low = float(np.quantile(source_scores, low_fraction))
    score_q_high = float(np.quantile(source_scores, 1.0 - high_fraction))

    target_score_codes = _score_codes(target_scores, score_q_low, score_q_high)
    source_score_codes = _score_codes(source_scores, score_q_low, score_q_high)

    rho, _ = spearmanr(source_scores, source_ranges)
    rho = float(0.0 if np.isnan(rho) else rho)
    conditional = abs(rho) > 0.3

    if conditional:
        target_consensus_codes = np.ones(target_ranges.shape[0], dtype=np.int8)
        for level_code in (0, 1, 2):
            source_mask = source_score_codes == level_code
            target_mask = target_score_codes == level_code
            if not np.any(source_mask) or not np.any(target_mask):
                continue
            part_q_low = float(np.quantile(source_ranges[source_mask], low_fraction))
            part_q_high = float(
                np.quantile(source_ranges[source_mask], 1.0 - high_fraction)
            )
            target_consensus_codes[target_mask] = _consensus_codes(
                target_ranges[target_mask], part_q_low, part_q_high
            )
    else:
        range_q_low = float(np.quantile(source_ranges, low_fraction))
        range_q_high = float(np.quantile(source_ranges, 1.0 - high_fraction))
        target_consensus_codes = _consensus_codes(
            target_ranges, range_q_low, range_q_high
        )

    quadrant_codes = target_score_codes * 3 + target_consensus_codes
    return quadrant_codes.astype(np.int8), rho, conditional


def run_bootstrap_stability_analysis(
    scores_df: pd.DataFrame,
    consensus_models: List[int],
    schemes: Sequence[str] | None = None,
    n_bootstrap: int = config.BOOTSTRAP_ITERATIONS,
    random_seed: int = config.SEED,
    density_decision_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
) -> Dict[str, object]:
    """Estimate book- and quadrant-level stability for alternative percentile schemes."""
    bootstrap_dir = output_dir or config.get_results_dir("bootstrap")
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = report_dir or config.get_results_dir("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    scheme_names = list(schemes or config.BOOTSTRAP_SCHEMES)
    decision_space = layer3.prepare_decision_space(scores_df, consensus_models)
    if decision_space.empty:
        report_path = reports_dir / "bootstrap_report.txt"
        report_path.write_text("No decision-space rows available.", encoding="utf-8")
        return {
            "report_path": str(report_path),
            "book_stability": pd.DataFrame(),
            "quadrant_summary": pd.DataFrame(),
            "scheme_summary": pd.DataFrame(),
            "chart_paths": [],
            "low_stability_books": [],
            "density_overlap": {},
        }

    valid_mask = ~decision_space["data_insufficient"].to_numpy(dtype=bool)
    valid_space = decision_space.loc[valid_mask].reset_index(drop=True)
    scores = valid_space["median_score"].to_numpy(dtype=float)
    ranges = valid_space["rank_iqr"].to_numpy(dtype=float)
    book_ids = valid_space["book_id"].to_numpy(dtype=int)
    rng = np.random.default_rng(random_seed)

    scheme_book_frames: List[pd.DataFrame] = []
    quadrant_rows: List[Dict[str, object]] = []
    scheme_rows: List[Dict[str, object]] = []
    default_low_stability_books: List[int] = []

    for scheme_name in scheme_names:
        _, low_fraction, high_fraction = layer3.get_scheme_fractions(scheme_name)
        baseline_codes, baseline_rho, baseline_conditional = _classify_codes(
            target_scores=scores,
            target_ranges=ranges,
            source_scores=scores,
            source_ranges=ranges,
            low_fraction=low_fraction,
            high_fraction=high_fraction,
        )
        counts = np.zeros((scores.shape[0], len(QUADRANT_ORDER)), dtype=np.uint16)

        for _ in range(int(n_bootstrap)):
            sample_idx = rng.integers(0, scores.shape[0], size=scores.shape[0])
            sample_scores = scores[sample_idx]
            sample_ranges = ranges[sample_idx]
            boot_codes, _, _ = _classify_codes(
                target_scores=scores,
                target_ranges=ranges,
                source_scores=sample_scores,
                source_ranges=sample_ranges,
                low_fraction=low_fraction,
                high_fraction=high_fraction,
            )
            counts[np.arange(scores.shape[0]), boot_codes] += 1

        modal_codes = counts.argmax(axis=1)
        stability_rates = counts.max(axis=1) / float(n_bootstrap)
        baseline_match = counts[np.arange(scores.shape[0]), baseline_codes] / float(
            n_bootstrap
        )

        scheme_book_df = pd.DataFrame(
            {
                "book_id": book_ids,
                "scheme_name": scheme_name,
                "baseline_quadrant": [
                    CODE_TO_QUADRANT[int(code)] for code in baseline_codes
                ],
                "modal_quadrant": [CODE_TO_QUADRANT[int(code)] for code in modal_codes],
                "stability_rate": stability_rates,
                "baseline_match_rate": baseline_match,
                "baseline_rho": baseline_rho,
                "baseline_conditional": baseline_conditional,
            }
        )
        scheme_book_frames.append(scheme_book_df)

        for quadrant, part in scheme_book_df.groupby("baseline_quadrant"):
            quadrant_rows.append(
                {
                    "scheme_name": scheme_name,
                    "quadrant": quadrant,
                    "book_count": int(len(part)),
                    "mean_stability": float(part["stability_rate"].mean()),
                    "median_stability": float(part["stability_rate"].median()),
                }
            )

        scheme_rows.append(
            {
                "scheme_name": scheme_name,
                "overall_mean_stability": float(stability_rates.mean()),
                "overall_median_stability": float(np.median(stability_rates)),
                "baseline_self_match": float(baseline_match.mean()),
                "min_stability": float(stability_rates.min()),
            }
        )

        if scheme_name == config.DEFAULT_PERCENTILE_SCHEME:
            cutoff = float(np.quantile(stability_rates, 0.20))
            default_low_stability_books = (
                scheme_book_df.loc[
                    scheme_book_df["stability_rate"] <= cutoff, "book_id"
                ]
                .astype(int)
                .tolist()
            )

    book_stability_df = pd.concat(scheme_book_frames, ignore_index=True)
    quadrant_summary_df = pd.DataFrame(quadrant_rows).sort_values(
        ["scheme_name", "quadrant"]
    )
    scheme_summary_df = pd.DataFrame(scheme_rows).sort_values(
        "overall_mean_stability", ascending=False
    )
    scheme_summary_df["scheme_style"] = scheme_summary_df["scheme_name"].map(
        lambda x: SCHEME_LABELS.get(str(x), "custom")
    )

    book_stability_path = bootstrap_dir / "bootstrap_book_stability.csv"
    quadrant_summary_path = bootstrap_dir / "bootstrap_quadrant_stability.csv"
    scheme_summary_path = bootstrap_dir / "bootstrap_scheme_summary.csv"
    book_stability_df.to_csv(book_stability_path, index=False, encoding="utf-8-sig")
    quadrant_summary_df.to_csv(quadrant_summary_path, index=False, encoding="utf-8-sig")
    scheme_summary_df.to_csv(scheme_summary_path, index=False, encoding="utf-8-sig")

    chart_paths = [
        visualizations.plot_bootstrap_stability_heatmap(
            quadrant_summary_df, output_dir=bootstrap_dir
        ),
        visualizations.plot_bootstrap_scheme_summary(
            scheme_summary_df, output_dir=bootstrap_dir
        ),
    ]
    chart_paths = [path for path in chart_paths if path]

    density_overlap: Dict[str, float | int] = {}
    if density_decision_df is not None and not density_decision_df.empty:
        density_controversy = set(
            density_decision_df.loc[
                density_decision_df["density_level"] == "低共识", "book_id"
            ].astype(int)
        )
        low_stability_set = set(default_low_stability_books)
        overlap = density_controversy & low_stability_set
        union = density_controversy | low_stability_set

        density_overlap = {
            "controversy_count": len(density_controversy),
            "low_stability_count": len(low_stability_set),
            "overlap_count": len(overlap),
            "jaccard": (len(overlap) / len(union)) if union else 0.0,
            "controversy_recall_of_low_stability": (
                len(overlap) / len(low_stability_set) if low_stability_set else 0.0
            ),
        }

    report_lines: List[str] = ["== Bootstrap Boundary Stability Analysis =="]
    report_lines.append(
        f"n_bootstrap={n_bootstrap}, schemes={scheme_names}, default_scheme={config.DEFAULT_PERCENTILE_SCHEME}"
    )
    report_lines.append("\n-- Scheme Summary --")
    report_lines.append(
        scheme_summary_df.to_string(index=False)
        if not scheme_summary_df.empty
        else "No scheme summary."
    )
    report_lines.append("\n-- Quadrant Stability Summary --")
    report_lines.append(
        quadrant_summary_df.to_string(index=False)
        if not quadrant_summary_df.empty
        else "No quadrant summary."
    )
    report_lines.append("\n-- Output Files --")
    report_lines.append(str(book_stability_path))
    report_lines.append(str(quadrant_summary_path))
    report_lines.append(str(scheme_summary_path))
    report_lines.extend(chart_paths)

    if density_overlap:
        report_lines.append("\n-- Density Controversy vs Bootstrap Instability --")
        report_lines.append(
            ", ".join(
                [
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in density_overlap.items()
                ]
            )
        )

    report_path = reports_dir / "bootstrap_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "report_path": str(report_path),
        "book_stability": book_stability_df,
        "quadrant_summary": quadrant_summary_df,
        "scheme_summary": scheme_summary_df,
        "chart_paths": chart_paths,
        "low_stability_books": default_low_stability_books,
        "density_overlap": density_overlap,
    }
