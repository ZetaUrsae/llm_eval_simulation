"""Threshold sensitivity analysis for consensus and decision layers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from . import config, density_quadrant, visualizations
from . import layer2_consensus as layer2
from . import layer3_decision as layer3


def _build_shared_consensus_path(
    scores_df: pd.DataFrame,
    reliable_models: List[int],
    thresholds: Sequence[float],
) -> Dict[str, object]:
    """Run the greedy consensus search once at the strictest threshold."""
    max_threshold = max(float(threshold) for threshold in thresholds)
    return layer2.find_consensus_subset(
        scores_df=scores_df,
        reliable_models=reliable_models,
        theta_con=max_threshold,
        include_mds=False,
        run_local_search=False,
        selection_sample_n=2500,
    )


def _derive_threshold_result(
    scores_df: pd.DataFrame,
    reliable_models: List[int],
    threshold: float,
    shared_path: Dict[str, object],
) -> Dict[str, object]:
    """Derive one threshold result from the shared greedy removal trajectory."""
    iteration_log = shared_path.get("iteration_log", [])
    log_df = pd.DataFrame(iteration_log)
    initial_w = (
        float(log_df["w_before"].iloc[0])
        if (not log_df.empty and "w_before" in log_df.columns)
        else float(shared_path.get("final_w", 0.0))
    )

    selected_models = list(reliable_models)
    selected_w = initial_w
    stop_idx = -1
    for idx, row in log_df.iterrows():
        row_w = float(row.get("w_after", selected_w))
        remaining_models = row.get("remaining_models", selected_models)
        if isinstance(remaining_models, list):
            selected_models = [int(model_id) for model_id in remaining_models]
        selected_w = row_w
        stop_idx = idx
        if row_w >= threshold:
            break

    removed_models = sorted(set(reliable_models) - set(selected_models))
    truncated_log = (
        log_df.iloc[: stop_idx + 1].copy() if stop_idx >= 0 else pd.DataFrame()
    )

    if config.DEFAULT_DECISION_METHOD == "density":
        decision_df = density_quadrant.generate_density_decision_matrix(
            scores_df=scores_df,
            consensus_models=selected_models,
            scheme_name=config.DEFAULT_PERCENTILE_SCHEME,
        )
    else:
        decision_df = layer3.generate_decision_matrix(
            scores_df,
            selected_models,
            scheme_name=config.DEFAULT_PERCENTILE_SCHEME,
        )
    quadrant_counts = (
        decision_df["quadrant"].value_counts().sort_index()
        if not decision_df.empty
        else pd.Series(dtype=int)
    )

    removed_group_counts = (
        pd.Series([config.model_group(m) for m in removed_models])
        .value_counts()
        .sort_index()
    )

    return {
        "threshold": threshold,
        "initial_w": initial_w,
        "final_w": selected_w,
        "removed_models": removed_models,
        "removed_count": len(removed_models),
        "consensus_models": selected_models,
        "consensus_size": len(selected_models),
        "iteration_log": truncated_log,
        "layer2_result": shared_path,
        "decision_df": decision_df,
        "quadrant_counts": quadrant_counts,
        "direct_accept_count": int(quadrant_counts.get("高分_高共识", 0)),
        "low_low_count": int(quadrant_counts.get("低分_低共识", 0)),
        "removed_group_counts": removed_group_counts,
    }


def run_sensitivity_analysis(
    scores_df: pd.DataFrame,
    reliable_models: List[int],
    thresholds: Sequence[float] = tuple(config.THRESHOLD_SCAN_VALUES),
    output_dir: Path | None = None,
    report_dir: Path | None = None,
) -> Dict[str, object]:
    """Run threshold sensitivity analysis and write sensitivity report.

    Args:
        scores_df: Long-form score table.
        reliable_models: Retained models from layer1.
        thresholds: Threshold values to compare.

    Returns:
        Dictionary with report path, per-threshold results, and generated chart paths.
    """
    sensitivity_dir = output_dir or config.get_results_dir("sensitivity")
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = report_dir or config.get_results_dir("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    keys = sorted(float(t) for t in thresholds)
    shared_path = _build_shared_consensus_path(
        scores_df=scores_df,
        reliable_models=reliable_models,
        thresholds=keys,
    )

    threshold_results: Dict[float, Dict[str, object]] = {}
    for threshold in keys:
        threshold_results[float(threshold)] = _derive_threshold_result(
            scores_df=scores_df,
            reliable_models=reliable_models,
            threshold=float(threshold),
            shared_path=shared_path,
        )
    key_metrics = pd.DataFrame(
        {
            "threshold": keys,
            "initial_w": [threshold_results[t]["initial_w"] for t in keys],
            "final_w": [threshold_results[t]["final_w"] for t in keys],
            "retained_models": [threshold_results[t]["consensus_size"] for t in keys],
            "removed_models": [threshold_results[t]["removed_count"] for t in keys],
            "consensus_size": [threshold_results[t]["consensus_size"] for t in keys],
            "high_high_count": [
                threshold_results[t]["direct_accept_count"] for t in keys
            ],
            "low_low_count": [threshold_results[t]["low_low_count"] for t in keys],
        }
    )

    all_removed = sorted(
        {model_id for t in keys for model_id in threshold_results[t]["removed_models"]}
    )
    removed_rows: List[Dict[str, object]] = []
    for model_id in all_removed:
        row: Dict[str, object] = {
            "model_id": model_id,
            "group": config.model_group(model_id),
        }
        for t in keys:
            row[f"removed_at_{t:.2f}"] = (
                model_id in threshold_results[t]["removed_models"]
            )
        removed_rows.append(row)
    removed_df = pd.DataFrame(removed_rows)

    all_quadrants = sorted(
        {
            q
            for t in keys
            for q in threshold_results[t]["quadrant_counts"].index.tolist()
        }
    )
    quadrant_comp = pd.DataFrame({"quadrant": all_quadrants})
    for t in keys:
        counts = threshold_results[t]["quadrant_counts"]
        quadrant_comp[f"count_theta_{t:.2f}"] = [
            int(counts.get(q, 0)) for q in all_quadrants
        ]

    analysis_lines: List[str] = []
    if len(keys) >= 2:
        recommended = float(config.THETA_CON)
        nearest_idx = min(
            range(len(keys)), key=lambda idx: abs(keys[idx] - recommended)
        )
        analysis_lines.append(
            f"Default threshold marker is theta={recommended:.2f}; nearest scanned point is theta={keys[nearest_idx]:.2f}."
        )

        w_diff = key_metrics["final_w"].diff().fillna(0.0)
        model_diff = key_metrics["retained_models"].diff().fillna(0.0)
        score = (w_diff.abs() / model_diff.abs().replace(0.0, 1.0)).replace(
            [pd.NA, float("inf")], 0.0
        )
        elbow_idx = int(score.iloc[1:].idxmax()) if len(score) > 1 else 0
        elbow_threshold = float(key_metrics.loc[elbow_idx, "threshold"])
        analysis_lines.append(
            f"Largest marginal W-per-model change appears near theta={elbow_threshold:.2f}."
        )

        focus_row = key_metrics.loc[key_metrics["threshold"] == recommended]
        if not focus_row.empty:
            row = focus_row.iloc[0]
            analysis_lines.append(
                f"At theta={recommended:.2f}, retained_models={int(row['retained_models'])}, final_w={row['final_w']:.3f}, 高分_高共识={int(row['high_high_count'])}, 低分_低共识={int(row['low_low_count'])}."
            )

    chart_paths: List[str] = []
    scan_chart = visualizations.plot_threshold_scan_overview(
        key_metrics,
        default_threshold=config.THETA_CON,
        output_dir=sensitivity_dir,
    )
    if scan_chart:
        chart_paths.append(scan_chart)

    if 0.70 in threshold_results and 0.80 in threshold_results:
        q_cmp = visualizations.plot_quadrant_comparison(
            threshold_results[0.70]["quadrant_counts"],
            threshold_results[0.80]["quadrant_counts"],
            output_dir=sensitivity_dir,
        )
        if q_cmp:
            chart_paths.append(q_cmp)

    report_lines: List[str] = ["== Threshold Sensitivity Analysis =="]
    report_lines.append("\n-- Key Metrics --")
    report_lines.append(key_metrics.to_string(index=False))

    report_lines.append("\n-- Removed Models and Group Mapping --")
    report_lines.append(
        removed_df.to_string(index=False)
        if not removed_df.empty
        else "No removed models."
    )

    report_lines.append("\n-- Removed Group Counts --")
    for t in keys:
        grp_counts = threshold_results[t]["removed_group_counts"]
        report_lines.append(f"theta={t:.2f}")
        report_lines.append(grp_counts.to_string() if not grp_counts.empty else "None")

    report_lines.append("\n-- Quadrant Count Comparison --")
    report_lines.append(
        quadrant_comp.to_string(index=False)
        if not quadrant_comp.empty
        else "No quadrant results."
    )

    report_lines.append("\n-- Automated Summary --")
    report_lines.extend(
        analysis_lines
        if analysis_lines
        else ["Not enough thresholds for comparative summary."]
    )

    report_lines.append("\n-- Charts --")
    report_lines.extend(
        chart_paths if chart_paths else ["No comparison charts generated."]
    )

    report_path = reports_dir / "sensitivity_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "report_path": str(report_path),
        "chart_paths": chart_paths,
        "results": threshold_results,
        "key_metrics": key_metrics,
        "quadrant_comparison": quadrant_comp,
        "analysis_lines": analysis_lines,
    }
