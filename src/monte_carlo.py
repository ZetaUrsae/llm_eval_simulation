"""Monte Carlo multi-seed experiment runner for full pipeline robustness checks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from . import config, data_simulator, density_quadrant, visualizations
from . import layer1_reliability as layer1
from . import layer2_consensus as layer2
from . import layer3_decision as layer3
from . import layer4_skill_diagnosis as layer4


def _summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/std/min/max table for key Monte Carlo metrics."""
    numeric_cols = [
        "reliable_model_count",
        "consensus_model_count",
        "final_w",
        "direct_accept_count",
        "clear_reject_count",
    ]
    rows: List[Dict[str, float | str]] = []
    for col in numeric_cols:
        series = metrics_df[col]
        rows.append(
            {
                "metric": col,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)


def run_monte_carlo_experiment(
    seeds: Sequence[int] | None = None,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
) -> Dict[str, object]:
    """Run full pipeline logic across multiple random seeds and report variability."""
    seed_list = list(seeds or config.MONTE_CARLO_SEEDS)
    monte_carlo_dir = output_dir or config.get_results_dir("monte_carlo")
    monte_carlo_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = report_dir or config.get_results_dir("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float | int]] = []
    for seed in seed_list:
        scores_df, skill_df = data_simulator.generate_data(seed=seed)
        reliable_models, _, _ = layer1.filter_reliable_models(scores_df)
        layer2_result = layer2.find_consensus_subset(scores_df, reliable_models)
        consensus_models = layer2_result.get("consensus_models", [])
        final_w = float(layer2_result.get("final_w", 0.0))

        if config.DEFAULT_DECISION_METHOD == "density":
            decision_df = density_quadrant.generate_density_decision_matrix(
                scores_df=scores_df,
                consensus_models=consensus_models,
                scheme_name=config.DEFAULT_PERCENTILE_SCHEME,
            )
        else:
            decision_df = layer3.generate_decision_matrix(
                scores_df=scores_df,
                consensus_models=consensus_models,
                scheme_name=config.DEFAULT_PERCENTILE_SCHEME,
            )

        # Keep layer4 and optional analyses in the loop to preserve full-flow execution semantics.
        disputed = decision_df[
            decision_df["action_label"].isin(
                ["重点评估/专家介入", "暂缓决策，补充信息"]
            )
        ].copy()
        if not disputed.empty:
            layer4.skill_diagnosis(
                disputed_books=disputed.head(5),
                skill_df=skill_df,
                consensus_models=consensus_models,
                results_dir=monte_carlo_dir,
            )

        quadrant_counts = decision_df["quadrant"].value_counts()
        rows.append(
            {
                "seed": int(seed),
                "reliable_model_count": int(len(reliable_models)),
                "consensus_model_count": int(len(consensus_models)),
                "final_w": final_w,
                "direct_accept_count": int(quadrant_counts.get("高分_高共识", 0)),
                "clear_reject_count": int(quadrant_counts.get("低分_高共识", 0)),
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    summary_df = _summarize_metrics(metrics_df)

    metrics_csv = monte_carlo_dir / "monte_carlo_seed_metrics.csv"
    summary_csv = monte_carlo_dir / "monte_carlo_summary_stats.csv"
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    chart_paths = [
        visualizations.plot_monte_carlo_high_high_hist(
            metrics_df=metrics_df,
            output_dir=monte_carlo_dir,
        )
    ]
    chart_paths = [path for path in chart_paths if path]

    report_lines: List[str] = ["== Monte Carlo Multi-Seed Experiment =="]
    report_lines.append(f"seeds={seed_list}")
    report_lines.append("\n-- Per-Seed Metrics --")
    report_lines.append(metrics_df.to_string(index=False))
    report_lines.append("\n-- Summary (mean/std/min/max) --")
    report_lines.append(summary_df.to_string(index=False))
    report_lines.append("\n-- Output Files --")
    report_lines.append(str(metrics_csv))
    report_lines.append(str(summary_csv))
    report_lines.extend(chart_paths)

    report_path = reports_dir / "monte_carlo_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "report_path": str(report_path),
        "metrics_df": metrics_df,
        "summary_df": summary_df,
        "chart_paths": chart_paths,
    }
