"""Pipeline orchestration for the full simulation workflow."""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from . import (
    bootstrap_stability,
    config,
    data_simulator,
    density_quadrant,
    layer1,
    layer2,
    layer3,
    layer4,
    monte_carlo,
    threshold_sensitivity,
    visualizations,
)


def _safe_log_df(iteration_log: List[dict]) -> pd.DataFrame:
    """Convert iteration log list into a DataFrame safely."""
    if not iteration_log:
        return pd.DataFrame(columns=["step", "w_after"])
    return pd.DataFrame(iteration_log)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the main workflow."""
    parser = argparse.ArgumentParser(
        description="Run the llm_eval_simulation analysis pipeline."
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="Run the additional multi-seed Monte Carlo robustness experiment.",
    )
    return parser


def run_pipeline(enable_monte_carlo: bool = False) -> Path:
    """Run the full simulation pipeline and return the summary report path."""
    output_dirs: Dict[str, Path] = config.ensure_results_dirs()

    summary_lines: List[str] = []
    chart_paths: List[str] = []
    stage_timing: List[tuple[str, float]] = []

    try:
        t0 = time.perf_counter()
        scores_df, skill_df = data_simulator.generate_data()
        stage_timing.append(("data_simulation", time.perf_counter() - t0))

        summary_lines.append("== Data Simulation ==")
        summary_lines.append(
            f"N_BOOKS={config.N_BOOKS}, N_MODELS={config.N_MODELS}, N_ROUNDS={config.N_ROUNDS}, N_SKILLS={config.N_SKILLS}"
        )
        summary_lines.append(f"Scores rows: {len(scores_df)}")
        summary_lines.append(f"Skill rows: {len(skill_df)}")

        t1 = time.perf_counter()
        reliable_models, icc_df, diag = layer1.filter_reliable_models(scores_df)
        stage_timing.append(("layer1_reliability", time.perf_counter() - t1))

        icc_path = visualizations.plot_icc_bar(icc_df, output_dir=output_dirs["layer1"])
        chart_paths.append(icc_path)
        missing_path = visualizations.plot_missing_heatmap(
            diag.get("missing_profile", pd.DataFrame()),
            output_dir=output_dirs["layer1"],
        )
        if missing_path:
            chart_paths.append(missing_path)

        summary_lines.append("\n== Layer1 Reliability ==")
        summary_lines.append("ICC summary:")
        summary_lines.append(icc_df.to_string(index=False))
        summary_lines.append("Distribution diagnostics:")
        summary_lines.append(diag["distribution_table"].to_string(index=False))
        if "status_table" in diag:
            summary_lines.append("Status table:")
            summary_lines.append(diag["status_table"].to_string(index=False))
        if "parallel_log" in diag:
            p_log = diag["parallel_log"]
            summary_lines.append("Parallel ICC log:")
            summary_lines.append(
                (
                    f"models={p_log.get('model_count')}, backend={p_log.get('backend')}, "
                    f"workers={p_log.get('workers')}, parallel_elapsed_sec={p_log.get('parallel_elapsed_sec', 0.0):.3f}, "
                    f"estimated_serial_sec={p_log.get('estimated_serial_sec', 0.0):.3f}, "
                    f"speedup={p_log.get('speedup', 0.0):.2f}"
                )
            )
        if diag["warnings"]:
            summary_lines.extend([f"Warning: {w}" for w in diag["warnings"]])
        summary_lines.append(f"Reliable models: {reliable_models}")
        summary_lines.append(f"Excluded models: {diag['excluded_models']}")
        summary_lines.append(f"Excluded no ICC: {diag.get('excluded_no_icc', [])}")

        report_path = output_dirs["reports"] / "summary_report.txt"
        if len(reliable_models) == 0:
            summary_lines.append("No model passed layer1. Pipeline stopped.")
            summary_lines.append("\n== Stage Timing (sec) ==")
            summary_lines.extend([f"{name}: {sec:.3f}" for name, sec in stage_timing])
            report_path.write_text("\n".join(summary_lines), encoding="utf-8")
            print(
                "No model passed layer1 reliability screening. See results/reports/summary_report.txt"
            )
            return report_path

        t2 = time.perf_counter()
        layer2_result = layer2.find_consensus_subset(scores_df, reliable_models)
        stage_timing.append(("layer2_consensus", time.perf_counter() - t2))

        greedy_subset = layer2_result["consensus_models"]
        exhaustive_best = layer2_result["exhaustive_best"]
        iteration_log = layer2_result["iteration_log"]

        mds_path = visualizations.plot_mds(
            layer2_result["mds_coords"], output_dir=output_dirs["layer2"]
        )
        if mds_path:
            chart_paths.append(mds_path)

        log_df = _safe_log_df(iteration_log)
        w_path = visualizations.plot_w_trajectory(
            log_df, output_dir=output_dirs["layer2"]
        )
        if w_path:
            chart_paths.append(w_path)

        summary_lines.append("\n== Layer2 Consensus ==")
        summary_lines.append(
            f"Greedy result: models={greedy_subset}, W={layer2_result['final_w']:.3f}, p={layer2_result['final_p']:.6f}"
        )
        summary_lines.append(
            "Exhaustive best: "
            f"models={exhaustive_best['models']}, size={exhaustive_best['size']}, "
            f"W={exhaustive_best['w']:.3f}, mean_rho={exhaustive_best['mean_rho']:.3f}, "
            f"method={exhaustive_best.get('method', 'n/a')}, evaluated={exhaustive_best.get('evaluated', 'n/a')}"
        )
        if not log_df.empty:
            summary_lines.append("Greedy iterations:")
            summary_lines.append(log_df.to_string(index=False))

        consensus_models = greedy_subset
        theta_con = float(layer2_result.get("theta_con", config.THETA_CON))
        if not consensus_models and exhaustive_best["size"] >= 3:
            if float(exhaustive_best.get("w", 0.0)) >= theta_con:
                consensus_models = exhaustive_best["models"]
                summary_lines.append(
                    "Greedy subset did not meet threshold; fallback to exhaustive best subset."
                )
            else:
                summary_lines.append(
                    "Greedy subset did not meet threshold; exhaustive best subset also below threshold."
                )

        if len(consensus_models) == 0:
            summary_lines.append(
                "Consensus subset empty. Pipeline stopped before decision matrix."
            )
            summary_lines.append("\n== Stage Timing (sec) ==")
            summary_lines.extend([f"{name}: {sec:.3f}" for name, sec in stage_timing])
            report_path.write_text("\n".join(summary_lines), encoding="utf-8")
            print("No valid consensus subset. See results/reports/summary_report.txt")
            return report_path

        t3 = time.perf_counter()
        percentile_df = layer3.generate_decision_matrix(
            scores_df=scores_df,
            consensus_models=consensus_models,
            scheme_name=config.DEFAULT_PERCENTILE_SCHEME,
        )
        density_df = density_quadrant.generate_density_decision_matrix(
            scores_df=scores_df,
            consensus_models=consensus_models,
            scheme_name=config.DEFAULT_PERCENTILE_SCHEME,
        )

        if config.DEFAULT_DECISION_METHOD == "density":
            decision_df = density_df
            summary_lines.append("\n== Layer3 Decision Matrix ==")
            summary_lines.append("Primary method: density")
        else:
            decision_df = percentile_df
            summary_lines.append("\n== Layer3 Decision Matrix ==")
            summary_lines.append("Primary method: percentile")

        stage_timing.append(("layer3_decision", time.perf_counter() - t3))

        percentile_csv = output_dirs["layer3"] / "decision_matrix_percentile.csv"
        density_csv = output_dirs["layer3"] / "decision_matrix_density.csv"
        decision_csv = output_dirs["layer3"] / "decision_matrix_primary.csv"
        percentile_df.to_csv(percentile_csv, index=False, encoding="utf-8-sig")
        density_df.to_csv(density_csv, index=False, encoding="utf-8-sig")
        decision_df.to_csv(decision_csv, index=False, encoding="utf-8-sig")

        percentile_scatter = visualizations.plot_decision_scatter(
            percentile_df,
            output_dir=output_dirs["layer3"],
            file_name="layer3_decision_scatter_percentile.png",
            title="Layer3 Percentile Decision Matrix",
        )
        density_scatter = visualizations.plot_decision_scatter(
            density_df,
            output_dir=output_dirs["layer3"],
            file_name="layer3_decision_scatter_density.png",
            title="Layer3 Density-Based Decision Matrix",
        )
        density_compare = visualizations.plot_density_quadrant_comparison(
            percentile_df=percentile_df,
            density_df=density_df,
            output_dir=output_dirs["layer3"],
        )

        for path in [percentile_scatter, density_scatter, density_compare]:
            if path:
                chart_paths.append(path)

        quadrant_counts = decision_df["quadrant"].value_counts().sort_index()
        summary_lines.append(f"Consensus models used: {consensus_models}")
        summary_lines.append("Quadrant counts:")
        summary_lines.append(quadrant_counts.to_string())
        if "data_insufficient" in decision_df.columns:
            insufficient_n = int(decision_df["data_insufficient"].sum())
            summary_lines.append(f"Data-insufficient books: {insufficient_n}")
        summary_lines.append(f"Primary decision CSV: {decision_csv}")
        summary_lines.append(f"Percentile decision CSV: {percentile_csv}")
        summary_lines.append(f"Density decision CSV: {density_csv}")

        t3b = time.perf_counter()
        bootstrap_result = bootstrap_stability.run_bootstrap_stability_analysis(
            scores_df=scores_df,
            consensus_models=consensus_models,
            density_decision_df=density_df,
            output_dir=output_dirs["bootstrap"],
            report_dir=output_dirs["reports"],
        )
        stage_timing.append(("bootstrap_stability", time.perf_counter() - t3b))
        summary_lines.append("\n== Bootstrap Stability ==")
        summary_lines.append(f"Bootstrap report: {bootstrap_result.get('report_path')}")
        scheme_summary = bootstrap_result.get("scheme_summary", pd.DataFrame())
        if isinstance(scheme_summary, pd.DataFrame) and not scheme_summary.empty:
            summary_lines.append("Scheme summary:")
            summary_lines.append(scheme_summary.to_string(index=False))
        density_overlap_bootstrap = bootstrap_result.get("density_overlap", {})
        if density_overlap_bootstrap:
            summary_lines.append("Density controversy vs instability:")
            summary_lines.append(
                ", ".join(
                    [
                        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in density_overlap_bootstrap.items()
                    ]
                )
            )
        chart_paths.extend(
            [str(path) for path in bootstrap_result.get("chart_paths", [])]
        )

        density_overlap = density_quadrant.summarize_density_overlap(
            percentile_df=percentile_df,
            density_df=density_df,
            low_stability_books=bootstrap_result.get("low_stability_books", []),
        )
        summary_lines.append("\n== Density vs Percentile Overlap ==")
        summary_lines.append(
            ", ".join(
                [
                    f"{key}={value:.3f}"
                    if isinstance(value, float)
                    else f"{key}={value}"
                    for key, value in density_overlap.items()
                ]
            )
        )

        disputed = decision_df[
            decision_df["action_label"].isin(
                ["重点评估/专家介入", "暂缓决策，补充信息"]
            )
        ].copy()

        summary_lines.append("\n== Layer4 Skill Diagnosis ==")
        if disputed.empty:
            summary_lines.append("No disputed books for layer4 diagnosis.")
        else:
            selected_disputed = disputed.head(max(2, min(len(disputed), 10)))
            t4 = time.perf_counter()
            diagnosis_df, text_reports, radar_paths = layer4.skill_diagnosis(
                selected_disputed,
                skill_df,
                consensus_models,
                results_dir=output_dirs["layer4"],
            )
            stage_timing.append(("layer4_skill_diagnosis", time.perf_counter() - t4))

            if diagnosis_df.empty:
                summary_lines.append("No valid skill diagnosis output generated.")
            else:
                diagnosis_csv = output_dirs["layer4"] / "layer4_skill_diagnosis.csv"
                diagnosis_df.to_csv(diagnosis_csv, index=False, encoding="utf-8-sig")
                summary_lines.append(
                    f"Diagnosed books: {sorted(diagnosis_df['book_id'].unique().tolist())}"
                )
                summary_lines.extend(text_reports)
                summary_lines.append(f"Diagnosis CSV: {diagnosis_csv}")
                chart_paths.extend(radar_paths)

        t5 = time.perf_counter()
        sensitivity_result = threshold_sensitivity.run_sensitivity_analysis(
            scores_df=scores_df,
            reliable_models=reliable_models,
            output_dir=output_dirs["sensitivity"],
            report_dir=output_dirs["reports"],
        )
        stage_timing.append(("threshold_sensitivity", time.perf_counter() - t5))

        sensitivity_report_path = sensitivity_result.get("report_path")
        if sensitivity_report_path:
            summary_lines.append("\n== Sensitivity Analysis ==")
            summary_lines.append(f"Sensitivity report: {sensitivity_report_path}")
            key_metrics = sensitivity_result.get("key_metrics")
            if isinstance(key_metrics, pd.DataFrame) and not key_metrics.empty:
                summary_lines.append("Key metrics:")
                summary_lines.append(key_metrics.to_string(index=False))
            analysis_lines = sensitivity_result.get("analysis_lines", [])
            if analysis_lines:
                summary_lines.append("Summary:")
                summary_lines.extend([str(line) for line in analysis_lines])

        sensitivity_charts = sensitivity_result.get("chart_paths", [])
        if sensitivity_charts:
            chart_paths.extend([str(path) for path in sensitivity_charts])

        if enable_monte_carlo:
            t6 = time.perf_counter()
            mc_result = monte_carlo.run_monte_carlo_experiment(
                seeds=config.MONTE_CARLO_SEEDS,
                output_dir=output_dirs["monte_carlo"],
                report_dir=output_dirs["reports"],
            )
            stage_timing.append(("monte_carlo", time.perf_counter() - t6))
            summary_lines.append("\n== Monte Carlo ==")
            summary_lines.append(f"Monte Carlo report: {mc_result.get('report_path')}")
            mc_summary = mc_result.get("summary_df", pd.DataFrame())
            if isinstance(mc_summary, pd.DataFrame) and not mc_summary.empty:
                summary_lines.append(mc_summary.to_string(index=False))
            chart_paths.extend([str(path) for path in mc_result.get("chart_paths", [])])

        summary_lines.append("\n== Stage Timing (sec) ==")
        summary_lines.extend([f"{name}: {sec:.3f}" for name, sec in stage_timing])

        total_sec = sum(sec for _, sec in stage_timing)
        summary_lines.append(f"total_pipeline_time: {total_sec:.3f}")

        summary_lines.append("\n== Generated Charts ==")
        for path in chart_paths:
            summary_lines.append(path)

        report_path.write_text("\n".join(summary_lines), encoding="utf-8")

        print(
            "Pipeline completed. Summary report generated at results/reports/summary_report.txt"
        )
        print("Reliable models:", reliable_models)
        print("Consensus models:", consensus_models)
        print("Quadrant distribution:")
        print(quadrant_counts)
        return report_path

    except Exception as exc:  # pragma: no cover
        report_path = output_dirs["reports"] / "summary_report.txt"
        msg = f"Pipeline failed with error: {exc}\n\n{traceback.format_exc()}"
        report_path.write_text(msg, encoding="utf-8")
        print(msg)
        return report_path


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and run the full pipeline."""
    args = build_parser().parse_args(argv)
    run_pipeline(enable_monte_carlo=bool(args.monte_carlo))