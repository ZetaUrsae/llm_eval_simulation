"""Compare ICC(2,1) vs ICC(3,1) impacts on layer1-3 pipeline outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, data_simulator, layer1
from . import layer2_consensus as layer2
from . import layer3_decision as layer3


def _plot_icc_diff_distribution(model_cmp_df: pd.DataFrame, output_path: Path) -> str:
    """Plot ICC(2,1)-ICC(3,1) distribution across models."""
    diffs = model_cmp_df["icc_diff_2_1_minus_3_1"].dropna()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if diffs.empty:
        ax.text(0.5, 0.5, "No valid ICC differences", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(
            diffs,
            bins=min(24, max(8, int(np.sqrt(len(diffs))))),
            color="#4c78a8",
            alpha=0.85,
        )
        ax.axvline(
            float(diffs.mean()),
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"mean={float(diffs.mean()):.6f}",
        )
        ax.set_xlabel("ICC(2,1) - ICC(3,1)")
        ax.set_ylabel("Model Count")
        ax.set_title("Distribution of ICC Differences Across Models")
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def _resolve_consensus_models(
    layer2_result: Dict[str, object],
) -> tuple[List[int], float]:
    """Resolve consensus subset with the same fallback logic used by main pipeline."""
    consensus_models = [int(mid) for mid in layer2_result.get("consensus_models", [])]
    final_w = float(layer2_result.get("final_w", 0.0))
    if consensus_models:
        return consensus_models, final_w

    exhaustive_best = layer2_result.get("exhaustive_best", {})
    if int(exhaustive_best.get("size", 0)) >= 3:
        models = [int(mid) for mid in exhaustive_best.get("models", [])]
        w_val = float(exhaustive_best.get("w", 0.0))
        return models, w_val

    return [], final_w


def _run_layer123_pipeline(
    scores_df: pd.DataFrame,
    icc_type: str,
) -> Dict[str, object]:
    """Run layer1-3 once under a chosen ICC type."""
    reliable_models, icc_df, diagnostics = layer1.filter_reliable_models(
        scores_df=scores_df,
        icc_type=icc_type,
    )

    layer2_result = layer2.find_consensus_subset(
        scores_df=scores_df,
        reliable_models=reliable_models,
    )
    consensus_models, final_w = _resolve_consensus_models(layer2_result)

    decision_df = layer3.generate_decision_matrix(
        scores_df=scores_df,
        consensus_models=consensus_models,
        scheme_name=config.DEFAULT_PERCENTILE_SCHEME,
    )
    high_high_n = int(
        (decision_df.get("quadrant", pd.Series(dtype=str)) == "高分_高共识").sum()
    )

    return {
        "icc_type": icc_type,
        "reliable_models": reliable_models,
        "icc_df": icc_df.copy(),
        "diagnostics": diagnostics,
        "layer2_result": layer2_result,
        "consensus_models": consensus_models,
        "final_w": final_w,
        "decision_df": decision_df,
        "high_high_count": high_high_n,
    }


def _build_model_level_comparison(
    run_21: Dict[str, object],
    run_31: Dict[str, object],
) -> pd.DataFrame:
    """Build per-model ICC value comparison table."""
    left = run_21["icc_df"][
        ["model_id", "icc_value", "ci95_low", "ci95_high", "status"]
    ].copy()
    right = run_31["icc_df"][
        ["model_id", "icc_value", "ci95_low", "ci95_high", "status"]
    ].copy()
    left = left.rename(
        columns={
            "icc_value": "icc_2_1",
            "ci95_low": "icc_2_1_ci95_low",
            "ci95_high": "icc_2_1_ci95_high",
            "status": "status_2_1",
        }
    )
    right = right.rename(
        columns={
            "icc_value": "icc_3_1",
            "ci95_low": "icc_3_1_ci95_low",
            "ci95_high": "icc_3_1_ci95_high",
            "status": "status_3_1",
        }
    )

    merged = left.merge(right, on="model_id", how="outer").sort_values("model_id")
    merged["icc_diff_2_1_minus_3_1"] = merged["icc_2_1"] - merged["icc_3_1"]

    pass_states = {"retained", "boundary_retained"}
    merged["pass_2_1"] = merged["status_2_1"].isin(pass_states)
    merged["pass_3_1"] = merged["status_3_1"].isin(pass_states)
    merged["pass_mismatch"] = merged["pass_2_1"] != merged["pass_3_1"]
    merged["boundary_model"] = (merged["status_2_1"] == "boundary_retained") | (
        merged["status_3_1"] == "boundary_retained"
    )

    return merged.reset_index(drop=True)


def _build_layer23_comparison(
    run_21: Dict[str, object], run_31: Dict[str, object]
) -> pd.DataFrame:
    """Build a compact layer2/layer3 metric comparison table."""
    rows = [
        {
            "icc_type": "ICC(2,1)",
            "layer1_passed_models": len(run_21["reliable_models"]),
            "layer2_consensus_size": len(run_21["consensus_models"]),
            "layer2_final_w": float(run_21["final_w"]),
            "layer3_high_high_count": int(run_21["high_high_count"]),
        },
        {
            "icc_type": "ICC(3,1)",
            "layer1_passed_models": len(run_31["reliable_models"]),
            "layer2_consensus_size": len(run_31["consensus_models"]),
            "layer2_final_w": float(run_31["final_w"]),
            "layer3_high_high_count": int(run_31["high_high_count"]),
        },
    ]
    df = pd.DataFrame(rows)
    diff_row = {
        "icc_type": "DIFF(2,1)-(3,1)",
        "layer1_passed_models": int(
            df.loc[0, "layer1_passed_models"] - df.loc[1, "layer1_passed_models"]
        ),
        "layer2_consensus_size": int(
            df.loc[0, "layer2_consensus_size"] - df.loc[1, "layer2_consensus_size"]
        ),
        "layer2_final_w": float(
            df.loc[0, "layer2_final_w"] - df.loc[1, "layer2_final_w"]
        ),
        "layer3_high_high_count": int(
            df.loc[0, "layer3_high_high_count"] - df.loc[1, "layer3_high_high_count"]
        ),
    }
    return pd.concat([df, pd.DataFrame([diff_row])], ignore_index=True)


def run_icc_comparison_experiment(seed: int = 42) -> Dict[str, object]:
    """Run ICC(2,1) vs ICC(3,1) comparison on the same simulated score data."""
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "results" / "icc_comparison"
    reports_dir = project_root / "results" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    scores_df, _ = data_simulator.generate_data(seed=int(seed))

    run_21 = _run_layer123_pipeline(scores_df=scores_df, icc_type="ICC(2,1)")
    run_31 = _run_layer123_pipeline(scores_df=scores_df, icc_type="ICC(3,1)")

    model_cmp_df = _build_model_level_comparison(run_21, run_31)
    layer23_cmp_df = _build_layer23_comparison(run_21, run_31)
    mismatch_df = model_cmp_df[model_cmp_df["pass_mismatch"]].copy()

    valid_diffs = model_cmp_df["icc_diff_2_1_minus_3_1"].dropna()
    desc = {
        "icc_2_1_mean": float(model_cmp_df["icc_2_1"].mean(skipna=True)),
        "icc_2_1_std": float(model_cmp_df["icc_2_1"].std(skipna=True, ddof=0)),
        "icc_3_1_mean": float(model_cmp_df["icc_3_1"].mean(skipna=True)),
        "icc_3_1_std": float(model_cmp_df["icc_3_1"].std(skipna=True, ddof=0)),
        "diff_mean": float(valid_diffs.mean())
        if not valid_diffs.empty
        else float("nan"),
        "diff_std": float(valid_diffs.std(ddof=0))
        if not valid_diffs.empty
        else float("nan"),
        "max_abs_diff": float(valid_diffs.abs().max())
        if not valid_diffs.empty
        else float("nan"),
    }

    model_csv = out_dir / "model_icc_comparison.csv"
    mismatch_csv = out_dir / "layer1_pass_mismatch.csv"
    layer23_csv = out_dir / "layer23_metrics_comparison.csv"
    chart_path = out_dir / "icc_diff_distribution.png"
    model_cmp_df.to_csv(model_csv, index=False, encoding="utf-8-sig")
    mismatch_df.to_csv(mismatch_csv, index=False, encoding="utf-8-sig")
    layer23_cmp_df.to_csv(layer23_csv, index=False, encoding="utf-8-sig")
    chart_file = _plot_icc_diff_distribution(
        model_cmp_df=model_cmp_df, output_path=chart_path
    )

    analysis_lines: List[str] = []
    analysis_lines.append(
        f"Layer1 pass count: ICC(2,1)={len(run_21['reliable_models'])}, ICC(3,1)={len(run_31['reliable_models'])}."
    )
    analysis_lines.append(
        f"Layer1 pass mismatch models: {mismatch_df['model_id'].astype(int).tolist()}."
    )
    analysis_lines.append(
        f"Layer2 consensus size: ICC(2,1)={len(run_21['consensus_models'])}, ICC(3,1)={len(run_31['consensus_models'])}."
    )
    analysis_lines.append(
        f"Layer2 final W: ICC(2,1)={run_21['final_w']:.6f}, ICC(3,1)={run_31['final_w']:.6f}."
    )
    analysis_lines.append(
        f"Layer3 high-score/high-consensus count: ICC(2,1)={run_21['high_high_count']}, ICC(3,1)={run_31['high_high_count']}."
    )

    report_lines: List[str] = ["== ICC Comparison Experiment Report =="]
    report_lines.append(
        f"seed={seed}, books={config.N_BOOKS}, models={config.N_MODELS}, same_scores_data=True"
    )
    report_lines.append("\n-- Descriptive Statistics for ICC Values --")
    report_lines.append(
        ", ".join(
            [
                f"icc_2_1_mean={desc['icc_2_1_mean']:.6f}",
                f"icc_2_1_std={desc['icc_2_1_std']:.6f}",
                f"icc_3_1_mean={desc['icc_3_1_mean']:.6f}",
                f"icc_3_1_std={desc['icc_3_1_std']:.6f}",
                f"diff_mean={desc['diff_mean']:.6f}",
                f"diff_std={desc['diff_std']:.6f}",
                f"max_abs_diff={desc['max_abs_diff']:.6f}",
            ]
        )
    )

    report_lines.append(
        "\n-- Layer1 Screening Difference Table (including boundary flag) --"
    )
    show_cols = [
        "model_id",
        "icc_2_1",
        "icc_3_1",
        "icc_diff_2_1_minus_3_1",
        "status_2_1",
        "status_3_1",
        "boundary_model",
        "pass_mismatch",
    ]
    report_lines.append(model_cmp_df[show_cols].to_string(index=False))

    report_lines.append("\n-- Layer2 and Layer3 Key Metric Comparison --")
    report_lines.append(layer23_cmp_df.to_string(index=False))

    report_lines.append("\n-- Short Analysis Summary --")
    report_lines.extend([f"- {line}" for line in analysis_lines])

    report_lines.append("\n-- Output Files --")
    report_lines.append(str(model_csv))
    report_lines.append(str(mismatch_csv))
    report_lines.append(str(layer23_csv))
    report_lines.append(str(chart_file))

    report_path = reports_dir / "icc_comparison_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "report_path": str(report_path),
        "model_comparison_csv": str(model_csv),
        "mismatch_csv": str(mismatch_csv),
        "layer23_comparison_csv": str(layer23_csv),
        "chart_path": str(chart_file),
        "layer23_df": layer23_cmp_df,
        "mismatch_df": mismatch_df,
    }


def main() -> None:
    """Entrypoint for command: python -m src.icc_comparison_experiment."""
    result = run_icc_comparison_experiment(seed=42)
    print("ICC comparison experiment completed.")
    print(f"Report: {result['report_path']}")


if __name__ == "__main__":
    main()
