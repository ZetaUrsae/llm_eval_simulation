"""Small-scale scenario experiments for the four-layer evaluation framework.

This module adds three practical scenarios with 3/6/9 models while keeping
10,000 books and the same latent quality generation logic as the main run.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src import config, layer1, layer2, layer3, layer4, visualizations
from src import data_simulator as ds


@dataclass(frozen=True)
class ScenarioModelSpec:
    """One model setting in a small-scale scenario."""

    model_id: int
    noise_std: float
    bias: float
    scale: float
    reversal_fraction: float = 0.0
    reversal_mode: str = "none"
    note: str = ""


@dataclass(frozen=True)
class ScenarioConfig:
    """Scenario-level definition and expectations."""

    scenario_id: int
    name: str
    research_question: str
    models: Tuple[ScenarioModelSpec, ...]
    expected_layer1_excluded: Tuple[int, ...]
    expected_layer2_disruptor: int | None = None


def _build_scenarios() -> List[ScenarioConfig]:
    """Create the 3/6/9-model small-scale experiment scenarios."""
    scenario1 = ScenarioConfig(
        scenario_id=1,
        name="3-model minimum viable setup",
        research_question=(
            "After removing low-reliability model(s), does the framework trigger "
            "fallback behavior when layer2 needs >=3 models?"
        ),
        models=(
            ScenarioModelSpec(
                0, noise_std=3.0, bias=0.0, scale=1.0, note="high reliability baseline"
            ),
            ScenarioModelSpec(
                1, noise_std=6.0, bias=0.0, scale=1.0, note="medium reliability"
            ),
            ScenarioModelSpec(
                2,
                noise_std=14.0,
                bias=0.0,
                scale=1.0,
                note="low reliability; expected layer1 exclusion",
            ),
        ),
        expected_layer1_excluded=(2,),
    )

    scenario2 = ScenarioConfig(
        scenario_id=2,
        name="6-model common setup",
        research_question=(
            "Can layer1 remove low-reliability model first, then layer2 greedy "
            "identify the directed disruptor?"
        ),
        models=(
            ScenarioModelSpec(
                0, noise_std=3.0, bias=0.0, scale=1.0, note="high reliability baseline"
            ),
            ScenarioModelSpec(
                1, noise_std=4.0, bias=0.0, scale=1.0, note="high reliability baseline"
            ),
            ScenarioModelSpec(
                2, noise_std=6.0, bias=0.0, scale=1.0, note="medium reliability"
            ),
            ScenarioModelSpec(
                3,
                noise_std=14.0,
                bias=0.0,
                scale=1.0,
                note="low reliability; expected layer1 exclusion",
            ),
            ScenarioModelSpec(
                4,
                noise_std=4.0,
                bias=0.0,
                scale=1.0,
                reversal_fraction=0.3,
                reversal_mode="directed_top",
                note="directed disruptor; expected layer2 exclusion",
            ),
            ScenarioModelSpec(
                5,
                noise_std=4.0,
                bias=3.0,
                scale=1.0,
                note="normal model with mild positive bias",
            ),
        ),
        expected_layer1_excluded=(3,),
        expected_layer2_disruptor=4,
    )

    scenario3 = ScenarioConfig(
        scenario_id=3,
        name="9-model sufficient setup",
        research_question=(
            "With co-existing issues, can layer1/layer2 process them in order, "
            "while keeping scale-biased but directionally consistent model(s)?"
        ),
        models=(
            ScenarioModelSpec(
                0, noise_std=3.0, bias=0.0, scale=1.0, note="high reliability baseline"
            ),
            ScenarioModelSpec(
                1, noise_std=3.5, bias=1.0, scale=1.0, note="high reliability baseline"
            ),
            ScenarioModelSpec(
                2, noise_std=4.0, bias=2.0, scale=1.0, note="high reliability baseline"
            ),
            ScenarioModelSpec(
                3, noise_std=6.0, bias=0.0, scale=1.0, note="medium reliability"
            ),
            ScenarioModelSpec(
                4,
                noise_std=14.0,
                bias=0.0,
                scale=1.0,
                note="low reliability; expected layer1 exclusion",
            ),
            ScenarioModelSpec(
                5,
                noise_std=4.0,
                bias=0.0,
                scale=1.0,
                reversal_fraction=0.3,
                reversal_mode="directed_top",
                note="directed disruptor; expected layer2 exclusion",
            ),
            ScenarioModelSpec(
                6, noise_std=3.5, bias=-10.0, scale=0.7, note="scale-biased model"
            ),
            ScenarioModelSpec(
                7, noise_std=4.0, bias=2.0, scale=1.0, note="normal model"
            ),
            ScenarioModelSpec(
                8, noise_std=4.0, bias=3.0, scale=1.0, note="normal model"
            ),
        ),
        expected_layer1_excluded=(4,),
        expected_layer2_disruptor=5,
    )
    return [scenario1, scenario2, scenario3]


def _build_directional_reversal_mask(
    true_quality: np.ndarray,
    reversal_fraction: float,
    direction: str = "high",
) -> np.ndarray:
    """Build a deterministic directional reversal mask by quality ordering."""
    n_books = len(true_quality)
    n_pick = int(round(n_books * float(reversal_fraction)))
    n_pick = max(0, min(n_books, n_pick))
    if n_pick == 0:
        return np.zeros(n_books, dtype=bool)

    order = np.argsort(true_quality)
    if direction == "high":
        idx = order[-n_pick:]
    else:
        idx = order[:n_pick]
    mask = np.zeros(n_books, dtype=bool)
    mask[idx] = True
    return mask


def _simulate_scores_for_scenario(
    scenario: ScenarioConfig,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate scores/skills using the same latent generation logic as main run."""
    rng = np.random.default_rng(int(seed))

    n_books = int(config.N_BOOKS)
    book_ids = np.arange(n_books)
    true_quality = ds._sample_true_quality(rng)

    sigma_dim = rng.uniform(4.0, 12.0, size=config.N_SKILLS)
    dim_offsets = rng.normal(
        loc=0.0,
        scale=sigma_dim,
        size=(n_books, config.N_SKILLS),
    )
    true_dims = np.clip(true_quality[:, None] + dim_offsets, 0.0, 100.0)

    score_frames: List[pd.DataFrame] = []
    skill_frames: List[pd.DataFrame] = []

    for spec in scenario.models:
        model_profile = ds.ModelProfile(
            noise_std=float(spec.noise_std),
            bias=float(spec.bias),
            scale=float(spec.scale),
            reversal_fraction=float(spec.reversal_fraction),
            reversal_mode=str(spec.reversal_mode),
            missing_mode="none",
            missing_fraction=0.0,
            hetero_mode="none",
            round2_drift=0.0,
            skill_weights=np.ones(config.N_SKILLS, dtype=np.float32),
        )

        if model_profile.reversal_mode == "directed_top":
            reversal_mask = _build_directional_reversal_mask(
                true_quality=true_quality,
                reversal_fraction=model_profile.reversal_fraction,
                direction="high",
            )
        else:
            reversal_mask = ds._build_reversal_mask(
                rng,
                model_profile.reversal_mode,
                model_profile.reversal_fraction,
                true_quality,
            )

        mapped_quality = np.where(reversal_mask, 100.0 - true_quality, true_quality)

        round_scores = []
        for round_idx in range(1, config.N_ROUNDS + 1):
            noise = ds._heteroscedastic_noise(
                rng,
                model_profile.noise_std,
                true_quality,
                model_profile.hetero_mode,
            )
            drift = model_profile.round2_drift if round_idx == 2 else 0.0
            base = mapped_quality + model_profile.bias + drift
            scaled_base = (base - 50.0) * model_profile.scale + 50.0
            score = np.clip(scaled_base + noise, 0.0, 100.0)
            round_scores.append(score.astype(np.float32))

        score_r1, score_r2 = ds._apply_missing(
            rng,
            round_scores[0].copy(),
            round_scores[1].copy(),
            model_profile.missing_mode,
            model_profile.missing_fraction,
        )

        for round_idx, score in enumerate([score_r1, score_r2], start=1):
            score_frames.append(
                pd.DataFrame(
                    {
                        "book_id": book_ids.astype(np.int32),
                        "model_id": np.full(
                            n_books, int(spec.model_id), dtype=np.int16
                        ),
                        "round": np.full(n_books, round_idx, dtype=np.int8),
                        "score": score,
                    }
                )
            )

        for skill_idx, skill_name in enumerate(config.SKILLS):
            skill_noise = rng.normal(0.0, 2.0, size=n_books)
            skill_score = np.clip(
                true_dims[:, skill_idx] * model_profile.skill_weights[skill_idx]
                + skill_noise,
                0.0,
                100.0,
            )
            skill_frames.append(
                pd.DataFrame(
                    {
                        "book_id": book_ids.astype(np.int32),
                        "model_id": np.full(
                            n_books, int(spec.model_id), dtype=np.int16
                        ),
                        "skill": np.full(n_books, skill_name),
                        "score": skill_score.astype(np.float32),
                    }
                )
            )

    scores_df = pd.concat(score_frames, ignore_index=True)
    scores_df["model_group"] = scores_df["model_id"].astype(int).map(config.model_group)

    skill_df = pd.concat(skill_frames, ignore_index=True)
    skill_df["model_group"] = skill_df["model_id"].astype(int).map(config.model_group)
    return scores_df, skill_df


def _safe_literal_eval_list(raw: str) -> List[int]:
    """Parse list-like text safely; return empty list on failure."""
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    result: List[int] = []
    for item in parsed:
        try:
            result.append(int(item))
        except Exception:
            continue
    return result


def _load_large_scale_baseline(project_root: Path) -> Dict[str, object]:
    """Load 100-model baseline metrics from existing main-experiment artifacts."""
    summary_path = project_root / "results" / "reports" / "summary_report.txt"
    decision_path = project_root / "results" / "layer3" / "decision_matrix_primary.csv"
    if not summary_path.exists():
        return {}

    text = summary_path.read_text(encoding="utf-8")

    reliable_match = re.search(r"Reliable models:\s*(\[.*?\])", text)
    consensus_match = re.search(r"Consensus models used:\s*(\[.*?\])", text)
    excluded_match = re.search(r"Excluded models:\s*(\[.*?\])", text)
    w_match = re.search(r"Greedy result:.*?W=([0-9.]+)", text)

    reliable_models = _safe_literal_eval_list(
        reliable_match.group(1) if reliable_match else "[]"
    )
    consensus_models = _safe_literal_eval_list(
        consensus_match.group(1) if consensus_match else "[]"
    )
    excluded_models = _safe_literal_eval_list(
        excluded_match.group(1) if excluded_match else "[]"
    )

    direct_accept_count = 0
    if decision_path.exists():
        ddf = pd.read_csv(decision_path)
        if "quadrant" in ddf.columns:
            direct_accept_count = int((ddf["quadrant"] == "高分_高共识").sum())

    layer2_removed = sorted(set(reliable_models) - set(consensus_models))
    return {
        "scenario": "Main-100",
        "total_models": 100,
        "passed_layer1": len(reliable_models),
        "consensus_models": len(consensus_models),
        "final_w": float(w_match.group(1)) if w_match else float("nan"),
        "recommended_count": direct_accept_count,
        "removed_models": {
            "layer1": excluded_models,
            "layer2": layer2_removed,
        },
    }


def _scenario_report_path(results_dir: Path, scenario_id: int) -> Path:
    return results_dir / f"small_scale_scenario{scenario_id}_report.txt"


def _run_one_scenario(
    scenario: ScenarioConfig,
    output_dir: Path,
    seed: int,
) -> Dict[str, object]:
    """Run one small-scale scenario and save report/csv/plot artifacts."""
    scores_df, skill_df = _simulate_scores_for_scenario(scenario=scenario, seed=seed)

    reliable_models, icc_df, diagnostics = layer1.filter_reliable_models(scores_df)
    status_df = diagnostics.get("status_table", pd.DataFrame()).copy()

    layer1_excluded = []
    if not status_df.empty:
        layer1_excluded = (
            status_df[~status_df["status"].isin(["retained", "boundary_retained"])][
                "model_id"
            ]
            .astype(int)
            .tolist()
        )

    layer2_removed: List[int] = []
    layer2_removed_reason = ""
    layer2_skipped = False
    layer2_fallback = False
    final_w = float("nan")

    if len(reliable_models) >= 3:
        layer2_result = layer2.find_consensus_subset(
            scores_df=scores_df, reliable_models=reliable_models
        )
        consensus_models = list(layer2_result.get("consensus_models", []))
        exhaustive_best = layer2_result.get("exhaustive_best", {})
        if (not consensus_models) and int(exhaustive_best.get("size", 0)) >= 3:
            consensus_models = [int(mid) for mid in exhaustive_best.get("models", [])]
            layer2_fallback = True
            final_w = float(exhaustive_best.get("w", float("nan")))
        else:
            final_w = float(layer2_result.get("final_w", float("nan")))

        iter_log = pd.DataFrame(layer2_result.get("iteration_log", []))
        if not iter_log.empty and "removed_model" in iter_log.columns:
            layer2_removed = (
                iter_log[iter_log["removed_model"].notna()]["removed_model"]
                .astype(int)
                .tolist()
            )
            if layer2_removed:
                layer2_removed_reason = "greedy_remove_low_mean_rho_high_variance"
    else:
        # Fallback for practical minimum setup: skip standard layer2 and use
        # layer1-retained models for layer3 decision projection.
        layer2_skipped = True
        layer2_fallback = True
        consensus_models = list(reliable_models)

    decision_df = layer3.generate_decision_matrix(
        scores_df=scores_df,
        consensus_models=consensus_models,
        scheme_name=config.DEFAULT_PERCENTILE_SCHEME,
    )

    decision_csv_path = (
        output_dir / f"small_scale_scenario{scenario.scenario_id}_decision_matrix.csv"
    )
    decision_df.to_csv(decision_csv_path, index=False, encoding="utf-8-sig")

    scatter_path = visualizations.plot_decision_scatter(
        decision_df=decision_df,
        output_dir=output_dir,
        file_name=f"small_scale_scenario{scenario.scenario_id}_decision_scatter.png",
        title=f"Small Scale Scenario {scenario.scenario_id} Decision Matrix",
    )

    quadrant_counts = (
        decision_df["quadrant"].value_counts().sort_index()
        if not decision_df.empty
        else pd.Series(dtype=int)
    )
    recommended_count = int(quadrant_counts.get("高分_高共识", 0))

    layer4_info = ""
    if (not layer2_skipped) and consensus_models:
        disputed = decision_df[
            decision_df["action_label"].isin(
                ["重点评估/专家介入", "暂缓决策，补充信息"]
            )
        ].copy()
        if not disputed.empty:
            diagnosis_df, text_reports, _ = layer4.skill_diagnosis(
                disputed_books=disputed.head(max(2, min(len(disputed), 10))),
                skill_df=skill_df,
                consensus_models=consensus_models,
                results_dir=output_dir,
            )
            if not diagnosis_df.empty:
                diag_csv = (
                    output_dir
                    / f"small_scale_scenario{scenario.scenario_id}_layer4_diagnosis.csv"
                )
                diagnosis_df.to_csv(diag_csv, index=False, encoding="utf-8-sig")
                layer4_info = f"Diagnosed books: {sorted(diagnosis_df['book_id'].astype(int).unique().tolist())}"
                if text_reports:
                    layer4_info += "\n" + "\n".join(text_reports)
        else:
            layer4_info = "No disputed books for layer4 diagnosis."
    elif layer2_skipped:
        layer4_info = (
            "Layer4 skipped because layer2 was skipped under minimum-model fallback."
        )

    report_lines: List[str] = []
    report_lines.append(
        f"== Small Scale Scenario {scenario.scenario_id}: {scenario.name} =="
    )
    report_lines.append(
        f"Seed: {seed}, Books: {config.N_BOOKS}, Models: {len(scenario.models)}"
    )
    report_lines.append(f"Research question: {scenario.research_question}")

    report_lines.append("\n-- Model Configuration --")
    cfg_df = pd.DataFrame(
        [
            {
                "model_id": spec.model_id,
                "noise_std": spec.noise_std,
                "bias": spec.bias,
                "scale": spec.scale,
                "reversal_fraction": spec.reversal_fraction,
                "reversal_mode": spec.reversal_mode,
                "note": spec.note,
            }
            for spec in scenario.models
        ]
    )
    report_lines.append(cfg_df.to_string(index=False))

    report_lines.append("\n-- Layer1 ICC Screening --")
    report_lines.append(icc_df.to_string(index=False))
    report_lines.append(f"Retained models: {reliable_models}")
    report_lines.append(f"Excluded models: {layer1_excluded}")

    report_lines.append("\n-- Layer2 Consensus --")
    report_lines.append(f"Layer2 skipped: {layer2_skipped}")
    report_lines.append(f"Layer2 fallback used: {layer2_fallback}")
    report_lines.append(f"Layer2 removed models: {layer2_removed}")
    if layer2_removed_reason:
        report_lines.append(f"Layer2 remove reason: {layer2_removed_reason}")
    report_lines.append(f"Consensus models for decision: {consensus_models}")
    report_lines.append(f"Final W: {final_w if np.isfinite(final_w) else 'N/A'}")

    report_lines.append("\n-- Layer3 Decision Matrix --")
    report_lines.append("Quadrant counts:")
    report_lines.append(
        quadrant_counts.to_string()
        if not quadrant_counts.empty
        else "No quadrant output."
    )
    report_lines.append(
        f"Recommended (high score + high consensus): {recommended_count}"
    )

    report_lines.append("\n-- Layer4 Diagnosis --")
    report_lines.append(
        layer4_info if layer4_info else "Layer4 executed with no persistent output."
    )

    report_lines.append("\n-- Output Files --")
    report_lines.append(str(decision_csv_path))
    report_lines.append(str(scatter_path) if scatter_path else "No scatter generated.")

    scenario_report_path = _scenario_report_path(output_dir, scenario.scenario_id)
    scenario_report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.name,
        "total_models": len(scenario.models),
        "layer1_retained": len(reliable_models),
        "layer1_excluded": layer1_excluded,
        "layer2_skipped": layer2_skipped,
        "layer2_fallback": layer2_fallback,
        "layer2_removed": layer2_removed,
        "consensus_models": len(consensus_models),
        "final_w": final_w,
        "recommended_count": recommended_count,
        "quadrant_counts": quadrant_counts,
        "scenario_report": str(scenario_report_path),
        "decision_csv": str(decision_csv_path),
        "scatter": str(scatter_path) if scatter_path else "",
        "expected_layer1_excluded": list(scenario.expected_layer1_excluded),
        "expected_layer2_disruptor": scenario.expected_layer2_disruptor,
    }


def _make_comparison_report(
    project_root: Path,
    scenario_results: List[Dict[str, object]],
) -> Path:
    """Write cross-scenario summary and comparison with the 100-model baseline."""
    reports_dir = project_root / "results" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_large_scale_baseline(project_root)
    baseline_recommended = int(baseline.get("recommended_count", 0)) if baseline else 0

    rows = []
    for result in scenario_results:
        removed_desc = f"L1:{result['layer1_excluded']}"
        if result["layer2_removed"]:
            removed_desc += f"; L2:{result['layer2_removed']}(greedy)"
        if result["layer2_skipped"]:
            removed_desc += "; L2 skipped (fallback)"

        rows.append(
            {
                "scenario": f"S{result['scenario_id']}",
                "total_models": result["total_models"],
                "passed_layer1": result["layer1_retained"],
                "consensus_models": result["consensus_models"],
                "final_w": (
                    float(result["final_w"])
                    if np.isfinite(result["final_w"])
                    else np.nan
                ),
                "recommended_count": result["recommended_count"],
                "removed_models_reason": removed_desc,
            }
        )

    comparison_df = pd.DataFrame(rows)

    analysis_lines: List[str] = []
    analysis_lines.append("1) Layer1 screening accuracy in small-scale scenarios")
    for result in scenario_results:
        expected = set(int(v) for v in result["expected_layer1_excluded"])
        actual = set(int(v) for v in result["layer1_excluded"])
        ok = expected.issubset(actual)
        analysis_lines.append(
            f"- S{result['scenario_id']}: expected low-reliability exclusion {sorted(expected)}, actual {sorted(actual)}, matched={ok}."
        )

    analysis_lines.append("2) Directed disruptor detection under greedy layer2")
    for result in scenario_results:
        disruptor = result["expected_layer2_disruptor"]
        if disruptor is None:
            analysis_lines.append(
                f"- S{result['scenario_id']}: no layer2 disruptor configured (minimum setup)."
            )
            continue
        removed = set(int(v) for v in result["layer2_removed"])
        analysis_lines.append(
            f"- S{result['scenario_id']}: expected disruptor={disruptor}, removed_in_layer2={sorted(removed)}, detected={int(disruptor) in removed}."
        )

    analysis_lines.append(
        "3) Recommendation count comparability with main 100-model run"
    )
    if baseline:
        analysis_lines.append(
            f"- Main-100 recommended_count={baseline_recommended}, final_w={baseline.get('final_w')}"
        )
        for result in scenario_results:
            value = int(result["recommended_count"])
            ratio = (
                (value / baseline_recommended) if baseline_recommended else float("nan")
            )
            analysis_lines.append(
                f"- S{result['scenario_id']} recommended_count={value}, ratio_vs_main={ratio:.3f}."
            )
    else:
        analysis_lines.append(
            "- Main-100 baseline artifacts not found, recommendation comparability unavailable."
        )

    analysis_lines.append("4) Are small-vs-large differences within reasonable range?")
    if baseline:
        base_w = float(baseline.get("final_w", float("nan")))
        for result in scenario_results:
            w = (
                float(result["final_w"])
                if np.isfinite(result["final_w"])
                else float("nan")
            )
            if np.isfinite(w) and np.isfinite(base_w):
                delta = abs(w - base_w)
                analysis_lines.append(
                    f"- S{result['scenario_id']}: |W - W_main|={delta:.3f} (W={w:.3f}, W_main={base_w:.3f})."
                )
            else:
                analysis_lines.append(
                    f"- S{result['scenario_id']}: W not directly comparable (fallback path or missing value)."
                )
    else:
        analysis_lines.append("- Baseline W unavailable from main artifacts.")

    analysis_lines.append("5) Was fallback correctly triggered in 3-model setup?")
    s1 = next(
        (item for item in scenario_results if int(item["scenario_id"]) == 1), None
    )
    if s1 is not None:
        analysis_lines.append(
            f"- S1 layer2_skipped={s1['layer2_skipped']}, layer2_fallback={s1['layer2_fallback']}, consensus_models_for_decision={s1['consensus_models']}."
        )
    else:
        analysis_lines.append("- S1 result missing.")

    lines: List[str] = ["== Small Scale Comparison Report =="]
    lines.append("\n-- Scenario Key Metrics Table --")
    lines.append(
        comparison_df.to_string(index=False)
        if not comparison_df.empty
        else "No scenario metrics available."
    )

    lines.append("\n-- Main-100 Baseline --")
    if baseline:
        lines.append(str(baseline))
    else:
        lines.append("Main-100 baseline not found from existing artifacts.")

    lines.append("\n-- Analytical Notes --")
    lines.extend(analysis_lines)

    report_path = reports_dir / "small_scale_comparison_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_small_scale_experiments(seed: int = config.SEED) -> Dict[str, object]:
    """Run all small-scale scenarios and write all requested artifacts."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "results" / "small_scale"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _build_scenarios()
    scenario_results: List[Dict[str, object]] = []
    for scenario in scenarios:
        # Offset seed by scenario id to avoid accidental clone-like draws.
        scenario_seed = int(seed + scenario.scenario_id * 100)
        scenario_results.append(
            _run_one_scenario(
                scenario=scenario,
                output_dir=output_dir,
                seed=scenario_seed,
            )
        )

    comparison_report = _make_comparison_report(
        project_root=project_root,
        scenario_results=scenario_results,
    )

    return {
        "output_dir": str(output_dir),
        "scenario_results": scenario_results,
        "comparison_report": str(comparison_report),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run small-scale 3/6/9 model experiments for llm_eval_simulation."
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED, help="Base random seed."
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_small_scale_experiments(seed=int(args.seed))

    print("Small-scale experiments completed.")
    print(f"Output directory: {result['output_dir']}")
    print(f"Comparison report: {result['comparison_report']}")


if __name__ == "__main__":
    main()
