"""Alternative layer3 partitioning methods: implementation, testing, and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

from . import bootstrap_stability, config, data_simulator
from . import layer1_reliability as layer1
from . import layer2_consensus as layer2
from . import layer3_decision as layer3


@dataclass(frozen=True)
class MethodResult:
    """Container for high-high recommendation result."""

    method: str
    high_high_count: int
    jaccard_vs_baseline: float
    details: str


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    l_set = set(int(v) for v in left)
    r_set = set(int(v) for v in right)
    union = l_set | r_set
    if not union:
        return 0.0
    return len(l_set & r_set) / len(union)


def _prepare_context() -> Tuple[pd.DataFrame, List[int], pd.DataFrame, pd.DataFrame]:
    """Run the core upstream pipeline to get layer3-ready inputs."""
    scores_df, _ = data_simulator.generate_data(seed=config.SEED)
    reliable_models, _, _ = layer1.filter_reliable_models(scores_df)
    layer2_result = layer2.find_consensus_subset(scores_df, reliable_models)
    consensus_models = layer2_result.get("consensus_models", [])

    decision_space = layer3.prepare_decision_space(scores_df, consensus_models)
    baseline_df = layer3.generate_decision_matrix(
        scores_df=scores_df,
        consensus_models=consensus_models,
        scheme_name=config.DEFAULT_PERCENTILE_SCHEME,
    )
    return scores_df, consensus_models, decision_space, baseline_df


def _search_bootstrap_optimal_scheme(
    decision_space: pd.DataFrame,
    high_grid: Sequence[float],
    low_grid: Sequence[float],
    n_bootstrap: int = 120,
    random_seed: int = config.SEED,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Grid-search percentile schemes by mean bootstrap stability."""
    valid_space = decision_space[~decision_space["data_insufficient"]].copy()
    scores = valid_space["median_score"].to_numpy(dtype=float)
    ranges = valid_space["rank_iqr"].to_numpy(dtype=float)
    n_books = scores.shape[0]
    rng = np.random.default_rng(random_seed)

    rows: List[Dict[str, float]] = []
    for p_high in high_grid:
        for p_low in low_grid:
            # Skip invalid partitions where middle band disappears.
            if p_high + p_low >= 0.95:
                continue

            baseline_codes, _, _ = bootstrap_stability._classify_codes(
                target_scores=scores,
                target_ranges=ranges,
                source_scores=scores,
                source_ranges=ranges,
                low_fraction=p_low,
                high_fraction=p_high,
            )

            counts = np.zeros((n_books, 9), dtype=np.uint16)
            for _ in range(int(n_bootstrap)):
                sample_idx = rng.integers(0, n_books, size=n_books)
                sample_scores = scores[sample_idx]
                sample_ranges = ranges[sample_idx]
                boot_codes, _, _ = bootstrap_stability._classify_codes(
                    target_scores=scores,
                    target_ranges=ranges,
                    source_scores=sample_scores,
                    source_ranges=sample_ranges,
                    low_fraction=p_low,
                    high_fraction=p_high,
                )
                counts[np.arange(n_books), boot_codes] += 1

            stability = counts.max(axis=1) / float(n_bootstrap)
            rows.append(
                {
                    "p_high": p_high,
                    "p_low": p_low,
                    "mean_stability": float(stability.mean()),
                    "median_stability": float(np.median(stability)),
                    "high_high_count": int(np.sum(baseline_codes == 6)),
                }
            )

    search_df = pd.DataFrame(rows).sort_values(
        ["mean_stability", "median_stability"], ascending=False
    )
    best = search_df.iloc[0].to_dict() if not search_df.empty else {}
    return search_df, best


def _run_gmm_method(
    decision_space: pd.DataFrame, random_seed: int = config.SEED
) -> Dict[str, object]:
    """Fit GMM on score-IQR plane and derive high-high cluster."""
    valid = decision_space[~decision_space["data_insufficient"]].copy()
    x = valid[["median_score", "rank_iqr"]].to_numpy(dtype=float)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=0)
    x_std = np.where(x_std == 0.0, 1.0, x_std)
    x_scaled = (x - x_mean) / x_std

    best_bic = float("inf")
    best_model: GaussianMixture | None = None
    for n_components in [3, 4, 5, 6]:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=5,
            random_state=random_seed,
        )
        model.fit(x_scaled)
        bic = float(model.bic(x_scaled))
        if bic < best_bic:
            best_bic = bic
            best_model = model

    if best_model is None:
        raise RuntimeError("GMM fitting failed.")

    labels = best_model.predict(x_scaled)
    valid["gmm_cluster"] = labels

    # Choose the cluster with highest score and lowest IQR in standardized utility.
    cluster_stats = (
        valid.groupby("gmm_cluster")[["median_score", "rank_iqr"]].mean().reset_index()
    )
    score_z = (cluster_stats["median_score"] - cluster_stats["median_score"].mean()) / (
        cluster_stats["median_score"].std(ddof=0) or 1.0
    )
    iqr_z = (cluster_stats["rank_iqr"] - cluster_stats["rank_iqr"].mean()) / (
        cluster_stats["rank_iqr"].std(ddof=0) or 1.0
    )
    cluster_stats["utility"] = score_z - iqr_z
    high_high_cluster = int(
        cluster_stats.sort_values("utility", ascending=False).iloc[0]["gmm_cluster"]
    )

    high_high_ids = (
        valid.loc[valid["gmm_cluster"] == high_high_cluster, "book_id"]
        .astype(int)
        .tolist()
    )
    return {
        "valid_df": valid,
        "cluster_stats": cluster_stats,
        "high_high_cluster": high_high_cluster,
        "high_high_ids": high_high_ids,
        "best_bic": best_bic,
        "n_components": int(best_model.n_components),
    }


def _run_consensus_depth_method(
    scores_df: pd.DataFrame,
    consensus_models: List[int],
    depth_threshold: float = 0.70,
    top_rank_ratio: float = 0.50,
    high_score_fraction: float = 0.20,
) -> Dict[str, object]:
    """Define majority consensus by top-rank vote depth and extract high-high books."""
    round1 = scores_df[scores_df["round"] == 1]
    wide = round1.pivot(index="book_id", columns="model_id", values="score")
    wide = wide[consensus_models]

    rank_df = wide.rank(axis=0, ascending=False, method="average")
    top_cutoff = float(len(wide) * top_rank_ratio)
    top_half_vote_ratio = (rank_df <= top_cutoff).mean(axis=1, skipna=True)
    median_score = wide.median(axis=1, skipna=True)
    missing_fraction = wide.isna().mean(axis=1)

    high_score_cutoff = float(median_score.quantile(1.0 - high_score_fraction))
    candidate_mask = (
        (median_score >= high_score_cutoff)
        & (top_half_vote_ratio >= depth_threshold)
        & (missing_fraction <= 0.5)
    )

    detail_df = pd.DataFrame(
        {
            "book_id": wide.index.astype(int),
            "median_score": median_score.values,
            "top_half_vote_ratio": top_half_vote_ratio.values,
            "missing_fraction": missing_fraction.values,
            "is_high_high": candidate_mask.values,
        }
    )

    high_high_ids = (
        detail_df.loc[detail_df["is_high_high"], "book_id"].astype(int).tolist()
    )
    return {
        "detail_df": detail_df,
        "high_high_ids": high_high_ids,
        "high_score_cutoff": high_score_cutoff,
        "depth_threshold": depth_threshold,
    }


def _plot_bootstrap_search_heatmap(search_df: pd.DataFrame, output_dir: Path) -> str:
    pivot = search_df.pivot(index="p_low", columns="p_high", values="mean_stability")
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax)
    ax.set_title("Bootstrap Mean Stability over (p_high, p_low) Grid")
    ax.set_xlabel("p_high")
    ax.set_ylabel("p_low")
    path = output_dir / "bootstrap_grid_search_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _plot_gmm_clusters(
    valid_df: pd.DataFrame, high_high_cluster: int, output_dir: Path
) -> str:
    sample_df = valid_df.sample(n=min(len(valid_df), 3000), random_state=config.SEED)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    sns.scatterplot(
        data=sample_df,
        x="median_score",
        y="rank_iqr",
        hue="gmm_cluster",
        palette="tab10",
        s=18,
        alpha=0.65,
        linewidth=0,
        ax=ax,
    )
    ax.set_title(
        f"GMM Clusters on Score-IQR Plane (High-High Cluster={high_high_cluster})"
    )
    ax.set_xlabel("Median Score")
    ax.set_ylabel("Rank IQR")
    path = output_dir / "gmm_cluster_scatter.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _plot_method_comparison(
    results: List[MethodResult], output_dir: Path
) -> Tuple[str, str]:
    comp_df = pd.DataFrame(
        {
            "method": [r.method for r in results],
            "high_high_count": [r.high_high_count for r in results],
            "jaccard": [r.jaccard_vs_baseline for r in results],
        }
    )

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=comp_df, x="method", y="high_high_count", palette="Set2", ax=ax1)
    ax1.set_title("High-High Recommendation Count by Method")
    ax1.set_xlabel("Method")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=20)
    p1 = output_dir / "alternative_high_high_counts.png"
    fig1.tight_layout()
    fig1.savefig(p1, dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(data=comp_df, x="method", y="jaccard", palette="Set3", ax=ax2)
    ax2.set_title("Jaccard Overlap with Baseline Percentile")
    ax2.set_xlabel("Method")
    ax2.set_ylabel("Jaccard")
    ax2.set_ylim(0.0, 1.0)
    ax2.tick_params(axis="x", rotation=20)
    p2 = output_dir / "alternative_jaccard_vs_baseline.png"
    fig2.tight_layout()
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)

    return str(p1), str(p2)


def run_alternative_methods_report() -> Dict[str, object]:
    """Run selected alternative methods and write report + visualizations."""
    alternative_dir = _ensure_dir(config.RESULTS_DIR / "alternative")
    reports_dir = _ensure_dir(config.get_results_dir("reports"))

    scores_df, consensus_models, decision_space, baseline_df = _prepare_context()
    baseline_high_high = set(
        baseline_df.loc[baseline_df["quadrant"] == "高分_高共识", "book_id"].astype(int)
    )

    # Method 1: Bootstrap-driven percentile threshold search.
    search_df, best = _search_bootstrap_optimal_scheme(
        decision_space=decision_space,
        high_grid=[0.10, 0.15, 0.20, 0.25, 0.30],
        low_grid=[0.30, 0.35, 0.40, 0.45, 0.50],
        n_bootstrap=120,
        random_seed=config.SEED,
    )
    best_low = float(best["p_low"])
    best_high = float(best["p_high"])
    optimal_df = layer3.classify_prepared_space(
        decision_space=decision_space,
        low_fraction=best_low,
        high_fraction=best_high,
        scheme_name=f"auto_{int(best_high * 100)}/{int(best_low * 100)}",
    )
    optimal_high_high = set(
        optimal_df.loc[optimal_df["quadrant"] == "高分_高共识", "book_id"].astype(int)
    )

    # Method 2: GMM clustering.
    gmm_result = _run_gmm_method(decision_space=decision_space, random_seed=config.SEED)
    gmm_high_high = set(int(v) for v in gmm_result["high_high_ids"])

    # Method 3: Consensus depth metric.
    depth_result = _run_consensus_depth_method(
        scores_df=scores_df,
        consensus_models=consensus_models,
        depth_threshold=0.70,
        top_rank_ratio=0.50,
        high_score_fraction=0.20,
    )
    depth_high_high = set(int(v) for v in depth_result["high_high_ids"])

    method_results = [
        MethodResult(
            method="Baseline Percentile",
            high_high_count=len(baseline_high_high),
            jaccard_vs_baseline=1.0,
            details="Default 20/40/40 percentile with rank IQR consensus.",
        ),
        MethodResult(
            method="Bootstrap-Optimal Percentile",
            high_high_count=len(optimal_high_high),
            jaccard_vs_baseline=_jaccard(optimal_high_high, baseline_high_high),
            details=(
                f"Best grid point: p_high={best_high:.2f}, p_low={best_low:.2f}, "
                f"mean_stability={float(best['mean_stability']):.4f}."
            ),
        ),
        MethodResult(
            method="GMM Cluster Mapping",
            high_high_count=len(gmm_high_high),
            jaccard_vs_baseline=_jaccard(gmm_high_high, baseline_high_high),
            details=(
                f"Selected cluster={gmm_result['high_high_cluster']}, "
                f"n_components={gmm_result['n_components']}, BIC={float(gmm_result['best_bic']):.2f}."
            ),
        ),
        MethodResult(
            method="Consensus Depth (70% in Top50%)",
            high_high_count=len(depth_high_high),
            jaccard_vs_baseline=_jaccard(depth_high_high, baseline_high_high),
            details=(
                f"Depth threshold={float(depth_result['depth_threshold']):.2f}, "
                f"score cutoff={float(depth_result['high_score_cutoff']):.2f}."
            ),
        ),
    ]

    heatmap_path = _plot_bootstrap_search_heatmap(
        search_df=search_df, output_dir=alternative_dir
    )
    gmm_plot_path = _plot_gmm_clusters(
        valid_df=gmm_result["valid_df"],
        high_high_cluster=int(gmm_result["high_high_cluster"]),
        output_dir=alternative_dir,
    )
    comp_count_path, comp_jaccard_path = _plot_method_comparison(
        results=method_results,
        output_dir=alternative_dir,
    )

    report_lines: List[str] = ["== Alternative Layer3 Methods Report =="]
    report_lines.append("\n-- Expert Methodology Screening --")
    report_lines.append(
        "Besides the requested methods, two additional candidates are noteworthy: "
        "(1) rank-biserial consensus using pairwise top-vs-bottom dominance, "
        "(2) quantile-regression contour partitioning for score-dependent consensus boundaries."
    )
    report_lines.append(
        "For this framework goal, the most actionable methods are: Bootstrap-optimal percentile "
        "(keeps policy continuity), GMM clustering (captures latent evaluator structure), and "
        "consensus-depth voting (direct majority semantics)."
    )

    report_lines.append("\n-- Baseline Context --")
    report_lines.append(
        f"consensus_models={len(consensus_models)}, baseline_high_high={len(baseline_high_high)}"
    )

    report_lines.append("\n-- Method Comparison --")
    for result in method_results:
        report_lines.append(
            f"{result.method}: high_high_count={result.high_high_count}, "
            f"jaccard_vs_baseline={result.jaccard_vs_baseline:.4f}."
        )
        report_lines.append(f"  detail: {result.details}")

    report_lines.append("\n-- Bootstrap Grid Top 10 (mean_stability) --")
    report_lines.append(search_df.head(10).to_string(index=False))

    report_lines.append("\n-- Practical Interpretation --")
    report_lines.append(
        "Bootstrap-optimal percentile is best for production default tuning because it preserves "
        "the existing interpretability and action labels while adding data-driven parameter evidence."
    )
    report_lines.append(
        "GMM is best for exploratory analysis and subgroup discovery, especially when evaluator coalitions "
        "are non-uniform or multi-modal."
    )
    report_lines.append(
        "Consensus-depth voting is the most intuitive majority definition for stakeholder communication, "
        "but it introduces threshold sensitivity and should be paired with robustness checks."
    )

    report_lines.append("\n-- Recommendation for Default Method --")
    report_lines.append(
        "Keep percentile + rank IQR as default, but attach an annual/bootstrap-based parameter retuning step "
        "(p_high, p_low) using the grid-search protocol as a governance mechanism."
    )
    report_lines.append(
        "Use GMM and consensus-depth as complementary diagnostics rather than replacing the default policy rule."
    )

    report_lines.append("\n-- Generated Figures --")
    report_lines.append(heatmap_path)
    report_lines.append(gmm_plot_path)
    report_lines.append(comp_count_path)
    report_lines.append(comp_jaccard_path)

    report_path = reports_dir / "alternative_methods_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "report_path": str(report_path),
        "figure_paths": [
            heatmap_path,
            gmm_plot_path,
            comp_count_path,
            comp_jaccard_path,
        ],
        "search_df": search_df,
        "method_results": method_results,
    }
