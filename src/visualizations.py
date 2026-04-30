"""Visualization helpers for the simulation pipeline."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse

from . import config


def _configure_plotting() -> None:
    """Configure fonts and suppress non-actionable plotting warnings."""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    selected_fonts = [font for font in preferred_fonts if font in available_fonts]
    if not selected_fonts:
        selected_fonts = ["DejaVu Sans"]

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = selected_fonts
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


_configure_plotting()
sns.set_theme(style="whitegrid", context="talk")

GROUP_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
    "D": "#d62728",
    "E": "#9467bd",
    "F": "#8c564b",
    "G": "#e377c2",
    "H": "#7f7f7f",
    "I": "#bcbd22",
    "J": "#17becf",
}

PLOT_LABEL_MAP = {
    "高分_高共识": "High Score / High Consensus",
    "高分_中共识": "High Score / Mid Consensus",
    "高分_低共识": "High Score / Low Consensus",
    "中分_高共识": "Mid Score / High Consensus",
    "中分_中共识": "Mid Score / Mid Consensus",
    "中分_低共识": "Mid Score / Low Consensus",
    "低分_高共识": "Low Score / High Consensus",
    "低分_中共识": "Low Score / Mid Consensus",
    "低分_低共识": "Low Score / Low Consensus",
    "高共识": "High Density",
    "中共识": "Mid Density",
    "低共识": "Low Density",
    "数据不足": "Insufficient Data",
    "直接接受": "Accept Directly",
    "建议复核": "Review",
    "重点评估/专家介入": "Expert Review",
    "储备/待定": "Reserve",
    "酌情考虑": "Conditional Consideration",
    "暂缓决策，补充信息": "Defer / Add Evidence",
    "明确不推荐": "Reject",
    "建议弃用": "Discard",
    "搁置/需决策者裁定": "Hold",
    "学术价值": "Scholarly Value",
    "主题相关性": "Topical Relevance",
    "可读性": "Readability",
    "权威性/可信度": "Authority / Credibility",
    "馆藏适配度": "Collection Fit",
}


def _plot_label(text: str) -> str:
    """Translate plot labels to ASCII-friendly text when available."""
    return PLOT_LABEL_MAP.get(text, text)


def _ensure_dir(output_dir: Path | str) -> Path:
    """Ensure output directory exists and return Path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_figure(fig: plt.Figure, path: Path) -> str:
    """Save figures while suppressing non-actionable rendering warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_icc_bar(icc_df: pd.DataFrame, output_dir: Path | str = "results") -> str:
    """Plot ICC bars for 100 models grouped by A-J with status markers."""
    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(18, 7))

    icc_df = icc_df.copy()
    if "group" not in icc_df.columns:
        icc_df["group"] = icc_df["model_id"].astype(int).map(config.model_group)
    icc_df = icc_df.sort_values("model_id")

    colors = icc_df["group"].map(GROUP_COLORS).fillna("#4c566a")

    yerr = np.vstack(
        [
            (icc_df["icc_2_1"].fillna(0.0) - icc_df["ci95_low"].fillna(0.0)).clip(
                lower=0.0
            ),
            (icc_df["ci95_high"].fillna(0.0) - icc_df["icc_2_1"].fillna(0.0)).clip(
                lower=0.0
            ),
        ]
    )

    x = np.arange(len(icc_df))
    ax.bar(x, icc_df["icc_2_1"].fillna(0.0), color=colors, alpha=0.85)
    ax.errorbar(
        x,
        icc_df["icc_2_1"].fillna(0.0),
        yerr=yerr,
        fmt="none",
        ecolor="black",
        capsize=2,
        linewidth=1.0,
    )
    excluded_mask = icc_df["status"].isin(["excluded", "excluded_no_icc"])
    boundary_mask = icc_df["status"] == "boundary_retained"
    ax.scatter(
        x[excluded_mask],
        np.zeros(excluded_mask.sum()) + 0.02,
        marker="x",
        s=28,
        color="red",
        label="excluded",
    )
    ax.scatter(
        x[boundary_mask],
        icc_df.loc[boundary_mask, "icc_2_1"].fillna(0.0),
        marker="o",
        s=18,
        edgecolor="black",
        facecolor="none",
        label="boundary",
    )

    ax.axhline(
        config.THETA_REL,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"theta_rel={config.THETA_REL:.2f}",
    )

    for boundary in range(10, 100, 10):
        ax.axvline(boundary - 0.5, color="#999999", linestyle=":", linewidth=0.8)

    ax.set_title("Layer1 ICC Screening (100 Models, Grouped A-J)")
    ax.set_xlabel("Model ID")
    ax.set_ylabel("ICC")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, len(icc_df), 5))
    ax.set_xticklabels(icc_df["model_id"].astype(int).iloc[::5])
    ax.legend(loc="lower right", ncol=2, fontsize=9)

    path = out / "layer1_icc_bar.png"
    return _save_figure(fig, path)


def plot_mds(
    mds_coords: pd.DataFrame, output_dir: Path | str = "results"
) -> Optional[str]:
    """Plot 2D MDS embedding for model-level consensus distances."""
    if mds_coords.empty:
        return None

    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_df = mds_coords.copy()
    if "group" not in plot_df.columns:
        plot_df["group"] = plot_df["model_id"].astype(int).map(config.model_group)

    sns.scatterplot(
        data=plot_df,
        x="mds_x",
        y="mds_y",
        hue="group",
        palette=GROUP_COLORS,
        style="group",
        s=70,
        ax=ax,
        legend=True,
    )
    key_models = set([30, 31, 32, 33, 90, 91, 94, 96, 98])
    for _, row in plot_df.iterrows():
        mid = int(row["model_id"])
        if row["group"] == "E" or mid in key_models:
            ax.text(
                row["mds_x"] + 0.01,
                row["mds_y"] + 0.01,
                f"M{mid}",
                fontsize=8,
            )

    e_df = plot_df[plot_df["group"] == "E"]
    if len(e_df) >= 2:
        ex, ey = e_df["mds_x"].mean(), e_df["mds_y"].mean()
        ew = max(0.1, float(e_df["mds_x"].std(ddof=0) * 6))
        eh = max(0.1, float(e_df["mds_y"].std(ddof=0) * 6))
        ellipse = Ellipse(
            (ex, ey),
            width=ew,
            height=eh,
            fill=False,
            linestyle="--",
            edgecolor="#6a4c93",
            linewidth=1.5,
        )
        ax.add_patch(ellipse)

    algo = plot_df["algo"].iloc[0] if "algo" in plot_df.columns else "mds"
    ax.set_title(f"Layer2 Model Embedding ({algo})")
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")
    ax.legend(loc="best", ncol=2, fontsize=8)

    path = out / "layer2_mds.png"
    return _save_figure(fig, path)


def plot_missing_heatmap(
    missing_df: pd.DataFrame, output_dir: Path | str = "results"
) -> Optional[str]:
    """Plot model-wise missingness heatmap for both rounds."""
    if missing_df.empty:
        return None

    out = _ensure_dir(output_dir)
    heat = missing_df.copy().sort_values("model_id")
    if "group" not in heat.columns:
        heat["group"] = heat["model_id"].astype(int).map(config.model_group)

    mat = heat[["missing_round1", "missing_round2", "missing_overall"]].to_numpy()
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        mat.T,
        cmap="rocket_r",
        cbar_kws={"label": "Missing Ratio"},
        ax=ax,
    )
    ax.set_yticklabels(["round1", "round2", "overall"], rotation=0)
    ax.set_xticks(np.arange(0, len(heat), 5) + 0.5)
    ax.set_xticklabels(heat["model_id"].astype(int).iloc[::5], rotation=45, ha="right")
    ax.set_title("Missingness Heatmap (Models x Rounds)")

    for boundary in range(10, len(heat), 10):
        ax.axvline(boundary, color="white", linestyle=":", linewidth=0.8)

    path = out / "layer1_missing_heatmap.png"
    return _save_figure(fig, path)


def plot_w_trajectory(
    log_df: pd.DataFrame, output_dir: Path | str = "results"
) -> Optional[str]:
    """Plot Kendall's W trajectory through greedy removal steps."""
    if log_df.empty:
        return None

    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=log_df, x="step", y="w_after", marker="o", linewidth=2, ax=ax)
    ax.axhline(
        config.THETA_CON,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"theta_con={config.THETA_CON:.2f}",
    )
    ax.set_title("Layer2 Greedy Kendall's W Trajectory")
    ax.set_xlabel("Iteration Step")
    ax.set_ylabel("Kendall's W")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")

    path = out / "layer2_w_trajectory.png"
    return _save_figure(fig, path)


def plot_w_trajectory_comparison(
    trajectories_70: pd.DataFrame,
    trajectories_80: pd.DataFrame,
    output_dir: Path | str = "results",
) -> Optional[str]:
    """Plot side-by-side trajectory comparison for theta=0.70 and theta=0.80."""
    if trajectories_70.empty and trajectories_80.empty:
        return None

    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5))

    if not trajectories_70.empty:
        sns.lineplot(
            data=trajectories_70,
            x="step",
            y="w_after",
            marker="o",
            linewidth=2,
            label="theta=0.70",
            ax=ax,
        )
    if not trajectories_80.empty:
        sns.lineplot(
            data=trajectories_80,
            x="step",
            y="w_after",
            marker="s",
            linewidth=2,
            label="theta=0.80",
            ax=ax,
        )

    ax.axhline(
        0.70, color="#555555", linestyle="--", linewidth=1, label="threshold 0.70"
    )
    ax.axhline(
        0.80, color="#222222", linestyle=":", linewidth=1, label="threshold 0.80"
    )
    ax.set_title("Layer2 Kendall's W Trajectory Comparison")
    ax.set_xlabel("Iteration Step")
    ax.set_ylabel("Kendall's W")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=2, fontsize=9)

    path = out / "sensitivity_w_trajectory_comparison.png"
    return _save_figure(fig, path)


def plot_quadrant_comparison(
    counts_70: pd.Series,
    counts_80: pd.Series,
    output_dir: Path | str = "results",
) -> Optional[str]:
    """Plot grouped bar chart comparing quadrant counts across two thresholds."""
    if counts_70.empty and counts_80.empty:
        return None

    out = _ensure_dir(output_dir)
    all_quadrants = sorted(
        set(counts_70.index.tolist()) | set(counts_80.index.tolist())
    )
    comp_df = pd.DataFrame(
        {
            "quadrant": all_quadrants,
            "theta_0_70": [int(counts_70.get(q, 0)) for q in all_quadrants],
            "theta_0_80": [int(counts_80.get(q, 0)) for q in all_quadrants],
        }
    )
    comp_df["quadrant_plot"] = comp_df["quadrant"].map(_plot_label)

    plot_df = comp_df.melt(
        id_vars=["quadrant", "quadrant_plot"],
        var_name="threshold",
        value_name="count",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="quadrant_plot",
        y="count",
        hue="threshold",
        palette=["#4c78a8", "#f58518"],
        ax=ax,
    )
    ax.set_title("Layer3 Quadrant Count Comparison (theta 0.70 vs 0.80)")
    ax.set_xlabel("Quadrant")
    ax.set_ylabel("Book Count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper right")

    path = out / "sensitivity_quadrant_comparison.png"
    return _save_figure(fig, path)


def plot_threshold_scan_overview(
    key_metrics: pd.DataFrame,
    default_threshold: float,
    output_dir: Path | str = "results",
) -> Optional[str]:
    """Plot four-panel threshold scan summary for layer2 and layer3 outputs."""
    if key_metrics.empty:
        return None

    out = _ensure_dir(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    metric_specs = [
        ("retained_models", "Retained Models", "#4c78a8"),
        ("final_w", "Final Kendall's W", "#f58518"),
        ("high_high_count", "High Score x High Consensus", "#54a24b"),
        ("low_low_count", "Low Score x Low Consensus", "#e45756"),
    ]

    for ax, (column, title, color) in zip(axes.flat, metric_specs):
        sns.lineplot(
            data=key_metrics, x="threshold", y=column, marker="o", color=color, ax=ax
        )
        ax.axvline(default_threshold, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Consensus Threshold")
        ax.set_ylabel(column)

    fig.suptitle("Threshold Scan Overview", y=0.98)
    path = out / "sensitivity_threshold_scan.png"
    return _save_figure(fig, path)


def plot_bootstrap_stability_heatmap(
    quadrant_summary_df: pd.DataFrame,
    output_dir: Path | str = "results",
) -> Optional[str]:
    """Plot a scheme-by-quadrant heatmap of average bootstrap stability."""
    if quadrant_summary_df.empty:
        return None

    out = _ensure_dir(output_dir)
    pivot = quadrant_summary_df.pivot(
        index="scheme_name", columns="quadrant", values="mean_stability"
    ).fillna(0.0)
    pivot = pivot.rename(
        columns={column: _plot_label(column) for column in pivot.columns}
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title("Bootstrap Mean Stability by Scheme and Quadrant")
    ax.set_xlabel("Quadrant")
    ax.set_ylabel("Scheme")

    path = out / "bootstrap_stability_heatmap.png"
    return _save_figure(fig, path)


def plot_bootstrap_scheme_summary(
    scheme_summary_df: pd.DataFrame,
    output_dir: Path | str = "results",
) -> Optional[str]:
    """Plot overall bootstrap stability by percentile scheme."""
    if scheme_summary_df.empty:
        return None

    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=scheme_summary_df,
        x="scheme_name",
        y="overall_mean_stability",
        palette="crest",
        hue="scheme_name",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("Overall Bootstrap Stability by Percentile Scheme")
    ax.set_xlabel("Scheme")
    ax.set_ylabel("Mean Stability")
    ax.set_ylim(0.0, 1.0)

    path = out / "bootstrap_scheme_stability.png"
    return _save_figure(fig, path)


def plot_decision_scatter(
    decision_df: pd.DataFrame,
    output_dir: Path | str = "results",
    file_name: str = "layer3_decision_scatter.png",
    title: str = "Layer3 Nine-Quadrant Decision Matrix",
) -> Optional[str]:
    """Plot nine-quadrant decision scatter with partition lines and labels."""
    if decision_df.empty:
        return None

    out = _ensure_dir(output_dir)
    p_low = config.to_quantile_fraction(config.P_LOW)
    p_high = config.to_quantile_fraction(config.P_HIGH)
    valid = decision_df[decision_df["quadrant"] != "数据不足"]
    if valid.empty:
        return None

    x_low = float(valid["median_score"].quantile(p_low))
    x_high = float(valid["median_score"].quantile(1.0 - p_high))
    y_low = float(valid["rank_iqr"].quantile(p_low))
    y_high = float(valid["rank_iqr"].quantile(1.0 - p_high))

    fig, ax = plt.subplots(figsize=(11, 8))

    x_min, x_max = (
        float(decision_df["median_score"].min()),
        float(decision_df["median_score"].max()),
    )
    y_min, y_max = (
        float(decision_df["rank_iqr"].min()),
        float(decision_df["rank_iqr"].max()),
    )

    x_bins = [x_min, x_low, x_high, x_max]
    y_bins = [y_min, y_low, y_high, y_max]
    shades = ["#d9f0d3", "#f7f7f7", "#fde0dd"]
    for i in range(3):
        for j in range(3):
            ax.fill_between(
                [x_bins[i], x_bins[i + 1]],
                y_bins[j],
                y_bins[j + 1],
                color=shades[i],
                alpha=0.15,
                linewidth=0,
            )

    ax.hexbin(
        valid["median_score"],
        valid["rank_iqr"],
        gridsize=75,
        cmap="viridis",
        mincnt=1,
        alpha=0.75,
    )
    if "data_insufficient" in decision_df.columns:
        insuff = decision_df[decision_df["data_insufficient"]]
        if not insuff.empty:
            ax.scatter(
                insuff["median_score"],
                insuff["rank_iqr"],
                s=8,
                alpha=0.35,
                color="red",
                label="Insufficient Data",
            )

    ax.axvline(x_low, linestyle="--", color="black", linewidth=1)
    ax.axvline(x_high, linestyle="--", color="black", linewidth=1)
    ax.axhline(y_low, linestyle="--", color="black", linewidth=1)
    ax.axhline(y_high, linestyle="--", color="black", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Median Score")
    ax.set_ylabel("Rank IQR (Lower Means Stronger Majority Consensus)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    labels = {
        (2, 0): "High Score / High Consensus\nAccept",
        (2, 1): "High Score / Mid Consensus\nReview",
        (2, 2): "High Score / Low Consensus\nExpert Review",
        (1, 0): "Mid Score / High Consensus\nReserve",
        (1, 1): "Mid Score / Mid Consensus\nConsider",
        (1, 2): "Mid Score / Low Consensus\nDefer",
        (0, 0): "Low Score / High Consensus\nReject",
        (0, 1): "Low Score / Mid Consensus\nDiscard",
        (0, 2): "Low Score / Low Consensus\nHold",
    }

    x_centers = [
        (x_bins[0] + x_bins[1]) / 2,
        (x_bins[1] + x_bins[2]) / 2,
        (x_bins[2] + x_bins[3]) / 2,
    ]
    y_centers = [
        (y_bins[0] + y_bins[1]) / 2,
        (y_bins[1] + y_bins[2]) / 2,
        (y_bins[2] + y_bins[3]) / 2,
    ]

    for (x_idx, y_idx), text in labels.items():
        ax.text(
            x_centers[x_idx],
            y_centers[y_idx],
            text,
            fontsize=9,
            ha="center",
            va="center",
            alpha=0.75,
        )

    path = out / file_name
    return _save_figure(fig, path)


def plot_density_quadrant_comparison(
    percentile_df: pd.DataFrame,
    density_df: pd.DataFrame,
    output_dir: Path | str = "results",
) -> Optional[str]:
    """Plot percentile and KDE-based decision spaces side-by-side."""
    if percentile_df.empty or density_df.empty:
        return None

    out = _ensure_dir(output_dir)
    valid_percentile = percentile_df[percentile_df["quadrant"] != "数据不足"].copy()
    valid_density = density_df[density_df["quadrant"] != "数据不足"].copy()
    if valid_percentile.empty or valid_density.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)
    plot_percentile = valid_percentile.sample(
        n=min(len(valid_percentile), 2500), random_state=config.SEED
    )
    plot_density = valid_density.sample(
        n=min(len(valid_density), 2500), random_state=config.SEED
    )
    plot_density = plot_density.assign(
        density_level_plot=plot_density["density_level"].map(_plot_label)
    )
    axes[0].scatter(
        plot_percentile["median_score"],
        plot_percentile["rank_iqr"],
        s=18,
        alpha=0.40,
        linewidths=0,
        color="#4c78a8",
    )
    p_low = config.to_quantile_fraction(config.P_LOW)
    p_high = config.to_quantile_fraction(config.P_HIGH)
    axes[0].axvline(
        valid_percentile["median_score"].quantile(p_low),
        linestyle="--",
        color="black",
        linewidth=1,
    )
    axes[0].axvline(
        valid_percentile["median_score"].quantile(1.0 - p_high),
        linestyle="--",
        color="black",
        linewidth=1,
    )
    axes[0].axhline(
        valid_percentile["rank_iqr"].quantile(p_low),
        linestyle="--",
        color="black",
        linewidth=1,
    )
    axes[0].axhline(
        valid_percentile["rank_iqr"].quantile(1.0 - p_high),
        linestyle="--",
        color="black",
        linewidth=1,
    )
    axes[0].set_title("Percentile Nine-Quadrant")
    axes[0].set_xlabel("Median Score")
    axes[0].set_ylabel("Rank IQR")

    axes[1].tricontour(
        plot_density["median_score"],
        plot_density["rank_iqr"],
        plot_density["density"],
        levels=5,
        colors="#1d3557",
        linewidths=1.1,
    )
    sns.scatterplot(
        data=plot_density,
        x="median_score",
        y="rank_iqr",
        hue="density_level_plot",
        palette={
            "High Density": "#2a9d8f",
            "Mid Density": "#e9c46a",
            "Low Density": "#e76f51",
        },
        s=18,
        alpha=0.50,
        linewidth=0,
        ax=axes[1],
    )
    axes[1].set_title("Density-Contour Decision Space")
    axes[1].set_xlabel("Median Score")
    axes[1].set_ylabel("Rank IQR")
    axes[1].legend(loc="upper right", fontsize=8, title="Density Level")

    path = out / "density_quadrant_comparison.png"
    return _save_figure(fig, path)


def plot_skill_radar(
    book_id: int,
    skill_scores: pd.DataFrame,
    output_dir: Path | str = "results",
    highlight_skill: Optional[str] = None,
) -> str:
    """Plot radar chart for one disputed book's mean skill scores."""
    out = _ensure_dir(output_dir)

    ordered = skill_scores.sort_values("skill")
    labels = [_plot_label(label) for label in ordered["skill"].tolist()]
    values = ordered["mean_score"].tolist()

    values_cycle = values + values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles_cycle = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})
    ax.plot(angles_cycle, values_cycle, color="#1d3557", linewidth=2)
    ax.fill(angles_cycle, values_cycle, color="#457b9d", alpha=0.25)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    if highlight_skill:
        ax.set_title(
            f"Layer4 Skill Radar - Book {book_id} (max dispute: {_plot_label(highlight_skill)})"
        )
    else:
        ax.set_title(f"Layer4 Skill Radar - Book {book_id}")

    path = out / f"layer4_skill_radar_book_{book_id}.png"
    return _save_figure(fig, path)


def plot_monte_carlo_high_high_hist(
    metrics_df: pd.DataFrame,
    output_dir: Path | str = "results",
) -> Optional[str]:
    """Plot histogram of high-score high-consensus counts across Monte Carlo seeds."""
    if metrics_df.empty or "direct_accept_count" not in metrics_df.columns:
        return None

    out = _ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        metrics_df["direct_accept_count"],
        bins=min(8, max(3, len(metrics_df))),
        kde=True,
        color="#2a9d8f",
        ax=ax,
    )
    ax.set_title("Monte Carlo Distribution of Direct-Accept Books")
    ax.set_xlabel("High Score x High Consensus Count")
    ax.set_ylabel("Frequency")

    path = out / "monte_carlo_high_high_hist.png"
    return _save_figure(fig, path)
