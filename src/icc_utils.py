"""Fast ICC utility functions used by reliability analysis."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import f as f_dist


def _prepare_two_way_anova_terms(
    round1_scores: np.ndarray,
    round2_scores: np.ndarray,
) -> Tuple[np.ndarray, int, int, float, float, float, int, int]:
    """Prepare balanced two-way ANOVA mean-square terms for paired scores.

    Returns:
        Tuple of (data, n, k, msb, msj, mse, df1, df2).
    """
    r1 = np.asarray(round1_scores, dtype=float)
    r2 = np.asarray(round2_scores, dtype=float)
    paired_mask = (~np.isnan(r1)) & (~np.isnan(r2))

    n = int(paired_mask.sum())
    k = 2
    if n < 5:
        return np.empty((0, 0)), n, k, np.nan, np.nan, np.nan, 0, 0

    data = np.column_stack([r1[paired_mask], r2[paired_mask]])
    grand_mean = float(data.mean())
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ssb = float(k * np.square(row_means - grand_mean).sum())
    ssj = float(n * np.square(col_means - grand_mean).sum())
    sse = float(
        np.square(data - row_means[:, None] - col_means[None, :] + grand_mean).sum()
    )

    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    if df1 <= 0 or df2 <= 0:
        return np.empty((0, 0)), n, k, np.nan, np.nan, np.nan, df1, df2

    msb = ssb / df1
    msj = ssj / (k - 1)
    mse = sse / df2

    if np.isclose(mse, 0.0):
        if np.allclose(data[:, 0], data[:, 1], equal_nan=True):
            mse = 0.0
        else:
            mse = np.finfo(float).eps

    return data, n, k, msb, msj, mse, df1, df2


def compute_icc_a_1(
    round1_scores: np.ndarray, round2_scores: np.ndarray
) -> Tuple[float, float, float, int]:
    """Compute ICC(A,1) and its 95% CI from paired round scores.

    This implements the same ANOVA-based ICC(A,1) formulation used by Pingouin,
    but directly on NumPy arrays to avoid repeated DataFrame construction.

    Args:
        round1_scores: Round-1 numeric scores.
        round2_scores: Round-2 numeric scores.

    Returns:
        Tuple of (icc, ci_low, ci_high, n_pairs).
    """
    data, n, k, msb, msj, mse, df1, df2 = _prepare_two_way_anova_terms(
        round1_scores=round1_scores,
        round2_scores=round2_scores,
    )
    if n < 5 or df1 <= 0 or df2 <= 0:
        return float("nan"), float("nan"), float("nan"), n

    if np.isclose(mse, 0.0) and np.allclose(data[:, 0], data[:, 1], equal_nan=True):
        return 1.0, 1.0, 1.0, n

    denom = msb + (k - 1) * mse + k * (msj - mse) / n
    if np.isclose(denom, 0.0):
        return float("nan"), float("nan"), float("nan"), n

    icc2 = float((msb - mse) / denom)

    alpha = 0.05
    try:
        f2 = msb / mse
        fj = msj / mse
        vn = df2 * (k * icc2 * fj + n * (1 + (k - 1) * icc2) - k * icc2) ** 2
        vd = (
            df1 * (k**2) * (icc2**2) * (fj**2)
            + (n * (1 + (k - 1) * icc2) - k * icc2) ** 2
        )
        v = vn / vd if vd > 0 else np.nan

        if not np.isfinite(v) or v <= 0:
            return icc2, float("nan"), float("nan"), n

        f2u = f_dist.ppf(1 - alpha / 2, df1, v)
        f2l = f_dist.ppf(1 - alpha / 2, v, df1)

        ci_low = (
            n * (msb - f2u * mse) / (f2u * (k * msj + (k * n - k - n) * mse) + n * msb)
        )
        ci_high = (
            n * (f2l * msb - mse) / (k * msj + (k * n - k - n) * mse + n * f2l * msb)
        )
        ci_low = float(np.clip(ci_low, -1.0, 1.0))
        ci_high = float(np.clip(ci_high, -1.0, 1.0))
    except Exception:
        ci_low, ci_high = float("nan"), float("nan")

    return icc2, ci_low, ci_high, n


def compute_icc_c_1(
    round1_scores: np.ndarray,
    round2_scores: np.ndarray,
) -> Tuple[float, float, float, int]:
    """Compute ICC(C,1)=ICC(3,1) and its 95% CI from paired round scores.

    Formula:
        ICC(3,1) = (MS_R - MS_E) / (MS_R + (k-1)*MS_E), where k=2.

    Unlike ICC(A,1), the denominator excludes the rater-column mean-square term.

    Args:
        round1_scores: Round-1 numeric scores.
        round2_scores: Round-2 numeric scores.

    Returns:
        Tuple of (icc, ci_low, ci_high, n_pairs).
    """
    data, n, k, msb, _, mse, df1, df2 = _prepare_two_way_anova_terms(
        round1_scores=round1_scores,
        round2_scores=round2_scores,
    )
    if n < 5 or df1 <= 0 or df2 <= 0:
        return float("nan"), float("nan"), float("nan"), n

    if np.isclose(mse, 0.0) and np.allclose(data[:, 0], data[:, 1], equal_nan=True):
        return 1.0, 1.0, 1.0, n

    denom = msb + (k - 1) * mse
    if np.isclose(denom, 0.0):
        return float("nan"), float("nan"), float("nan"), n

    icc3 = float((msb - mse) / denom)

    alpha = 0.05
    try:
        f_stat = msb / mse
        if (not np.isfinite(f_stat)) or f_stat <= 0:
            return icc3, float("nan"), float("nan"), n

        f_low = f_stat / f_dist.ppf(1 - alpha / 2, df1, df2)
        f_high = f_stat * f_dist.ppf(1 - alpha / 2, df2, df1)

        ci_low = (f_low - 1.0) / (f_low + (k - 1.0))
        ci_high = (f_high - 1.0) / (f_high + (k - 1.0))
        ci_low = float(np.clip(ci_low, -1.0, 1.0))
        ci_high = float(np.clip(ci_high, -1.0, 1.0))
    except Exception:
        ci_low, ci_high = float("nan"), float("nan")

    return icc3, ci_low, ci_high, n
