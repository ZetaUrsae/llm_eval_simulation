"""Data simulation module for total and skill-level LLM scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config


def _sample_true_quality(rng: np.random.Generator) -> np.ndarray:
    """Sample latent book quality from a detailed Gaussian mixture."""
    mixture = rng.choice(
        np.arange(7), size=config.N_BOOKS, p=[0.03, 0.07, 0.15, 0.35, 0.25, 0.12, 0.03]
    )
    means = np.array([96.0, 87.0, 75.0, 55.0, 40.0, 25.0, 10.0])
    stds = np.array([2.0, 5.0, 7.0, 10.0, 9.0, 10.0, 8.0])
    quality = rng.normal(loc=means[mixture], scale=stds[mixture])
    return np.clip(quality, 0.0, 100.0)


@dataclass
class ModelProfile:
    """Model-level behavior parameters for simulation."""

    noise_std: float
    bias: float
    scale: float
    reversal_fraction: float
    reversal_mode: str
    missing_mode: str
    missing_fraction: float
    hetero_mode: str
    round2_drift: float
    skill_weights: np.ndarray


def _default_skill_weights() -> np.ndarray:
    """Return default unit weights for 5 skill dimensions."""
    return np.ones(config.N_SKILLS, dtype=np.float32)


def _build_model_profiles(rng: np.random.Generator) -> Dict[int, ModelProfile]:
    """Construct 100 model profiles across A-J groups."""
    profiles: Dict[int, ModelProfile] = {}

    for model_id in range(config.N_MODELS):
        profiles[model_id] = ModelProfile(
            noise_std=4.0,
            bias=0.0,
            scale=1.0,
            reversal_fraction=0.0,
            reversal_mode="none",
            missing_mode="none",
            missing_fraction=0.0,
            hetero_mode="none",
            round2_drift=0.0,
            skill_weights=_default_skill_weights(),
        )

    bias_choices = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    for model_id in range(0, 10):
        profiles[model_id].noise_std = float(rng.uniform(2.0, 4.0))
        profiles[model_id].bias = float(rng.choice(bias_choices))
        profiles[model_id].scale = float(rng.uniform(0.95, 1.05))

    noise_gradient = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    for i, model_id in enumerate(range(10, 20)):
        profiles[model_id].noise_std = float(noise_gradient[i])

    c_biases = [15, 10, 5, 3, 0, 0, -3, -5, -10, -15]
    for i, model_id in enumerate(range(20, 30)):
        profiles[model_id].noise_std = 4.0
        profiles[model_id].bias = float(c_biases[i])
        profiles[model_id].scale = float(rng.uniform(0.5, 1.5))

    for model_id in range(30, 40):
        profiles[model_id].noise_std = float(rng.uniform(3.0, 5.0))
    profiles[30].reversal_mode, profiles[30].reversal_fraction = "tails20", 0.4
    profiles[31].reversal_mode, profiles[31].reversal_fraction = "tails20", 0.4
    profiles[32].reversal_mode, profiles[32].reversal_fraction = "random", 0.3
    profiles[33].reversal_mode, profiles[33].reversal_fraction = "random", 0.3
    profiles[34].reversal_mode, profiles[34].reversal_fraction = "extreme5", 0.1
    profiles[35].reversal_mode, profiles[35].reversal_fraction = "extreme5", 0.1
    profiles[36].reversal_mode, profiles[36].reversal_fraction = "middle40", 0.4
    profiles[37].reversal_mode, profiles[37].reversal_fraction = "middle40", 0.4
    profiles[38].bias = 20.0
    profiles[39].bias = -20.0

    for model_id in range(40, 50):
        profiles[model_id].noise_std = 4.0
    flow1 = np.array([2.5, 2.2, 0.4, 1.6, 0.6], dtype=np.float32)
    flow2 = np.array([0.5, 0.6, 2.4, 2.2, 2.5], dtype=np.float32)
    for model_id in range(40, 45):
        profiles[model_id].skill_weights = flow1
    for model_id in range(45, 50):
        profiles[model_id].skill_weights = flow2

    for model_id in range(50, 60):
        profiles[model_id].noise_std = float(rng.uniform(3.0, 5.0))
    for model_id, miss in zip(range(50, 54), [0.1, 0.2, 0.3, 0.4]):
        profiles[model_id].missing_mode = "round1_only"
        profiles[model_id].missing_fraction = miss
    profiles[54].missing_mode, profiles[54].missing_fraction = "both_rounds", 0.5
    profiles[55].missing_mode, profiles[55].missing_fraction = "both_rounds", 0.7
    profiles[56].missing_mode = "round2_all"
    profiles[57].missing_mode = "round2_all"
    profiles[58].missing_mode, profiles[58].missing_fraction = "both_rounds", 0.3
    profiles[59].missing_mode, profiles[59].missing_fraction = "both_rounds", 0.3

    for model_id in range(60, 70):
        profiles[model_id].noise_std = 3.0
        profiles[model_id].hetero_mode = (
            "low_tail" if model_id % 2 == 0 else "high_tail"
        )

    for i, model_id in enumerate(range(70, 80)):
        profiles[model_id].noise_std = 3.0
        profiles[model_id].round2_drift = float(2.0 + (6.0 / 9.0) * i)

    for model_id in range(80, 90):
        profiles[model_id].noise_std = float(rng.uniform(2.0, 4.0))
        profiles[model_id].bias = float(rng.choice(bias_choices))
        profiles[model_id].scale = float(rng.uniform(0.95, 1.05))

    profiles[90].noise_std = 2.0
    profiles[90].reversal_mode = "tails20"
    profiles[90].reversal_fraction = 0.4

    profiles[91].noise_std = 24.0
    profiles[92].bias = 25.0
    profiles[93].bias = -25.0
    profiles[93].scale = 0.3

    profiles[94].reversal_mode = "random"
    profiles[94].reversal_fraction = 0.3
    profiles[94].missing_mode = "both_rounds"
    profiles[94].missing_fraction = 0.3

    profiles[95].hetero_mode = "low_tail"
    profiles[95].round2_drift = 6.0
    profiles[96].noise_std = 16.0
    profiles[96].reversal_mode = "random"
    profiles[96].reversal_fraction = 0.3

    profiles[97].reversal_mode = "extreme5"
    profiles[97].reversal_fraction = 0.1
    profiles[97].missing_mode = "round1_only"
    profiles[97].missing_fraction = 0.2

    profiles[98].skill_weights = np.array([0.3, 0.4, 2.8, 1.8, 3.8], dtype=np.float32)
    profiles[99].noise_std = 3.0
    profiles[99].bias = 0.5
    profiles[99].scale = 1.0

    return profiles


def _build_reversal_mask(
    rng: np.random.Generator,
    reversal_mode: str,
    reversal_fraction: float,
    true_quality: np.ndarray,
) -> np.ndarray:
    """Construct reversal mask by model mode."""
    n_books = len(true_quality)
    if reversal_mode == "none" or reversal_fraction <= 0:
        return np.zeros(n_books, dtype=bool)

    q05, q20, q30, q70, q80, q95 = np.quantile(
        true_quality, [0.05, 0.20, 0.30, 0.70, 0.80, 0.95]
    )

    if reversal_mode == "tails20":
        return (true_quality <= q20) | (true_quality >= q80)
    if reversal_mode == "extreme5":
        return (true_quality <= q05) | (true_quality >= q95)
    if reversal_mode == "middle40":
        return (true_quality >= q30) & (true_quality <= q70)
    if reversal_mode == "random":
        n_pick = int(round(n_books * reversal_fraction))
        idx = rng.choice(n_books, size=n_pick, replace=False)
        mask = np.zeros(n_books, dtype=bool)
        mask[idx] = True
        return mask
    return np.zeros(n_books, dtype=bool)


def _heteroscedastic_noise(
    rng: np.random.Generator,
    base_std: float,
    true_quality: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Create noise vector with optional quality-dependent heteroscedasticity."""
    noise = rng.normal(0.0, base_std, size=true_quality.shape[0])
    if mode == "low_tail":
        mask = true_quality < 40.0
        noise[mask] = rng.normal(0.0, base_std * 3.0, size=int(mask.sum()))
    elif mode == "high_tail":
        mask = true_quality > 85.0
        noise[mask] = rng.normal(0.0, base_std * 4.0, size=int(mask.sum()))
    return noise


def _apply_missing(
    rng: np.random.Generator,
    scores_round1: np.ndarray,
    scores_round2: np.ndarray,
    missing_mode: str,
    missing_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject missingness pattern into round scores."""
    n_books = scores_round1.shape[0]
    if missing_mode == "none":
        return scores_round1, scores_round2

    if missing_mode == "round1_only" and missing_fraction > 0:
        miss = rng.random(n_books) < missing_fraction
        scores_round1[miss] = np.nan
    elif missing_mode == "round2_all":
        scores_round2[:] = np.nan
    elif missing_mode == "both_rounds" and missing_fraction > 0:
        miss1 = rng.random(n_books) < missing_fraction
        miss2 = rng.random(n_books) < missing_fraction
        scores_round1[miss1] = np.nan
        scores_round2[miss2] = np.nan

    return scores_round1, scores_round2


def generate_data(seed: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate total-score and skill-score long tables.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - scores_df columns: book_id, model_id, round, score
            - skill_df columns: book_id, model_id, skill, score
    """
    rng = np.random.default_rng(config.SEED if seed is None else int(seed))

    book_ids = np.arange(config.N_BOOKS)
    true_quality = _sample_true_quality(rng)

    sigma_dim = rng.uniform(4.0, 12.0, size=config.N_SKILLS)
    dim_offsets = rng.normal(
        loc=0.0, scale=sigma_dim, size=(config.N_BOOKS, config.N_SKILLS)
    )
    true_dims = np.clip(true_quality[:, None] + dim_offsets, 0.0, 100.0)

    profiles = _build_model_profiles(rng)

    score_frames = []
    skill_frames = []

    for model_id in range(config.N_MODELS):
        p = profiles[model_id]
        reversal_mask = _build_reversal_mask(
            rng, p.reversal_mode, p.reversal_fraction, true_quality
        )

        latent_quality = true_quality.copy()
        if config.model_group(model_id) == "E" or model_id == 98:
            weighted = (true_dims * p.skill_weights[None, :]).sum(axis=1) / max(
                float(p.skill_weights.sum()), 1e-9
            )
            latent_quality = np.clip(weighted, 0.0, 100.0)

        mapped_quality = np.where(reversal_mask, 100.0 - true_quality, latent_quality)

        round_scores = []
        for round_idx in range(1, config.N_ROUNDS + 1):
            noise = _heteroscedastic_noise(
                rng, p.noise_std, true_quality, p.hetero_mode
            )
            drift = p.round2_drift if round_idx == 2 else 0.0
            base = mapped_quality + p.bias + drift
            scaled_base = (base - 50.0) * p.scale + 50.0
            score = np.clip(scaled_base + noise, 0.0, 100.0)
            round_scores.append(score.astype(np.float32))

        score_r1, score_r2 = _apply_missing(
            rng,
            round_scores[0].copy(),
            round_scores[1].copy(),
            p.missing_mode,
            p.missing_fraction,
        )

        for round_idx, score in enumerate([score_r1, score_r2], start=1):
            frame = pd.DataFrame(
                {
                    "book_id": book_ids.astype(np.int32),
                    "model_id": np.full(config.N_BOOKS, model_id, dtype=np.int16),
                    "round": np.full(config.N_BOOKS, round_idx, dtype=np.int8),
                    "score": score,
                }
            )
            score_frames.append(frame)

        for skill_idx, skill_name in enumerate(config.SKILLS):
            skill_noise = rng.normal(0.0, 2.0, size=config.N_BOOKS)
            skill_score = np.clip(
                true_dims[:, skill_idx] * p.skill_weights[skill_idx] + skill_noise,
                0.0,
                100.0,
            )
            skill_frame = pd.DataFrame(
                {
                    "book_id": book_ids.astype(np.int32),
                    "model_id": np.full(config.N_BOOKS, model_id, dtype=np.int16),
                    "skill": np.full(config.N_BOOKS, skill_name),
                    "score": skill_score.astype(np.float32),
                }
            )
            skill_frames.append(skill_frame)

    scores_df = pd.concat(score_frames, ignore_index=True)
    scores_df["model_group"] = scores_df["model_id"].astype(int).map(config.model_group)

    skill_df = pd.concat(skill_frames, ignore_index=True)
    skill_df["model_group"] = skill_df["model_id"].astype(int).map(config.model_group)
    return scores_df, skill_df
