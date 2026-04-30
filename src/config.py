"""Global configuration for the LLM evaluation simulation project."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

SEED: int = 42
RANDOM_SEED: int = SEED

N_BOOKS: int = 10000
N_MODELS: int = 100
N_ROUNDS: int = 2
N_JOBS: int = -1

THETA_REL: float = 0.80
THETA_CON: float = 0.80

P_HIGH: float = 20.0
P_LOW: float = 40.0

DEFAULT_PERCENTILE_SCHEME: str = "20/40/40"
DEFAULT_DECISION_METHOD: str = "percentile"
BOOTSTRAP_ITERATIONS: int = 1000
BOOTSTRAP_SCHEMES: List[str] = ["15/35/50", "20/40/40", "25/50/25"]
THRESHOLD_SCAN_VALUES: List[float] = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
MONTE_CARLO_SEEDS: List[int] = [42, 52, 62, 72, 82, 92, 102, 112, 122, 132]
DENSITY_KDE_SAMPLE_N: int = 6000

SKILLS: List[str] = [
    "学术价值",
    "主题相关性",
    "可读性",
    "权威性/可信度",
    "馆藏适配度",
]
N_SKILLS: int = 5

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
RESULTS_DIR: Path = PROJECT_ROOT / "results"
RESULTS_SUBDIRS: Dict[str, str] = {
    "layer1": "layer1",
    "layer2": "layer2",
    "layer3": "layer3",
    "layer4": "layer4",
    "sensitivity": "sensitivity",
    "bootstrap": "bootstrap",
    "monte_carlo": "monte_carlo",
    "reports": "reports",
}

MODEL_GROUPS: Dict[str, tuple[int, int]] = {
    "A": (0, 9),
    "B": (10, 19),
    "C": (20, 29),
    "D": (30, 39),
    "E": (40, 49),
    "F": (50, 59),
    "G": (60, 69),
    "H": (70, 79),
    "I": (80, 89),
    "J": (90, 99),
}


def model_group(model_id: int) -> str:
    """Map model id to group label A-J."""
    group_idx = int(model_id) // 10
    return chr(ord("A") + group_idx)


def to_quantile_fraction(value: float) -> float:
    """Convert percentile-like value to [0, 1] fraction."""
    if value > 1.0:
        return value / 100.0
    return value


def get_results_dir(section: str) -> Path:
    """Return a section-specific output directory under results/."""
    if section not in RESULTS_SUBDIRS:
        raise KeyError(f"Unknown results section: {section}")
    return RESULTS_DIR / RESULTS_SUBDIRS[section]


def ensure_results_dirs() -> Dict[str, Path]:
    """Create all configured output folders and return their paths."""
    dirs: Dict[str, Path] = {}
    for key in RESULTS_SUBDIRS:
        path = get_results_dir(key)
        path.mkdir(parents=True, exist_ok=True)
        dirs[key] = path
    return dirs
