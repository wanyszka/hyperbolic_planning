"""Core modules: environment, data generation, and datasets."""

from .environment import (
    EnvConfig,
    RandomWalkEnv,
    generate_random_trajectory,
    generate_dataset,
    reconstruct_actions,
)
from .data_generation import (
    DEFAULT_REGIMES,
    generate_regime_dataset,
    generate_all_regimes,
    save_dataset,
    load_dataset,
    load_all_regimes,
    analyze_trajectory_characteristics,
)

__all__ = [
    # Environment
    "EnvConfig",
    "RandomWalkEnv",
    "generate_random_trajectory",
    "generate_dataset",
    "reconstruct_actions",
    # Data generation
    "DEFAULT_REGIMES",
    "generate_regime_dataset",
    "generate_all_regimes",
    "save_dataset",
    "load_dataset",
    "load_all_regimes",
    "analyze_trajectory_characteristics",
]
