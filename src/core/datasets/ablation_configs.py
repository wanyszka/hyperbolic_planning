"""Ablation study configurations for in-batch contrastive learning experiments."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from itertools import product

from .trajectory_dataset import (
    NegativeFormationType,
    AnchorSamplingConfig,
    InBatchSamplingConfig,
)

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from src.training.trainer import TrainingConfig


def _get_training_config():
    """Lazy import of TrainingConfig to avoid circular import."""
    from src.training.trainer import TrainingConfig
    return TrainingConfig


@dataclass
class AblationConfig:
    """Complete configuration for a single ablation experiment."""

    name: str
    description: str

    # Dataset config
    sampling_config: InBatchSamplingConfig

    # Training config (use Any to avoid import issues, actual type is TrainingConfig)
    training_config: Any

    def __repr__(self) -> str:
        return f"AblationConfig(name='{self.name}')"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "name": self.name,
            "description": self.description,
            "negative_formation": self.training_config.negative_formation.value,
            "swap_probability": self.training_config.swap_probability,
            "use_geometric": self.sampling_config.anchor_sampling.use_geometric,
            "geometric_p": self.sampling_config.anchor_sampling.geometric_p,
            "min_anchor_length": self.sampling_config.anchor_sampling.min_anchor_length,
        }


def create_ablation_config(
    name: str,
    description: str = "",
    # Negative formation
    negative_formation: NegativeFormationType = NegativeFormationType.PLAIN,
    swap_probability: float = 0.5,
    # Anchor sampling
    use_geometric: bool = False,
    geometric_p: float = 0.3,
    min_anchor_length: int = 1,
    # Training params (can override defaults)
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 0.001,
    temperature: float = 0.1,
    **training_kwargs,
) -> AblationConfig:
    """
    Create a single ablation configuration.

    Args:
        name: Unique identifier for this config
        description: Human-readable description
        negative_formation: Type of in-batch negative formation
        swap_probability: Probability of swapping coords (for MIXED_SWAPPED)
        use_geometric: Use geometric distribution for anchor length
        geometric_p: Geometric distribution parameter
        min_anchor_length: Minimum anchor span in indices
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        temperature: InfoNCE temperature
        **training_kwargs: Additional TrainingConfig parameters

    Returns:
        AblationConfig instance
    """
    sampling_config = InBatchSamplingConfig(
        anchor_sampling=AnchorSamplingConfig(
            use_geometric=use_geometric,
            geometric_p=geometric_p,
            min_anchor_length=min_anchor_length,
        ),
        experiment_name=name,
    )

    TrainingConfig = _get_training_config()
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        temperature=temperature,
        negative_formation=negative_formation,
        swap_probability=swap_probability,
        **training_kwargs,
    )

    return AblationConfig(
        name=name,
        description=description,
        sampling_config=sampling_config,
        training_config=training_config,
    )


# =============================================================================
# Pre-defined ablation configurations
# =============================================================================

def get_baseline_config(**kwargs) -> AblationConfig:
    """Baseline: plain negatives, uniform anchor sampling."""
    return create_ablation_config(
        name="baseline",
        description="Plain in-batch negatives with uniform anchor sampling",
        negative_formation=NegativeFormationType.PLAIN,
        use_geometric=False,
        **kwargs,
    )


def get_negative_formation_ablations(**kwargs) -> List[AblationConfig]:
    """Ablation over negative formation strategies (holding anchor sampling fixed)."""
    configs = [
        create_ablation_config(
            name="neg_plain",
            description="Plain: positives from other samples as negatives",
            negative_formation=NegativeFormationType.PLAIN,
            use_geometric=False,
            **kwargs,
        ),
        create_ablation_config(
            name="neg_mixed",
            description="Mixed: cross-mix coordinates from different positives",
            negative_formation=NegativeFormationType.MIXED,
            use_geometric=False,
            **kwargs,
        ),
        create_ablation_config(
            name="neg_mixed_swap_0.3",
            description="Mixed + 30% swap probability",
            negative_formation=NegativeFormationType.MIXED_SWAPPED,
            swap_probability=0.3,
            use_geometric=False,
            **kwargs,
        ),
        create_ablation_config(
            name="neg_mixed_swap_0.5",
            description="Mixed + 50% swap probability",
            negative_formation=NegativeFormationType.MIXED_SWAPPED,
            swap_probability=0.5,
            use_geometric=False,
            **kwargs,
        ),
        create_ablation_config(
            name="neg_mixed_swap_0.7",
            description="Mixed + 70% swap probability",
            negative_formation=NegativeFormationType.MIXED_SWAPPED,
            swap_probability=0.7,
            use_geometric=False,
            **kwargs,
        ),
    ]
    return configs


def get_anchor_sampling_ablations(**kwargs) -> List[AblationConfig]:
    """Ablation over anchor sampling strategies (holding negative formation fixed)."""
    configs = [
        create_ablation_config(
            name="anchor_uniform",
            description="Uniform anchor sampling",
            negative_formation=NegativeFormationType.PLAIN,
            use_geometric=False,
            **kwargs,
        ),
        create_ablation_config(
            name="anchor_geo_p0.2",
            description="Geometric sampling p=0.2 (longer anchors)",
            negative_formation=NegativeFormationType.PLAIN,
            use_geometric=True,
            geometric_p=0.2,
            **kwargs,
        ),
        create_ablation_config(
            name="anchor_geo_p0.3",
            description="Geometric sampling p=0.3 (medium)",
            negative_formation=NegativeFormationType.PLAIN,
            use_geometric=True,
            geometric_p=0.3,
            **kwargs,
        ),
        create_ablation_config(
            name="anchor_geo_p0.5",
            description="Geometric sampling p=0.5 (shorter anchors)",
            negative_formation=NegativeFormationType.PLAIN,
            use_geometric=True,
            geometric_p=0.5,
            **kwargs,
        ),
    ]
    return configs


def get_full_ablation_grid(**kwargs) -> List[AblationConfig]:
    """
    Full factorial ablation grid over all parameters.

    Grid:
    - negative_formation: PLAIN, MIXED, MIXED_SWAPPED (swap=0.5)
    - use_geometric: False, True (p=0.3)

    Returns 6 configurations.
    """
    neg_configs = [
        (NegativeFormationType.PLAIN, 0.0, "plain"),
        (NegativeFormationType.MIXED, 0.0, "mixed"),
        (NegativeFormationType.MIXED_SWAPPED, 0.5, "mixed_swap"),
    ]

    geo_configs = [
        (False, 0.3, "uniform"),
        (True, 0.3, "geo"),
    ]

    configs = []
    for (neg_type, swap_prob, neg_name), (use_geo, geo_p, geo_name) in product(neg_configs, geo_configs):
        name = f"{neg_name}_{geo_name}"
        desc = f"Negative: {neg_name}, Anchor: {geo_name}"

        configs.append(create_ablation_config(
            name=name,
            description=desc,
            negative_formation=neg_type,
            swap_probability=swap_prob,
            use_geometric=use_geo,
            geometric_p=geo_p,
            **kwargs,
        ))

    return configs


def get_extended_ablation_grid(**kwargs) -> List[AblationConfig]:
    """
    Extended ablation grid with more parameter variations.

    Grid:
    - negative_formation: PLAIN, MIXED, MIXED_SWAPPED (swap=0.3, 0.5, 0.7)
    - use_geometric: False, True (p=0.2, 0.3, 0.5)

    Returns 5 * 4 = 20 configurations.
    """
    neg_configs = [
        (NegativeFormationType.PLAIN, 0.0, "plain"),
        (NegativeFormationType.MIXED, 0.0, "mixed"),
        (NegativeFormationType.MIXED_SWAPPED, 0.3, "swap0.3"),
        (NegativeFormationType.MIXED_SWAPPED, 0.5, "swap0.5"),
        (NegativeFormationType.MIXED_SWAPPED, 0.7, "swap0.7"),
    ]

    geo_configs = [
        (False, 0.3, "uniform"),
        (True, 0.2, "geo0.2"),
        (True, 0.3, "geo0.3"),
        (True, 0.5, "geo0.5"),
    ]

    configs = []
    for (neg_type, swap_prob, neg_name), (use_geo, geo_p, geo_name) in product(neg_configs, geo_configs):
        name = f"{neg_name}_{geo_name}"
        desc = f"Negative: {neg_name}, Anchor: {geo_name}"

        configs.append(create_ablation_config(
            name=name,
            description=desc,
            negative_formation=neg_type,
            swap_probability=swap_prob,
            use_geometric=use_geo,
            geometric_p=geo_p,
            **kwargs,
        ))

    return configs


# =============================================================================
# Convenience function to get all configs as a dictionary
# =============================================================================

def get_ablation_suite(
    suite: str = "full",
    **kwargs,
) -> Dict[str, AblationConfig]:
    """
    Get a suite of ablation configurations.

    Args:
        suite: One of "negative", "anchor", "full", "extended"
        **kwargs: Override default training parameters

    Returns:
        Dictionary mapping config names to AblationConfig instances
    """
    if suite == "negative":
        configs = get_negative_formation_ablations(**kwargs)
    elif suite == "anchor":
        configs = get_anchor_sampling_ablations(**kwargs)
    elif suite == "full":
        configs = get_full_ablation_grid(**kwargs)
    elif suite == "extended":
        configs = get_extended_ablation_grid(**kwargs)
    else:
        raise ValueError(f"Unknown suite: {suite}. Use 'negative', 'anchor', 'full', or 'extended'")

    return {c.name: c for c in configs}


# =============================================================================
# Pretty printing for notebooks
# =============================================================================

def print_ablation_summary(configs: List[AblationConfig]) -> None:
    """Print a summary table of ablation configurations."""
    print(f"{'Name':<25} {'Negative':<15} {'Swap':<6} {'Geo':<6} {'Geo_p':<6}")
    print("-" * 60)
    for c in configs:
        neg = c.training_config.negative_formation.value
        swap = c.training_config.swap_probability
        geo = c.sampling_config.anchor_sampling.use_geometric
        geo_p = c.sampling_config.anchor_sampling.geometric_p
        print(f"{c.name:<25} {neg:<15} {swap:<6.2f} {str(geo):<6} {geo_p:<6.2f}")
