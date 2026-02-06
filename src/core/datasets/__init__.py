"""Dataset classes for contrastive learning and policy training."""

from .interval_dataset import IntervalDataset, IntervalDatasetInverted
from .trajectory_dataset import (
    TrajectoryContrastiveDataset,
    InBatchContrastiveDataset,
    PolicyTrainingDataset,
    create_train_val_test_split,
    create_contrastive_dataloader,
    create_policy_dataloader,
    create_in_batch_dataloader,
    # Config classes
    NegativeFormationType,
    AnchorSamplingConfig,
    InBatchSamplingConfig,
)
from .ablation_configs import (
    AblationConfig,
    create_ablation_config,
    get_baseline_config,
    get_negative_formation_ablations,
    get_anchor_sampling_ablations,
    get_full_ablation_grid,
    get_extended_ablation_grid,
    get_ablation_suite,
    print_ablation_summary,
)

__all__ = [
    # Interval datasets
    "IntervalDataset",
    "IntervalDatasetInverted",
    # Trajectory datasets
    "TrajectoryContrastiveDataset",
    "InBatchContrastiveDataset",
    "PolicyTrainingDataset",
    # Config classes
    "NegativeFormationType",
    "AnchorSamplingConfig",
    "InBatchSamplingConfig",
    # Ablation configs
    "AblationConfig",
    "create_ablation_config",
    "get_baseline_config",
    "get_negative_formation_ablations",
    "get_anchor_sampling_ablations",
    "get_full_ablation_grid",
    "get_extended_ablation_grid",
    "get_ablation_suite",
    "print_ablation_summary",
    # Utilities
    "create_train_val_test_split",
    "create_contrastive_dataloader",
    "create_policy_dataloader",
    "create_in_batch_dataloader",
]
