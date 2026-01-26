"""Training infrastructure: losses and training loops."""

from .losses import info_nce_loss, hyperbolic_info_nce_loss
from .trainer import (
    TrainingConfig,
    TrainingResult,
    train_model,
    train_encoder,
    train_policy,
)

__all__ = [
    # Losses
    "info_nce_loss",
    "hyperbolic_info_nce_loss",
    # Training
    "TrainingConfig",
    "TrainingResult",
    "train_model",
    "train_encoder",
    "train_policy",
]
