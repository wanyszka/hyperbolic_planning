"""Hyperbolic Planning - Interval Representations for Goal-Conditioned Planning."""

from .core import EnvConfig, RandomWalkEnv
from .models import (
    EuclideanIntervalEncoder,
    HyperbolicIntervalEncoder,
    GCBCSinglePolicy,
    GCBCIntervalPolicy,
    create_encoder,
)
from .training import TrainingConfig, train_encoder, train_policy
from .evaluation import evaluate_policy, PolicyMetrics

__version__ = "0.1.0"

__all__ = [
    # Core
    "EnvConfig",
    "RandomWalkEnv",
    # Models
    "EuclideanIntervalEncoder",
    "HyperbolicIntervalEncoder",
    "GCBCSinglePolicy",
    "GCBCIntervalPolicy",
    "create_encoder",
    # Training
    "TrainingConfig",
    "train_encoder",
    "train_policy",
    # Evaluation
    "evaluate_policy",
    "PolicyMetrics",
]
