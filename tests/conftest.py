"""Shared pytest fixtures for hyperbolic planning tests."""

import pytest
import torch
import numpy as np
from typing import List, Tuple

from src.core import EnvConfig, RandomWalkEnv
from src.models import (
    EuclideanIntervalEncoder,
    HyperbolicIntervalEncoder,
    GCBCSinglePolicy,
    GCBCIntervalPolicy,
)


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def device() -> str:
    """Return compute device (CPU for testing)."""
    return "cpu"


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def env_config() -> EnvConfig:
    """Return default environment configuration."""
    return EnvConfig(n_bins=100, max_steps=200)


@pytest.fixture
def small_env_config() -> EnvConfig:
    """Return small environment configuration for fast tests."""
    return EnvConfig(n_bins=10, max_steps=50)


@pytest.fixture
def random_walk_env(env_config: EnvConfig) -> RandomWalkEnv:
    """Return a RandomWalkEnv instance."""
    return RandomWalkEnv(env_config)


@pytest.fixture
def small_env(small_env_config: EnvConfig) -> RandomWalkEnv:
    """Return a small RandomWalkEnv instance for fast tests."""
    return RandomWalkEnv(small_env_config)


# =============================================================================
# Trajectory Fixtures
# =============================================================================

@pytest.fixture
def sample_trajectories() -> List[List[float]]:
    """Return sample trajectories for testing."""
    np.random.seed(42)
    trajectories = []
    for _ in range(50):
        length = np.random.randint(20, 80)
        # Create a trajectory going roughly from 0 to 1
        traj = np.linspace(0, 1, length)
        # Add some noise
        noise = np.random.randn(length) * 0.02
        traj = np.clip(traj + noise, 0, 1)
        trajectories.append(traj.tolist())
    return trajectories


@pytest.fixture
def sample_actions(sample_trajectories: List[List[float]]) -> List[List[int]]:
    """Return sample actions corresponding to trajectories."""
    actions = []
    for traj in sample_trajectories:
        # Compute actions from trajectory differences
        acts = []
        for i in range(len(traj) - 1):
            delta = traj[i + 1] - traj[i]
            if delta > 0.005:
                acts.append(2)  # Right
            elif delta < -0.005:
                acts.append(0)  # Left
            else:
                acts.append(1)  # Stay
        actions.append(acts)
    return actions


@pytest.fixture
def short_trajectories() -> List[List[float]]:
    """Return short trajectories for edge case testing."""
    return [
        [0.0, 0.1, 0.2, 0.3],
        [0.0, 0.05, 0.1, 0.15, 0.2],
        [0.5, 0.6, 0.7],
    ]


@pytest.fixture
def monotonic_trajectories() -> List[List[float]]:
    """Return strictly monotonic trajectories."""
    return [
        list(np.linspace(0, 1, 100)),
        list(np.linspace(0, 0.5, 50)),
        list(np.linspace(0.2, 0.8, 60)),
    ]


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def euclidean_encoder(device: str) -> EuclideanIntervalEncoder:
    """Return an EuclideanIntervalEncoder instance."""
    return EuclideanIntervalEncoder(embedding_dim=8).to(device)


@pytest.fixture
def euclidean_encoder_2d(device: str) -> EuclideanIntervalEncoder:
    """Return a 2D EuclideanIntervalEncoder instance."""
    return EuclideanIntervalEncoder(embedding_dim=2).to(device)


@pytest.fixture
def hyperbolic_encoder(device: str) -> HyperbolicIntervalEncoder:
    """Return a HyperbolicIntervalEncoder instance."""
    return HyperbolicIntervalEncoder(embedding_dim=8).to(device)


@pytest.fixture
def hyperbolic_encoder_2d(device: str) -> HyperbolicIntervalEncoder:
    """Return a 2D HyperbolicIntervalEncoder instance."""
    return HyperbolicIntervalEncoder(embedding_dim=2).to(device)


@pytest.fixture
def gcbc_single_policy(device: str) -> GCBCSinglePolicy:
    """Return a GCBCSinglePolicy instance."""
    return GCBCSinglePolicy(n_actions=3, hidden_sizes=[32, 32]).to(device)


@pytest.fixture
def gcbc_interval_policy(
    euclidean_encoder: EuclideanIntervalEncoder, device: str
) -> GCBCIntervalPolicy:
    """Return a GCBCIntervalPolicy instance with frozen encoder."""
    return GCBCIntervalPolicy(
        encoder=euclidean_encoder,
        n_actions=3,
        hidden_sizes=[32, 32],
        freeze_encoder=True,
    ).to(device)


# =============================================================================
# Tensor Fixtures
# =============================================================================

@pytest.fixture
def sample_intervals(device: str) -> torch.Tensor:
    """Return sample interval tensors for testing."""
    torch.manual_seed(42)
    intervals = torch.rand(32, 2)
    # Ensure start <= end
    intervals = torch.sort(intervals, dim=1)[0]
    return intervals.to(device)


@pytest.fixture
def sample_states(device: str) -> torch.Tensor:
    """Return sample state tensors."""
    torch.manual_seed(42)
    return torch.rand(32, 1).to(device)


@pytest.fixture
def sample_goals(device: str) -> torch.Tensor:
    """Return sample goal tensors."""
    return torch.ones(32, 1).to(device)


# =============================================================================
# Hyperbolic Manifold Fixtures
# =============================================================================

@pytest.fixture
def poincare_manifold():
    """Return a Poincaré ball manifold."""
    from hypll.manifolds.poincare_ball import PoincareBall, Curvature
    curvature = Curvature(value=1.0, requires_grad=False)
    return PoincareBall(c=curvature)


# =============================================================================
# Dataset Fixtures
# =============================================================================

@pytest.fixture
def interval_dataset():
    """Return an IntervalDataset instance."""
    from src.core.datasets import IntervalDataset
    return IntervalDataset(num_samples=100, num_negatives=5, num_points=20, seed=42)


@pytest.fixture
def trajectory_contrastive_dataset(sample_trajectories: List[List[float]]):
    """Return a TrajectoryContrastiveDataset instance."""
    from src.core.datasets import TrajectoryContrastiveDataset
    return TrajectoryContrastiveDataset(
        trajectories=sample_trajectories,
        num_samples=100,
        n_negatives=5,
        seed=42,
    )


@pytest.fixture
def policy_training_dataset(
    sample_trajectories: List[List[float]], sample_actions: List[List[int]]
):
    """Return a PolicyTrainingDataset instance."""
    from src.core.datasets import PolicyTrainingDataset
    return PolicyTrainingDataset(
        trajectories=sample_trajectories,
        actions=sample_actions,
        goal_type="final",
        seed=42,
    )


# =============================================================================
# Utility Functions for Tests
# =============================================================================

def assert_tensor_in_range(tensor: torch.Tensor, min_val: float, max_val: float):
    """Assert all values in tensor are within specified range."""
    assert (tensor >= min_val).all(), f"Values below {min_val} found"
    assert (tensor <= max_val).all(), f"Values above {max_val} found"


def assert_valid_intervals(intervals: torch.Tensor):
    """Assert intervals have start <= end."""
    assert (intervals[:, 0] <= intervals[:, 1] + 1e-6).all(), "Invalid intervals: start > end"


def create_dummy_trajectories(
    n_trajectories: int = 10,
    min_length: int = 20,
    max_length: int = 50,
    seed: int = 42,
) -> Tuple[List[List[float]], List[List[int]]]:
    """Create dummy trajectories and actions for testing."""
    np.random.seed(seed)
    trajectories = []
    actions = []

    for _ in range(n_trajectories):
        length = np.random.randint(min_length, max_length)
        traj = list(np.linspace(0, 1, length))
        trajectories.append(traj)
        # All right actions since trajectory is monotonic
        actions.append([2] * (length - 1))

    return trajectories, actions
