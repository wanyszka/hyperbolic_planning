"""Tests for metrics.py - Representation and policy evaluation metrics."""

import pytest
import numpy as np
import torch
from typing import List

from src.evaluation import (
    RepresentationMetrics,
    PolicyMetrics,
    compute_hyperbolic_distance,
    compute_hyperbolic_norm,
    compute_temporal_geometric_alignment,
    compute_norm_length_correlation,
    compute_angle_midpoint_correlation,
    compute_embedding_statistics,
    compute_all_representation_metrics,
    compute_action_entropy,
    compute_policy_metrics,
)
from src.models import EuclideanIntervalEncoder, HyperbolicIntervalEncoder


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_trajectories() -> List[List[float]]:
    """Create sample trajectories for testing."""
    np.random.seed(42)
    trajectories = []
    for _ in range(50):
        length = np.random.randint(20, 80)
        traj = np.linspace(0, 1, length)
        noise = np.random.randn(length) * 0.02
        traj = np.clip(traj + noise, 0, 1)
        trajectories.append(traj.tolist())
    return trajectories


@pytest.fixture
def euclidean_encoder():
    """Create an Euclidean encoder for testing."""
    return EuclideanIntervalEncoder(embedding_dim=2)


@pytest.fixture
def euclidean_encoder_8d():
    """Create an Euclidean encoder with 8D embeddings."""
    return EuclideanIntervalEncoder(embedding_dim=8)


@pytest.fixture
def hyperbolic_encoder():
    """Create a Hyperbolic encoder for testing."""
    return HyperbolicIntervalEncoder(embedding_dim=2)


# =============================================================================
# Dataclass Tests
# =============================================================================

class TestRepresentationMetrics:
    """Tests for RepresentationMetrics dataclass."""

    def test_creation(self):
        """Test creating RepresentationMetrics."""
        metrics = RepresentationMetrics(
            temporal_geometric_alignment=0.85,
            norm_length_correlation=0.7,
            angle_midpoint_correlation=0.6,
            embedding_norm_mean=1.5,
            embedding_norm_std=0.3,
        )

        assert metrics.temporal_geometric_alignment == 0.85
        assert metrics.norm_length_correlation == 0.7
        assert metrics.angle_midpoint_correlation == 0.6

    def test_none_angle_correlation(self):
        """Test that angle_midpoint_correlation can be None."""
        metrics = RepresentationMetrics(
            temporal_geometric_alignment=0.85,
            norm_length_correlation=0.7,
            angle_midpoint_correlation=None,  # For non-2D embeddings
            embedding_norm_mean=1.5,
            embedding_norm_std=0.3,
        )

        assert metrics.angle_midpoint_correlation is None


class TestPolicyMetrics:
    """Tests for PolicyMetrics dataclass."""

    def test_creation(self):
        """Test creating PolicyMetrics."""
        metrics = PolicyMetrics(
            success_rate=0.8,
            mean_steps=110.5,
            median_steps=108.0,
            std_steps=15.2,
            path_efficiency=0.9,
            action_entropy=0.5,
        )

        assert metrics.success_rate == 0.8
        assert metrics.mean_steps == 110.5
        assert metrics.path_efficiency == 0.9


# =============================================================================
# Hyperbolic Distance Tests
# =============================================================================

class TestComputeHyperbolicDistance:
    """Tests for compute_hyperbolic_distance function."""

    def test_distance_from_origin(self):
        """Test distance from origin."""
        x = torch.zeros(5, 2)
        y = torch.tensor([[0.5, 0.0]] * 5)

        distances = compute_hyperbolic_distance(x, y)

        assert distances.shape == (5,)
        assert torch.all(distances > 0)

    def test_zero_distance_same_point(self):
        """Test that distance to same point is zero."""
        x = torch.tensor([[0.3, 0.2]] * 5)
        y = x.clone()

        distances = compute_hyperbolic_distance(x, y)

        assert torch.allclose(distances, torch.zeros(5), atol=1e-5)

    def test_symmetry(self):
        """Test that distance is symmetric."""
        x = torch.tensor([[0.2, 0.1], [0.3, 0.4]])
        y = torch.tensor([[0.5, 0.3], [-0.2, 0.1]])

        d_xy = compute_hyperbolic_distance(x, y)
        d_yx = compute_hyperbolic_distance(y, x)

        assert torch.allclose(d_xy, d_yx, atol=1e-5)

    def test_larger_distance_for_farther_points(self):
        """Test that farther points have larger distances."""
        origin = torch.zeros(1, 2)
        close = torch.tensor([[0.2, 0.0]])
        far = torch.tensor([[0.7, 0.0]])

        d_close = compute_hyperbolic_distance(origin, close)
        d_far = compute_hyperbolic_distance(origin, far)

        assert d_far > d_close

    def test_handles_batched_input(self):
        """Test with batched input."""
        x = torch.randn(100, 4) * 0.3  # Keep within Poincaré ball
        y = torch.randn(100, 4) * 0.3

        distances = compute_hyperbolic_distance(x, y)

        assert distances.shape == (100,)
        assert torch.all(torch.isfinite(distances))


# =============================================================================
# Hyperbolic Norm Tests
# =============================================================================

class TestComputeHyperbolicNorm:
    """Tests for compute_hyperbolic_norm function."""

    def test_origin_has_zero_norm(self):
        """Test that origin has zero hyperbolic norm."""
        x = torch.zeros(5, 2)

        norms = compute_hyperbolic_norm(x)

        assert torch.allclose(norms, torch.zeros(5), atol=1e-5)

    def test_positive_norms(self):
        """Test that non-origin points have positive norms."""
        x = torch.tensor([[0.3, 0.2], [0.5, 0.0], [-0.2, 0.4]])

        norms = compute_hyperbolic_norm(x)

        assert torch.all(norms > 0)

    def test_larger_euclidean_norm_gives_larger_hyperbolic_norm(self):
        """Test that points farther from origin have larger hyperbolic norm."""
        close = torch.tensor([[0.1, 0.0]])
        far = torch.tensor([[0.8, 0.0]])

        norm_close = compute_hyperbolic_norm(close)
        norm_far = compute_hyperbolic_norm(far)

        assert norm_far > norm_close

    def test_numerical_stability_near_boundary(self):
        """Test numerical stability for points near boundary."""
        # Points very close to the boundary (||x|| close to 1)
        x = torch.tensor([[0.99, 0.0], [0.0, 0.99], [0.7, 0.7]])

        norms = compute_hyperbolic_norm(x)

        assert torch.all(torch.isfinite(norms))
        assert torch.all(norms > 0)


# =============================================================================
# Temporal-Geometric Alignment Tests
# =============================================================================

class TestTemporalGeometricAlignment:
    """Tests for compute_temporal_geometric_alignment function."""

    def test_returns_valid_fraction(self, euclidean_encoder, sample_trajectories):
        """Test that alignment is between 0 and 1."""
        alignment = compute_temporal_geometric_alignment(
            euclidean_encoder,
            sample_trajectories,
            n_samples=500,
            seed=42,
        )

        assert 0.0 <= alignment <= 1.0

    def test_monotonic_trajectories_high_alignment(self):
        """Test that strictly monotonic trajectories have high alignment."""
        # Create strictly monotonic trajectories
        mono_trajs = [list(np.linspace(0, 1, 50)) for _ in range(20)]

        encoder = EuclideanIntervalEncoder(embedding_dim=2)

        alignment = compute_temporal_geometric_alignment(
            encoder, mono_trajs, n_samples=500, seed=42
        )

        # For monotonic trajectories, temporal containment implies geometric containment
        assert alignment > 0.8

    def test_reproducibility(self, euclidean_encoder, sample_trajectories):
        """Test that same seed gives same result."""
        a1 = compute_temporal_geometric_alignment(
            euclidean_encoder, sample_trajectories, n_samples=100, seed=42
        )
        a2 = compute_temporal_geometric_alignment(
            euclidean_encoder, sample_trajectories, n_samples=100, seed=42
        )

        assert a1 == a2


# =============================================================================
# Norm-Length Correlation Tests
# =============================================================================

class TestNormLengthCorrelation:
    """Tests for compute_norm_length_correlation function."""

    def test_returns_valid_correlation(self, euclidean_encoder, sample_trajectories):
        """Test that correlation is between -1 and 1."""
        correlation = compute_norm_length_correlation(
            euclidean_encoder,
            sample_trajectories,
            n_samples=500,
            seed=42,
        )

        assert -1.0 <= correlation <= 1.0

    def test_reproducibility(self, euclidean_encoder, sample_trajectories):
        """Test that same seed gives same result."""
        c1 = compute_norm_length_correlation(
            euclidean_encoder, sample_trajectories, n_samples=100, seed=42
        )
        c2 = compute_norm_length_correlation(
            euclidean_encoder, sample_trajectories, n_samples=100, seed=42
        )

        assert c1 == c2

    def test_works_with_hyperbolic_encoder(self, hyperbolic_encoder, sample_trajectories):
        """Test that it works with hyperbolic encoder."""
        correlation = compute_norm_length_correlation(
            hyperbolic_encoder,
            sample_trajectories,
            n_samples=500,
            seed=42,
        )

        assert -1.0 <= correlation <= 1.0


# =============================================================================
# Angle-Midpoint Correlation Tests
# =============================================================================

class TestAngleMidpointCorrelation:
    """Tests for compute_angle_midpoint_correlation function."""

    def test_returns_valid_correlation_for_2d(self, euclidean_encoder, sample_trajectories):
        """Test that correlation is between -1 and 1 for 2D encoder."""
        correlation = compute_angle_midpoint_correlation(
            euclidean_encoder,  # 2D
            sample_trajectories,
            n_samples=500,
            seed=42,
        )

        assert correlation is not None
        assert -1.0 <= correlation <= 1.0

    def test_returns_none_for_non_2d(self, euclidean_encoder_8d, sample_trajectories):
        """Test that returns None for non-2D encoder."""
        correlation = compute_angle_midpoint_correlation(
            euclidean_encoder_8d,  # 8D
            sample_trajectories,
            n_samples=100,
        )

        assert correlation is None

    def test_works_with_hyperbolic_encoder(self, hyperbolic_encoder, sample_trajectories):
        """Test that it works with 2D hyperbolic encoder."""
        correlation = compute_angle_midpoint_correlation(
            hyperbolic_encoder,  # 2D
            sample_trajectories,
            n_samples=500,
            seed=42,
        )

        assert correlation is not None
        assert -1.0 <= correlation <= 1.0



# =============================================================================
# Embedding Statistics Tests
# =============================================================================

class TestEmbeddingStatistics:
    """Tests for compute_embedding_statistics function."""

    def test_returns_all_statistics(self, euclidean_encoder, sample_trajectories):
        """Test that all expected statistics are returned."""
        stats = compute_embedding_statistics(
            euclidean_encoder,
            sample_trajectories,
            n_samples=500,
            seed=42,
        )

        expected_keys = [
            "euclidean_norm_mean",
            "euclidean_norm_std",
            "euclidean_norm_min",
            "euclidean_norm_max",
            "hyperbolic_norm_mean",
            "hyperbolic_norm_std",
        ]

        for key in expected_keys:
            assert key in stats

    def test_positive_statistics(self, euclidean_encoder, sample_trajectories):
        """Test that norm statistics are positive."""
        stats = compute_embedding_statistics(
            euclidean_encoder,
            sample_trajectories,
            n_samples=500,
        )

        assert stats["euclidean_norm_mean"] >= 0
        assert stats["euclidean_norm_std"] >= 0
        assert stats["euclidean_norm_min"] >= 0

    def test_valid_ranges(self, euclidean_encoder, sample_trajectories):
        """Test that min <= mean <= max."""
        stats = compute_embedding_statistics(
            euclidean_encoder,
            sample_trajectories,
            n_samples=500,
        )

        assert stats["euclidean_norm_min"] <= stats["euclidean_norm_mean"]
        assert stats["euclidean_norm_mean"] <= stats["euclidean_norm_max"]


# =============================================================================
# All Representation Metrics Tests
# =============================================================================

class TestComputeAllRepresentationMetrics:
    """Tests for compute_all_representation_metrics function."""

    def test_returns_representation_metrics(self, euclidean_encoder, sample_trajectories):
        """Test that function returns RepresentationMetrics."""
        metrics = compute_all_representation_metrics(
            euclidean_encoder,
            sample_trajectories,
            n_samples=500,
            seed=42,
        )

        assert isinstance(metrics, RepresentationMetrics)

    def test_all_fields_populated(self, euclidean_encoder, sample_trajectories):
        """Test that all fields are populated."""
        metrics = compute_all_representation_metrics(
            euclidean_encoder,
            sample_trajectories,
            n_samples=500,
        )

        assert metrics.temporal_geometric_alignment is not None
        assert metrics.norm_length_correlation is not None
        # angle_midpoint_correlation can be None for non-2D
        assert metrics.embedding_norm_mean is not None
        assert metrics.embedding_norm_std is not None

    def test_works_with_hyperbolic_encoder(self, hyperbolic_encoder, sample_trajectories):
        """Test that it works with hyperbolic encoder."""
        metrics = compute_all_representation_metrics(
            hyperbolic_encoder,
            sample_trajectories,
            n_samples=500,
        )

        assert isinstance(metrics, RepresentationMetrics)


# =============================================================================
# Action Entropy Tests
# =============================================================================

class TestComputeActionEntropy:
    """Tests for compute_action_entropy function."""

    def test_uniform_distribution_max_entropy(self):
        """Test that uniform distribution gives maximum entropy."""
        # Uniform over 3 actions
        probs = torch.ones(100, 3) / 3

        entropy = compute_action_entropy(probs)

        # Max entropy for 3 actions is log(3) ≈ 1.099
        expected_max = np.log(3)
        assert abs(entropy - expected_max) < 0.01

    def test_deterministic_distribution_zero_entropy(self):
        """Test that deterministic distribution gives zero entropy."""
        # Always pick first action
        probs = torch.zeros(100, 3)
        probs[:, 0] = 1.0

        entropy = compute_action_entropy(probs)

        assert abs(entropy) < 1e-5

    def test_intermediate_entropy(self):
        """Test intermediate entropy values."""
        # Some uncertainty but not uniform
        probs = torch.tensor([[0.7, 0.2, 0.1]] * 50, dtype=torch.float32)

        entropy = compute_action_entropy(probs)

        # Should be between 0 and log(3)
        assert 0 < entropy < np.log(3)


# =============================================================================
# Policy Metrics Tests
# =============================================================================

class TestComputePolicyMetrics:
    """Tests for compute_policy_metrics function."""

    def test_basic_metrics(self):
        """Test basic policy metrics computation."""
        success_flags = [True, True, False, True, False]
        steps = [100, 110, 200, 105, 200]  # Failed episodes have max_steps

        metrics = compute_policy_metrics(success_flags, steps, n_bins=100)

        assert metrics.success_rate == 0.6  # 3/5
        assert metrics.mean_steps > 0
        assert metrics.path_efficiency > 0

    def test_all_success(self):
        """Test metrics when all episodes succeed."""
        success_flags = [True] * 10
        steps = [100, 105, 110, 108, 102, 99, 115, 120, 95, 100]

        metrics = compute_policy_metrics(success_flags, steps, n_bins=100)

        assert metrics.success_rate == 1.0
        assert 95 <= metrics.mean_steps <= 120

    def test_all_failure(self):
        """Test metrics when all episodes fail."""
        success_flags = [False] * 10
        steps = [200] * 10

        metrics = compute_policy_metrics(success_flags, steps, n_bins=100)

        assert metrics.success_rate == 0.0
        assert metrics.mean_steps == float('inf')
        assert metrics.path_efficiency == 0.0

    def test_path_efficiency_calculation(self):
        """Test path efficiency is computed correctly."""
        success_flags = [True, True]
        steps = [100, 200]  # Optimal is 100, second is twice as long

        metrics = compute_policy_metrics(success_flags, steps, n_bins=100)

        # efficiency = n_bins / steps
        # (100/100 + 100/200) / 2 = (1.0 + 0.5) / 2 = 0.75
        assert abs(metrics.path_efficiency - 0.75) < 1e-5

    def test_with_action_entropies(self):
        """Test metrics with action entropies provided."""
        success_flags = [True] * 5
        steps = [100] * 5
        entropies = [0.5, 0.6, 0.4, 0.5, 0.5]

        metrics = compute_policy_metrics(
            success_flags, steps, n_bins=100, action_entropies=entropies
        )

        assert abs(metrics.action_entropy - 0.5) < 1e-5

    def test_empty_input(self):
        """Test handling of empty input."""
        metrics = compute_policy_metrics([], [], n_bins=100)

        assert metrics.success_rate == 0.0
