"""Tests for model.py - Interval encoders and policy networks."""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.models import (
    manifold_map,
    EuclideanIntervalEncoder,
    HyperbolicIntervalEncoder,
    create_encoder,
    PolicyNetwork,
    GCBCSinglePolicy,
    GCBCIntervalPolicy,
)
from hypll.tensors.manifold_tensor import ManifoldTensor


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_intervals():
    """Create sample interval tensors."""
    torch.manual_seed(42)
    intervals = torch.rand(32, 2)
    # Ensure start <= end by sorting
    intervals = torch.sort(intervals, dim=1)[0]
    return intervals


@pytest.fixture
def euclidean_encoder_2d():
    """Create a 2D Euclidean encoder."""
    return EuclideanIntervalEncoder(embedding_dim=2)


@pytest.fixture
def euclidean_encoder_8d():
    """Create an 8D Euclidean encoder."""
    return EuclideanIntervalEncoder(embedding_dim=8)


@pytest.fixture
def hyperbolic_encoder_2d():
    """Create a 2D Hyperbolic encoder."""
    return HyperbolicIntervalEncoder(embedding_dim=2)


@pytest.fixture
def hyperbolic_encoder_8d():
    """Create an 8D Hyperbolic encoder."""
    return HyperbolicIntervalEncoder(embedding_dim=8)


# =============================================================================
# EuclideanIntervalEncoder Tests
# =============================================================================

class TestEuclideanIntervalEncoder:
    """Tests for EuclideanIntervalEncoder."""

    def test_default_initialization(self):
        """Test default initialization."""
        encoder = EuclideanIntervalEncoder()
        assert encoder.embedding_dim == 2
        assert encoder.hidden_sizes == [128, 128, 128, 128]
        assert encoder.manifold is None

    def test_custom_initialization(self):
        """Test custom initialization."""
        encoder = EuclideanIntervalEncoder(
            embedding_dim=16,
            hidden_sizes=[64, 64],
            activation="tanh",
        )
        assert encoder.embedding_dim == 16
        assert encoder.hidden_sizes == [64, 64]

    def test_forward_shape(self, euclidean_encoder_2d, sample_intervals):
        """Test output shape."""
        output = euclidean_encoder_2d(sample_intervals)
        assert output.shape == (32, 2)

    def test_forward_8d_shape(self, euclidean_encoder_8d, sample_intervals):
        """Test 8D output shape."""
        output = euclidean_encoder_8d(sample_intervals)
        assert output.shape == (32, 8)

    def test_output_is_tensor(self, euclidean_encoder_2d, sample_intervals):
        """Test that output is a regular tensor (not ManifoldTensor)."""
        output = euclidean_encoder_2d(sample_intervals)
        assert isinstance(output, torch.Tensor)
        assert not isinstance(output, ManifoldTensor)

    def test_batch_processing(self, euclidean_encoder_2d):
        """Test with different batch sizes."""
        for batch_size in [1, 16, 64]:
            intervals = torch.rand(batch_size, 2)
            output = euclidean_encoder_2d(intervals)
            assert output.shape == (batch_size, 2)

    def test_differentiable(self, euclidean_encoder_2d, sample_intervals):
        """Test that output is differentiable."""
        output = euclidean_encoder_2d(sample_intervals)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for param in euclidean_encoder_2d.parameters():
            assert param.grad is not None

    def test_invalid_activation_raises(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            EuclideanIntervalEncoder(activation="invalid")


# =============================================================================
# HyperbolicIntervalEncoder Tests
# =============================================================================

class TestHyperbolicIntervalEncoder:
    """Tests for HyperbolicIntervalEncoder."""

    def test_default_initialization(self):
        """Test default initialization."""
        encoder = HyperbolicIntervalEncoder()
        assert encoder.embedding_dim == 2
        assert encoder.manifold is not None

    def test_custom_initialization(self):
        """Test custom initialization."""
        encoder = HyperbolicIntervalEncoder(
            embedding_dim=16,
            c=2.0,
            euc_width=64,
            hyp_width=32,
        )
        assert encoder.embedding_dim == 16

    def test_forward_returns_manifold_tensor(self, hyperbolic_encoder_2d, sample_intervals):
        """Test that forward returns ManifoldTensor."""
        output = hyperbolic_encoder_2d(sample_intervals)
        assert isinstance(output, ManifoldTensor)

    def test_forward_shape(self, hyperbolic_encoder_2d, sample_intervals):
        """Test output tensor shape."""
        output = hyperbolic_encoder_2d(sample_intervals)
        assert output.tensor.shape == (32, 2)

    def test_output_in_poincare_ball(self, hyperbolic_encoder_2d, sample_intervals):
        """Test that output lies within Poincaré ball (norm < 1)."""
        output = hyperbolic_encoder_2d(sample_intervals)
        norms = torch.norm(output.tensor, dim=-1)
        assert torch.all(norms < 1.0)

    def test_get_euclidean_embeddings(self, hyperbolic_encoder_2d, sample_intervals):
        """Test get_euclidean_embeddings method."""
        emb = hyperbolic_encoder_2d.get_euclidean_embeddings(sample_intervals)
        assert isinstance(emb, torch.Tensor)
        assert not isinstance(emb, ManifoldTensor)
        assert emb.shape == (32, 2)

    def test_batch_processing(self, hyperbolic_encoder_2d):
        """Test with different batch sizes."""
        for batch_size in [1, 16, 64]:
            intervals = torch.rand(batch_size, 2)
            output = hyperbolic_encoder_2d(intervals)
            assert output.tensor.shape == (batch_size, 2)


# =============================================================================
# Create Encoder Factory Tests
# =============================================================================

class TestCreateEncoder:
    """Tests for create_encoder factory function."""

    def test_creates_euclidean(self):
        """Test creating Euclidean encoder."""
        encoder = create_encoder("euclidean", embedding_dim=4)
        assert isinstance(encoder, EuclideanIntervalEncoder)
        assert encoder.embedding_dim == 4

    def test_creates_hyperbolic(self):
        """Test creating Hyperbolic encoder."""
        encoder = create_encoder("hyperbolic", embedding_dim=4)
        assert isinstance(encoder, HyperbolicIntervalEncoder)
        assert encoder.embedding_dim == 4

    def test_passes_hidden_sizes_euclidean(self):
        """Test that hidden_sizes are passed to Euclidean encoder."""
        encoder = create_encoder(
            "euclidean",
            embedding_dim=2,
            hidden_sizes=[32, 32],
        )
        assert encoder.hidden_sizes == [32, 32]

    def test_passes_curvature_hyperbolic(self):
        """Test that curvature is passed to Hyperbolic encoder."""
        encoder = create_encoder(
            "hyperbolic",
            embedding_dim=2,
            curvature=2.0,
        )
        # Can check manifold exists
        assert encoder.manifold is not None

    def test_invalid_geometry_raises(self):
        """Test that invalid geometry raises error."""
        with pytest.raises(ValueError, match="Unknown geometry"):
            create_encoder("invalid")


# =============================================================================
# PolicyNetwork Tests
# =============================================================================

class TestPolicyNetwork:
    """Tests for PolicyNetwork."""

    def test_initialization(self):
        """Test initialization."""
        policy = PolicyNetwork(input_dim=4, n_actions=3)
        assert policy.input_dim == 4
        assert policy.n_actions == 3

    def test_forward_shape(self):
        """Test forward output shape."""
        policy = PolicyNetwork(input_dim=4, n_actions=3)
        x = torch.rand(32, 4)
        logits = policy(x)
        assert logits.shape == (32, 3)

    def test_get_action_deterministic(self):
        """Test deterministic action selection."""
        policy = PolicyNetwork(input_dim=2, n_actions=3)
        x = torch.rand(10, 2)
        actions = policy.get_action(x, deterministic=True)

        assert actions.shape == (10,)
        assert torch.all(actions >= 0)
        assert torch.all(actions < 3)

    def test_get_action_stochastic(self):
        """Test stochastic action selection."""
        policy = PolicyNetwork(input_dim=2, n_actions=3)
        x = torch.rand(10, 2)
        actions = policy.get_action(x, deterministic=False)

        assert actions.shape == (10,)
        assert torch.all(actions >= 0)
        assert torch.all(actions < 3)

    def test_get_action_probs_sum_to_one(self):
        """Test that action probabilities sum to 1."""
        policy = PolicyNetwork(input_dim=2, n_actions=3)
        x = torch.rand(10, 2)
        probs = policy.get_action_probs(x)

        assert probs.shape == (10, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(10), atol=1e-5)

    def test_custom_hidden_sizes(self):
        """Test with custom hidden sizes."""
        policy = PolicyNetwork(
            input_dim=2,
            n_actions=5,
            hidden_sizes=[128, 64, 32],
        )
        x = torch.rand(8, 2)
        logits = policy(x)
        assert logits.shape == (8, 5)


# =============================================================================
# GCBCSinglePolicy Tests
# =============================================================================

class TestGCBCSinglePolicy:
    """Tests for GCBCSinglePolicy."""

    def test_initialization(self):
        """Test initialization."""
        policy = GCBCSinglePolicy(n_actions=3)
        assert policy.policy.input_dim == 2  # [state, goal]

    def test_forward_with_1d_input(self):
        """Test forward with 1D state/goal."""
        policy = GCBCSinglePolicy(n_actions=3)
        state = torch.rand(10)  # 1D
        goal = torch.rand(10)
        logits = policy(state, goal)
        assert logits.shape == (10, 3)

    def test_forward_with_2d_input(self):
        """Test forward with 2D state/goal."""
        policy = GCBCSinglePolicy(n_actions=3)
        state = torch.rand(10, 1)  # 2D
        goal = torch.rand(10, 1)
        logits = policy(state, goal)
        assert logits.shape == (10, 3)

    def test_get_action(self):
        """Test get_action method."""
        policy = GCBCSinglePolicy(n_actions=3)
        state = torch.rand(5, 1)
        goal = torch.ones(5, 1)
        actions = policy.get_action(state, goal)

        assert actions.shape == (5,)
        assert torch.all(actions >= 0)
        assert torch.all(actions < 3)


# =============================================================================
# GCBCIntervalPolicy Tests
# =============================================================================

class TestGCBCIntervalPolicy:
    """Tests for GCBCIntervalPolicy."""

    def test_initialization_with_euclidean(self, euclidean_encoder_2d):
        """Test initialization with Euclidean encoder."""
        policy = GCBCIntervalPolicy(
            encoder=euclidean_encoder_2d,
            n_actions=3,
            freeze_encoder=True,
        )
        assert policy.policy.input_dim == 2  # embedding_dim

    def test_initialization_with_hyperbolic(self, hyperbolic_encoder_2d):
        """Test initialization with Hyperbolic encoder."""
        policy = GCBCIntervalPolicy(
            encoder=hyperbolic_encoder_2d,
            n_actions=3,
        )
        assert policy.policy.input_dim == 2

    def test_forward_shape(self, euclidean_encoder_2d):
        """Test forward output shape."""
        policy = GCBCIntervalPolicy(encoder=euclidean_encoder_2d, n_actions=3)
        state = torch.rand(10, 1)
        goal = torch.rand(10, 1)
        logits = policy(state, goal)
        assert logits.shape == (10, 3)

    def test_frozen_encoder(self, euclidean_encoder_2d):
        """Test that encoder parameters are frozen when freeze_encoder=True."""
        policy = GCBCIntervalPolicy(
            encoder=euclidean_encoder_2d,
            n_actions=3,
            freeze_encoder=True,
        )

        for param in policy.encoder.parameters():
            assert not param.requires_grad

    def test_unfrozen_encoder(self, euclidean_encoder_2d):
        """Test that encoder parameters are trainable when freeze_encoder=False."""
        policy = GCBCIntervalPolicy(
            encoder=euclidean_encoder_2d,
            n_actions=3,
            freeze_encoder=False,
        )

        for param in policy.encoder.parameters():
            assert param.requires_grad

    def test_get_action(self, euclidean_encoder_2d):
        """Test get_action method."""
        policy = GCBCIntervalPolicy(encoder=euclidean_encoder_2d, n_actions=3)
        state = torch.rand(5, 1)
        goal = torch.ones(5, 1)
        actions = policy.get_action(state, goal)

        assert actions.shape == (5,)
        assert torch.all(actions >= 0)
        assert torch.all(actions < 3)

    def test_works_with_hyperbolic_encoder(self, hyperbolic_encoder_2d):
        """Test that policy works with hyperbolic encoder."""
        policy = GCBCIntervalPolicy(
            encoder=hyperbolic_encoder_2d,
            n_actions=3,
        )
        state = torch.rand(8, 1)
        goal = torch.rand(8, 1)
        logits = policy(state, goal)
        actions = policy.get_action(state, goal)

        assert logits.shape == (8, 3)
        assert actions.shape == (8,)


# =============================================================================
# Integration Tests
# =============================================================================

class TestEncoderPolicyIntegration:
    """Integration tests for encoder-policy combinations."""

    def test_euclidean_single_forward_backward(self):
        """Test Euclidean encoder with single policy forward/backward."""
        encoder = EuclideanIntervalEncoder(embedding_dim=8)
        policy = GCBCIntervalPolicy(encoder=encoder, n_actions=3, freeze_encoder=False)

        state = torch.rand(16, 1)
        goal = torch.rand(16, 1)
        target = torch.randint(0, 3, (16,))

        logits = policy(state, goal)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        # Check gradients flow through
        for param in policy.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_hyperbolic_single_forward_backward(self):
        """Test Hyperbolic encoder with single policy forward/backward."""
        encoder = HyperbolicIntervalEncoder(embedding_dim=8)
        policy = GCBCIntervalPolicy(encoder=encoder, n_actions=3, freeze_encoder=False)

        state = torch.rand(16, 1)
        goal = torch.rand(16, 1)
        target = torch.randint(0, 3, (16,))

        logits = policy(state, goal)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        # Check policy gradients exist
        for param in policy.policy.parameters():
            assert param.grad is not None

    def test_different_embedding_dims(self):
        """Test with different embedding dimensions."""
        for dim in [2, 8, 16]:
            euc_encoder = EuclideanIntervalEncoder(embedding_dim=dim)
            euc_policy = GCBCIntervalPolicy(encoder=euc_encoder, n_actions=3)

            hyp_encoder = HyperbolicIntervalEncoder(embedding_dim=dim)
            hyp_policy = GCBCIntervalPolicy(encoder=hyp_encoder, n_actions=3)

            state = torch.rand(4, 1)
            goal = torch.rand(4, 1)

            euc_logits = euc_policy(state, goal)
            hyp_logits = hyp_policy(state, goal)

            assert euc_logits.shape == (4, 3)
            assert hyp_logits.shape == (4, 3)


# =============================================================================
# Manifold Map Tests
# =============================================================================

class TestManifoldMap:
    """Tests for manifold_map function."""

    def test_maps_to_poincare_ball(self):
        """Test that manifold_map produces points in Poincaré ball."""
        from hypll.manifolds.poincare_ball import PoincareBall, Curvature

        curvature = Curvature(value=1.0, requires_grad=False)
        manifold = PoincareBall(c=curvature)

        x = torch.randn(32, 8) * 0.1  # Small magnitudes to stay in ball
        result = manifold_map(x, manifold)

        assert isinstance(result, ManifoldTensor)
        norms = torch.norm(result.tensor, dim=-1)
        assert torch.all(norms < 1.0)

    def test_origin_maps_to_near_origin(self):
        """Test that origin maps to near origin."""
        from hypll.manifolds.poincare_ball import PoincareBall, Curvature

        curvature = Curvature(value=1.0, requires_grad=False)
        manifold = PoincareBall(c=curvature)

        x = torch.zeros(4, 2)
        result = manifold_map(x, manifold)

        # Origin should map to (near) origin
        assert torch.allclose(result.tensor, torch.zeros(4, 2), atol=1e-5)
