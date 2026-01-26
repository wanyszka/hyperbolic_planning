"""Tests for loss module."""

import pytest
import torch

from src.training import info_nce_loss, hyperbolic_info_nce_loss


class TestInfoNCELoss:
    """Tests for Euclidean info_nce_loss function."""

    def test_output_is_scalar(self):
        """Test that loss is a scalar tensor."""
        anchor = torch.randn(8, 4)
        positive = torch.randn(8, 4)
        negatives = torch.randn(8, 5, 4)

        loss = info_nce_loss(anchor, positive, negatives)

        assert loss.dim() == 0  # scalar

    def test_loss_is_non_negative(self):
        """Test that loss is non-negative."""
        anchor = torch.randn(8, 4)
        positive = torch.randn(8, 4)
        negatives = torch.randn(8, 5, 4)

        loss = info_nce_loss(anchor, positive, negatives)

        assert loss >= 0

    def test_loss_is_differentiable(self):
        """Test that loss supports backpropagation."""
        anchor = torch.randn(8, 4, requires_grad=True)
        positive = torch.randn(8, 4, requires_grad=True)
        negatives = torch.randn(8, 5, 4, requires_grad=True)

        loss = info_nce_loss(anchor, positive, negatives)
        loss.backward()

        assert anchor.grad is not None
        assert positive.grad is not None
        assert negatives.grad is not None

    def test_perfect_positive_low_loss(self):
        """Test that identical anchor-positive gives lower loss than random."""
        anchor = torch.randn(8, 4)
        positive_perfect = anchor.clone()
        positive_random = torch.randn(8, 4)
        negatives = torch.randn(8, 5, 4) * 10  # Far negatives

        loss_perfect = info_nce_loss(anchor, positive_perfect, negatives)
        loss_random = info_nce_loss(anchor, positive_random, negatives)

        assert loss_perfect < loss_random

    def test_different_temperatures(self):
        """Test loss with different temperature values."""
        anchor = torch.randn(8, 4)
        positive = torch.randn(8, 4)
        negatives = torch.randn(8, 5, 4)

        loss_low_temp = info_nce_loss(anchor, positive, negatives, temperature=0.1)
        loss_high_temp = info_nce_loss(anchor, positive, negatives, temperature=1.0)

        # Both should be valid scalars
        assert loss_low_temp.dim() == 0
        assert loss_high_temp.dim() == 0

    def test_different_batch_sizes(self):
        """Test loss with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            anchor = torch.randn(batch_size, 4)
            positive = torch.randn(batch_size, 4)
            negatives = torch.randn(batch_size, 5, 4)

            loss = info_nce_loss(anchor, positive, negatives)

            assert loss.dim() == 0

    def test_different_num_negatives(self):
        """Test loss with different numbers of negatives."""
        for num_neg in [1, 3, 5, 10]:
            anchor = torch.randn(8, 4)
            positive = torch.randn(8, 4)
            negatives = torch.randn(8, num_neg, 4)

            loss = info_nce_loss(anchor, positive, negatives)

            assert loss.dim() == 0


class TestHyperbolicInfoNCELoss:
    """Tests for hyperbolic_info_nce_loss function."""

    @pytest.fixture
    def manifold(self):
        """Create a Poincaré ball manifold."""
        from hypll.manifolds.poincare_ball import PoincareBall, Curvature
        return PoincareBall(c=Curvature(1.0))

    @pytest.fixture
    def sample_embeddings(self, manifold):
        """Create sample ManifoldTensor embeddings."""
        from hypll.tensors.manifold_tensor import ManifoldTensor

        # Create points well inside the ball (norm < 1) to avoid numerical issues
        anchor = torch.randn(8, 4) * 0.1
        positive = torch.randn(8, 4) * 0.1
        negatives = torch.randn(8, 5, 4) * 0.1

        return (
            ManifoldTensor(anchor, manifold=manifold),
            ManifoldTensor(positive, manifold=manifold),
            ManifoldTensor(negatives, manifold=manifold),
        )

    def test_output_is_scalar(self, manifold, sample_embeddings):
        """Test that loss is a scalar tensor."""
        anchor, positive, negatives = sample_embeddings

        loss = hyperbolic_info_nce_loss(anchor, positive, negatives, manifold)

        assert loss.dim() == 0

    def test_loss_is_non_negative(self, manifold, sample_embeddings):
        """Test that loss is non-negative."""
        anchor, positive, negatives = sample_embeddings

        loss = hyperbolic_info_nce_loss(anchor, positive, negatives, manifold)

        assert loss >= 0

    def test_loss_is_differentiable(self, manifold):
        """Test that loss supports backpropagation."""
        from hypll.tensors.manifold_tensor import ManifoldTensor

        # Create leaf tensors well inside the ball
        anchor_t = torch.randn(8, 4) * 0.1
        anchor_t.requires_grad_(True)

        anchor = ManifoldTensor(anchor_t, manifold=manifold)
        positive = ManifoldTensor(torch.randn(8, 4) * 0.1, manifold=manifold)
        negatives = ManifoldTensor(torch.randn(8, 5, 4) * 0.1, manifold=manifold)

        loss = hyperbolic_info_nce_loss(anchor, positive, negatives, manifold)

        # Check that loss has grad_fn (is differentiable)
        assert loss.grad_fn is not None

        loss.backward()

        # Check gradients flow back to the leaf tensor
        assert anchor_t.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
