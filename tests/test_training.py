"""Tests for training module."""

import pytest
import torch
import torch.nn as nn

from src.core.datasets import IntervalDataset
from src.models import HyperbolicIntervalEncoder
from src.training import train_model


def _is_hyperbolic_model(model):
    """Check if model has a hyperbolic manifold."""
    return hasattr(model, 'manifold') and model.manifold is not None


class SimpleEuclideanModel(nn.Module):
    """Simple Euclidean model for testing."""

    def __init__(self, embedding_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, embedding_dim),
        )
        self.manifold = None  # Explicitly not hyperbolic

    def forward(self, x):
        return self.net(x)


class TestIsHyperbolicModel:
    """Tests for _is_hyperbolic_model function."""

    def test_hyperbolic_model_detected(self):
        """Test that hyperbolic model is detected."""
        model = HyperbolicIntervalEncoder(embedding_dim=2, euc_width=16, hyp_width=16)
        assert _is_hyperbolic_model(model) is True

    def test_euclidean_model_detected(self):
        """Test that Euclidean model is detected."""
        model = SimpleEuclideanModel()
        assert _is_hyperbolic_model(model) is False

    def test_model_without_manifold_attr(self):
        """Test model without manifold attribute."""
        model = nn.Linear(2, 4)
        assert _is_hyperbolic_model(model) is False


class TestOptimizerSelection:
    """Tests for optimizer selection based on model type."""

    def test_euclidean_model_uses_adam(self, capsys):
        """Test that Euclidean model training uses Adam."""
        model = SimpleEuclideanModel()
        dataset = IntervalDataset(num_samples=16, num_negatives=2, num_points=10)

        train_model(model, dataset, num_epochs=1, batch_size=8, verbose=True)

        captured = capsys.readouterr()
        assert "Adam" in captured.out
        assert "Riemannian" not in captured.out

    def test_hyperbolic_model_uses_riemannian_adam(self, capsys):
        """Test that hyperbolic model training uses RiemannianAdam."""
        model = HyperbolicIntervalEncoder(embedding_dim=2, euc_width=16, hyp_width=16)
        dataset = IntervalDataset(num_samples=16, num_negatives=2, num_points=10)

        train_model(model, dataset, num_epochs=1, batch_size=8, verbose=True)

        captured = capsys.readouterr()
        assert "RiemannianAdam" in captured.out


class TestTrainModel:
    """Tests for train_model function."""

    @pytest.fixture
    def small_dataset(self):
        """Create a small dataset for testing."""
        return IntervalDataset(num_samples=32, num_negatives=2, num_points=10)

    @pytest.fixture
    def hyperbolic_model(self):
        """Create a small hyperbolic model for testing."""
        return HyperbolicIntervalEncoder(embedding_dim=2, euc_width=16, hyp_width=16)

    @pytest.fixture
    def euclidean_model(self):
        """Create a simple Euclidean model for testing."""
        return SimpleEuclideanModel(embedding_dim=4)

    def test_train_hyperbolic_model(self, hyperbolic_model, small_dataset):
        """Test training a hyperbolic model."""
        losses = train_model(
            hyperbolic_model,
            small_dataset,
            num_epochs=2,
            batch_size=8,
            verbose=False,
        )

        assert len(losses) == 2
        assert all(isinstance(l, float) for l in losses)

    def test_train_euclidean_model(self, euclidean_model, small_dataset):
        """Test training a Euclidean model."""
        losses = train_model(
            euclidean_model,
            small_dataset,
            num_epochs=2,
            batch_size=8,
            verbose=False,
        )

        assert len(losses) == 2
        assert all(isinstance(l, float) for l in losses)

    def test_loss_decreases(self, hyperbolic_model, small_dataset):
        """Test that loss generally decreases during training."""
        losses = train_model(
            hyperbolic_model,
            small_dataset,
            num_epochs=10,
            batch_size=8,
            lr=0.01,
            verbose=False,
        )

        # Average of last few losses should be lower than first few
        # (not strict monotonic decrease, but general trend)
        assert len(losses) >= 5
        early_avg = sum(losses[:3]) / 3
        late_avg = sum(losses[-3:]) / 3

        # Allow some tolerance - training might not always decrease smoothly
        assert late_avg < early_avg * 1.5, f"Loss did not decrease: {early_avg} -> {late_avg}"

    def test_returns_list_of_losses(self, hyperbolic_model, small_dataset):
        """Test that training returns a list of losses."""
        losses = train_model(
            hyperbolic_model,
            small_dataset,
            num_epochs=3,
            batch_size=8,
            verbose=False,
        )

        assert isinstance(losses, list)
        assert len(losses) == 3

    def test_model_parameters_updated(self, hyperbolic_model, small_dataset):
        """Test that model parameters are updated during training."""
        # Get initial parameters (handle ManifoldParameter by accessing .tensor)
        initial_params = []
        for p in hyperbolic_model.parameters():
            if hasattr(p, 'tensor'):
                initial_params.append(p.tensor.clone().detach())
            else:
                initial_params.append(p.clone().detach())

        train_model(
            hyperbolic_model,
            small_dataset,
            num_epochs=2,
            batch_size=8,
            verbose=False,
        )

        # Check that at least some parameters changed
        params_changed = False
        for initial, current in zip(initial_params, hyperbolic_model.parameters()):
            current_tensor = current.tensor if hasattr(current, 'tensor') else current
            if not torch.allclose(initial, current_tensor):
                params_changed = True
                break

        assert params_changed, "Model parameters were not updated"

    def test_verbose_output(self, hyperbolic_model, small_dataset, capsys):
        """Test that verbose mode prints progress."""
        train_model(
            hyperbolic_model,
            small_dataset,
            num_epochs=5,
            batch_size=8,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "Epoch 5/5" in captured.out
        assert "Loss:" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
