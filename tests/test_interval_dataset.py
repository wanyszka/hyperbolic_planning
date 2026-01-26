"""Tests for dataset module."""

import pytest
import torch

from src.core.datasets import IntervalDataset, IntervalDatasetInverted


class TestIntervalDataset:
    """Tests for IntervalDataset class."""

    def test_init_creates_correct_shapes(self):
        """Test that dataset initializes with correct tensor shapes."""
        dataset = IntervalDataset(num_samples=100, num_negatives=3, num_points=10)

        assert len(dataset) == 100
        assert dataset.anchors.shape == (100, 2)
        assert dataset.positives.shape == (100, 2)
        assert dataset.negatives_list.shape == (100, 3, 2)

    def test_getitem_returns_correct_types(self):
        """Test that __getitem__ returns tensors."""
        dataset = IntervalDataset(num_samples=10, num_negatives=2, num_points=10)
        anchor, positive, negatives = dataset[0]

        assert isinstance(anchor, torch.Tensor)
        assert isinstance(positive, torch.Tensor)
        assert isinstance(negatives, torch.Tensor)
        assert anchor.shape == (2,)
        assert positive.shape == (2,)
        assert negatives.shape == (2, 2)

    def test_intervals_are_valid(self):
        """Test that all intervals have i <= j."""
        dataset = IntervalDataset(num_samples=100, num_negatives=5, num_points=20)

        for i in range(len(dataset)):
            anchor, positive, negatives = dataset[i]
            assert anchor[0] <= anchor[1], f"Invalid anchor at {i}: {anchor}"
            assert positive[0] <= positive[1], f"Invalid positive at {i}: {positive}"
            for neg in negatives:
                assert neg[0] <= neg[1], f"Invalid negative at {i}: {neg}"

    def test_intervals_in_unit_range(self):
        """Test that all interval values are in [0, 1]."""
        dataset = IntervalDataset(num_samples=100, num_negatives=5, num_points=20)

        assert (dataset.anchors >= 0).all() and (dataset.anchors <= 1).all()
        assert (dataset.positives >= 0).all() and (dataset.positives <= 1).all()
        assert (dataset.negatives_list >= 0).all() and (dataset.negatives_list <= 1).all()

    def test_positive_is_subinterval_of_anchor(self):
        """Test that positive is a subinterval of anchor."""
        dataset = IntervalDataset(num_samples=100, num_negatives=2, num_points=20, endpoint_prob=0)

        for i in range(len(dataset)):
            anchor, positive, _ = dataset[i]
            # positive should satisfy: anchor[0] <= positive[0] <= positive[1] <= anchor[1]
            assert anchor[0] <= positive[0] + 1e-6, f"Failed at {i}: {anchor}, {positive}"
            assert positive[1] <= anchor[1] + 1e-6, f"Failed at {i}: {anchor}, {positive}"

    def test_seed_reproducibility(self):
        """Test that same seed produces same dataset."""
        dataset1 = IntervalDataset(num_samples=50, num_negatives=2, num_points=10, seed=42)
        dataset2 = IntervalDataset(num_samples=50, num_negatives=2, num_points=10, seed=42)

        assert torch.allclose(dataset1.anchors, dataset2.anchors)
        assert torch.allclose(dataset1.positives, dataset2.positives)
        assert torch.allclose(dataset1.negatives_list, dataset2.negatives_list)

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different datasets."""
        dataset1 = IntervalDataset(num_samples=50, num_negatives=2, num_points=10, seed=42)
        dataset2 = IntervalDataset(num_samples=50, num_negatives=2, num_points=10, seed=123)

        assert not torch.allclose(dataset1.anchors, dataset2.anchors)


class TestIntervalDatasetInverted:
    """Tests for IntervalDatasetInverted class."""

    def test_init_creates_correct_shapes(self):
        """Test that dataset initializes with correct tensor shapes."""
        dataset = IntervalDatasetInverted(num_samples=100, num_negatives=3, num_points=10)

        assert len(dataset) == 100
        assert dataset.anchors.shape == (100, 2)
        assert dataset.positives.shape == (100, 2)
        assert dataset.negatives_list.shape == (100, 3, 2)

    def test_positive_is_superinterval_of_anchor(self):
        """Test that positive is a superinterval of anchor."""
        dataset = IntervalDatasetInverted(num_samples=100, num_negatives=2, num_points=20)

        for i in range(len(dataset)):
            anchor, positive, _ = dataset[i]
            # positive should satisfy: positive[0] <= anchor[0] <= anchor[1] <= positive[1]
            assert positive[0] <= anchor[0] + 1e-6, f"Failed at {i}: {anchor}, {positive}"
            assert anchor[1] <= positive[1] + 1e-6, f"Failed at {i}: {anchor}, {positive}"

    def test_intervals_are_valid(self):
        """Test that all intervals have i <= j."""
        dataset = IntervalDatasetInverted(num_samples=100, num_negatives=5, num_points=20)

        for i in range(len(dataset)):
            anchor, positive, negatives = dataset[i]
            assert anchor[0] <= anchor[1]
            assert positive[0] <= positive[1]
            for neg in negatives:
                assert neg[0] <= neg[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
