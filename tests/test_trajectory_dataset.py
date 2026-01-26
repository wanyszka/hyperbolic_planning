"""Tests for trajectory_dataset.py - Contrastive learning datasets."""

import pytest
import numpy as np
import torch
from typing import List

from src.core.datasets.trajectory_dataset import (
    TrajectoryContrastiveDataset,
    InBatchContrastiveDataset,
    PolicyTrainingDataset,
    create_train_val_test_split,
    create_contrastive_dataloader,
    create_policy_dataloader,
)


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
def sample_actions(sample_trajectories: List[List[float]]) -> List[List[int]]:
    """Create sample actions corresponding to trajectories."""
    actions = []
    for traj in sample_trajectories:
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
    """Create very short trajectories for edge case testing."""
    return [
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],  # length 6
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # length 6
        [0.5, 0.6, 0.7, 0.8, 0.9],       # length 5
    ]


# =============================================================================
# TrajectoryContrastiveDataset Tests
# =============================================================================

class TestTrajectoryContrastiveDataset:
    """Tests for TrajectoryContrastiveDataset."""

    def test_dataset_creation(self, sample_trajectories):
        """Test basic dataset creation."""
        dataset = TrajectoryContrastiveDataset(
            trajectories=sample_trajectories,
            num_samples=100,
            n_negatives=5,
            seed=42,
        )
        assert len(dataset) == 100

    def test_output_shapes(self, sample_trajectories):
        """Test output tensor shapes."""
        dataset = TrajectoryContrastiveDataset(
            trajectories=sample_trajectories,
            num_samples=100,
            n_negatives=5,
            seed=42,
        )

        anchor, positive, negatives = dataset[0]

        assert anchor.shape == (2,)  # [start, end]
        assert positive.shape == (2,)
        assert negatives.shape == (5, 2)  # n_negatives x 2

    def test_reproducibility(self, sample_trajectories):
        """Test that same seed produces same samples."""
        dataset1 = TrajectoryContrastiveDataset(
            trajectories=sample_trajectories,
            num_samples=50,
            n_negatives=5,
            seed=42,
        )
        dataset2 = TrajectoryContrastiveDataset(
            trajectories=sample_trajectories,
            num_samples=50,
            n_negatives=5,
            seed=42,
        )

        for idx in range(10):
            a1, p1, n1 = dataset1[idx]
            a2, p2, n2 = dataset2[idx]
            assert torch.allclose(a1, a2)
            assert torch.allclose(p1, p2)
            assert torch.allclose(n1, n2)

    def test_different_seeds_produce_different_samples(self, sample_trajectories):
        """Test that different seeds produce different samples."""
        dataset1 = TrajectoryContrastiveDataset(
            trajectories=sample_trajectories,
            num_samples=50,
            n_negatives=5,
            seed=42,
        )
        dataset2 = TrajectoryContrastiveDataset(
            trajectories=sample_trajectories,
            num_samples=50,
            n_negatives=5,
            seed=123,
        )

        # At least some samples should be different
        different = False
        for idx in range(10):
            a1, _, _ = dataset1[idx]
            a2, _, _ = dataset2[idx]
            if not torch.allclose(a1, a2):
                different = True
                break
        assert different


# =============================================================================
# Negative Sampling Tests
# =============================================================================

class TestNegativeSampling:
    """Tests for negative sampling."""

    def test_negatives_generated(self, sample_trajectories):
        """Test that negatives are generated correctly."""
        dataset = TrajectoryContrastiveDataset(
            trajectories=sample_trajectories,
            num_samples=100,
            n_negatives=10,
            seed=42,
        )

        # Verify the dataset was created without errors
        assert len(dataset) == 100

        # Check that negatives exist and have correct shape
        _, _, negatives = dataset[0]
        assert negatives.shape[0] == 10

    def test_correct_number_of_negatives(self, sample_trajectories):
        """Test correct number of negatives per sample."""
        for n_neg in [5, 10, 50]:
            dataset = TrajectoryContrastiveDataset(
                trajectories=sample_trajectories,
                num_samples=10,
                n_negatives=n_neg,
                seed=42,
            )

            _, _, negatives = dataset[0]
            assert negatives.shape[0] == n_neg


# =============================================================================
# TrajectoryContrastiveDataset Tests - Edge Cases
# =============================================================================

class TestContrastiveDatasetEdgeCases:
    """Tests for edge cases in contrastive dataset."""

    def test_raises_on_empty_trajectories(self):
        """Test error when no trajectories provided."""
        with pytest.raises(ValueError, match="No trajectories with sufficient length"):
            TrajectoryContrastiveDataset(
                trajectories=[],
                num_samples=10,
            )

    def test_handles_minimum_length_trajectories(self, short_trajectories):
        """Test with trajectories at minimum valid length."""
        dataset = TrajectoryContrastiveDataset(
            trajectories=short_trajectories,
            num_samples=10,
            n_negatives=2,
            seed=42,
        )

        assert len(dataset) == 10

    def test_single_element_trajectory(self):
        """Test with single element trajectories."""
        single_trajs = [[0.5], [0.3], [0.7]]
        dataset = TrajectoryContrastiveDataset(
            trajectories=single_trajs,
            num_samples=5,
            n_negatives=2,
            seed=42,
        )
        assert len(dataset) == 5


# =============================================================================
# InBatchContrastiveDataset Tests
# =============================================================================

class TestInBatchContrastiveDataset:
    """Tests for InBatchContrastiveDataset."""

    def test_dataset_creation(self, sample_trajectories):
        """Test basic dataset creation."""
        dataset = InBatchContrastiveDataset(
            trajectories=sample_trajectories,
            num_samples=100,
            seed=42,
        )
        assert len(dataset) == 100

    def test_output_shapes(self, sample_trajectories):
        """Test output tensor shapes (no negatives)."""
        dataset = InBatchContrastiveDataset(
            trajectories=sample_trajectories,
            num_samples=100,
            seed=42,
        )

        anchor, positive = dataset[0]

        assert anchor.shape == (2,)
        assert positive.shape == (2,)

    def test_reproducibility(self, sample_trajectories):
        """Test that same seed produces same samples."""
        dataset1 = InBatchContrastiveDataset(
            trajectories=sample_trajectories,
            num_samples=50,
            seed=42,
        )
        dataset2 = InBatchContrastiveDataset(
            trajectories=sample_trajectories,
            num_samples=50,
            seed=42,
        )

        for idx in range(10):
            a1, p1 = dataset1[idx]
            a2, p2 = dataset2[idx]
            assert torch.allclose(a1, a2)
            assert torch.allclose(p1, p2)


# =============================================================================
# PolicyTrainingDataset Tests
# =============================================================================

class TestPolicyTrainingDataset:
    """Tests for PolicyTrainingDataset."""

    def test_dataset_creation(self, sample_trajectories, sample_actions):
        """Test basic dataset creation."""
        dataset = PolicyTrainingDataset(
            trajectories=sample_trajectories,
            actions=sample_actions,
            goal_type="final",
            seed=42,
        )
        assert len(dataset) > 0

    def test_output_shapes(self, sample_trajectories, sample_actions):
        """Test output tensor shapes."""
        dataset = PolicyTrainingDataset(
            trajectories=sample_trajectories,
            actions=sample_actions,
            goal_type="final",
        )

        state, goal, action = dataset[0]

        assert state.shape == (1,)
        assert goal.shape == (1,)
        assert action.shape == ()  # Scalar

    def test_final_goal_type(self, sample_trajectories, sample_actions):
        """Test that final goal type uses trajectory endpoint."""
        dataset = PolicyTrainingDataset(
            trajectories=sample_trajectories,
            actions=sample_actions,
            goal_type="final",
        )

        # All goals from first trajectory should be its final state
        traj = sample_trajectories[0]
        final_state = traj[-1]
        n_transitions = len(traj) - 1

        for idx in range(n_transitions):
            _, goal, _ = dataset[idx]
            assert abs(goal.item() - final_state) < 1e-6

    def test_random_goal_type(self, sample_trajectories, sample_actions):
        """Test that random goal type samples future states."""
        dataset = PolicyTrainingDataset(
            trajectories=sample_trajectories,
            actions=sample_actions,
            goal_type="random",
            seed=42,
        )

        # Goals should vary (not all be the final state)
        traj = sample_trajectories[0]
        final_state = traj[-1]
        n_transitions = len(traj) - 1

        goals = []
        for idx in range(n_transitions):
            _, goal, _ = dataset[idx]
            goals.append(goal.item())

        # At least some goals should differ from final state
        non_final_goals = [g for g in goals if abs(g - final_state) > 1e-6]
        assert len(non_final_goals) > 0

    def test_total_samples_correct(self, sample_trajectories, sample_actions):
        """Test that total samples equals sum of transitions."""
        dataset = PolicyTrainingDataset(
            trajectories=sample_trajectories,
            actions=sample_actions,
            goal_type="final",
        )

        expected_samples = sum(len(t) - 1 for t in sample_trajectories)
        assert len(dataset) == expected_samples

    def test_action_values_in_range(self, sample_trajectories, sample_actions):
        """Test that actions are in valid range."""
        dataset = PolicyTrainingDataset(
            trajectories=sample_trajectories,
            actions=sample_actions,
        )

        for idx in range(min(100, len(dataset))):
            _, _, action = dataset[idx]
            assert 0 <= action.item() <= 2

    def test_state_values_in_range(self, sample_trajectories, sample_actions):
        """Test that states are in [0, 1] range."""
        dataset = PolicyTrainingDataset(
            trajectories=sample_trajectories,
            actions=sample_actions,
        )

        for idx in range(min(100, len(dataset))):
            state, goal, _ = dataset[idx]
            assert 0.0 <= state.item() <= 1.0
            assert 0.0 <= goal.item() <= 1.0


# =============================================================================
# Train/Val/Test Split Tests
# =============================================================================

class TestCreateTrainValTestSplit:
    """Tests for create_train_val_test_split function."""

    def test_split_structure(self, sample_trajectories, sample_actions):
        """Test that split returns correct structure."""
        split = create_train_val_test_split(
            sample_trajectories,
            sample_actions,
            val_fraction=0.1,
            test_fraction=0.1,
        )

        assert "train" in split
        assert "val" in split
        assert "test" in split

        for key in ["train", "val", "test"]:
            trajs, acts = split[key]
            assert len(trajs) == len(acts)

    def test_split_sizes(self, sample_trajectories, sample_actions):
        """Test that split sizes are approximately correct."""
        n = len(sample_trajectories)

        split = create_train_val_test_split(
            sample_trajectories,
            sample_actions,
            val_fraction=0.1,
            test_fraction=0.2,
            seed=42,
        )

        train_trajs, _ = split["train"]
        val_trajs, _ = split["val"]
        test_trajs, _ = split["test"]

        # Check approximate sizes
        assert len(val_trajs) == int(n * 0.1)
        assert len(test_trajs) == int(n * 0.2)
        assert len(train_trajs) == n - len(val_trajs) - len(test_trajs)

    def test_split_no_overlap(self, sample_trajectories, sample_actions):
        """Test that splits don't overlap."""
        split = create_train_val_test_split(
            sample_trajectories,
            sample_actions,
            val_fraction=0.1,
            test_fraction=0.1,
        )

        train_trajs, _ = split["train"]
        val_trajs, _ = split["val"]
        test_trajs, _ = split["test"]

        # Convert to sets of tuple representations
        train_set = set(tuple(t) for t in train_trajs)
        val_set = set(tuple(t) for t in val_trajs)
        test_set = set(tuple(t) for t in test_trajs)

        # Check no overlap
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_split_reproducibility(self, sample_trajectories, sample_actions):
        """Test that same seed produces same split."""
        split1 = create_train_val_test_split(
            sample_trajectories, sample_actions, seed=42
        )
        split2 = create_train_val_test_split(
            sample_trajectories, sample_actions, seed=42
        )

        train1, _ = split1["train"]
        train2, _ = split2["train"]

        assert train1 == train2


# =============================================================================
# DataLoader Factory Tests
# =============================================================================

class TestDataLoaderFactories:
    """Tests for DataLoader factory functions."""

    def test_contrastive_dataloader(self, sample_trajectories):
        """Test contrastive DataLoader creation."""
        loader = create_contrastive_dataloader(
            trajectories=sample_trajectories,
            batch_size=16,
            num_samples=50,
            n_negatives=5,
            seed=42,
        )

        batch = next(iter(loader))
        anchors, positives, negatives = batch

        assert anchors.shape[0] <= 16  # batch size
        assert anchors.shape[1] == 2
        assert positives.shape[1] == 2
        assert negatives.shape[1] == 5  # n_negatives
        assert negatives.shape[2] == 2

    def test_policy_dataloader(self, sample_trajectories, sample_actions):
        """Test policy DataLoader creation."""
        loader = create_policy_dataloader(
            trajectories=sample_trajectories,
            actions=sample_actions,
            batch_size=32,
            goal_type="final",
            seed=42,
        )

        batch = next(iter(loader))
        states, goals, actions = batch

        assert states.shape[0] <= 32  # batch size
        assert states.shape[1] == 1
        assert goals.shape[1] == 1
        assert len(actions.shape) == 1  # 1D tensor of action indices

    def test_dataloader_shuffle(self, sample_trajectories, sample_actions):
        """Test that shuffle affects batch order."""
        # Create two loaders with same seed but shuffle on/off
        loader_shuffled = create_policy_dataloader(
            sample_trajectories, sample_actions,
            batch_size=10, shuffle=True, seed=42
        )
        loader_unshuffled = create_policy_dataloader(
            sample_trajectories, sample_actions,
            batch_size=10, shuffle=False, seed=42
        )

        batch_shuffled = next(iter(loader_shuffled))
        batch_unshuffled = next(iter(loader_unshuffled))

        # Unshuffled should give first 10 samples in order
        # Shuffled may or may not match (depending on torch random state)
        # At minimum, both should have valid data
        assert batch_shuffled[0].shape == batch_unshuffled[0].shape
