"""Trajectory-based dataset classes for contrastive learning with temporal containment."""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Literal, Optional



class TrajectoryContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning on trajectory intervals.

    Generates (anchor, positive) pairs where positive is temporally contained
    within anchor. Negatives are sampled from the same trajectory but are
    NOT subintervals of the anchor.

    Args:
        trajectories: List of trajectory state sequences
        num_samples: Number of contrastive samples to generate
        sampling_config: Configuration for sampling strategy
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        trajectories: List[List[float]],
        num_samples: int = 10000,
        n_negatives: int = 255,
        seed: int = 42,
    ):
        self.trajectories = trajectories
        self.num_samples = num_samples
        self.n_negatives = n_negatives
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Precompute trajectory lengths for efficient sampling
        self.traj_lengths = [len(t) for t in trajectories]
        self.valid_traj_indices = [
            i for i, length in enumerate(self.traj_lengths)
            if length >= 1
        ]

        if len(self.valid_traj_indices) == 0:
            raise ValueError("No trajectories with sufficient length for sampling")

        # Generate all samples upfront
        self.anchors, self.positives, self.negatives = self._generate_all_samples()

    def _generate_all_samples(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate all contrastive samples."""
        anchors = []
        positives = []
        negatives_list = []

        for _ in range(self.num_samples):
            anchor, positive, negs = self._generate_single_sample()
            anchors.append(anchor)
            positives.append(positive)
            negatives_list.append(negs)

        return (
            torch.tensor(anchors, dtype=torch.float32),
            torch.tensor(positives, dtype=torch.float32),
            torch.tensor(negatives_list, dtype=torch.float32),
        )

    def _generate_single_sample(self) -> Tuple[List[float], List[float], List[List[float]]]:
        """
        Generate a single (anchor, positive, negatives) tuple.

        Anchor: interval [i, j] from trajectory
        Positive: subinterval [k, l] where i <= k <= l <= j
        Negative: interval from same trajectory that is NOT a subinterval of anchor
        """
        # Sample trajectory
        traj_idx = np.random.choice(self.valid_traj_indices)
        traj = self.trajectories[traj_idx]
        T = len(traj) - 1  # Last valid index

        # Sample anchor interval [i, j] where i <= j
        j = np.random.randint(0, T + 1)
        i = np.random.randint(0, j + 1)

        anchor = [traj[i], traj[j]]

        # Sample positive (subinterval): i <= k <= l <= j
        l = np.random.randint(i, j + 1)
        k = np.random.randint(i, l + 1)

        positive = [traj[k], traj[l]]

        # Sample negatives (non-subintervals from same trajectory)
        negatives = []
        for _ in range(self.n_negatives):
            neg = self._sample_negative_from_trajectory(traj, i, j, T)
            negatives.append(neg)

        return anchor, positive, negatives

    def _sample_negative_from_trajectory(
        self,
        traj: List[float],
        anchor_i: int,
        anchor_j: int,
        T: int,
        max_attempts: int = 1000,
    ) -> List[float]:
        """
        Sample a negative interval that is NOT a subinterval of the anchor.

        An interval [k, l] is a subinterval of [i, j] if i <= k and l <= j.
        """
        for _ in range(max_attempts):
            # Sample random interval from trajectory (like interval_dataset)
            l = np.random.randint(0, T + 1)
            k = np.random.randint(0, l + 1)

            # Check if it's NOT a subinterval (temporal containment)
            is_subinterval = (anchor_i <= k) and (l <= anchor_j)

            if not is_subinterval:
                return [traj[k], traj[l]]

        # Fallback: return any non-identical interval
        return [traj[0], traj[T // 2]] if anchor_i > 0 else [traj[T // 2], traj[T]]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.anchors[idx], self.positives[idx], self.negatives[idx]


class InBatchContrastiveDataset(Dataset):
    """
    Dataset that generates interval pairs for in-batch contrastive learning.

    Each sample is an (anchor, positive) pair. Negatives are formed from
    other samples in the batch during training.

    Args:
        trajectories: List of trajectory state sequences
        num_samples: Number of samples to generate
        sampling_config: Configuration for sampling strategy
        seed: Random seed
    """

    def __init__(
        self,
        trajectories: List[List[float]],
        num_samples: int = 10000,
        seed: int = 42,
    ):
        self.trajectories = trajectories
        self.num_samples = num_samples

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.traj_lengths = [len(t) for t in trajectories]
        self.valid_traj_indices = [
            i for i, length in enumerate(self.traj_lengths)
            if length >= 1
        ]

        # Generate samples
        self.anchors, self.positives = self._generate_samples()

    def _generate_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all (anchor, positive) pairs."""
        anchors = []
        positives = []

        for _ in range(self.num_samples):
            anchor, positive = self._generate_pair()
            anchors.append(anchor)
            positives.append(positive)

        return (
            torch.tensor(anchors, dtype=torch.float32),
            torch.tensor(positives, dtype=torch.float32),
        )

    def _generate_pair(self) -> Tuple[List[float], List[float]]:
        """Generate a single (anchor, positive) pair."""
        traj_idx = np.random.choice(self.valid_traj_indices)
        traj = self.trajectories[traj_idx]
        T = len(traj) - 1

        # Sample anchor interval [i, j] where i <= j
        j = np.random.randint(0, T + 1)
        i = np.random.randint(0, j + 1)

        # Sample positive (subinterval): i <= k <= l <= j
        l = np.random.randint(i, j + 1)
        k = np.random.randint(i, l + 1)

        return [traj[i], traj[j]], [traj[k], traj[l]]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.anchors[idx], self.positives[idx]


class PolicyTrainingDataset(Dataset):
    """
    Dataset for training goal-conditioned policies.

    Extracts (state, goal, action) tuples from trajectories.

    Args:
        trajectories: List of trajectory state sequences
        actions: List of action sequences (parallel to trajectories)
        goal_type: "final" (use trajectory endpoint) or "random" (sample from future states)
        max_samples_per_trajectory: Maximum samples to extract per trajectory (None = all).
            Useful for limiting dataset size when trajectories are very long.
        seed: Random seed
    """

    def __init__(
        self,
        trajectories: List[List[float]],
        actions: List[List[int]],
        goal_type: Literal["final", "random"] = "final",
        max_samples_per_trajectory: Optional[int] = None,
        seed: int = 42,
    ):
        self.trajectories = trajectories
        self.actions_list = actions
        self.goal_type = goal_type
        self.max_samples_per_trajectory = max_samples_per_trajectory

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Extract all (state, goal, action) tuples
        self.states, self.goals, self.actions = self._extract_samples()

    def _extract_samples(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract training samples from all trajectories."""
        states = []
        goals = []
        actions = []

        for traj, acts in zip(self.trajectories, self.actions_list):
            T = len(traj) - 1  # Number of transitions
            if T <= 0:
                continue
            final_state = traj[-1]

            # Determine which timesteps to sample
            if self.max_samples_per_trajectory is not None and T > self.max_samples_per_trajectory:
                # Subsample: half evenly spaced, half random for diversity
                n_samples = self.max_samples_per_trajectory
                n_even = n_samples // 2
                n_random = n_samples - n_even
                even_indices = np.linspace(0, T - 1, n_even, dtype=int).tolist()
                random_indices = np.random.choice(T, size=min(n_random, T), replace=False).tolist()
                timesteps = sorted(set(even_indices + random_indices))
            else:
                timesteps = range(T)

            for t in timesteps:
                state = traj[t]
                action = acts[t]

                if self.goal_type == "final":
                    goal = final_state
                else:  # random future state
                    future_idx = np.random.randint(t + 1, len(traj))
                    goal = traj[future_idx]

                states.append(state)
                goals.append(goal)
                actions.append(action)

        return (
            torch.tensor(states, dtype=torch.float32).unsqueeze(1),
            torch.tensor(goals, dtype=torch.float32).unsqueeze(1),
            torch.tensor(actions, dtype=torch.long),
        )

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.goals[idx], self.actions[idx]


def create_train_val_test_split(
    trajectories: List[List[float]],
    actions: List[List[int]],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, Tuple[List, List]]:
    """
    Split trajectories into train/val/test sets.

    Args:
        trajectories: List of trajectories
        actions: List of action sequences
        val_fraction: Fraction for validation
        test_fraction: Fraction for test
        seed: Random seed

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing (trajectories, actions)
    """
    np.random.seed(seed)

    n = len(trajectories)
    indices = np.random.permutation(n)

    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": (
            [trajectories[i] for i in train_idx],
            [actions[i] for i in train_idx],
        ),
        "val": (
            [trajectories[i] for i in val_idx],
            [actions[i] for i in val_idx],
        ),
        "test": (
            [trajectories[i] for i in test_idx],
            [actions[i] for i in test_idx],
        ),
    }


def create_contrastive_dataloader(
    trajectories: List[List[float]],
    batch_size: int = 256,
    num_samples: int = 10000,
    n_negatives: int = 255,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for contrastive learning.

    Args:
        trajectories: List of trajectories
        batch_size: Batch size
        num_samples: Number of contrastive samples
        n_negatives: Number of negatives per anchor
        seed: Random seed
        num_workers: DataLoader workers

    Returns:
        DataLoader instance
    """

    dataset = TrajectoryContrastiveDataset(
        trajectories=trajectories,
        num_samples=num_samples,
        n_negatives=n_negatives,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def create_policy_dataloader(
    trajectories: List[List[float]],
    actions: List[List[int]],
    batch_size: int = 256,
    goal_type: Literal["final", "random"] = "final",
    max_samples_per_trajectory: Optional[int] = None,
    seed: int = 42,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for policy training.

    Args:
        trajectories: List of trajectories
        actions: List of action sequences
        batch_size: Batch size
        goal_type: "final" or "random"
        max_samples_per_trajectory: Maximum samples per trajectory (None = all)
        seed: Random seed
        num_workers: DataLoader workers
        shuffle: Whether to shuffle

    Returns:
        DataLoader instance
    """
    dataset = PolicyTrainingDataset(
        trajectories=trajectories,
        actions=actions,
        goal_type=goal_type,
        max_samples_per_trajectory=max_samples_per_trajectory,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )