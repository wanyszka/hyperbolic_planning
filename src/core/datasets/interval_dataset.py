"""Dataset module for hyperbolic interval encoding."""

import torch
import numpy as np
from torch.utils.data import Dataset


class IntervalDataset(Dataset):
    """
    Generate hierarchical interval pairs for contrastive learning.

    Anchor is PARENT, positive is CHILD (subinterval), negatives are non-subintervals.
    This places smaller/more specific intervals closer to the origin.

    Args:
        num_samples: Number of training samples to generate
        num_negatives: Number of negative examples per sample
        num_points: Number of grid points in [0, 1]
        seed: Random seed for reproducibility
        endpoint_prob: Probability of sampling endpoint as positive
    """

    def __init__(
        self,
        num_samples: int = 10000,
        num_negatives: int = 5,
        num_points: int = 100,
        seed: int = 0,
        endpoint_prob: float = 0.0,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.num_samples = num_samples
        self.num_negatives = num_negatives
        self.num_points = num_points
        self.endpoint_prob = endpoint_prob
        self.grid = torch.linspace(0, 1, num_points)

        self.anchors, self.positives, self.negatives_list = self._generate_samples()

    def _generate_samples(self):
        """Generate all training samples."""
        anchors, positives, negatives_list = [], [], []

        for _ in range(self.num_samples):
            anchor, positive, negatives = self._generate_single_sample()
            anchors.append(anchor)
            positives.append(positive)
            negatives_list.append(negatives)

        return (
            torch.tensor(anchors, dtype=torch.float32),
            torch.tensor(positives, dtype=torch.float32),
            torch.tensor(negatives_list, dtype=torch.float32),
        )

    def _generate_single_sample(self):
        """Generate a single (anchor, positive, negatives) tuple."""
        j_idx = torch.randint(0, self.num_points, (1,)).item()
        i_idx = torch.randint(0, j_idx + 1, (1,)).item()

        i = self.grid[i_idx].item()
        j = self.grid[j_idx].item()

        positive = self._sample_positive(i_idx, j_idx, i, j)
        negatives = [self._sample_negative(i_idx, j_idx) for _ in range(self.num_negatives)]

        return [i, j], positive, negatives

    def _sample_positive(self, i_idx: int, j_idx: int, i: float, j: float):
        """Sample a positive (subinterval) example."""
        if torch.rand(1).item() < self.endpoint_prob:
            if torch.rand(1).item() < 0.5:
                return [i, i]
            return [j, j]

        j_pos_idx = torch.randint(i_idx, j_idx + 1, (1,)).item()
        i_pos_idx = torch.randint(i_idx, j_pos_idx + 1, (1,)).item()

        return [self.grid[i_pos_idx].item(), self.grid[j_pos_idx].item()]

    def _sample_negative(self, i_idx: int, j_idx: int, max_attempts: int = 1000):
        """Sample a negative (non-subinterval) example."""
        for _ in range(max_attempts):
            j_neg_idx = torch.randint(0, self.num_points, (1,)).item()
            i_neg_idx = torch.randint(0, j_neg_idx + 1, (1,)).item()

            is_subinterval = i_idx <= i_neg_idx and j_neg_idx <= j_idx
            if not is_subinterval:
                return [self.grid[i_neg_idx].item(), self.grid[j_neg_idx].item()]

        return [self.grid[i_neg_idx].item(), self.grid[j_neg_idx].item()]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.anchors[idx], self.positives[idx], self.negatives_list[idx]


class IntervalDatasetInverted(Dataset):
    """
    Inverted hierarchical interval pairs for contrastive learning.

    Anchor is CHILD (subinterval), positive is PARENT (superinterval), negatives are non-superintervals.
    This places larger/more general intervals closer to origin.

    Args:
        num_samples: Number of training samples to generate
        num_negatives: Number of negative examples per sample
        num_points: Number of grid points in [0, 1]
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        num_samples: int = 10000,
        num_negatives: int = 5,
        num_points: int = 100,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.num_samples = num_samples
        self.num_negatives = num_negatives
        self.num_points = num_points
        self.grid = torch.linspace(0, 1, num_points)

        self.anchors, self.positives, self.negatives_list = self._generate_samples()

    def _generate_samples(self):
        """Generate all training samples."""
        anchors, positives, negatives_list = [], [], []

        for _ in range(self.num_samples):
            anchor, positive, negatives = self._generate_single_sample()
            anchors.append(anchor)
            positives.append(positive)
            negatives_list.append(negatives)

        return (
            torch.tensor(anchors, dtype=torch.float32),
            torch.tensor(positives, dtype=torch.float32),
            torch.tensor(negatives_list, dtype=torch.float32),
        )

    def _generate_single_sample(self):
        """Generate a single (anchor, positive, negatives) tuple."""
        # Sample anchor interval
        j_idx = torch.randint(0, self.num_points, (1,)).item()
        i_idx = torch.randint(0, j_idx + 1, (1,)).item()

        i_anchor = self.grid[i_idx].item()
        j_anchor = self.grid[j_idx].item()

        # Positive is a PARENT (superinterval): i_pos <= i_anchor <= j_anchor <= j_pos
        i_pos_idx = torch.randint(0, i_idx + 1, (1,)).item()
        j_pos_idx = torch.randint(j_idx, self.num_points, (1,)).item()

        i_pos = self.grid[i_pos_idx].item()
        j_pos = self.grid[j_pos_idx].item()

        # Negatives: intervals that do NOT contain the anchor
        negatives = [self._sample_negative(i_idx, j_idx) for _ in range(self.num_negatives)]

        return [i_anchor, j_anchor], [i_pos, j_pos], negatives

    def _sample_negative(self, i_idx: int, j_idx: int, max_attempts: int = 1000):
        """Sample a negative (non-superinterval) example."""
        for _ in range(max_attempts):
            j_neg_idx = torch.randint(0, self.num_points, (1,)).item()
            i_neg_idx = torch.randint(0, j_neg_idx + 1, (1,)).item()

            # Valid negative must NOT be a superinterval of anchor
            is_superinterval = i_neg_idx <= i_idx and j_idx <= j_neg_idx
            if not is_superinterval:
                return [self.grid[i_neg_idx].item(), self.grid[j_neg_idx].item()]

        return [self.grid[i_neg_idx].item(), self.grid[j_neg_idx].item()]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.anchors[idx], self.positives[idx], self.negatives_list[idx]
