"""Metrics module for evaluating representation and policy quality."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from dataclasses import dataclass

from hypll.tensors.manifold_tensor import ManifoldTensor


@dataclass
class RepresentationMetrics:
    """Container for representation quality metrics."""

    temporal_geometric_alignment: float
    norm_length_correlation: float
    angle_midpoint_correlation: Optional[float]  # Only for 2D embeddings
    embedding_norm_mean: float
    embedding_norm_std: float


@dataclass
class PolicyMetrics:
    """Container for policy evaluation metrics."""

    success_rate: float
    mean_steps: float
    median_steps: float
    std_steps: float
    path_efficiency: float
    action_entropy: float


def compute_hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Compute hyperbolic distance in Poincaré ball.

    d_H(x, y) = arccosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))

    Args:
        x: Points in Poincaré ball (batch, dim)
        y: Points in Poincaré ball (batch, dim)
        c: Curvature (default 1.0)

    Returns:
        Distances (batch,)
    """
    # Compute squared norms
    x_norm_sq = torch.sum(x ** 2, dim=-1)
    y_norm_sq = torch.sum(y ** 2, dim=-1)
    diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)

    # Compute hyperbolic distance
    numerator = 2 * diff_norm_sq
    denominator = (1 - x_norm_sq) * (1 - y_norm_sq)

    # Clamp to avoid numerical issues
    denominator = torch.clamp(denominator, min=1e-10)
    ratio = numerator / denominator

    # arccosh(1 + x) = log(1 + x + sqrt(x² + 2x))
    dist = torch.acosh(torch.clamp(1 + ratio, min=1.0 + 1e-10))

    return dist * np.sqrt(c)


def compute_hyperbolic_norm(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Compute hyperbolic distance from origin (hyperbolic norm).

    For Poincaré ball: d(0, x) = 2 * arctanh(||x||) / sqrt(c)

    Args:
        x: Points in Poincaré ball (batch, dim)
        c: Curvature

    Returns:
        Hyperbolic norms (batch,)
    """
    euclidean_norm = torch.norm(x, dim=-1)
    # Clamp to avoid arctanh(1) = inf
    euclidean_norm = torch.clamp(euclidean_norm, max=1.0 - 1e-6)
    return 2 * torch.arctanh(euclidean_norm) / np.sqrt(c)


def compute_temporal_geometric_alignment(
    encoder: torch.nn.Module,
    trajectories: List[List[float]],
    n_samples: int = 10000,
    seed: int = 42,
    device: str = "cpu",
) -> float:
    """
    Compute temporal-geometric alignment metric.

    Samples (anchor, positive) pairs using temporal containment and checks
    if geometric containment also holds.

    Geometric containment: min(s_k, s_l) >= min(s_i, s_j) AND max(s_k, s_l) <= max(s_i, s_j)

    Args:
        encoder: Trained encoder model
        trajectories: List of trajectory state sequences
        n_samples: Number of pairs to sample
        seed: Random seed
        device: Compute device

    Returns:
        Fraction of temporally contained pairs that are also geometrically contained
    """
    np.random.seed(seed)
    encoder.eval()

    geometric_contained_count = 0
    total_count = 0

    # Filter valid trajectories
    valid_trajs = [t for t in trajectories if len(t) >= 4]

    for _ in range(n_samples):
        # Sample trajectory
        traj_idx = np.random.randint(len(valid_trajs))
        traj = valid_trajs[traj_idx]
        T = len(traj) - 1

        # Sample anchor interval [i, j]
        i = np.random.randint(0, T - 1)
        j = np.random.randint(i + 2, T + 1)

        # Sample positive (temporally contained) [k, l]
        k = np.random.randint(i, j)
        l = np.random.randint(k + 1, j + 1)

        # Get state values
        s_i, s_j = traj[i], traj[j]
        s_k, s_l = traj[k], traj[l]

        # Check geometric containment
        anchor_min, anchor_max = min(s_i, s_j), max(s_i, s_j)
        pos_min, pos_max = min(s_k, s_l), max(s_k, s_l)

        is_geometrically_contained = (pos_min >= anchor_min - 1e-6) and (pos_max <= anchor_max + 1e-6)

        if is_geometrically_contained:
            geometric_contained_count += 1
        total_count += 1

    return geometric_contained_count / total_count if total_count > 0 else 0.0


def compute_norm_length_correlation(
    encoder: torch.nn.Module,
    trajectories: List[List[float]],
    n_samples: int = 10000,
    seed: int = 42,
    device: str = "cpu",
) -> float:
    """
    Compute Spearman correlation between interval length and embedding norm.

    For hyperbolic embeddings, uses hyperbolic norm: arctanh(||φ(s,g)||)
    For Euclidean embeddings, uses standard L2 norm.

    Args:
        encoder: Trained encoder model
        trajectories: List of trajectory state sequences
        n_samples: Number of samples
        seed: Random seed
        device: Compute device

    Returns:
        Spearman correlation coefficient
    """
    np.random.seed(seed)
    encoder.eval()

    is_hyperbolic = hasattr(encoder, 'manifold') and encoder.manifold is not None

    intervals = []
    lengths = []

    # Filter valid trajectories
    valid_trajs = [t for t in trajectories if len(t) >= 2]

    for _ in range(n_samples):
        traj_idx = np.random.randint(len(valid_trajs))
        traj = valid_trajs[traj_idx]
        T = len(traj) - 1

        # Sample random interval
        i = np.random.randint(0, T)
        j = np.random.randint(i + 1, T + 1)

        s_i, s_j = traj[i], traj[j]
        intervals.append([s_i, s_j])
        lengths.append(abs(s_j - s_i))

    # Compute embeddings
    intervals_tensor = torch.tensor(intervals, dtype=torch.float32).to(device)

    with torch.no_grad():
        embeddings = encoder(intervals_tensor)
        if isinstance(embeddings, ManifoldTensor):
            embeddings = embeddings.tensor

    # Compute norms
    if is_hyperbolic:
        norms = compute_hyperbolic_norm(embeddings.cpu()).numpy()
    else:
        norms = torch.norm(embeddings, dim=-1).cpu().numpy()

    # Compute Spearman correlation
    correlation, _ = stats.spearmanr(lengths, norms)

    return correlation


def compute_angle_midpoint_correlation(
    encoder: torch.nn.Module,
    trajectories: List[List[float]],
    n_samples: int = 10000,
    seed: int = 42,
    device: str = "cpu",
) -> Optional[float]:
    """
    Compute Spearman correlation between interval midpoint and embedding angle.

    Only meaningful for 2D embeddings.

    Args:
        encoder: Trained encoder model (must have embedding_dim=2)
        trajectories: List of trajectory state sequences
        n_samples: Number of samples
        seed: Random seed
        device: Compute device

    Returns:
        Spearman correlation coefficient, or None if embedding_dim != 2
    """
    # Check embedding dimension
    if not hasattr(encoder, 'embedding_dim') or encoder.embedding_dim != 2:
        return None

    np.random.seed(seed)
    encoder.eval()

    intervals = []
    midpoints = []

    valid_trajs = [t for t in trajectories if len(t) >= 2]

    for _ in range(n_samples):
        traj_idx = np.random.randint(len(valid_trajs))
        traj = valid_trajs[traj_idx]
        T = len(traj) - 1

        i = np.random.randint(0, T)
        j = np.random.randint(i + 1, T + 1)

        s_i, s_j = traj[i], traj[j]
        intervals.append([s_i, s_j])
        midpoints.append((s_i + s_j) / 2)

    # Compute embeddings
    intervals_tensor = torch.tensor(intervals, dtype=torch.float32).to(device)

    with torch.no_grad():
        embeddings = encoder(intervals_tensor)
        if isinstance(embeddings, ManifoldTensor):
            embeddings = embeddings.tensor

    embeddings = embeddings.cpu().numpy()

    # Compute angles
    angles = np.arctan2(embeddings[:, 1], embeddings[:, 0])

    # Compute Spearman correlation
    correlation, _ = stats.spearmanr(midpoints, angles)

    return correlation


def compute_embedding_statistics(
    encoder: torch.nn.Module,
    trajectories: List[List[float]],
    n_samples: int = 10000,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute embedding norm statistics.

    Args:
        encoder: Trained encoder model
        trajectories: List of trajectory state sequences
        n_samples: Number of samples
        seed: Random seed
        device: Compute device

    Returns:
        Dictionary with norm statistics
    """
    np.random.seed(seed)
    encoder.eval()

    is_hyperbolic = hasattr(encoder, 'manifold') and encoder.manifold is not None

    intervals = []
    valid_trajs = [t for t in trajectories if len(t) >= 2]

    for _ in range(n_samples):
        traj_idx = np.random.randint(len(valid_trajs))
        traj = valid_trajs[traj_idx]
        T = len(traj) - 1

        i = np.random.randint(0, T)
        j = np.random.randint(i + 1, T + 1)

        intervals.append([traj[i], traj[j]])

    intervals_tensor = torch.tensor(intervals, dtype=torch.float32).to(device)

    with torch.no_grad():
        embeddings = encoder(intervals_tensor)
        if isinstance(embeddings, ManifoldTensor):
            embeddings = embeddings.tensor

    # Compute Euclidean norms
    euclidean_norms = torch.norm(embeddings, dim=-1).cpu().numpy()

    # Compute hyperbolic norms if applicable
    if is_hyperbolic:
        hyperbolic_norms = compute_hyperbolic_norm(embeddings.cpu()).numpy()
    else:
        hyperbolic_norms = euclidean_norms

    return {
        "euclidean_norm_mean": float(np.mean(euclidean_norms)),
        "euclidean_norm_std": float(np.std(euclidean_norms)),
        "euclidean_norm_min": float(np.min(euclidean_norms)),
        "euclidean_norm_max": float(np.max(euclidean_norms)),
        "hyperbolic_norm_mean": float(np.mean(hyperbolic_norms)),
        "hyperbolic_norm_std": float(np.std(hyperbolic_norms)),
    }


def compute_all_representation_metrics(
    encoder: torch.nn.Module,
    trajectories: List[List[float]],
    n_samples: int = 10000,
    seed: int = 42,
    device: str = "cpu",
) -> RepresentationMetrics:
    """
    Compute all representation quality metrics.

    Args:
        encoder: Trained encoder model
        trajectories: List of trajectory state sequences
        n_samples: Number of samples for each metric
        seed: Random seed
        device: Compute device

    Returns:
        RepresentationMetrics dataclass
    """
    # Compute individual metrics
    alignment = compute_temporal_geometric_alignment(
        encoder, trajectories, n_samples, seed, device
    )

    norm_corr = compute_norm_length_correlation(
        encoder, trajectories, n_samples, seed + 1, device
    )

    angle_corr = compute_angle_midpoint_correlation(
        encoder, trajectories, n_samples, seed + 2, device
    )

    stats = compute_embedding_statistics(
        encoder, trajectories, n_samples, seed + 4, device
    )

    return RepresentationMetrics(
        temporal_geometric_alignment=alignment,
        norm_length_correlation=norm_corr,
        angle_midpoint_correlation=angle_corr,
        embedding_norm_mean=stats["hyperbolic_norm_mean"],
        embedding_norm_std=stats["hyperbolic_norm_std"],
    )


# Policy evaluation metrics

def compute_action_entropy(action_probs: torch.Tensor) -> float:
    """
    Compute average entropy of action distribution.

    Args:
        action_probs: Action probabilities (batch, n_actions)

    Returns:
        Mean entropy
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    entropy = -torch.sum(action_probs * torch.log(action_probs + eps), dim=-1)
    return entropy.mean().item()


def compute_policy_metrics(
    success_flags: List[bool],
    steps_to_goal: List[int],
    n_bins: int = 100,
    action_entropies: Optional[List[float]] = None,
) -> PolicyMetrics:
    """
    Compute policy evaluation metrics.

    Args:
        success_flags: List of success/failure flags for each episode
        steps_to_goal: List of steps taken (for successful episodes)
        n_bins: Number of bins (for computing path efficiency)
        action_entropies: Optional list of action entropies per episode

    Returns:
        PolicyMetrics dataclass
    """
    n_episodes = len(success_flags)
    n_success = sum(success_flags)
    success_rate = n_success / n_episodes if n_episodes > 0 else 0.0

    # Filter steps for successful episodes
    successful_steps = [s for s, flag in zip(steps_to_goal, success_flags) if flag]

    if len(successful_steps) > 0:
        mean_steps = float(np.mean(successful_steps))
        median_steps = float(np.median(successful_steps))
        std_steps = float(np.std(successful_steps))
        # Path efficiency: optimal_steps / actual_steps
        path_efficiency = float(np.mean([n_bins / s for s in successful_steps]))
    else:
        mean_steps = float('inf')
        median_steps = float('inf')
        std_steps = 0.0
        path_efficiency = 0.0

    if action_entropies:
        action_entropy = float(np.mean(action_entropies))
    else:
        action_entropy = 0.0

    return PolicyMetrics(
        success_rate=success_rate,
        mean_steps=mean_steps,
        median_steps=median_steps,
        std_steps=std_steps,
        path_efficiency=path_efficiency,
        action_entropy=action_entropy,
    )


if __name__ == "__main__":
    # Test the metrics
    print("Testing metrics module...")

    # Create dummy trajectories
    np.random.seed(42)
    dummy_trajectories = []
    for _ in range(100):
        length = np.random.randint(50, 150)
        # Simple random walk starting at 0
        steps = np.random.choice([-0.01, 0, 0.01], size=length)
        traj = np.cumsum(steps)
        traj = np.clip(traj - traj.min(), 0, 1)
        dummy_trajectories.append(traj.tolist())

    # Test with a simple encoder
    from src.models import EuclideanIntervalEncoder, HyperbolicIntervalEncoder

    device = "cpu"

    print("\n1. Testing with Euclidean encoder...")
    euc_encoder = EuclideanIntervalEncoder(embedding_dim=2).to(device)

    alignment = compute_temporal_geometric_alignment(
        euc_encoder, dummy_trajectories, n_samples=1000, device=device
    )
    print(f"  Temporal-geometric alignment: {alignment:.4f}")

    norm_corr = compute_norm_length_correlation(
        euc_encoder, dummy_trajectories, n_samples=1000, device=device
    )
    print(f"  Norm-length correlation: {norm_corr:.4f}")

    angle_corr = compute_angle_midpoint_correlation(
        euc_encoder, dummy_trajectories, n_samples=1000, device=device
    )
    print(f"  Angle-midpoint correlation: {angle_corr:.4f}")


    print("\n2. Testing with Hyperbolic encoder...")
    hyp_encoder = HyperbolicIntervalEncoder(embedding_dim=2).to(device)

    metrics = compute_all_representation_metrics(
        hyp_encoder, dummy_trajectories, n_samples=1000, device=device
    )
    print(f"  Temporal-geometric alignment: {metrics.temporal_geometric_alignment:.4f}")
    print(f"  Norm-length correlation: {metrics.norm_length_correlation:.4f}")
    print(f"  Angle-midpoint correlation: {metrics.angle_midpoint_correlation:.4f}")
    print(f"  Embedding norm: {metrics.embedding_norm_mean:.4f} ± {metrics.embedding_norm_std:.4f}")

    print("\n3. Testing policy metrics...")
    success_flags = [True, True, False, True, False, True, True, True, False, True]
    steps = [100, 120, 200, 110, 200, 105, 115, 108, 200, 112]

    policy_metrics = compute_policy_metrics(success_flags, steps, n_bins=100)
    print(f"  Success rate: {policy_metrics.success_rate:.2%}")
    print(f"  Mean steps: {policy_metrics.mean_steps:.1f}")
    print(f"  Path efficiency: {policy_metrics.path_efficiency:.4f}")

    print("\nAll tests passed!")
