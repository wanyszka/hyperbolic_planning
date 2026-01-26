"""Visualization module for hyperbolic interval embeddings."""

import numpy as np
import torch
import geoopt
import matplotlib.pyplot as plt
from pathlib import Path
from hypll.manifolds.poincare_ball import PoincareBall, Curvature
from hypll.tensors.manifold_tensor import ManifoldTensor


def get_device():
    """Get the current compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_geodesic_points(x, y, curvature: float = 1.0, num_points: int = 100):
    """
    Compute points along the geodesic between two points in a Poincaré ball.

    Args:
        x: Starting point coordinates
        y: Ending point coordinates
        curvature: Curvature of the Poincaré ball
        num_points: Number of points along the geodesic

    Returns:
        numpy array of shape (num_points, dim) containing geodesic points
    """
    manifold = geoopt.PoincareBall(c=curvature)

    x_tensor = geoopt.ManifoldTensor(
        torch.tensor(x, dtype=torch.float64), manifold=manifold
    )
    y_tensor = geoopt.ManifoldTensor(
        torch.tensor(y, dtype=torch.float64), manifold=manifold
    )

    if x_tensor.dim() == 1:
        x_tensor = x_tensor.unsqueeze(0)
    if y_tensor.dim() == 1:
        y_tensor = y_tensor.unsqueeze(0)

    t_values = torch.linspace(0, 1, num_points)
    geodesic_points = manifold.geodesic(t_values.unsqueeze(-1), x_tensor, y_tensor)

    return geodesic_points.squeeze(1).numpy()


def hyperbolic_norm(embedding, manifold):
    """Compute hyperbolic distance from origin."""
    origin = ManifoldTensor(
        torch.zeros(embedding.tensor.shape[-1]), manifold=manifold
    )
    return manifold.dist(embedding, origin).item()


def hyperbolic_norms(embeddings, manifold):
    """Compute hyperbolic norms for multiple embeddings."""
    return [hyperbolic_norm(emb, manifold) for emb in embeddings]


def embed_intervals(model, intervals, device=None):
    """
    Embed a list of intervals using the model.

    Args:
        model: HyperbolicIntervalEncoder model
        intervals: List of [i, j] interval pairs
        device: Compute device

    Returns:
        Dict mapping (i_idx, j_idx) to numpy embeddings
    """
    if device is None:
        device = get_device()

    model.eval()
    embeddings = {}

    with torch.no_grad():
        for i_idx, j_idx in intervals:
            interval_tensor = torch.tensor(
                [[i_idx, j_idx]], dtype=torch.float32
            ).to(device)
            emb = model(interval_tensor)
            if isinstance(emb, ManifoldTensor):
                emb = emb.tensor
            embeddings[(i_idx, j_idx)] = emb.squeeze(0).cpu().numpy()

    return embeddings


def plot_poincare_disk(ax, title=""):
    """Draw the Poincaré disk boundary and origin."""
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.scatter([0], [0], s=80, c='black', marker='+', linewidth=2, zorder=10)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    if title:
        ax.set_title(title, fontsize=14)


def test_specificity_gradient(model, num_points: int = 10, save_path: str = None):
    """
    Test and visualize the specificity gradient (norm vs duration relationship).

    Args:
        model: Trained HyperbolicIntervalEncoder
        num_points: Number of grid points
        save_path: Optional path to save figure
    """
    device = get_device()
    model.eval()

    intervals, durations, mid_points = [], [], []

    for a in range(num_points + 1):
        for b in range(a, num_points + 1):
            intervals.append([a / num_points, b / num_points])
            durations.append((b - a) / num_points)
            mid_points.append((a + b) / (2 * num_points))

    intervals_tensor = torch.tensor(intervals, dtype=torch.float32).to(device)

    with torch.no_grad():
        embeddings = model(intervals_tensor)
        if isinstance(embeddings, ManifoldTensor):
            norms = torch.tensor(hyperbolic_norms(embeddings, model.manifold))
        else:
            norms = torch.norm(embeddings, dim=-1).cpu()

    durations = np.array(durations)
    norms = norms.numpy() if hasattr(norms, 'numpy') else np.array(norms)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax = axes[0]
    scatter = ax.scatter(durations, norms, alpha=0.5, s=20, c=mid_points, cmap='viridis')
    ax.set_xlabel('Duration |b-a|', fontsize=12)
    ax.set_ylabel('Norm ||φ(a,b)||', fontsize=12)
    ax.set_title('Specificity Gradient', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='Mid Points (a+b)/2')
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(durations, norms, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(durations.min(), durations.max(), 100)
    ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, label=f'slope={z[0]:.3f}')
    ax.legend()

    # Binned statistics
    ax = axes[1]
    bins = np.linspace(durations.min(), durations.max(), 10)
    bin_means, bin_stds, bin_centers = [], [], []

    for i in range(len(bins) - 1):
        mask = (durations >= bins[i]) & (durations < bins[i + 1])
        if mask.sum() > 0:
            bin_means.append(norms[mask].mean())
            bin_stds.append(norms[mask].std())
            bin_centers.append((bins[i] + bins[i + 1]) / 2)

    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=5)
    ax.set_xlabel('Duration (binned)')
    ax.set_ylabel('Mean Norm')
    ax.set_title('Aggregated Gradient')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

    # Print statistics
    corr = np.corrcoef(durations, norms)[0, 1]
    print(f"\nCorrelation: {corr:.4f}")
    print(f"Expected: Negative (longer → smaller norm)")
    print(f"Result: {'✓ VALIDATED' if corr < -0.1 else '✗ NEEDS TUNING'}")


def test_geodesic(model, a=0.2, b=0.8, c=0.4, d=0.6, save_path: str = None):
    """
    Test the vague middle hypothesis via geodesic visualization.

    Args:
        model: Trained HyperbolicIntervalEncoder
        a, b: Endpoints of the parent interval [a, b]
        c, d: Endpoints of the subtask interval [c, d]
        save_path: Optional path to save figure
    """
    device = get_device()
    model.eval()

    # Create test intervals
    test_intervals = {
        's1': torch.tensor([[a, a]], dtype=torch.float32),
        's2': torch.tensor([[b, b]], dtype=torch.float32),
        'task': torch.tensor([[a, b]], dtype=torch.float32),
        'subtask': torch.tensor([[c, d]], dtype=torch.float32),
    }

    embeddings = {}
    with torch.no_grad():
        for name, interval in test_intervals.items():
            emb = model(interval.to(device))
            if isinstance(emb, ManifoldTensor):
                emb = emb.tensor
            embeddings[name] = emb.squeeze(0).cpu().numpy()

    z1, z2 = embeddings['s1'], embeddings['s2']
    z_task, z_subtask = embeddings['task'], embeddings['subtask']

    # Compute geodesic and find apex
    geodesic = compute_geodesic_points(z1, z2, curvature=1.0)
    geodesic_norms = np.linalg.norm(geodesic, axis=1)
    apex_idx = np.argmin(geodesic_norms)
    z_apex = geodesic[apex_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_poincare_disk(ax, 'Geodesic Vague Middle Test')

    ax.plot(geodesic[:, 0], geodesic[:, 1], 'b-', linewidth=2, alpha=0.6, label='Geodesic')
    ax.scatter(*z1, s=200, c='red', marker='o', edgecolors='black', linewidth=2, label=f's₁ [{a},{a}]', zorder=5)
    ax.scatter(*z2, s=200, c='red', marker='o', edgecolors='black', linewidth=2, label=f's₂ [{b},{b}]', zorder=5)
    ax.scatter(*z_task, s=300, c='green', marker='*', edgecolors='black', linewidth=2, label=f'Task [{a},{b}]', zorder=6)
    ax.scatter(*z_subtask, s=300, c='blue', marker='*', edgecolors='black', linewidth=2, label=f'Subtask [{c},{d}]', zorder=6)
    ax.scatter(*z_apex, s=200, c='orange', marker='D', edgecolors='black', linewidth=2, label='Apex', zorder=5)

    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

    # Print norms
    print(f"\nNorm z₁: {np.linalg.norm(z1):.4f}")
    print(f"Norm z₂: {np.linalg.norm(z2):.4f}")
    print(f"Norm task: {np.linalg.norm(z_task):.4f}")
    print(f"Norm subtask: {np.linalg.norm(z_subtask):.4f}")
    print(f"Norm apex: {np.linalg.norm(z_apex):.4f}")


def plot_all_intervals_with_geodesics(model, num_points: int = 3, curvature: float = 1.0):
    """
    Plot all intervals [i, j] where i <= j with geodesics connecting atomic intervals.

    Args:
        model: Trained HyperbolicIntervalEncoder
        num_points: Number of grid points
        curvature: Curvature of the Poincaré ball

    Returns:
        Dict of embeddings
    """
    device = get_device()
    model.eval()

    grid = np.linspace(0, 1, num_points + 1)
    intervals = [(i, j) for i in range(num_points + 1) for j in range(i, num_points + 1)]

    # Embed all intervals
    embeddings = {}
    with torch.no_grad():
        for i_idx, j_idx in intervals:
            interval = torch.tensor([[grid[i_idx], grid[j_idx]]], dtype=torch.float32).to(device)
            emb = model(interval)
            if isinstance(emb, ManifoldTensor):
                emb = emb.tensor
            embeddings[(i_idx, j_idx)] = emb.squeeze(0).cpu().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_poincare_disk(ax)

    # Draw geodesics between atomic endpoints for non-atomic intervals
    for i_idx, j_idx in intervals:
        if i_idx < j_idx:
            z_left = embeddings[(i_idx, i_idx)]
            z_right = embeddings[(j_idx, j_idx)]
            geodesic = compute_geodesic_points(z_left, z_right, curvature)
            ax.plot(geodesic[:, 0], geodesic[:, 1], 'b-', linewidth=1, alpha=0.3, zorder=1)

    # Draw all interval points
    for i_idx, j_idx in intervals:
        z = embeddings[(i_idx, j_idx)]
        i_val, j_val = grid[i_idx], grid[j_idx]

        if i_idx == j_idx:
            ax.scatter(*z, s=150, c='red', marker='o', edgecolors='black', linewidth=1.5, zorder=5)
            label = f'[{i_val:.2f}]'
        else:
            ax.scatter(*z, s=120, c='green', marker='s', edgecolors='black', linewidth=1.5, zorder=4)
            label = f'[{i_val:.2f},{j_val:.2f}]'

        ax.annotate(label, z, textcoords="offset points", xytext=(5, 5), fontsize=8, zorder=11)

    ax.set_title(f'All Intervals & Geodesics (N={num_points})\nRed: atomic [i,i]  |  Green: non-atomic [i,j]', fontsize=14)
    plt.tight_layout()
    plt.show()

    return embeddings

