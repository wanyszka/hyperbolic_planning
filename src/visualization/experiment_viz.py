"""Visualization module for experiment results and analysis."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from scipy import stats

from hypll.tensors.manifold_tensor import ManifoldTensor


def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (10, 8),
        'figure.dpi': 100,
    })


# =============================================================================
# Representation Visualizations
# =============================================================================

def plot_embedding_space_2d(
    encoder: torch.nn.Module,
    trajectories: List[List[float]],
    n_samples: int = 2000,
    save_path: Optional[str] = None,
    title: str = "Interval Embeddings",
    device: str = "cpu",
) -> plt.Figure:
    """
    Plot 2D embedding space colored by interval midpoint and sized by length.

    Args:
        encoder: Trained encoder (must have embedding_dim=2)
        trajectories: List of trajectories
        n_samples: Number of intervals to sample
        save_path: Path to save figure
        title: Plot title
        device: Compute device

    Returns:
        Matplotlib figure
    """
    encoder.eval()
    np.random.seed(42)

    # Sample intervals
    intervals = []
    midpoints = []
    lengths = []

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
        lengths.append(abs(s_j - s_i))

    # Get embeddings
    intervals_tensor = torch.tensor(intervals, dtype=torch.float32).to(device)

    with torch.no_grad():
        embeddings = encoder(intervals_tensor)
        if isinstance(embeddings, ManifoldTensor):
            embeddings = embeddings.tensor
        embeddings = embeddings.cpu().numpy()

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw Poincaré disk boundary if hyperbolic
    is_hyperbolic = hasattr(encoder, 'manifold') and encoder.manifold is not None
    if is_hyperbolic:
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2, linestyle='--')
        ax.add_patch(circle)
        ax.scatter([0], [0], s=100, c='black', marker='+', linewidth=2, zorder=10)

    # Normalize sizes
    sizes = np.array(lengths)
    sizes = 20 + 200 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-8)

    # Plot embeddings
    scatter = ax.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=midpoints, cmap='viridis',
        s=sizes, alpha=0.6,
        edgecolors='white', linewidth=0.5,
    )

    plt.colorbar(scatter, ax=ax, label='Interval Midpoint (s+g)/2')

    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    ax.set_title(f'{title}\n(point size ∝ interval length)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if is_hyperbolic:
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_norm_vs_length(
    encoder: torch.nn.Module,
    trajectories: List[List[float]],
    n_samples: int = 5000,
    save_path: Optional[str] = None,
    title: str = "Embedding Norm vs Interval Length",
    device: str = "cpu",
) -> Tuple[plt.Figure, float]:
    """
    Plot embedding norm versus interval length with correlation.

    Args:
        encoder: Trained encoder
        trajectories: List of trajectories
        n_samples: Number of samples
        save_path: Path to save figure
        title: Plot title
        device: Compute device

    Returns:
        Tuple of (figure, Spearman correlation)
    """
    encoder.eval()
    np.random.seed(42)

    is_hyperbolic = hasattr(encoder, 'manifold') and encoder.manifold is not None

    intervals = []
    lengths = []

    valid_trajs = [t for t in trajectories if len(t) >= 2]

    for _ in range(n_samples):
        traj_idx = np.random.randint(len(valid_trajs))
        traj = valid_trajs[traj_idx]
        T = len(traj) - 1

        i = np.random.randint(0, T)
        j = np.random.randint(i + 1, T + 1)

        s_i, s_j = traj[i], traj[j]
        intervals.append([s_i, s_j])
        lengths.append(abs(s_j - s_i))

    intervals_tensor = torch.tensor(intervals, dtype=torch.float32).to(device)

    with torch.no_grad():
        embeddings = encoder(intervals_tensor)
        if isinstance(embeddings, ManifoldTensor):
            embeddings = embeddings.tensor
        embeddings = embeddings.cpu()

    # Compute norms
    if is_hyperbolic:
        euclidean_norm = torch.norm(embeddings, dim=-1)
        euclidean_norm = torch.clamp(euclidean_norm, max=1.0 - 1e-6)
        norms = (2 * torch.arctanh(euclidean_norm)).numpy()
    else:
        norms = torch.norm(embeddings, dim=-1).numpy()

    lengths = np.array(lengths)

    # Compute correlation
    corr, p_value = stats.spearmanr(lengths, norms)

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(lengths, norms, alpha=0.3, s=10)

    # Add trend line
    z = np.polyfit(lengths, norms, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(lengths.min(), lengths.max(), 100)
    ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, label=f'Linear fit (slope={z[0]:.3f})')

    ax.set_xlabel('Interval Length |g - s|')
    ax.set_ylabel('Embedding Norm' + (' (hyperbolic)' if is_hyperbolic else ' (Euclidean)'))
    ax.set_title(f'Spearman ρ = {corr:.4f} (p = {p_value:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Binned statistics
    ax = axes[1]
    n_bins = 15
    bin_edges = np.linspace(lengths.min(), lengths.max(), n_bins + 1)
    bin_centers = []
    bin_means = []
    bin_stds = []

    for i in range(n_bins):
        mask = (lengths >= bin_edges[i]) & (lengths < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_means.append(norms[mask].mean())
            bin_stds.append(norms[mask].std())

    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=5, capthick=2)
    ax.set_xlabel('Interval Length (binned)')
    ax.set_ylabel('Mean Norm')
    ax.set_title('Aggregated Norm vs Length')
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, corr


def plot_angle_vs_midpoint(
    encoder: torch.nn.Module,
    trajectories: List[List[float]],
    n_samples: int = 5000,
    save_path: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[Optional[plt.Figure], Optional[float]]:
    """
    Plot embedding angle versus interval midpoint (for 2D embeddings only).

    Args:
        encoder: Trained encoder (must have embedding_dim=2)
        trajectories: List of trajectories
        n_samples: Number of samples
        save_path: Path to save figure
        device: Compute device

    Returns:
        Tuple of (figure, Spearman correlation) or (None, None) if not 2D
    """
    if not hasattr(encoder, 'embedding_dim') or encoder.embedding_dim != 2:
        return None, None

    encoder.eval()
    np.random.seed(42)

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

    intervals_tensor = torch.tensor(intervals, dtype=torch.float32).to(device)

    with torch.no_grad():
        embeddings = encoder(intervals_tensor)
        if isinstance(embeddings, ManifoldTensor):
            embeddings = embeddings.tensor
        embeddings = embeddings.cpu().numpy()

    # Compute angles
    angles = np.arctan2(embeddings[:, 1], embeddings[:, 0])
    midpoints = np.array(midpoints)

    # Compute correlation
    corr, p_value = stats.spearmanr(midpoints, angles)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(midpoints, angles, c=midpoints, cmap='viridis', alpha=0.4, s=10)
    plt.colorbar(scatter, ax=ax, label='Interval Midpoint')

    # Add trend line
    z = np.polyfit(midpoints, angles, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(midpoints.min(), midpoints.max(), 100)
    ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, label=f'Linear fit (slope={z[0]:.3f})')

    ax.set_xlabel('Interval Midpoint (s + g) / 2')
    ax.set_ylabel('Embedding Angle (radians)')
    ax.set_title(f'Angle vs Midpoint\nSpearman ρ = {corr:.4f} (p = {p_value:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, corr


# =============================================================================
# Policy Visualizations
# =============================================================================

def plot_learning_curves(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Learning Curves",
) -> plt.Figure:
    """
    Plot training and validation loss curves for multiple methods.

    Args:
        results: Dict mapping method names to dicts with 'train_losses' and 'val_losses'
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10.colors

    # Training loss
    ax = axes[0]
    for i, (name, data) in enumerate(results.items()):
        if 'train_losses' in data:
            ax.plot(data['train_losses'], label=name, color=colors[i % len(colors)])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation loss
    ax = axes[1]
    for i, (name, data) in enumerate(results.items()):
        if 'val_losses' in data and data['val_losses']:
            ax.plot(data['val_losses'], label=name, color=colors[i % len(colors)])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_success_rate_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Success Rate Comparison",
) -> plt.Figure:
    """
    Plot success rate comparison across methods and data regimes.

    Args:
        results: Nested dict: results[method][regime] = success_rate
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    methods = list(results.keys())
    regimes = list(results[methods[0]].keys()) if methods else []

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(regimes))
    width = 0.8 / len(methods)
    colors = plt.cm.tab10.colors

    for i, method in enumerate(methods):
        success_rates = [results[method].get(regime, 0) for regime in regimes]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, success_rates, width, label=method, color=colors[i % len(colors)])

        # Add value labels
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f'{rate:.1%}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Data Regime')
    ax.set_ylabel('Success Rate')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_sample_trajectories(
    trajectories: Dict[str, List[List[float]]],
    n_samples: int = 10,
    n_bins: int = 100,
    save_path: Optional[str] = None,
    title: str = "Sample Trajectories",
) -> plt.Figure:
    """
    Plot sample rollout trajectories for multiple methods.

    Args:
        trajectories: Dict mapping method names to lists of trajectories
        n_samples: Number of trajectories to plot per method
        n_bins: Number of bins (for optimal trajectory)
        save_path: Path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    n_methods = len(trajectories)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5), squeeze=False)
    axes = axes[0]

    colors = plt.cm.tab10.colors

    for ax, (method, trajs) in zip(axes, trajectories.items()):
        # Plot optimal trajectory
        optimal_x = range(n_bins + 1)
        optimal_y = np.linspace(0, 1, n_bins + 1)
        ax.plot(optimal_x, optimal_y, 'k--', linewidth=2, label='Optimal', alpha=0.7)

        # Plot sample trajectories
        for i, traj in enumerate(trajs[:n_samples]):
            color = colors[i % len(colors)]
            ax.plot(range(len(traj)), traj, alpha=0.6, linewidth=1, color=color)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('State')
        ax.set_title(method)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_action_distribution_heatmap(
    policy: torch.nn.Module,
    n_states: int = 50,
    goal: float = 1.0,
    save_path: Optional[str] = None,
    device: str = "cpu",
) -> plt.Figure:
    """
    Plot heatmap of action distribution across states.

    Args:
        policy: Trained policy
        n_states: Number of states to sample
        goal: Goal state
        save_path: Path to save figure
        device: Compute device

    Returns:
        Matplotlib figure
    """
    policy.eval()

    states = np.linspace(0, 1, n_states)
    action_probs = np.zeros((3, n_states))  # 3 actions x n_states

    with torch.no_grad():
        for i, state in enumerate(states):
            state_tensor = torch.tensor([[state]], dtype=torch.float32).to(device)
            goal_tensor = torch.tensor([[goal]], dtype=torch.float32).to(device)

            logits = policy(state_tensor, goal_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            action_probs[:, i] = probs

    fig, ax = plt.subplots(figsize=(12, 4))

    im = ax.imshow(action_probs, aspect='auto', cmap='Blues',
                   extent=[0, 1, -0.5, 2.5], origin='lower')

    plt.colorbar(im, ax=ax, label='Probability')

    ax.set_xlabel('State')
    ax.set_ylabel('Action')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Left (-1)', 'Stay (0)', 'Right (+1)'])
    ax.set_title(f'Action Distribution π(a|s, g={goal})')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Results Tables
# =============================================================================

def create_representation_table(results: Dict[str, Dict]) -> str:
    """
    Create a formatted table of representation quality metrics.

    Args:
        results: Dict mapping condition names to metric dicts

    Returns:
        Formatted string table
    """
    header = f"{'Condition':<30} {'Alignment':>10} {'Norm-Len':>10} {'Angle-Mid':>10} {'AUROC':>10}"
    separator = "-" * 75

    lines = [separator, header, separator]

    for condition, metrics in results.items():
        alignment = metrics.get('temporal_geometric_alignment', 0)
        norm_corr = metrics.get('norm_length_correlation', 0)
        angle_corr = metrics.get('angle_midpoint_correlation', None)
        auroc = metrics.get('containment_auroc', 0)

        angle_str = f"{angle_corr:.4f}" if angle_corr is not None else "N/A"

        line = f"{condition:<30} {alignment:>10.4f} {norm_corr:>10.4f} {angle_str:>10} {auroc:>10.4f}"
        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)


def create_policy_table(results: Dict[str, Dict]) -> str:
    """
    Create a formatted table of policy performance metrics.

    Args:
        results: Dict mapping method names to metric dicts

    Returns:
        Formatted string table
    """
    header = f"{'Method':<25} {'Success%':>10} {'Steps':>10} {'Efficiency':>12}"
    separator = "-" * 60

    lines = [separator, header, separator]

    for method, metrics in results.items():
        success = metrics.get('success_rate', 0) * 100
        steps = metrics.get('mean_steps', float('inf'))
        efficiency = metrics.get('path_efficiency', 0)

        steps_str = f"{steps:.1f}" if steps != float('inf') else "N/A"

        line = f"{method:<25} {success:>9.1f}% {steps_str:>10} {efficiency:>12.4f}"
        lines.append(line)

    lines.append(separator)
    return "\n".join(lines)


def save_all_figures(figures: Dict[str, plt.Figure], output_dir: str) -> None:
    """
    Save all figures to a directory.

    Args:
        figures: Dict mapping names to matplotlib figures
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        filepath = output_path / f"{name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)



if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization module...")
    setup_plotting_style()

    from src.models import EuclideanIntervalEncoder, HyperbolicIntervalEncoder

    device = "cpu"

    # Create dummy trajectories
    np.random.seed(42)
    dummy_trajectories = []
    for _ in range(100):
        length = np.random.randint(50, 150)
        traj = np.linspace(0, 1, length)
        # Add some noise
        traj = traj + np.random.randn(length) * 0.02
        traj = np.clip(traj, 0, 1)
        dummy_trajectories.append(traj.tolist())

    # Test with Euclidean encoder
    print("\n1. Testing 2D embedding space plot...")
    encoder = EuclideanIntervalEncoder(embedding_dim=2).to(device)

    fig = plot_embedding_space_2d(
        encoder, dummy_trajectories,
        n_samples=500, device=device,
        title="Euclidean Embeddings (Untrained)"
    )
    plt.close(fig)
    print("  Done!")

    # Test norm vs length plot
    print("\n2. Testing norm vs length plot...")
    fig, corr = plot_norm_vs_length(
        encoder, dummy_trajectories,
        n_samples=1000, device=device
    )
    plt.close(fig)
    print(f"  Correlation: {corr:.4f}")

    # Test angle vs midpoint
    print("\n3. Testing angle vs midpoint plot...")
    fig, corr = plot_angle_vs_midpoint(
        encoder, dummy_trajectories,
        n_samples=1000, device=device
    )
    if fig:
        plt.close(fig)
        print(f"  Correlation: {corr:.4f}")
    else:
        print("  Skipped (requires 2D embeddings)")

    # Test learning curves
    print("\n4. Testing learning curves plot...")
    dummy_results = {
        "GCBC-Single": {"train_losses": np.random.rand(50).cumsum()[::-1].tolist(),
                        "val_losses": np.random.rand(50).cumsum()[::-1].tolist()},
        "GCBC-Interval": {"train_losses": (np.random.rand(50).cumsum()[::-1] * 0.8).tolist(),
                          "val_losses": (np.random.rand(50).cumsum()[::-1] * 0.8).tolist()},
    }
    fig = plot_learning_curves(dummy_results)
    plt.close(fig)
    print("  Done!")

    # Test success rate comparison
    print("\n5. Testing success rate comparison plot...")
    comparison_results = {
        "GCBC-Single": {"tight": 0.85, "moderate": 0.70, "loose": 0.55, "very_loose": 0.40},
        "GCBC-Interval": {"tight": 0.92, "moderate": 0.82, "loose": 0.68, "very_loose": 0.52},
    }
    fig = plot_success_rate_comparison(comparison_results)
    plt.close(fig)
    print("  Done!")

    # Test tables
    print("\n6. Testing table creation...")
    rep_results = {
        "tight_temporal_hyp": {"temporal_geometric_alignment": 0.92, "norm_length_correlation": -0.75,
                               "angle_midpoint_correlation": 0.85, "containment_auroc": 0.88},
        "loose_temporal_euc": {"temporal_geometric_alignment": 0.78, "norm_length_correlation": -0.55,
                               "angle_midpoint_correlation": 0.70, "containment_auroc": 0.72},
    }
    print(create_representation_table(rep_results))

    policy_results = {
        "GCBC-Single": {"success_rate": 0.75, "mean_steps": 125.5, "path_efficiency": 0.80},
        "GCBC-Interval (frozen)": {"success_rate": 0.88, "mean_steps": 110.2, "path_efficiency": 0.91},
    }
    print(create_policy_table(policy_results))

    print("\nAll visualization tests passed!")
