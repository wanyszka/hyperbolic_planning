#!/usr/bin/env python3
"""
Generate all visualization figures from trained models and save to results/figures.

Usage:
    python -m scripts.generate_figures --config config/tight_frozen_experiment.yaml
"""

import argparse
import sys
import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Fix Windows console encoding issues
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.core.data_generation import load_dataset
from src.models import HyperbolicIntervalEncoder, EuclideanIntervalEncoder
from src.visualization.hyperbolic_viz import (
    test_specificity_gradient,
    plot_all_intervals_with_geodesics,
    test_geodesic,
)
from src.visualization.experiment_viz import (
    setup_plotting_style,
    plot_embedding_space_2d,
    plot_norm_vs_length,
    plot_angle_vs_midpoint,
)


def get_device():
    """Get compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(encoder_path: Path, device: str):
    """Load a trained encoder from checkpoint."""
    checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    geometry = config.get("geometry", "hyperbolic")
    embedding_dim = config.get("embedding_dim", 2)

    if geometry == "hyperbolic":
        encoder = HyperbolicIntervalEncoder(
            embedding_dim=embedding_dim,
            euc_width=config.get("euc_width", 128),
            hyp_width=config.get("hyp_width", 128),
        )
    else:
        hidden_sizes = config.get("hidden_sizes", [128, 128, 128, 128])
        encoder = EuclideanIntervalEncoder(
            embedding_dim=embedding_dim,
            hidden_sizes=hidden_sizes,
        )

    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.to(device)
    encoder.eval()

    return encoder, geometry


def generate_hyperbolic_figures(encoder, output_dir: Path, device: str):
    """Generate hyperbolic-specific visualizations."""
    print("\n--- Hyperbolic Visualizations ---")

    # Test specificity gradient
    print("Generating specificity gradient plot...")
    save_path = output_dir / "specificity_gradient.png"
    plt.ioff()
    try:
        test_specificity_gradient(encoder, num_points=10, save_path=str(save_path))
    except UnicodeEncodeError:
        pass  # Ignore console printing issues on Windows
    plt.close('all')

    # Plot all intervals with geodesics
    print("Generating intervals with geodesics plot...")
    try:
        embeddings = plot_all_intervals_with_geodesics(encoder, num_points=4, curvature=1.0)
        plt.savefig(output_dir / "intervals_with_geodesics.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / 'intervals_with_geodesics.png'}")
    except UnicodeEncodeError:
        pass
    plt.close('all')

    # Test geodesic (vague middle hypothesis)
    print("Generating geodesic test plot...")
    save_path = output_dir / "geodesic_vague_middle.png"
    try:
        test_geodesic(encoder, a=0.2, b=0.8, c=0.4, d=0.6, save_path=str(save_path))
    except UnicodeEncodeError:
        pass
    plt.close('all')


def generate_experiment_figures(encoder, trajectories, output_dir: Path, device: str, geometry: str):
    """Generate experiment visualization figures."""
    print("\n--- Experiment Visualizations ---")

    # Embedding space 2D
    print("Generating 2D embedding space plot...")
    fig = plot_embedding_space_2d(
        encoder, trajectories,
        n_samples=2000,
        save_path=str(output_dir / f"embedding_space_2d_{geometry}.png"),
        title=f"{geometry.capitalize()} Interval Embeddings",
        device=device,
    )
    plt.close(fig)

    # Norm vs length
    print("Generating norm vs length plot...")
    fig, corr = plot_norm_vs_length(
        encoder, trajectories,
        n_samples=5000,
        save_path=str(output_dir / f"norm_vs_length_{geometry}.png"),
        title=f"Embedding Norm vs Interval Length ({geometry.capitalize()})",
        device=device,
    )
    print(f"  Norm-Length correlation: {corr:.4f}")
    plt.close(fig)

    # Angle vs midpoint (only for 2D)
    if hasattr(encoder, 'embedding_dim') and encoder.embedding_dim == 2:
        print("Generating angle vs midpoint plot...")
        fig, corr = plot_angle_vs_midpoint(
            encoder, trajectories,
            n_samples=5000,
            save_path=str(output_dir / f"angle_vs_midpoint_{geometry}.png"),
            device=device,
        )
        if fig:
            print(f"  Angle-Midpoint correlation: {corr:.4f}")
            plt.close(fig)


def generate_additional_figures(encoder, trajectories, output_dir: Path, device: str, geometry: str):
    """Generate additional analysis figures."""
    print("\n--- Additional Analysis Figures ---")

    # Containment visualization
    print("Generating containment hierarchy plot...")
    fig = plot_containment_hierarchy(encoder, trajectories, device, geometry)
    fig.savefig(output_dir / f"containment_hierarchy_{geometry}.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'containment_hierarchy_{geometry}.png'}")
    plt.close(fig)

    # Embedding density
    print("Generating embedding density plot...")
    fig = plot_embedding_density(encoder, trajectories, device, geometry)
    fig.savefig(output_dir / f"embedding_density_{geometry}.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / f'embedding_density_{geometry}.png'}")
    plt.close(fig)


def plot_containment_hierarchy(encoder, trajectories, device: str, geometry: str):
    """Plot intervals showing containment relationships."""
    encoder.eval()
    np.random.seed(42)

    # Create nested intervals
    base_intervals = []
    labels = []
    colors = []

    # Full interval and nested subintervals
    levels = [
        ([0.0, 1.0], 'Full [0,1]', 'red'),
        ([0.1, 0.9], '[0.1,0.9]', 'orange'),
        ([0.2, 0.8], '[0.2,0.8]', 'yellow'),
        ([0.3, 0.7], '[0.3,0.7]', 'green'),
        ([0.4, 0.6], '[0.4,0.6]', 'blue'),
        ([0.45, 0.55], '[0.45,0.55]', 'purple'),
    ]

    for interval, label, color in levels:
        base_intervals.append(interval)
        labels.append(label)
        colors.append(color)

    # Add atomic states
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        base_intervals.append([s, s])
        labels.append(f'[{s}]')
        colors.append('gray')

    intervals_tensor = torch.tensor(base_intervals, dtype=torch.float32).to(device)

    with torch.no_grad():
        embeddings = encoder(intervals_tensor)
        if hasattr(embeddings, 'tensor'):
            embeddings = embeddings.tensor
        embeddings = embeddings.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw Poincare disk if hyperbolic
    if geometry == "hyperbolic":
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2, linestyle='--')
        ax.add_patch(circle)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)

    # Plot points
    for i, (emb, label, color) in enumerate(zip(embeddings, labels, colors)):
        ax.scatter(emb[0], emb[1], c=color, s=150, edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(label, emb, textcoords="offset points", xytext=(8, 8), fontsize=9)

    # Draw lines connecting nested intervals
    for i in range(len(levels) - 1):
        ax.plot([embeddings[i, 0], embeddings[i+1, 0]],
                [embeddings[i, 1], embeddings[i+1, 1]],
                'k-', alpha=0.3, linewidth=1)

    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    ax.set_title(f'Containment Hierarchy ({geometry.capitalize()})\nNested intervals should move toward origin')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return fig


def plot_embedding_density(encoder, trajectories, device: str, geometry: str):
    """Plot 2D histogram of embedding density."""
    encoder.eval()
    np.random.seed(42)

    # Sample many intervals
    intervals = []
    valid_trajs = [t for t in trajectories if len(t) >= 2]

    for _ in range(5000):
        traj_idx = np.random.randint(len(valid_trajs))
        traj = valid_trajs[traj_idx]
        T = len(traj) - 1
        i = np.random.randint(0, T)
        j = np.random.randint(i + 1, T + 1)
        intervals.append([traj[i], traj[j]])

    intervals_tensor = torch.tensor(intervals, dtype=torch.float32).to(device)

    with torch.no_grad():
        embeddings = encoder(intervals_tensor)
        if hasattr(embeddings, 'tensor'):
            embeddings = embeddings.tensor
        embeddings = embeddings.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))

    # 2D histogram
    h = ax.hist2d(embeddings[:, 0], embeddings[:, 1], bins=50, cmap='hot')
    plt.colorbar(h[3], ax=ax, label='Count')

    # Draw Poincare disk if hyperbolic
    if geometry == "hyperbolic":
        circle = plt.Circle((0, 0), 1, color='white', fill=False, linewidth=2, linestyle='--')
        ax.add_patch(circle)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)

    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    ax.set_title(f'Embedding Density ({geometry.capitalize()})')
    ax.set_aspect('equal')

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate visualization figures")
    parser.add_argument("--config", type=str, default="config/tight_frozen_experiment.yaml",
                        help="Path to config file")
    parser.add_argument("--output", type=str, default="results/figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Setup plotting style
    setup_plotting_style()
    plt.ioff()  # Turn off interactive mode

    # Load dataset
    data_config = config.get("data", {})
    data_dir = Path(data_config.get("data_dir", "data"))
    regimes = list(data_config.get("regimes", {}).keys())

    if not regimes:
        regimes = ["tight"]

    regime = regimes[0]
    print(f"\nLoading dataset: {regime}")

    dataset_path = data_dir / f"dataset_{regime}.pkl"
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("Please run data generation first:")
        print("  python -m scripts.generate_data --config <config>")
        return

    dataset = load_dataset(dataset_path)
    trajectories = dataset["trajectories"]
    print(f"Loaded {len(trajectories)} trajectories")

    # Find trained encoders
    results_dir = Path(config.get("experiment", {}).get("results_dir", "results"))
    rep_dir = results_dir / "representations"

    # Get geometry and embedding dim from config
    rep_config = config.get("representation", {})
    geometries = rep_config.get("geometries", ["hyperbolic"])
    embedding_dims = rep_config.get("embedding_dims", [2])

    for geometry in geometries:
        for dim in embedding_dims:
            encoder_name = f"{regime}_{geometry}_dim{dim}"
            encoder_path = rep_dir / encoder_name / "encoder_final.pt"

            if not encoder_path.exists():
                print(f"\nEncoder not found: {encoder_path}")
                print("Please run training first:")
                print("  python -m scripts.run_all_experiments --config <config>")
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {encoder_name}")
            print(f"{'='*60}")

            # Load encoder
            encoder, loaded_geometry = load_encoder(encoder_path, device)
            print(f"Loaded encoder: {loaded_geometry}, dim={encoder.embedding_dim}")

            # Generate figures
            generate_experiment_figures(encoder, trajectories, output_dir, device, geometry)

            if geometry == "hyperbolic":
                generate_hyperbolic_figures(encoder, output_dir, device)

            generate_additional_figures(encoder, trajectories, output_dir, device, geometry)

    print(f"\n{'='*60}")
    print(f"All figures saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
