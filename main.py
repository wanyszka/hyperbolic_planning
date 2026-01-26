"""Main script for training and evaluating hyperbolic interval encoder."""

import torch
import numpy as np

from src.core.datasets import IntervalDataset
from src.models import HyperbolicIntervalEncoder
from src.training import train_model
from src.visualization import (
    test_specificity_gradient,
    test_geodesic,
    plot_all_intervals_with_geodesics,
    analyze_geodesic_apex_conjecture,
)


def setup_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # Setup
    setup_seeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Create dataset
    print("[1/4] Creating dataset...")
    dataset = IntervalDataset(
        num_samples=1000,
        num_negatives=2,
        num_points=10,
        endpoint_prob=0,
    )

    # Create model
    print("[2/4] Initializing model...")
    model = HyperbolicIntervalEncoder(
        embedding_dim=2,
        c=1.0,
        euc_width=128,
        hyp_width=128,
    ).to(device)

    # Train
    print("[3/4] Training hyperbolic encoder...")
    losses = train_model(
        model,
        dataset,
        num_epochs=200,
        batch_size=32,
        lr=0.001,
        temperature=0.1,
        device=device,
    )

    # Evaluate
    print("\n[4/4] Running evaluations...")

    print("\n--- Specificity Gradient Test ---")
    test_specificity_gradient(model, num_points=10, save_path='specificity_gradient.png')

    print("\n--- Geodesic Test ---")
    test_geodesic(model, a=0.1, b=0.9, c=0.3, d=0.7, save_path='geodesic_test.png')

    print("\n--- All Intervals Visualization ---")
    embeddings = plot_all_intervals_with_geodesics(model, num_points=5)

    print("\n--- Geodesic Apex Conjecture Analysis ---")
    results, embeddings, grid, intervals = analyze_geodesic_apex_conjecture(model, num_points=5)

    print("\nDone!")


if __name__ == "__main__":
    main()
