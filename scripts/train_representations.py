#!/usr/bin/env python
"""Script to train interval representation encoders."""

import argparse
import torch
import yaml
from pathlib import Path
from typing import Dict, List

from src.core.data_generation import load_dataset
from src.core.datasets import (
    TrajectoryContrastiveDataset,
    create_train_val_test_split,
)
from src.models import create_encoder, EuclideanIntervalEncoder, HyperbolicIntervalEncoder
from src.training import train_encoder, TrainingConfig
from src.evaluation import compute_all_representation_metrics


def train_single_encoder(
    trajectories: List[List[float]],
    geometry: str,
    embedding_dim: int,
    config: Dict,
    device: str,
    save_dir: str,
    verbose: bool = True,
) -> Dict:
    """Train a single encoder configuration."""

    # Create dataset split
    split = create_train_val_test_split(
        trajectories,
        actions=[[0] * (len(t) - 1) for t in trajectories],  # Dummy actions
        val_fraction=config.get("val_fraction", 0.1),
        test_fraction=config.get("test_fraction", 0.1),
    )

    train_trajs, _ = split["train"]
    val_trajs, _ = split["val"]
    test_trajs, _ = split["test"]

    # Create datasets
    n_negatives = config.get("n_negatives", 255)

    train_dataset = TrajectoryContrastiveDataset(
        trajectories=train_trajs,
        num_samples=len(train_trajs) * 10,
        n_negatives=n_negatives,
        seed=config.get("seed", 42),
    )

    val_dataset = TrajectoryContrastiveDataset(
        trajectories=val_trajs,
        num_samples=len(val_trajs) * 5,
        n_negatives=n_negatives,
        seed=config.get("seed", 42) + 1,
    )

    # Create encoder
    hidden_sizes = config.get("hidden_sizes", [128, 128, 128, 128])

    if geometry == "euclidean":
        encoder = EuclideanIntervalEncoder(
            embedding_dim=embedding_dim,
            hidden_sizes=hidden_sizes,
        )
    else:
        encoder = HyperbolicIntervalEncoder(
            embedding_dim=embedding_dim,
            c=config.get("curvature", 1.0),
            euc_width=hidden_sizes[0],
            hyp_width=hidden_sizes[-1],
        )

    encoder = encoder.to(device)

    # Training config
    training_config = TrainingConfig(
        num_epochs=config.get("epochs", 100),
        batch_size=config.get("batch_size", 256),
        lr=config.get("learning_rate", 0.001),
        lr_min=config.get("lr_min", 1e-5),
        temperature=config.get("temperature", 0.1),
        weight_decay=config.get("weight_decay", 1e-4),
        early_stopping_patience=config.get("early_stopping_patience", 20),
        print_every=config.get("print_every", 5),
        save_every=config.get("save_every", 10),
    )

    # Train
    if verbose:
        print(f"\nTraining {geometry} encoder (dim={embedding_dim})")

    result = train_encoder(
        encoder,
        train_dataset,
        val_dataset,
        config=training_config,
        device=device,
        verbose=verbose,
        save_dir=save_dir,
    )

    # Evaluate representation quality
    encoder.eval()
    metrics = compute_all_representation_metrics(
        encoder,
        test_trajs,
        n_samples=5000,
        device=device,
    )

    # Save final model
    model_path = Path(save_dir) / f"encoder_final.pt"
    torch.save({
        "model_state_dict": encoder.state_dict(),
        "config": {
            "geometry": geometry,
            "embedding_dim": embedding_dim,
            "hidden_sizes": hidden_sizes,
        },
        "metrics": {
            "temporal_geometric_alignment": metrics.temporal_geometric_alignment,
            "norm_length_correlation": metrics.norm_length_correlation,
            "angle_midpoint_correlation": metrics.angle_midpoint_correlation,
        },
        "training": {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
        },
    }, model_path)

    if verbose:
        print(f"\nRepresentation Metrics:")
        print(f"  Temporal-Geometric Alignment: {metrics.temporal_geometric_alignment:.4f}")
        print(f"  Norm-Length Correlation: {metrics.norm_length_correlation:.4f}")
        if metrics.angle_midpoint_correlation is not None:
            print(f"  Angle-Midpoint Correlation: {metrics.angle_midpoint_correlation:.4f}")
        print(f"\nSaved encoder to: {model_path}")

    return {
        "model_path": str(model_path),
        "metrics": {
            "temporal_geometric_alignment": metrics.temporal_geometric_alignment,
            "norm_length_correlation": metrics.norm_length_correlation,
            "angle_midpoint_correlation": metrics.angle_midpoint_correlation,
        },
        "training": {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Train interval representation encoders")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/representations",
        help="Output directory for trained models",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="moderate",
        help="Data regime to use",
    )
    parser.add_argument(
        "--geometry",
        type=str,
        default="hyperbolic",
        choices=["euclidean", "hyperbolic"],
        help="Embedding geometry",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=2,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (auto-detected if not specified)",
    )

    args = parser.parse_args()

    # Determine device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        rep_config = config.get("representation", {}).get("training", {})
    else:
        rep_config = {
            "epochs": args.epochs,
            "batch_size": 256,
            "learning_rate": 0.001,
            "temperature": 0.1,
            "n_negatives": 255,
            "early_stopping_patience": 20,
        }

    # Load dataset
    dataset_path = Path(args.data_dir) / f"dataset_{args.regime}.pkl"
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Run 'python scripts/generate_data.py' first")
        return

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(str(dataset_path))
    trajectories = dataset["trajectories"]
    print(f"Loaded {len(trajectories)} trajectories")

    # Create output directory
    save_dir = Path(args.output_dir) / args.regime / f"{args.geometry}_dim{args.embedding_dim}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train encoder
    results = train_single_encoder(
        trajectories=trajectories,
        geometry=args.geometry,
        embedding_dim=args.embedding_dim,
        config=rep_config,
        device=device,
        save_dir=str(save_dir),
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
