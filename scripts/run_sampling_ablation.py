#!/usr/bin/env python
"""
Sampling Strategy Ablation Experiment Runner.

Runs ablation studies on:
1. Negative formation strategies (plain, mixed, mixed_swapped)
2. Anchor sampling strategies (uniform, geometric)

Each run creates a unique timestamped directory to prevent overwriting.

Usage:
    python -m scripts.run_sampling_ablation --config config/sampling_ablation.yaml
    python -m scripts.run_sampling_ablation --config config/sampling_ablation.yaml --suite full
    python -m scripts.run_sampling_ablation --config config/sampling_ablation.yaml --name my_experiment
"""

import argparse
import yaml
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import hashlib

from src.core.data_generation import load_dataset, load_all_regimes
from src.core.datasets import (
    InBatchContrastiveDataset,
    create_train_val_test_split,
    get_ablation_suite,
    get_full_ablation_grid,
    get_extended_ablation_grid,
    create_ablation_config,
    print_ablation_summary,
    NegativeFormationType,
    AblationConfig,
)
from src.models import EuclideanIntervalEncoder, HyperbolicIntervalEncoder
from src.training import train_encoder, TrainingConfig
from src.evaluation import compute_all_representation_metrics


def load_config(config_path: str) -> Dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_experiment_dir(base_dir: str, name: Optional[str] = None) -> Path:
    """
    Create a unique experiment directory with timestamp.

    Format: {base_dir}/{name}_{YYYYMMDD}_{HHMMSS}_{short_hash}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create short hash for uniqueness
    hash_input = f"{timestamp}_{np.random.randint(0, 10000)}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    if name:
        dir_name = f"{name}_{timestamp}_{short_hash}"
    else:
        dir_name = f"ablation_{timestamp}_{short_hash}"

    exp_dir = Path(base_dir) / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def save_experiment_metadata(
    exp_dir: Path,
    config: Dict,
    ablation_configs: List[AblationConfig],
    args: argparse.Namespace,
) -> None:
    """Save experiment metadata for reproducibility."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "experiment_dir": str(exp_dir),
        "command_line_args": vars(args),
        "config_file": config,
        "ablation_configs": [c.to_dict() for c in ablation_configs],
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }

    with open(exp_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Also save the original config
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_ablation_configs_from_yaml(config: Dict) -> List[AblationConfig]:
    """Parse ablation configurations from YAML config."""
    ablations = config.get("ablations", {})
    training_defaults = config.get("training_defaults", {})

    configs = []
    for name, ablation_cfg in ablations.items():
        # Merge with defaults
        training_params = {**training_defaults, **ablation_cfg.get("training", {})}
        sampling_params = ablation_cfg.get("sampling", {})

        # Parse negative formation type
        neg_type_str = training_params.get("negative_formation", "plain")
        neg_type = NegativeFormationType(neg_type_str)

        cfg = create_ablation_config(
            name=name,
            description=ablation_cfg.get("description", ""),
            negative_formation=neg_type,
            swap_probability=training_params.get("swap_probability", 0.5),
            use_geometric=sampling_params.get("use_geometric", False),
            geometric_p=sampling_params.get("geometric_p", 0.3),
            min_anchor_length=sampling_params.get("min_anchor_length", 1),
            num_epochs=training_params.get("num_epochs", 100),
            batch_size=training_params.get("batch_size", 256),
            lr=training_params.get("lr", 0.001),
            temperature=training_params.get("temperature", 0.1),
        )
        configs.append(cfg)

    return configs


def run_single_ablation(
    ablation_config: AblationConfig,
    trajectories: Dict[str, List],
    exp_dir: Path,
    geometry: str,
    embedding_dim: int,
    device: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single ablation configuration."""
    name = ablation_config.name

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"  Negative: {ablation_config.training_config.negative_formation.value}")
        print(f"  Swap prob: {ablation_config.training_config.swap_probability}")
        print(f"  Geometric: {ablation_config.sampling_config.anchor_sampling.use_geometric}")
        print(f"{'='*60}")

    # Create output directory for this ablation
    ablation_dir = exp_dir / "ablations" / name
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # Split data
    train_trajs = trajectories["train"]
    val_trajs = trajectories["val"]
    test_trajs = trajectories["test"]

    # Create datasets
    train_dataset = InBatchContrastiveDataset(
        trajectories=train_trajs,
        num_samples=len(train_trajs) * 5,
        config=ablation_config.sampling_config,
        seed=42,
    )

    val_dataset = InBatchContrastiveDataset(
        trajectories=val_trajs,
        num_samples=len(val_trajs) * 2,
        config=ablation_config.sampling_config,
        seed=43,
    )

    # Create model
    if geometry == "euclidean":
        model = EuclideanIntervalEncoder(embedding_dim=embedding_dim)
    else:
        model = HyperbolicIntervalEncoder(embedding_dim=embedding_dim)

    model = model.to(device)

    # Train
    result = train_encoder(
        model,
        train_dataset,
        val_dataset,
        config=ablation_config.training_config,
        device=device,
        verbose=verbose,
        save_dir=str(ablation_dir),
    )

    # Evaluate representation quality
    model.eval()
    metrics = compute_all_representation_metrics(
        model, test_trajs, n_samples=5000, device=device
    )

    # Save model and results
    torch.save({
        "model_state_dict": model.state_dict(),
        "ablation_config": ablation_config.to_dict(),
        "training_result": {
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
        },
        "metrics": {
            "temporal_geometric_alignment": metrics.temporal_geometric_alignment,
            "norm_length_correlation": metrics.norm_length_correlation,
            "angle_midpoint_correlation": metrics.angle_midpoint_correlation,
        },
    }, ablation_dir / "model_final.pt")

    # Save losses as CSV for easy plotting
    import csv
    with open(ablation_dir / "losses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (train_loss, val_loss) in enumerate(zip(
            result.train_losses,
            result.val_losses if result.val_losses else [None] * len(result.train_losses)
        )):
            writer.writerow([i, train_loss, val_loss])

    results = {
        "name": name,
        "config": ablation_config.to_dict(),
        "training": {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "final_train_loss": result.train_losses[-1] if result.train_losses else None,
        },
        "metrics": {
            "temporal_geometric_alignment": metrics.temporal_geometric_alignment,
            "norm_length_correlation": metrics.norm_length_correlation,
            "angle_midpoint_correlation": metrics.angle_midpoint_correlation,
        },
        "model_path": str(ablation_dir / "model_final.pt"),
    }

    if verbose:
        print(f"\nResults for {name}:")
        print(f"  Best val loss: {result.best_val_loss:.4f} (epoch {result.best_epoch})")
        print(f"  Alignment: {metrics.temporal_geometric_alignment:.4f}")
        print(f"  Norm-Length Corr: {metrics.norm_length_correlation:.4f}")

    return results


def run_ablation_experiment(
    config: Dict,
    ablation_configs: List[AblationConfig],
    exp_dir: Path,
    device: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full ablation experiment."""

    # Load data
    data_config = config.get("data", {})
    data_dir = data_config.get("data_dir", "data")
    regime = data_config.get("regime", "tight")

    if verbose:
        print(f"\nLoading data from {data_dir}, regime: {regime}")

    dataset = load_dataset(data_dir, regime)
    trajectories_all = dataset["trajectories"]
    actions_all = dataset["actions"]

    # Split data
    split = create_train_val_test_split(
        trajectories_all,
        actions_all,
        val_fraction=data_config.get("val_fraction", 0.1),
        test_fraction=data_config.get("test_fraction", 0.1),
        seed=config.get("seeds", {}).get("data_split", 42),
    )

    trajectories = {
        "train": split["train"][0],
        "val": split["val"][0],
        "test": split["test"][0],
    }

    if verbose:
        print(f"  Train: {len(trajectories['train'])} trajectories")
        print(f"  Val: {len(trajectories['val'])} trajectories")
        print(f"  Test: {len(trajectories['test'])} trajectories")

    # Model config
    model_config = config.get("model", {})
    geometry = model_config.get("geometry", "hyperbolic")
    embedding_dim = model_config.get("embedding_dim", 2)

    # Run all ablations
    all_results = {
        "experiment_dir": str(exp_dir),
        "timestamp": datetime.now().isoformat(),
        "data": {
            "regime": regime,
            "n_train": len(trajectories["train"]),
            "n_val": len(trajectories["val"]),
            "n_test": len(trajectories["test"]),
        },
        "model": {
            "geometry": geometry,
            "embedding_dim": embedding_dim,
        },
        "ablations": {},
    }

    for ablation_cfg in ablation_configs:
        result = run_single_ablation(
            ablation_cfg,
            trajectories,
            exp_dir,
            geometry,
            embedding_dim,
            device,
            verbose,
        )
        all_results["ablations"][ablation_cfg.name] = result

    # Save combined results
    with open(exp_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Create summary table
    if verbose:
        print("\n" + "=" * 70)
        print("ABLATION RESULTS SUMMARY")
        print("=" * 70)
        print(f"\n{'Name':<25} {'Alignment':<12} {'Norm-Len':<12} {'Best Val':<12}")
        print("-" * 70)
        for name, res in all_results["ablations"].items():
            align = res["metrics"]["temporal_geometric_alignment"]
            norm_len = res["metrics"]["norm_length_correlation"]
            val_loss = res["training"]["best_val_loss"]
            print(f"{name:<25} {align:<12.4f} {norm_len:<12.4f} {val_loss:<12.4f}")

    # Save summary as CSV
    import csv
    with open(exp_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name", "negative_formation", "swap_prob", "use_geometric", "geometric_p",
            "alignment", "norm_length_corr", "angle_midpoint_corr", "best_val_loss", "best_epoch"
        ])
        for name, res in all_results["ablations"].items():
            cfg = res["config"]
            metrics = res["metrics"]
            training = res["training"]
            writer.writerow([
                name,
                cfg["negative_formation"],
                cfg["swap_probability"],
                cfg["use_geometric"],
                cfg["geometric_p"],
                metrics["temporal_geometric_alignment"],
                metrics["norm_length_correlation"],
                metrics["angle_midpoint_correlation"],
                training["best_val_loss"],
                training["best_epoch"],
            ])

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling strategy ablation experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sampling_ablation.yaml",
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=["negative", "anchor", "full", "extended", "config"],
        default="config",
        help="Ablation suite to run (default: use config file)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom name for experiment directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sampling_ablations",
        help="Base output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (auto-detected if not specified)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick version with reduced epochs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ablation configs without running",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        print(f"Warning: Config file not found at {config_path}, using defaults")
        config = {}

    # Determine ablation configs
    if args.suite == "config":
        # Use configs from YAML file
        ablation_configs = get_ablation_configs_from_yaml(config)
    else:
        # Use pre-defined suite
        training_defaults = config.get("training_defaults", {})
        ablation_configs = list(get_ablation_suite(
            args.suite,
            num_epochs=training_defaults.get("num_epochs", 100),
            batch_size=training_defaults.get("batch_size", 256),
            lr=training_defaults.get("lr", 0.001),
            temperature=training_defaults.get("temperature", 0.1),
        ).values())

    # Quick mode overrides
    if args.quick:
        for cfg in ablation_configs:
            cfg.training_config.num_epochs = 10
            cfg.training_config.print_every = 2

    # Dry run - just show configs
    if args.dry_run:
        print("\nAblation configurations to run:")
        print_ablation_summary(ablation_configs)
        return

    # Create experiment directory
    exp_dir = create_experiment_dir(args.output_dir, args.name or args.suite)

    print("=" * 70)
    print("SAMPLING STRATEGY ABLATION EXPERIMENT")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Suite: {args.suite}")
    print(f"Output: {exp_dir}")
    print(f"Ablations: {len(ablation_configs)}")
    print("=" * 70)

    # Show ablation configs
    print("\nAblation configurations:")
    print_ablation_summary(ablation_configs)

    # Save metadata
    save_experiment_metadata(exp_dir, config, ablation_configs, args)

    # Run experiment
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    results = run_ablation_experiment(
        config,
        ablation_configs,
        exp_dir,
        device,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {exp_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
