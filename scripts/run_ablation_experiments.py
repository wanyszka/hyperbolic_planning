#!/usr/bin/env python3
"""
Run ablation experiments for hyperbolic interval encoders.

Each run creates a unique timestamped directory to prevent overwriting.

Usage:
    python -m scripts.run_ablation_experiments --config config/ablation_experiments.yaml
    python -m scripts.run_ablation_experiments --config config/ablation_experiments.yaml --ablation high_temp
    python -m scripts.run_ablation_experiments --config config/ablation_experiments.yaml --name my_experiment
"""

import argparse
import yaml
import json
import torch
import numpy as np
import hashlib
from datetime import datetime
from pathlib import Path
from itertools import product
from dataclasses import asdict

from src.core.data_generation import load_dataset
from src.core.datasets import TrajectoryContrastiveDataset, PolicyTrainingDataset, create_train_val_test_split
from src.models import HyperbolicIntervalEncoder, EuclideanIntervalEncoder, GCBCSinglePolicy, GCBCIntervalPolicy
from src.training import train_encoder, train_policy, TrainingConfig
from src.evaluation import compute_all_representation_metrics


def get_device():
    """Get compute device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def create_experiment_dir(base_dir: str, ablation_name: str, custom_name: str = None) -> Path:
    """
    Create a unique experiment directory with timestamp.

    Format: {base_dir}/{ablation_name}_{YYYYMMDD}_{HHMMSS}_{short_hash}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_input = f"{timestamp}_{np.random.randint(0, 10000)}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    if custom_name:
        dir_name = f"{custom_name}_{timestamp}_{short_hash}"
    else:
        dir_name = f"{ablation_name}_{timestamp}_{short_hash}"

    exp_dir = Path(base_dir) / dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_experiment_metadata(
    exp_dir: Path,
    ablation_name: str,
    ablation_config: dict,
    full_config: dict,
) -> None:
    """Save experiment metadata for reproducibility."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "experiment_dir": str(exp_dir),
        "ablation_name": ablation_name,
        "ablation_config": ablation_config,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }

    with open(exp_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(full_config, f, default_flow_style=False)


def run_single_ablation(
    ablation_name: str,
    ablation_config: dict,
    data_config: dict,
    policy_config: dict,
    eval_config: dict,
    seeds: dict,
    device: str,
    full_config: dict,
    custom_name: str = None,
    verbose: bool = True,
):
    """Run a single ablation experiment."""
    print(f"\n{'='*70}")
    print(f"ABLATION: {ablation_name}")
    print(f"Description: {ablation_config.get('description', 'N/A')}")
    print(f"{'='*70}")

    # Create unique timestamped directory
    base_results_dir = ablation_config.get("results_dir", f"results/{ablation_name}")
    results_dir = create_experiment_dir(base_results_dir, ablation_name, custom_name)
    print(f"Results directory: {results_dir}")

    # Save metadata
    save_experiment_metadata(results_dir, ablation_name, ablation_config, full_config)

    rep_config = ablation_config["representation"]
    rep_training = rep_config.get("training", {})
    rep_arch = rep_config.get("architecture", {})

    # Load dataset
    data_dir = Path(data_config.get("data_dir", "data"))
    regimes = list(data_config.get("regimes", {}).keys())
    regime = regimes[0] if regimes else "tight"

    dataset_path = data_dir / f"dataset_{regime}.pkl"
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return None

    print(f"\nLoading dataset: {regime}")
    dataset = load_dataset(dataset_path)
    trajectories = dataset["trajectories"]
    actions = dataset["actions"]
    print(f"Loaded {len(trajectories)} trajectories")

    # Train encoder
    geometries = rep_config.get("geometries", ["hyperbolic"])
    embedding_dims = rep_config.get("embedding_dims", [2])

    all_results = {"encoders": {}, "policies": {}}

    for geometry, dim in product(geometries, embedding_dims):
        encoder_name = f"{regime}_{geometry}_dim{dim}"
        encoder_dir = results_dir / "representations" / encoder_name
        encoder_dir.mkdir(parents=True, exist_ok=True)
        encoder_path = encoder_dir / "encoder_final.pt"

        if encoder_path.exists():
            print(f"\nSkipping encoder {encoder_name} (already exists)")
            checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
            all_results["encoders"][encoder_name] = checkpoint.get("metrics", {})
            continue

        print(f"\nTraining encoder: {encoder_name}")
        print(f"  Temperature: {rep_training.get('temperature', 0.1)}")
        print(f"  N_negatives: {rep_training.get('n_negatives', 15)}")

        # Create encoder
        if geometry == "hyperbolic":
            encoder = HyperbolicIntervalEncoder(
                embedding_dim=dim,
                euc_width=128,
                hyp_width=128,
            ).to(device)
        else:
            encoder = EuclideanIntervalEncoder(
                embedding_dim=dim,
                hidden_sizes=rep_arch.get("hidden_sizes", [128, 128, 128, 128]),
            ).to(device)

        # Create datasets
        n_train = rep_training.get("n_train_samples", 20000)
        n_val = rep_training.get("n_val_samples", 1000)

        train_dataset = TrajectoryContrastiveDataset(
            trajectories=trajectories,
            num_samples=n_train,
            n_negatives=rep_training.get("n_negatives", 15),
            seed=seeds.get("representation_training", 123),
        )
        val_dataset = TrajectoryContrastiveDataset(
            trajectories=trajectories,
            num_samples=n_val,
            n_negatives=rep_training.get("n_negatives", 15),
            seed=seeds.get("representation_training", 123) + 1,
        )

        # Training config
        train_config = TrainingConfig(
            num_epochs=rep_training.get("epochs", 100),
            batch_size=rep_training.get("batch_size", 256),
            lr=rep_training.get("learning_rate", 0.001),
            temperature=rep_training.get("temperature", 0.1),
            early_stopping_patience=rep_training.get("early_stopping_patience", 20),
        )

        # Train
        result = train_encoder(
            encoder, train_dataset, val_dataset,
            config=train_config, device=device, verbose=verbose,
        )

        # Compute metrics
        metrics = compute_all_representation_metrics(encoder, trajectories, device=device)
        print(f"  Alignment: {metrics.temporal_geometric_alignment:.4f}")
        print(f"  Norm-Length Corr: {metrics.norm_length_correlation:.4f}")

        # Save
        metrics_dict = asdict(metrics)
        torch.save({
            "model_state_dict": encoder.state_dict(),
            "config": {
                "geometry": geometry,
                "embedding_dim": dim,
                "ablation": ablation_name,
                "temperature": rep_training.get("temperature", 0.1),
                "n_negatives": rep_training.get("n_negatives", 15),
            },
            "metrics": metrics_dict,
            "training": {
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
            },
        }, encoder_path)
        print(f"  Saved: {encoder_path}")

        all_results["encoders"][encoder_name] = metrics_dict

    # Train policies
    print(f"\n{'='*70}")
    print("POLICY TRAINING")
    print(f"{'='*70}")

    policy_training = policy_config.get("training", {})

    split = create_train_val_test_split(
        trajectories, actions,
        val_fraction=data_config.get("validation_fraction", 0.1),
        test_fraction=data_config.get("test_fraction", 0.1),
    )

    train_trajs, train_acts = split["train"]
    val_trajs, val_acts = split["val"]

    max_samples = policy_training.get("max_samples_per_trajectory", None)
    train_dataset = PolicyTrainingDataset(train_trajs, train_acts, max_samples_per_trajectory=max_samples)
    val_dataset = PolicyTrainingDataset(val_trajs, val_acts, max_samples_per_trajectory=max_samples)

    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    policy_dir = results_dir / "policies" / regime
    policy_dir.mkdir(parents=True, exist_ok=True)

    policy_train_config = TrainingConfig(
        num_epochs=policy_training.get("epochs", 50),
        batch_size=policy_training.get("batch_size", 4096),
        lr=policy_training.get("learning_rate", 0.001),
        early_stopping_patience=policy_training.get("early_stopping_patience", 10),
    )

    # Train GCBC-Single
    single_path = policy_dir / "gcbc_single.pt"
    if not single_path.exists():
        print(f"\nTraining GCBC-Single...")
        single_policy = GCBCSinglePolicy(
            n_actions=3,
            hidden_sizes=policy_config.get("architecture", {}).get("hidden_sizes", [64, 64]),
        ).to(device)

        result = train_policy(
            single_policy, train_dataset, val_dataset,
            config=policy_train_config, device=device, verbose=verbose,
        )

        torch.save({
            "model_state_dict": single_policy.state_dict(),
            "config": {"regime": regime, "method": "gcbc_single", "ablation": ablation_name},
            "training": {"best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss},
        }, single_path)
        print(f"  Saved: {single_path}")

    # Train GCBC-Interval for each encoder
    encoder_modes = policy_config.get("encoder_modes", ["frozen"])

    for geometry, dim in product(geometries, embedding_dims):
        encoder_name = f"{regime}_{geometry}_dim{dim}"
        encoder_path = results_dir / "representations" / encoder_name / "encoder_final.pt"

        if not encoder_path.exists():
            continue

        # Load encoder
        checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
        if geometry == "hyperbolic":
            encoder = HyperbolicIntervalEncoder(embedding_dim=dim).to(device)
        else:
            encoder = EuclideanIntervalEncoder(embedding_dim=dim).to(device)
        encoder.load_state_dict(checkpoint["model_state_dict"])

        for mode in encoder_modes:
            freeze = (mode == "frozen")
            policy_name = f"gcbc_interval_{encoder_name}_{mode}"
            policy_path = policy_dir / f"{policy_name}.pt"

            if policy_path.exists():
                print(f"\nSkipping {policy_name} (already exists)")
                continue

            print(f"\nTraining {policy_name}...")

            interval_policy = GCBCIntervalPolicy(
                encoder=encoder,
                n_actions=3,
                hidden_sizes=policy_config.get("architecture", {}).get("hidden_sizes", [64, 64]),
                freeze_encoder=freeze,
            ).to(device)

            result = train_policy(
                interval_policy, train_dataset, val_dataset,
                config=policy_train_config, device=device, verbose=verbose,
            )

            torch.save({
                "model_state_dict": interval_policy.state_dict(),
                "config": {
                    "regime": regime,
                    "method": "gcbc_interval",
                    "encoder": encoder_name,
                    "freeze_encoder": freeze,
                    "ablation": ablation_name,
                },
                "training": {"best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss},
            }, policy_path)
            print(f"  Saved: {policy_path}")

    # Save combined results
    all_results["experiment_dir"] = str(results_dir)
    all_results["ablation_name"] = ablation_name
    all_results["timestamp"] = datetime.now().isoformat()

    with open(results_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_dir}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to ablation config")
    parser.add_argument("--ablation", type=str, default=None, help="Run specific ablation only")
    parser.add_argument("--name", type=str, default=None, help="Custom name for experiment directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = args.device if args.device != "auto" else get_device()

    print("=" * 70)
    print("ABLATION EXPERIMENTS")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print("=" * 70)

    data_config = config.get("data", {})
    policy_config = config.get("policy", {})
    eval_config = config.get("evaluation", {})
    seeds = config.get("seeds", {})
    ablations = config.get("ablations", {})
    verbose = config.get("logging", {}).get("verbose", True)

    if not ablations:
        print("No ablations defined in config!")
        return

    # Run ablations
    ablation_names = [args.ablation] if args.ablation else list(ablations.keys())

    for ablation_name in ablation_names:
        if ablation_name not in ablations:
            print(f"Ablation '{ablation_name}' not found in config!")
            continue

        run_single_ablation(
            ablation_name=ablation_name,
            ablation_config=ablations[ablation_name],
            data_config=data_config,
            policy_config=policy_config,
            eval_config=eval_config,
            seeds=seeds,
            device=device,
            full_config=config,
            custom_name=args.name,
            verbose=verbose,
        )

    print(f"\n{'='*70}")
    print("ALL ABLATIONS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
