#!/usr/bin/env python
"""
Main experiment runner script.

This script orchestrates the full experimental pipeline:
1. Generate datasets for all data regimes
2. Train representation encoders for all configurations
3. Train policies using GCBC-Single and GCBC-Interval
4. Evaluate all policies
5. Generate visualizations and result tables
"""

import argparse
import yaml
import json
import torch
import numpy as np
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional

from src.core.data_generation import generate_all_regimes, load_dataset, load_all_regimes
from src.core.datasets import (
    TrajectoryContrastiveDataset,
    PolicyTrainingDataset,
    create_train_val_test_split,
)
from src.models import (
    EuclideanIntervalEncoder,
    HyperbolicIntervalEncoder,
    GCBCSinglePolicy,
    GCBCIntervalPolicy,
)
from src.training import train_encoder, train_policy, TrainingConfig
from src.evaluation import (
    compute_all_representation_metrics,
    evaluate_policy,
    run_full_evaluation,
    compare_policies,
    save_evaluation_results,
    print_comparison_table,
)
from src.visualization import (
    setup_plotting_style,
    plot_embedding_space_2d,
    plot_norm_vs_length,
    plot_angle_vs_midpoint,
    plot_learning_curves,
    plot_success_rate_comparison,
    plot_sample_trajectories,
    create_representation_table,
    create_policy_table,
    save_all_figures,
)


def load_experiment_config(config_path: str) -> Dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_experiments(
    config: Dict,
    output_dir: str = "results",
    device: str = None,
    verbose: bool = True,
) -> Dict:
    """
    Run full experimental pipeline.

    Args:
        config: Experiment configuration dictionary
        output_dir: Base output directory
        device: Compute device
        verbose: Print progress

    Returns:
        Dictionary of all results
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "representations": {},
        "policies": {},
        "evaluations": {},
    }

    # Extract configuration
    env_config = config.get("environment", {})
    data_config = config.get("data", {})
    rep_config = config.get("representation", {})
    policy_config = config.get("policy", {})
    eval_config = config.get("evaluation", {})
    exp_config = config.get("experiment", {})

    n_bins = env_config.get("n_bins", 100)

    # Determine which configurations to run
    if exp_config.get("run_all_combinations", False):
        data_regimes = list(data_config.get("regimes", {}).keys())
        geometries = rep_config.get("geometries", ["hyperbolic"])
        embedding_dims = rep_config.get("embedding_dims", [2])
        encoder_modes = policy_config.get("encoder_modes", ["frozen"])
    else:
        selected = exp_config.get("selected_configs", {})
        data_regimes = selected.get("data_regimes", ["moderate"])
        geometries = selected.get("geometries", ["hyperbolic"])
        embedding_dims = selected.get("embedding_dims", [2])
        encoder_modes = selected.get("encoder_modes", ["frozen"])

    # Phase 1: Data Generation
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1: DATA GENERATION")
        print("=" * 70)

    # Use custom data directory if specified in config, otherwise use output_path/data
    custom_data_dir = data_config.get("data_dir")
    if custom_data_dir:
        data_dir = Path(custom_data_dir)
        if verbose:
            print(f"Using custom data directory: {data_dir}")
    else:
        data_dir = output_path / "data"

    # Check if data already exists
    existing_data = True
    for regime in data_regimes:
        if not (data_dir / f"dataset_{regime}.pkl").exists():
            existing_data = False
            break

    if existing_data:
        if verbose:
            print("Loading existing datasets...")
        datasets = load_all_regimes(str(data_dir), data_regimes)
    else:
        regimes_to_generate = {
            name: cfg for name, cfg in data_config.get("regimes", {}).items()
            if name in data_regimes
        }

        datasets = generate_all_regimes(
            regimes=regimes_to_generate,
            n_trajectories=data_config.get("n_trajectories", 10000),
            n_bins=n_bins,
            base_seed=config.get("seeds", {}).get("data_generation", 42),
            output_dir=str(data_dir),
            verbose=verbose,
        )

    # Phase 2: Representation Learning
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2: REPRESENTATION LEARNING")
        print("=" * 70)

    rep_training = rep_config.get("training", {})

    for regime in data_regimes:
        trajectories = datasets[regime]["trajectories"]

        # Split data
        split = create_train_val_test_split(
            trajectories,
            actions=[[0] * (len(t) - 1) for t in trajectories],
            val_fraction=data_config.get("validation_fraction", 0.1),
            test_fraction=data_config.get("test_fraction", 0.1),
        )

        train_trajs, _ = split["train"]
        val_trajs, _ = split["val"]
        test_trajs, _ = split["test"]

        for geometry, dim in product(geometries, embedding_dims):
            config_name = f"{regime}_{geometry}_dim{dim}"
            save_dir = output_path / "representations" / config_name
            encoder_path = save_dir / "encoder_final.pt"

            # Skip if encoder already trained
            if encoder_path.exists():
                if verbose:
                    print(f"\nSkipping {config_name} (already trained)")
                # Load existing metrics for results
                checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)
                all_results["representations"][config_name] = {
                    "metrics": checkpoint.get("metrics", {}),
                    "training": checkpoint.get("training", {}),
                    "model_path": str(encoder_path),
                }
                continue

            if verbose:
                print(f"\nTraining: {config_name}")

            # Create encoder
            if geometry == "euclidean":
                encoder = EuclideanIntervalEncoder(
                    embedding_dim=dim,
                    hidden_sizes=rep_config.get("architecture", {}).get("hidden_sizes", [128, 128, 128, 128]),
                )
            else:
                encoder = HyperbolicIntervalEncoder(
                    embedding_dim=dim,
                    c=1.0,
                    euc_width=128,
                    hyp_width=128,
                )

            encoder = encoder.to(device)

            # Create datasets
            n_negatives = rep_training.get("n_negatives", 255)

            train_dataset = TrajectoryContrastiveDataset(
                trajectories=train_trajs,
                num_samples=len(train_trajs) * 5,
                n_negatives=n_negatives,
                seed=config.get("seeds", {}).get("representation_training", 123),
            )

            val_dataset = TrajectoryContrastiveDataset(
                trajectories=val_trajs,
                num_samples=len(val_trajs) * 2,
                n_negatives=n_negatives,
                seed=config.get("seeds", {}).get("representation_training", 123) + 1,
            )

            # Training config
            training_config = TrainingConfig(
                num_epochs=rep_training.get("epochs", 100),
                batch_size=rep_training.get("batch_size", 256),
                lr=rep_training.get("learning_rate", 0.001),
                temperature=rep_training.get("temperature", 0.1),
                early_stopping_patience=rep_training.get("early_stopping_patience", 20),
            )

            # Train
            save_dir.mkdir(parents=True, exist_ok=True)
            result = train_encoder(
                encoder, train_dataset, val_dataset,
                config=training_config,
                device=device,
                verbose=verbose,
                save_dir=str(save_dir),
            )

            # Evaluate representation quality
            encoder.eval()
            metrics = compute_all_representation_metrics(
                encoder, test_trajs, n_samples=5000, device=device
            )

            # Save encoder with metrics
            torch.save({
                "model_state_dict": encoder.state_dict(),
                "config": {
                    "geometry": geometry,
                    "embedding_dim": dim,
                    "regime": regime,
                },
                "metrics": {
                    "temporal_geometric_alignment": metrics.temporal_geometric_alignment,
                    "norm_length_correlation": metrics.norm_length_correlation,
                    "angle_midpoint_correlation": metrics.angle_midpoint_correlation,
                },
                "training": {
                    "best_epoch": result.best_epoch,
                    "best_val_loss": result.best_val_loss,
                },
            }, save_dir / "encoder_final.pt")

            all_results["representations"][config_name] = {
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
                "model_path": str(save_dir / "encoder_final.pt"),
            }

            if verbose:
                print(f"  Alignment: {metrics.temporal_geometric_alignment:.4f}")
                print(f"  Norm-Length Corr: {metrics.norm_length_correlation:.4f}")

    # Phase 3: Policy Learning
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 3: POLICY LEARNING")
        print("=" * 70)

    policy_training = policy_config.get("training", {})

    for regime in data_regimes:
        trajectories = datasets[regime]["trajectories"]
        actions = datasets[regime]["actions"]

        split = create_train_val_test_split(
            trajectories, actions,
            val_fraction=data_config.get("validation_fraction", 0.1),
            test_fraction=data_config.get("test_fraction", 0.1),
        )

        train_trajs, train_acts = split["train"]
        val_trajs, val_acts = split["val"]

        max_samples_per_traj = policy_training.get("max_samples_per_trajectory", None)
        train_dataset = PolicyTrainingDataset(
            train_trajs, train_acts, max_samples_per_trajectory=max_samples_per_traj
        )
        val_dataset = PolicyTrainingDataset(
            val_trajs, val_acts, max_samples_per_trajectory=max_samples_per_traj
        )

        policy_save_dir = output_path / "policies" / regime
        policy_save_dir.mkdir(parents=True, exist_ok=True)

        # Policy training config (used for both GCBC-Single and GCBC-Interval)
        policy_train_config = TrainingConfig(
            num_epochs=policy_training.get("epochs", 50),
            batch_size=policy_training.get("batch_size", 256),
            lr=policy_training.get("learning_rate", 0.001),
            early_stopping_patience=policy_training.get("early_stopping_patience", 10),
        )

        single_policy_path = policy_save_dir / "gcbc_single.pt"

        # Train GCBC-Single (skip if exists)
        if single_policy_path.exists():
            if verbose:
                print(f"\nSkipping GCBC-Single for {regime} (already trained)")
            checkpoint = torch.load(single_policy_path, map_location=device, weights_only=False)
            all_results["policies"][f"{regime}_gcbc_single"] = {
                "training": checkpoint.get("training", {}),
                "model_path": str(single_policy_path),
            }
        else:
            if verbose:
                print(f"\nTraining GCBC-Single for {regime}...")

            single_policy = GCBCSinglePolicy(
                n_actions=3,
                hidden_sizes=policy_config.get("architecture", {}).get("hidden_sizes", [64, 64]),
            ).to(device)

            single_result = train_policy(
                single_policy, train_dataset, val_dataset,
                config=policy_train_config, device=device, verbose=verbose,
            )

            torch.save({
                "model_state_dict": single_policy.state_dict(),
                "config": {"regime": regime, "method": "gcbc_single"},
                "training": {
                    "best_epoch": single_result.best_epoch,
                    "best_val_loss": single_result.best_val_loss,
                },
            }, single_policy_path)

            all_results["policies"][f"{regime}_gcbc_single"] = {
                "training": {
                    "best_epoch": single_result.best_epoch,
                    "best_val_loss": single_result.best_val_loss,
                },
                "model_path": str(single_policy_path),
            }

        # Train GCBC-Interval for each encoder configuration
        for geometry, dim in product(geometries, embedding_dims):
            encoder_config_name = f"{regime}_{geometry}_dim{dim}"
            encoder_path = output_path / "representations" / encoder_config_name / "encoder_final.pt"

            if not encoder_path.exists():
                continue

            for mode in encoder_modes:
                freeze_encoder = (mode == "frozen")
                policy_name = f"{regime}_{geometry}_dim{dim}_{mode}"
                policy_path = policy_save_dir / f"gcbc_interval_{policy_name}.pt"

                # Skip if already trained
                if policy_path.exists():
                    if verbose:
                        print(f"\nSkipping GCBC-Interval: {policy_name} (already trained)")
                    checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
                    all_results["policies"][policy_name] = {
                        "training": checkpoint.get("training", {}),
                        "model_path": str(policy_path),
                    }
                    continue

                if verbose:
                    print(f"\nTraining GCBC-Interval: {policy_name}")

                # Load encoder
                checkpoint = torch.load(encoder_path, map_location=device, weights_only=False)

                if geometry == "euclidean":
                    encoder = EuclideanIntervalEncoder(embedding_dim=dim)
                else:
                    encoder = HyperbolicIntervalEncoder(embedding_dim=dim)

                encoder.load_state_dict(checkpoint["model_state_dict"])
                encoder = encoder.to(device)

                # Create policy
                interval_policy = GCBCIntervalPolicy(
                    encoder=encoder,
                    n_actions=3,
                    hidden_sizes=policy_config.get("architecture", {}).get("hidden_sizes", [64, 64]),
                    freeze_encoder=freeze_encoder,
                ).to(device)

                interval_result = train_policy(
                    interval_policy, train_dataset, val_dataset,
                    config=policy_train_config, device=device, verbose=verbose,
                )

                torch.save({
                    "model_state_dict": interval_policy.state_dict(),
                    "config": {
                        "regime": regime,
                        "method": "gcbc_interval",
                        "geometry": geometry,
                        "embedding_dim": dim,
                        "encoder_mode": mode,
                    },
                    "training": {
                        "best_epoch": interval_result.best_epoch,
                        "best_val_loss": interval_result.best_val_loss,
                    },
                }, policy_path)

                all_results["policies"][policy_name] = {
                    "training": {
                        "best_epoch": interval_result.best_epoch,
                        "best_val_loss": interval_result.best_val_loss,
                    },
                    "model_path": str(policy_path),
                }

    # Phase 4: Evaluation
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 4: EVALUATION")
        print("=" * 70)

    for policy_name, policy_info in all_results["policies"].items():
        if verbose:
            print(f"\nEvaluating: {policy_name}")

        # Load policy
        checkpoint = torch.load(policy_info["model_path"], map_location=device, weights_only=False)
        policy_config_info = checkpoint["config"]

        if policy_config_info.get("method") == "gcbc_single":
            policy = GCBCSinglePolicy(n_actions=3)
            policy.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Need to reconstruct the encoder
            geometry = policy_config_info.get("geometry", "hyperbolic")
            dim = policy_config_info.get("embedding_dim", 2)

            if geometry == "euclidean":
                encoder = EuclideanIntervalEncoder(embedding_dim=dim)
            else:
                encoder = HyperbolicIntervalEncoder(embedding_dim=dim)

            policy = GCBCIntervalPolicy(
                encoder=encoder,
                n_actions=3,
                freeze_encoder=True,
            )
            policy.load_state_dict(checkpoint["model_state_dict"])

        policy = policy.to(device)

        # Evaluate
        eval_result = evaluate_policy(
            policy,
            n_episodes=eval_config.get("n_episodes", 100),
            n_bins=n_bins,
            max_steps=env_config.get("max_eval_steps", 200),
            deterministic=eval_config.get("deterministic_policy", True),
            device=device,
        )

        all_results["evaluations"][policy_name] = {
            "success_rate": eval_result.metrics.success_rate,
            "mean_steps": eval_result.metrics.mean_steps,
            "median_steps": eval_result.metrics.median_steps,
            "path_efficiency": eval_result.metrics.path_efficiency,
            "action_entropy": eval_result.metrics.action_entropy,
        }

        if verbose:
            print(f"  Success: {eval_result.metrics.success_rate:.2%}")
            print(f"  Steps: {eval_result.metrics.mean_steps:.1f}")

    # Save all results
    results_path = output_path / "experiment_results.json"

    def convert_for_json(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if np.isfinite(obj) else str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)

    if verbose:
        print(f"\nSaved results to: {results_path}")

    # Print summary tables
    if verbose:
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        print("\nRepresentation Quality:")
        rep_summary = {name: info["metrics"] for name, info in all_results["representations"].items()}
        print(create_representation_table(rep_summary))

        print("\nPolicy Performance:")
        print(create_policy_table(all_results["evaluations"]))

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run full experimental pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment_config.yaml",
        help="Path to experiment configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for all results",
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
        help="Run quick version with reduced parameters",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_experiment_config(str(config_path))
    else:
        print(f"Warning: Config file not found at {config_path}, using defaults")
        config = {}

    # Override for quick mode
    if args.quick:
        config.setdefault("representation", {}).setdefault("training", {})["epochs"] = 10
        config.setdefault("policy", {}).setdefault("training", {})["epochs"] = 10
        config.setdefault("data", {})["n_trajectories"] = 1000
        config.setdefault("evaluation", {})["n_episodes"] = 20

    print("=" * 70)
    print("INTERVAL REPRESENTATIONS FOR GOAL-CONDITIONED PLANNING")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device or 'auto'}")
    print("=" * 70)

    results = run_experiments(
        config=config,
        output_dir=args.output_dir,
        device=args.device,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
