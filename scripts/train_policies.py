#!/usr/bin/env python
"""Script to train goal-conditioned policies."""

import argparse
import torch
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.core.data_generation import load_dataset
from src.core.datasets import (
    PolicyTrainingDataset,
    create_train_val_test_split,
)
from src.models import (
    GCBCSinglePolicy,
    GCBCIntervalPolicy,
    EuclideanIntervalEncoder,
    HyperbolicIntervalEncoder,
)
from src.training import train_policy, TrainingConfig
from src.evaluation import evaluate_policy, run_full_evaluation


def train_gcbc_single(
    trajectories: List[List[float]],
    actions: List[List[int]],
    config: Dict,
    device: str,
    save_dir: str,
    verbose: bool = True,
) -> Dict:
    """Train GCBC-Single policy."""

    # Split data
    split = create_train_val_test_split(
        trajectories, actions,
        val_fraction=config.get("val_fraction", 0.1),
        test_fraction=config.get("test_fraction", 0.1),
    )

    train_trajs, train_acts = split["train"]
    val_trajs, val_acts = split["val"]

    # Create datasets
    train_dataset = PolicyTrainingDataset(train_trajs, train_acts, goal_type="final")
    val_dataset = PolicyTrainingDataset(val_trajs, val_acts, goal_type="final")

    # Create policy
    hidden_sizes = config.get("hidden_sizes", [64, 64])
    policy = GCBCSinglePolicy(n_actions=3, hidden_sizes=hidden_sizes).to(device)

    # Training config
    training_config = TrainingConfig(
        num_epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 256),
        lr=config.get("learning_rate", 0.001),
        lr_min=config.get("lr_min", 1e-5),
        early_stopping_patience=config.get("early_stopping_patience", 10),
        print_every=config.get("print_every", 5),
    )

    if verbose:
        print("\nTraining GCBC-Single policy...")

    result = train_policy(
        policy, train_dataset, val_dataset,
        config=training_config, device=device, verbose=verbose,
    )

    # Save model
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = save_path / "policy_gcbc_single.pt"
    torch.save({
        "model_state_dict": policy.state_dict(),
        "config": {"hidden_sizes": hidden_sizes},
        "training": {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
        },
    }, model_path)

    if verbose:
        print(f"\nSaved policy to: {model_path}")

    return {
        "policy": policy,
        "model_path": str(model_path),
        "training": {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
        },
    }


def train_gcbc_interval(
    trajectories: List[List[float]],
    actions: List[List[int]],
    encoder_path: str,
    encoder_type: str,
    embedding_dim: int,
    freeze_encoder: bool,
    config: Dict,
    device: str,
    save_dir: str,
    verbose: bool = True,
) -> Dict:
    """Train GCBC-Interval policy."""

    # Load encoder
    checkpoint = torch.load(encoder_path, map_location=device)

    if encoder_type == "euclidean":
        encoder = EuclideanIntervalEncoder(
            embedding_dim=embedding_dim,
            hidden_sizes=checkpoint["config"].get("hidden_sizes", [128, 128, 128, 128]),
        )
    else:
        encoder = HyperbolicIntervalEncoder(
            embedding_dim=embedding_dim,
            c=checkpoint["config"].get("curvature", 1.0),
        )

    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder = encoder.to(device)

    if verbose:
        print(f"Loaded encoder from: {encoder_path}")

    # Split data
    split = create_train_val_test_split(
        trajectories, actions,
        val_fraction=config.get("val_fraction", 0.1),
        test_fraction=config.get("test_fraction", 0.1),
    )

    train_trajs, train_acts = split["train"]
    val_trajs, val_acts = split["val"]

    # Create datasets
    train_dataset = PolicyTrainingDataset(train_trajs, train_acts, goal_type="final")
    val_dataset = PolicyTrainingDataset(val_trajs, val_acts, goal_type="final")

    # Create policy
    hidden_sizes = config.get("hidden_sizes", [64, 64])
    policy = GCBCIntervalPolicy(
        encoder=encoder,
        n_actions=3,
        hidden_sizes=hidden_sizes,
        freeze_encoder=freeze_encoder,
    ).to(device)

    # Training config
    training_config = TrainingConfig(
        num_epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 256),
        lr=config.get("learning_rate", 0.001),
        lr_min=config.get("lr_min", 1e-5),
        early_stopping_patience=config.get("early_stopping_patience", 10),
        print_every=config.get("print_every", 5),
    )

    freeze_str = "frozen" if freeze_encoder else "finetuned"
    if verbose:
        print(f"\nTraining GCBC-Interval policy ({freeze_str} encoder)...")

    result = train_policy(
        policy, train_dataset, val_dataset,
        config=training_config, device=device, verbose=verbose,
    )

    # Save model
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = save_path / f"policy_gcbc_interval_{freeze_str}.pt"
    torch.save({
        "model_state_dict": policy.state_dict(),
        "config": {
            "hidden_sizes": hidden_sizes,
            "freeze_encoder": freeze_encoder,
            "encoder_path": encoder_path,
            "encoder_type": encoder_type,
            "embedding_dim": embedding_dim,
        },
        "training": {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
        },
    }, model_path)

    if verbose:
        print(f"\nSaved policy to: {model_path}")

    return {
        "policy": policy,
        "model_path": str(model_path),
        "training": {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Train goal-conditioned policies")
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
        default="results/policies",
        help="Output directory for trained policies",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="moderate",
        help="Data regime to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="both",
        choices=["single", "interval", "both"],
        help="Policy method to train",
    )
    parser.add_argument(
        "--encoder-path",
        type=str,
        default=None,
        help="Path to trained encoder (required for interval method)",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="hyperbolic",
        choices=["euclidean", "hyperbolic"],
        help="Encoder geometry type",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=2,
        help="Encoder embedding dimension",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder during policy training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        policy_config = config.get("policy", {}).get("training", {})
    else:
        policy_config = {
            "epochs": args.epochs,
            "batch_size": 256,
            "learning_rate": 0.001,
            "hidden_sizes": [64, 64],
            "early_stopping_patience": 10,
        }

    # Load dataset
    dataset_path = Path(args.data_dir) / f"dataset_{args.regime}.pkl"
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(str(dataset_path))
    trajectories = dataset["trajectories"]
    actions = dataset["actions"]
    print(f"Loaded {len(trajectories)} trajectories")

    # Output directory
    save_dir = Path(args.output_dir) / args.regime
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Train GCBC-Single
    if args.method in ["single", "both"]:
        print("\n" + "=" * 60)
        print("Training GCBC-Single")
        print("=" * 60)

        single_result = train_gcbc_single(
            trajectories, actions,
            config=policy_config,
            device=device,
            save_dir=str(save_dir),
            verbose=True,
        )
        results["gcbc_single"] = single_result

        if args.evaluate:
            print("\nEvaluating GCBC-Single...")
            eval_result = evaluate_policy(
                single_result["policy"],
                n_episodes=100,
                device=device,
                verbose=True,
            )
            results["gcbc_single"]["evaluation"] = {
                "success_rate": eval_result.metrics.success_rate,
                "mean_steps": eval_result.metrics.mean_steps,
                "path_efficiency": eval_result.metrics.path_efficiency,
            }

    # Train GCBC-Interval
    if args.method in ["interval", "both"]:
        if args.encoder_path is None and args.method == "interval":
            print("Error: --encoder-path required for interval method")
            return

        if args.encoder_path:
            print("\n" + "=" * 60)
            print("Training GCBC-Interval")
            print("=" * 60)

            interval_result = train_gcbc_interval(
                trajectories, actions,
                encoder_path=args.encoder_path,
                encoder_type=args.encoder_type,
                embedding_dim=args.embedding_dim,
                freeze_encoder=args.freeze_encoder,
                config=policy_config,
                device=device,
                save_dir=str(save_dir),
                verbose=True,
            )

            freeze_str = "frozen" if args.freeze_encoder else "finetuned"
            results[f"gcbc_interval_{freeze_str}"] = interval_result

            if args.evaluate:
                print(f"\nEvaluating GCBC-Interval ({freeze_str})...")
                eval_result = evaluate_policy(
                    interval_result["policy"],
                    n_episodes=100,
                    device=device,
                    verbose=True,
                )
                results[f"gcbc_interval_{freeze_str}"]["evaluation"] = {
                    "success_rate": eval_result.metrics.success_rate,
                    "mean_steps": eval_result.metrics.mean_steps,
                    "path_efficiency": eval_result.metrics.path_efficiency,
                }

    # Save summary
    summary_path = save_dir / "training_summary.json"
    summary = {}
    for name, res in results.items():
        summary[name] = {
            "model_path": res.get("model_path"),
            "training": res.get("training"),
            "evaluation": res.get("evaluation"),
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: str(x) if not isinstance(x, (int, float, str, bool, type(None), list, dict)) else x)

    print(f"\nSaved summary to: {summary_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Print evaluation summary
    if args.evaluate:
        print("\nEvaluation Summary:")
        print("-" * 40)
        for name, res in results.items():
            if "evaluation" in res:
                eval_res = res["evaluation"]
                print(f"{name}:")
                print(f"  Success rate: {eval_res['success_rate']:.2%}")
                print(f"  Mean steps: {eval_res['mean_steps']:.1f}")
                print(f"  Path efficiency: {eval_res['path_efficiency']:.4f}")


if __name__ == "__main__":
    main()
