#!/usr/bin/env python
"""Script to evaluate trained policies."""

import argparse
import torch
import json
from pathlib import Path

from src.models import (
    GCBCSinglePolicy,
    GCBCIntervalPolicy,
    EuclideanIntervalEncoder,
    HyperbolicIntervalEncoder,
)
from src.evaluation import (
    run_full_evaluation,
    evaluate_generalization,
    compute_generalization_summary,
    save_evaluation_results,
)


def load_policy(policy_path: str, device: str) -> torch.nn.Module:
    """Load a policy from checkpoint."""
    checkpoint = torch.load(policy_path, map_location=device)
    config = checkpoint.get("config", {})

    method = config.get("method", "gcbc_single")

    if method == "gcbc_single":
        policy = GCBCSinglePolicy(
            n_actions=3,
            hidden_sizes=config.get("hidden_sizes", [64, 64]),
        )
        policy.load_state_dict(checkpoint["model_state_dict"])
    else:
        # GCBC-Interval - need to reconstruct encoder
        geometry = config.get("geometry", config.get("encoder_type", "hyperbolic"))
        dim = config.get("embedding_dim", 2)

        if geometry == "euclidean":
            encoder = EuclideanIntervalEncoder(embedding_dim=dim)
        else:
            encoder = HyperbolicIntervalEncoder(embedding_dim=dim)

        policy = GCBCIntervalPolicy(
            encoder=encoder,
            n_actions=3,
            hidden_sizes=config.get("hidden_sizes", [64, 64]),
            freeze_encoder=True,
        )
        policy.load_state_dict(checkpoint["model_state_dict"])

    return policy.to(device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policies")
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to trained policy checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluations",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=100,
        help="Number of discrete bins",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--generalization",
        action="store_true",
        help="Run generalization tests",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action selection",
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

    # Load policy
    print(f"Loading policy from: {args.policy_path}")
    policy = load_policy(args.policy_path, device)
    print("Policy loaded successfully")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run full evaluation
    print("\n" + "=" * 60)
    print("Running Full Evaluation")
    print("=" * 60)

    results = run_full_evaluation(
        policy,
        n_bins=args.n_bins,
        max_steps=args.max_steps,
        n_episodes=args.n_episodes,
        deterministic=not args.stochastic,
        device=device,
        verbose=True,
    )

    # Save results
    policy_name = Path(args.policy_path).stem
    results_path = output_dir / f"{policy_name}_evaluation.json"
    save_evaluation_results(results, str(results_path))
    print(f"\nSaved results to: {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    std = results.get("standard", {})
    print(f"\nStandard Task (0 -> 1):")
    print(f"  Success Rate: {std.get('success_rate', 0):.2%}")
    print(f"  Mean Steps: {std.get('mean_steps', 'N/A')}")
    print(f"  Path Efficiency: {std.get('path_efficiency', 0):.4f}")

    for test_type in ["start_states", "goal_states", "random_pairs"]:
        key = f"generalization_{test_type}"
        if key in results:
            gen = results[key]
            print(f"\nGeneralization ({test_type}):")
            print(f"  Mean Success: {gen.get('mean_success_rate', 0):.2%} ± {gen.get('std_success_rate', 0):.2%}")
            print(f"  Range: [{gen.get('min_success_rate', 0):.2%}, {gen.get('max_success_rate', 0):.2%}]")


if __name__ == "__main__":
    main()
