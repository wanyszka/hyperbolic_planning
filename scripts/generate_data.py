#!/usr/bin/env python
"""Script to generate trajectory datasets for all data regimes."""

import argparse
from pathlib import Path

from src.core.data_generation import generate_all_regimes, DEFAULT_REGIMES, load_config


def main():
    parser = argparse.ArgumentParser(description="Generate trajectory datasets")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=10000,
        help="Number of trajectories per regime",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=100,
        help="Number of discrete bins",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=None,
        help="Specific regimes to generate (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Determine regimes to generate
    if args.config:
        config = load_config(args.config)
        regimes = {}
        for name, cfg in config.get("data", {}).get("regimes", {}).items():
            regimes[name] = {"max_length": cfg["max_length"]}
        n_trajectories = config.get("data", {}).get("n_trajectories", args.n_trajectories)
        n_bins = config.get("environment", {}).get("n_bins", args.n_bins)
        seed = config.get("seeds", {}).get("data_generation", args.seed)
    else:
        regimes = DEFAULT_REGIMES
        if args.regimes:
            regimes = {k: v for k, v in regimes.items() if k in args.regimes}
        n_trajectories = args.n_trajectories
        n_bins = args.n_bins
        seed = args.seed

    print("=" * 60)
    print("Dataset Generation")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Trajectories per regime: {n_trajectories}")
    print(f"Number of bins: {n_bins}")
    print(f"Regimes: {list(regimes.keys())}")
    print("=" * 60)

    # Generate datasets
    datasets = generate_all_regimes(
        regimes=regimes,
        n_trajectories=n_trajectories,
        n_bins=n_bins,
        base_seed=seed,
        output_dir=args.output_dir,
        verbose=True,
    )

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
