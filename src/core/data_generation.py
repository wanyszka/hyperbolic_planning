"""Data generation module for creating trajectory datasets across different regimes."""

import pickle
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .environment import generate_dataset, EnvConfig


# Default regime configurations
DEFAULT_REGIMES = {
    "tight": {
        "max_length": 120,
        "description": "Nearly monotonic paths (slack factor 12)",
    },
    "moderate": {
        "max_length": 150,
        "description": "Some backtracking (slack factor 15)",
    },
    "loose": {
        "max_length": 200,
        "description": "Significant wandering (slack factor 20)",
    },
    "very_loose": {
        "max_length": 300,
        "description": "Heavy backtracking (slack factor 30)",
    },
}


def generate_regime_dataset(
    regime_name: str,
    max_length: int,
    n_trajectories: int = 10000,
    n_bins: int = 100,
    start: float = 0.0,
    goal: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Generate a dataset for a specific data regime.

    Args:
        regime_name: Name of the regime (e.g., "tight", "moderate")
        max_length: Maximum trajectory length for this regime
        n_trajectories: Number of successful trajectories
        n_bins: Number of discrete bins
        start: Starting state
        goal: Goal state
        seed: Random seed
        verbose: Print progress

    Returns:
        Dataset dictionary with trajectories, actions, and metadata
    """
    if verbose:
        slack_factor = max_length / n_bins
        print(f"\n{'='*60}")
        print(f"Generating '{regime_name}' dataset")
        print(f"  Max length: {max_length} (slack factor: {slack_factor:.1f})")
        print(f"{'='*60}")

    dataset = generate_dataset(
        n_trajectories=n_trajectories,
        max_length=max_length,
        n_bins=n_bins,
        start=start,
        goal=goal,
        seed=seed,
        verbose=verbose,
    )

    # Add regime-specific metadata
    dataset["metadata"]["regime_name"] = regime_name

    return dataset


def generate_all_regimes(
    regimes: Optional[Dict] = None,
    n_trajectories: int = 10000,
    n_bins: int = 10,
    start: float = 0.0,
    goal: float = 1.0,
    base_seed: int = 42,
    output_dir: Optional[str] = None,
    save_format: str = "pickle",
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Generate datasets for all data regimes.

    Args:
        regimes: Dictionary of regime configurations (uses DEFAULT_REGIMES if None)
        n_trajectories: Number of trajectories per regime
        n_bins: Number of discrete bins
        start: Starting state
        goal: Goal state
        base_seed: Base random seed (each regime gets base_seed + regime_index)
        output_dir: Directory to save datasets (None = don't save)
        save_format: "pickle" or "json"
        verbose: Print progress

    Returns:
        Dictionary mapping regime names to datasets
    """
    if regimes is None:
        regimes = DEFAULT_REGIMES

    datasets = {}

    for i, (regime_name, regime_config) in enumerate(regimes.items()):
        seed = base_seed + i * 1000  # Different seed for each regime

        dataset = generate_regime_dataset(
            regime_name=regime_name,
            max_length=regime_config["max_length"],
            n_trajectories=n_trajectories,
            n_bins=n_bins,
            start=start,
            goal=goal,
            seed=seed,
            verbose=verbose,
        )

        datasets[regime_name] = dataset

        if output_dir:
            save_dataset(dataset, regime_name, output_dir, save_format, verbose)

    if verbose:
        print_regime_summary(datasets)

    return datasets


def save_dataset(
    dataset: dict,
    regime_name: str,
    output_dir: str,
    save_format: str = "pickle",
    verbose: bool = True,
) -> str:
    """
    Save a dataset to disk.

    Args:
        dataset: Dataset dictionary
        regime_name: Name of the regime
        output_dir: Output directory
        save_format: "pickle" or "json"
        verbose: Print confirmation

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"dataset_{regime_name}.{save_format.replace('pickle', 'pkl')}"
    filepath = output_path / filename

    if save_format == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(dataset, f)
    elif save_format == "json":
        # Convert numpy types to Python types for JSON serialization
        json_dataset = _convert_for_json(dataset)
        with open(filepath, "w") as f:
            json.dump(json_dataset, f, indent=2)
    else:
        raise ValueError(f"Unknown save format: {save_format}")

    if verbose:
        print(f"  Saved: {filepath}")

    return str(filepath)


def load_dataset(filepath: str) -> dict:
    """
    Load a dataset from disk.

    Args:
        filepath: Path to dataset file

    Returns:
        Dataset dictionary
    """
    filepath = Path(filepath)

    if filepath.suffix == ".pkl":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif filepath.suffix == ".json":
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown file format: {filepath.suffix}")


def load_all_regimes(data_dir: str, regimes: Optional[List[str]] = None) -> Dict[str, dict]:
    """
    Load all regime datasets from a directory.

    Args:
        data_dir: Directory containing dataset files
        regimes: List of regime names to load (None = load all found)

    Returns:
        Dictionary mapping regime names to datasets
    """
    data_path = Path(data_dir)
    datasets = {}

    if regimes is None:
        # Find all dataset files
        files = list(data_path.glob("dataset_*.pkl")) + list(data_path.glob("dataset_*.json"))
        regimes = [f.stem.replace("dataset_", "") for f in files]

    for regime in regimes:
        # Try pickle first, then json
        pkl_path = data_path / f"dataset_{regime}.pkl"
        json_path = data_path / f"dataset_{regime}.json"

        if pkl_path.exists():
            datasets[regime] = load_dataset(pkl_path)
        elif json_path.exists():
            datasets[regime] = load_dataset(json_path)
        else:
            print(f"Warning: Dataset for regime '{regime}' not found")

    return datasets


def _convert_for_json(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def print_regime_summary(datasets: Dict[str, dict]) -> None:
    """Print summary statistics for all regimes."""
    print(f"\n{'='*80}")
    print("DATASET SUMMARY")
    print(f"{'='*80}")

    headers = ["Regime", "N Traj", "Max Len", "Slack", "Success%", "Len Mean", "Len Std", "Len Range"]
    print(f"{headers[0]:<12} {headers[1]:>8} {headers[2]:>8} {headers[3]:>6} {headers[4]:>9} {headers[5]:>9} {headers[6]:>8} {headers[7]:>12}")
    print("-" * 80)

    for regime_name, dataset in datasets.items():
        meta = dataset["metadata"]
        print(
            f"{regime_name:<12} "
            f"{meta['n_trajectories']:>8} "
            f"{meta['max_length']:>8} "
            f"{meta['slack_factor']:>6.1f} "
            f"{meta['success_rate']*100:>8.1f}% "
            f"{meta['length_mean']:>9.1f} "
            f"{meta['length_std']:>8.1f} "
            f"[{meta['length_min']:>3}, {meta['length_max']:>3}]"
        )


def analyze_trajectory_characteristics(dataset: dict) -> dict:
    """
    Analyze characteristics of trajectories in a dataset.

    Args:
        dataset: Dataset dictionary

    Returns:
        Dictionary of analysis results
    """
    trajectories = dataset["trajectories"]
    n_bins = dataset["metadata"]["n_bins"]

    # Compute per-trajectory statistics
    monotonicity_scores = []
    backtrack_counts = []
    progress_rates = []

    for traj in trajectories:
        traj = np.array(traj)

        # Monotonicity: fraction of steps that move toward goal
        diffs = np.diff(traj)
        goal_direction = 1.0  # Assuming goal is 1.0
        monotonic_steps = np.sum(diffs * goal_direction > 0)
        monotonicity = monotonic_steps / len(diffs) if len(diffs) > 0 else 1.0
        monotonicity_scores.append(monotonicity)

        # Backtrack count: number of times direction changes
        if len(diffs) > 1:
            signs = np.sign(diffs)
            sign_changes = np.sum(signs[1:] != signs[:-1])
            backtrack_counts.append(sign_changes)
        else:
            backtrack_counts.append(0)

        # Progress rate: (final - start) / steps
        if len(traj) > 1:
            progress = (traj[-1] - traj[0]) / (len(traj) - 1)
            progress_rates.append(progress)
        else:
            progress_rates.append(0)

    return {
        "monotonicity_mean": float(np.mean(monotonicity_scores)),
        "monotonicity_std": float(np.std(monotonicity_scores)),
        "backtrack_count_mean": float(np.mean(backtrack_counts)),
        "backtrack_count_std": float(np.std(backtrack_counts)),
        "progress_rate_mean": float(np.mean(progress_rates)),
        "progress_rate_std": float(np.std(progress_rates)),
        "optimal_length": n_bins,  # Minimum possible length
        "efficiency_mean": n_bins / dataset["metadata"]["length_mean"],
    }


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_from_config(config_path: str, output_dir: Optional[str] = None) -> Dict[str, dict]:
    """
    Generate datasets based on a configuration file.

    Args:
        config_path: Path to YAML config file
        output_dir: Override output directory (uses config if None)

    Returns:
        Dictionary of datasets
    """
    config = load_config(config_path)

    data_config = config.get("data", {})
    env_config = config.get("environment", {})

    # Build regimes from config
    regimes = {}
    for regime_name, regime_cfg in data_config.get("regimes", {}).items():
        regimes[regime_name] = {
            "max_length": regime_cfg["max_length"],
            "description": regime_cfg.get("description", ""),
        }

    if not regimes:
        regimes = DEFAULT_REGIMES

    # Determine output directory
    if output_dir is None:
        output_dir = config.get("experiment", {}).get("results_dir", "data")
        output_dir = Path(output_dir) / "datasets"

    return generate_all_regimes(
        regimes=regimes,
        n_trajectories=data_config.get("n_trajectories", 10000),
        n_bins=env_config.get("n_bins", 10),
        start=0.0,
        goal=1.0,
        base_seed=config.get("seeds", {}).get("data_generation", 42),
        output_dir=str(output_dir),
        save_format="pickle",
        verbose=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate trajectory datasets")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (optional)",
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
        "--regimes",
        nargs="+",
        default=None,
        help="Specific regimes to generate (default: all)",
    )

    args = parser.parse_args()

    if args.config:
        datasets = generate_from_config(args.config, args.output_dir)
    else:
        # Use specific regimes if provided, otherwise all
        regimes = DEFAULT_REGIMES
        if args.regimes:
            regimes = {k: v for k, v in DEFAULT_REGIMES.items() if k in args.regimes}

        datasets = generate_all_regimes(
            n_bins=10,
            regimes=regimes,
            n_trajectories=args.n_trajectories,
            output_dir=args.output_dir,
            verbose=True,
        )

    # Print analysis for each regime
    print("\n" + "="*80)
    print("TRAJECTORY CHARACTERISTICS ANALYSIS")
    print("="*80)

    for regime_name, dataset in datasets.items():
        analysis = analyze_trajectory_characteristics(dataset)
        print(f"\n{regime_name.upper()}:")
        print(f"  Monotonicity: {analysis['monotonicity_mean']:.3f} ± {analysis['monotonicity_std']:.3f}")
        print(f"  Backtracks:   {analysis['backtrack_count_mean']:.1f} ± {analysis['backtrack_count_std']:.1f}")
        print(f"  Efficiency:   {analysis['efficiency_mean']:.3f}")
