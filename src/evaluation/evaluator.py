"""Evaluation module for policy rollouts and comprehensive experiment evaluation."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

from ..core.environment import RandomWalkEnv, EnvConfig
from .metrics import PolicyMetrics, compute_policy_metrics, compute_action_entropy


@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""

    success: bool
    steps: int
    trajectory: List[float]
    actions: List[int]
    action_entropies: List[float] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Complete evaluation results."""

    metrics: PolicyMetrics
    episodes: List[EpisodeResult]
    config: Dict


def evaluate_policy(
    policy: torch.nn.Module,
    n_episodes: int = 100,
    start: float = 0.0,
    goal: float = 1.0,
    n_bins: int = 100,
    max_steps: int = 200,
    deterministic: bool = True,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = False,
) -> EvaluationResult:
    """
    Evaluate a trained policy in the random walk environment.

    Args:
        policy: Trained policy (GCBCSinglePolicy or GCBCIntervalPolicy)
        n_episodes: Number of evaluation episodes
        start: Starting state
        goal: Goal state
        n_bins: Number of bins in environment
        max_steps: Maximum steps per episode
        deterministic: Use argmax for action selection
        device: Compute device
        seed: Random seed
        verbose: Print progress

    Returns:
        EvaluationResult with metrics and episode details
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = EnvConfig(n_bins=n_bins, max_steps=max_steps)
    env = RandomWalkEnv(config)

    policy.eval()
    episodes = []

    for ep in range(n_episodes):
        episode = run_episode(
            policy, env, start, goal, deterministic, device
        )
        episodes.append(episode)

        if verbose and (ep + 1) % 10 == 0:
            success_rate = sum(e.success for e in episodes) / len(episodes)
            print(f"Episode {ep+1}/{n_episodes}, Success rate: {success_rate:.2%}")

    # Compute metrics
    success_flags = [e.success for e in episodes]
    steps_list = [e.steps for e in episodes]
    entropies = [np.mean(e.action_entropies) if e.action_entropies else 0 for e in episodes]

    metrics = compute_policy_metrics(
        success_flags=success_flags,
        steps_to_goal=steps_list,
        n_bins=n_bins,
        action_entropies=entropies,
    )

    eval_config = {
        "n_episodes": n_episodes,
        "start": start,
        "goal": goal,
        "n_bins": n_bins,
        "max_steps": max_steps,
        "deterministic": deterministic,
        "seed": seed,
    }

    return EvaluationResult(
        metrics=metrics,
        episodes=episodes,
        config=eval_config,
    )


def run_episode(
    policy: torch.nn.Module,
    env: RandomWalkEnv,
    start: float,
    goal: float,
    deterministic: bool = True,
    device: str = "cpu",
) -> EpisodeResult:
    """
    Run a single evaluation episode.

    Args:
        policy: Policy network
        env: Environment instance
        start: Starting state
        goal: Goal state
        deterministic: Use argmax for actions
        device: Compute device

    Returns:
        EpisodeResult with trajectory and metrics
    """
    state = env.reset(start=start, goal=goal)
    trajectory = [state]
    actions = []
    action_entropies = []

    done = False
    while not done:
        # Prepare inputs
        state_tensor = torch.tensor([[state]], dtype=torch.float32).to(device)
        goal_tensor = torch.tensor([[goal]], dtype=torch.float32).to(device)

        # Get action
        with torch.no_grad():
            logits = policy(state_tensor, goal_tensor)
            probs = torch.softmax(logits, dim=-1)

            if deterministic:
                action = logits.argmax(dim=-1).item()
            else:
                action = torch.multinomial(probs, num_samples=1).item()

            # Compute entropy
            entropy = compute_action_entropy(probs)
            action_entropies.append(entropy)

        # Take step
        state, done, info = env.step(action)
        trajectory.append(state)
        actions.append(action)

    return EpisodeResult(
        success=info["success"],
        steps=info["steps"],
        trajectory=trajectory,
        actions=actions,
        action_entropies=action_entropies,
    )


def evaluate_generalization(
    policy: torch.nn.Module,
    test_type: str = "start_states",
    n_episodes_per_config: int = 20,
    n_bins: int = 100,
    max_steps: int = 200,
    deterministic: bool = True,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, EvaluationResult]:
    """
    Evaluate policy generalization to unseen (start, goal) pairs.

    Args:
        policy: Trained policy
        test_type: "start_states", "goal_states", or "random_pairs"
        n_episodes_per_config: Episodes per (start, goal) configuration
        n_bins: Number of bins
        max_steps: Max steps per episode
        deterministic: Use argmax
        device: Compute device
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary mapping configuration names to EvaluationResults
    """
    results = {}

    if test_type == "start_states":
        # Different start states, goal = 1
        start_states = [0.0, 0.1, 0.2, 0.3, 0.4]
        goal = 1.0

        for start in start_states:
            if verbose:
                print(f"Evaluating start={start:.1f}, goal={goal:.1f}...")

            result = evaluate_policy(
                policy,
                n_episodes=n_episodes_per_config,
                start=start,
                goal=goal,
                n_bins=n_bins,
                max_steps=max_steps,
                deterministic=deterministic,
                device=device,
                seed=seed,
            )
            results[f"start_{start:.1f}"] = result

    elif test_type == "goal_states":
        # Different goal states, start = 0
        start = 0.0
        goal_states = [0.6, 0.7, 0.8, 0.9, 1.0]

        for goal in goal_states:
            if verbose:
                print(f"Evaluating start={start:.1f}, goal={goal:.1f}...")

            result = evaluate_policy(
                policy,
                n_episodes=n_episodes_per_config,
                start=start,
                goal=goal,
                n_bins=n_bins,
                max_steps=max_steps,
                deterministic=deterministic,
                device=device,
                seed=seed,
            )
            results[f"goal_{goal:.1f}"] = result

    elif test_type == "random_pairs":
        # Random (start, goal) pairs where start < goal
        np.random.seed(seed)
        n_pairs = 50

        for i in range(n_pairs):
            start = np.random.uniform(0, 0.5)
            goal = np.random.uniform(start + 0.2, 1.0)

            if verbose and (i + 1) % 10 == 0:
                print(f"Evaluating random pair {i+1}/{n_pairs}...")

            result = evaluate_policy(
                policy,
                n_episodes=n_episodes_per_config,
                start=start,
                goal=goal,
                n_bins=n_bins,
                max_steps=max_steps,
                deterministic=deterministic,
                device=device,
                seed=seed + i,
            )
            results[f"random_{i}"] = result

    else:
        raise ValueError(f"Unknown test type: {test_type}")

    return results


def compute_generalization_summary(
    generalization_results: Dict[str, EvaluationResult]
) -> Dict[str, float]:
    """
    Compute summary statistics for generalization evaluation.

    Args:
        generalization_results: Results from evaluate_generalization

    Returns:
        Summary statistics dictionary
    """
    success_rates = []
    mean_steps_list = []
    path_efficiencies = []

    for result in generalization_results.values():
        success_rates.append(result.metrics.success_rate)
        if result.metrics.mean_steps != float('inf'):
            mean_steps_list.append(result.metrics.mean_steps)
        if result.metrics.path_efficiency > 0:
            path_efficiencies.append(result.metrics.path_efficiency)

    return {
        "mean_success_rate": float(np.mean(success_rates)) if success_rates else 0.0,
        "std_success_rate": float(np.std(success_rates)) if success_rates else 0.0,
        "min_success_rate": float(np.min(success_rates)) if success_rates else 0.0,
        "max_success_rate": float(np.max(success_rates)) if success_rates else 0.0,
        "mean_steps": float(np.mean(mean_steps_list)) if mean_steps_list else float('inf'),
        "mean_path_efficiency": float(np.mean(path_efficiencies)) if path_efficiencies else 0.0,
        "n_configs": len(generalization_results),
    }


def run_full_evaluation(
    policy: torch.nn.Module,
    n_bins: int = 100,
    max_steps: int = 200,
    n_episodes: int = 100,
    deterministic: bool = True,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Run complete evaluation including standard task and all generalization tests.

    Args:
        policy: Trained policy
        n_bins: Number of bins
        max_steps: Max steps
        n_episodes: Episodes for standard evaluation
        deterministic: Use argmax
        device: Compute device
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary with all evaluation results
    """
    results = {}

    # Standard evaluation (0 -> 1)
    if verbose:
        print("\n" + "="*60)
        print("Standard Evaluation (start=0, goal=1)")
        print("="*60)

    standard_result = evaluate_policy(
        policy,
        n_episodes=n_episodes,
        start=0.0,
        goal=1.0,
        n_bins=n_bins,
        max_steps=max_steps,
        deterministic=deterministic,
        device=device,
        seed=seed,
        verbose=verbose,
    )
    results["standard"] = {
        "success_rate": standard_result.metrics.success_rate,
        "mean_steps": standard_result.metrics.mean_steps,
        "median_steps": standard_result.metrics.median_steps,
        "path_efficiency": standard_result.metrics.path_efficiency,
        "action_entropy": standard_result.metrics.action_entropy,
    }

    if verbose:
        print(f"\nStandard Results:")
        print(f"  Success rate: {standard_result.metrics.success_rate:.2%}")
        print(f"  Mean steps: {standard_result.metrics.mean_steps:.1f}")
        print(f"  Path efficiency: {standard_result.metrics.path_efficiency:.4f}")

    # Generalization tests
    for test_type in ["start_states", "goal_states", "random_pairs"]:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generalization Test: {test_type}")
            print("="*60)

        gen_results = evaluate_generalization(
            policy,
            test_type=test_type,
            n_episodes_per_config=20,
            n_bins=n_bins,
            max_steps=max_steps,
            deterministic=deterministic,
            device=device,
            seed=seed,
            verbose=verbose,
        )

        summary = compute_generalization_summary(gen_results)
        results[f"generalization_{test_type}"] = summary

        if verbose:
            print(f"\n{test_type} Summary:")
            print(f"  Mean success rate: {summary['mean_success_rate']:.2%} ± {summary['std_success_rate']:.2%}")
            print(f"  Range: [{summary['min_success_rate']:.2%}, {summary['max_success_rate']:.2%}]")

    return results


def compare_policies(
    policies: Dict[str, torch.nn.Module],
    n_bins: int = 100,
    max_steps: int = 200,
    n_episodes: int = 100,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Compare multiple policies.

    Args:
        policies: Dictionary mapping policy names to policy modules
        n_bins: Number of bins
        max_steps: Max steps
        n_episodes: Episodes per policy
        device: Compute device
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary mapping policy names to evaluation results
    """
    results = {}

    for name, policy in policies.items():
        if verbose:
            print(f"\n{'#'*60}")
            print(f"Evaluating: {name}")
            print("#"*60)

        results[name] = run_full_evaluation(
            policy,
            n_bins=n_bins,
            max_steps=max_steps,
            n_episodes=n_episodes,
            device=device,
            seed=seed,
            verbose=verbose,
        )

    return results


def save_evaluation_results(results: Dict, filepath: str) -> None:
    """Save evaluation results to JSON file."""
    # Convert to JSON-serializable format
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if np.isfinite(obj) else str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    serializable = convert(results)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)


def load_evaluation_results(filepath: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def print_comparison_table(results: Dict[str, Dict]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("POLICY COMPARISON")
    print("="*80)

    # Standard evaluation table
    print("\nStandard Task (0 -> 1):")
    print("-"*70)
    print(f"{'Policy':<25} {'Success%':>10} {'Steps':>10} {'Efficiency':>12}")
    print("-"*70)

    for name, res in results.items():
        std = res.get("standard", {})
        success = std.get("success_rate", 0) * 100
        steps = std.get("mean_steps", float('inf'))
        eff = std.get("path_efficiency", 0)

        steps_str = f"{steps:.1f}" if steps != float('inf') else "N/A"
        print(f"{name:<25} {success:>9.1f}% {steps_str:>10} {eff:>12.4f}")

    # Generalization summary
    print("\nGeneralization (Mean Success Rate):")
    print("-"*70)
    print(f"{'Policy':<25} {'Start States':>12} {'Goal States':>12} {'Random':>12}")
    print("-"*70)

    for name, res in results.items():
        start = res.get("generalization_start_states", {}).get("mean_success_rate", 0) * 100
        goal = res.get("generalization_goal_states", {}).get("mean_success_rate", 0) * 100
        random = res.get("generalization_random_pairs", {}).get("mean_success_rate", 0) * 100

        print(f"{name:<25} {start:>11.1f}% {goal:>11.1f}% {random:>11.1f}%")

    print("="*80)


if __name__ == "__main__":
    # Test evaluation
    print("Testing evaluation module...")

    from src.models import GCBCSinglePolicy, EuclideanIntervalEncoder, GCBCIntervalPolicy

    device = "cpu"

    # Create a simple policy
    policy = GCBCSinglePolicy(n_actions=3, hidden_sizes=[64, 64]).to(device)

    # Test single evaluation
    print("\n1. Testing single evaluation...")
    result = evaluate_policy(
        policy,
        n_episodes=10,
        start=0.0,
        goal=1.0,
        n_bins=100,
        max_steps=200,
        device=device,
        verbose=True,
    )

    print(f"\nResults:")
    print(f"  Success rate: {result.metrics.success_rate:.2%}")
    print(f"  Mean steps: {result.metrics.mean_steps:.1f}")

    # Test generalization
    print("\n2. Testing generalization evaluation...")
    gen_results = evaluate_generalization(
        policy,
        test_type="start_states",
        n_episodes_per_config=5,
        device=device,
        verbose=True,
    )

    summary = compute_generalization_summary(gen_results)
    print(f"\nGeneralization Summary:")
    print(f"  Mean success rate: {summary['mean_success_rate']:.2%}")

    # Test policy comparison
    print("\n3. Testing policy comparison...")
    encoder = EuclideanIntervalEncoder(embedding_dim=2).to(device)
    interval_policy = GCBCIntervalPolicy(encoder, n_actions=3).to(device)

    policies = {
        "GCBC-Single": policy,
        "GCBC-Interval": interval_policy,
    }

    # Quick comparison
    comparison = {}
    for name, pol in policies.items():
        result = evaluate_policy(pol, n_episodes=10, device=device)
        comparison[name] = {
            "standard": {
                "success_rate": result.metrics.success_rate,
                "mean_steps": result.metrics.mean_steps,
                "path_efficiency": result.metrics.path_efficiency,
            }
        }

    print("\nComparison Results:")
    for name, res in comparison.items():
        print(f"  {name}: {res['standard']['success_rate']:.2%} success")

    print("\nAll tests passed!")
