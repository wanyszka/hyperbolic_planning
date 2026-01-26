"""Tests for evaluation.py - Policy evaluation and rollouts."""

import pytest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path

from src.evaluation import (
    EpisodeResult,
    EvaluationResult,
    evaluate_policy,
    run_episode,
    evaluate_generalization,
    compute_generalization_summary,
    run_full_evaluation,
    compare_policies,
    save_evaluation_results,
    load_evaluation_results,
    PolicyMetrics,
)
from src.core import RandomWalkEnv, EnvConfig
from src.models import GCBCSinglePolicy, GCBCIntervalPolicy, EuclideanIntervalEncoder


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_policy():
    """Create a simple untrained policy for testing."""
    return GCBCSinglePolicy(n_actions=3, hidden_sizes=[32, 32])


@pytest.fixture
def interval_policy():
    """Create an interval-based policy for testing."""
    encoder = EuclideanIntervalEncoder(embedding_dim=2)
    return GCBCIntervalPolicy(encoder=encoder, n_actions=3, hidden_sizes=[32, 32])


@pytest.fixture
def env():
    """Create a test environment."""
    config = EnvConfig(n_bins=10, max_steps=50)
    return RandomWalkEnv(config)


# =============================================================================
# Dataclass Tests
# =============================================================================

class TestEpisodeResult:
    """Tests for EpisodeResult dataclass."""

    def test_creation(self):
        """Test creating EpisodeResult."""
        result = EpisodeResult(
            success=True,
            steps=100,
            trajectory=[0.0, 0.1, 0.2, 0.3],
            actions=[2, 2, 2],
            action_entropies=[0.5, 0.6, 0.5],
        )

        assert result.success is True
        assert result.steps == 100
        assert len(result.trajectory) == 4
        assert len(result.actions) == 3

    def test_default_entropies(self):
        """Test that action_entropies defaults to empty list."""
        result = EpisodeResult(
            success=False,
            steps=50,
            trajectory=[0.0],
            actions=[],
        )

        assert result.action_entropies == []


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self):
        """Test creating EvaluationResult."""
        metrics = PolicyMetrics(
            success_rate=0.8,
            mean_steps=100.0,
            median_steps=95.0,
            std_steps=10.0,
            path_efficiency=0.9,
            action_entropy=0.5,
        )

        result = EvaluationResult(
            metrics=metrics,
            episodes=[],
            config={"n_episodes": 100},
        )

        assert result.metrics.success_rate == 0.8
        assert result.config["n_episodes"] == 100


# =============================================================================
# Run Episode Tests
# =============================================================================

class TestRunEpisode:
    """Tests for run_episode function."""

    def test_runs_to_completion(self, simple_policy, env):
        """Test that episode runs to completion."""
        result = run_episode(
            simple_policy,
            env,
            start=0.0,
            goal=1.0,
            deterministic=True,
            device="cpu",
        )

        assert isinstance(result, EpisodeResult)
        assert len(result.trajectory) > 0
        assert len(result.actions) == len(result.trajectory) - 1
        assert result.steps > 0

    def test_trajectory_starts_at_start(self, simple_policy, env):
        """Test that trajectory starts at specified start."""
        result = run_episode(simple_policy, env, start=0.5, goal=1.0)

        assert result.trajectory[0] == 0.5

    def test_success_episode(self, simple_policy, env):
        """Test successful episode detection."""
        # Run multiple episodes, some might succeed
        success_count = 0
        for seed in range(20):
            torch.manual_seed(seed)
            result = run_episode(simple_policy, env, start=0.0, goal=1.0)
            if result.success:
                success_count += 1
                # Verify trajectory reached goal
                final_state = result.trajectory[-1]
                assert abs(final_state - 1.0) < env.config.success_threshold + 1e-6

    def test_timeout_episode(self, simple_policy):
        """Test timeout episode detection."""
        # Use very small max_steps to force timeout
        config = EnvConfig(n_bins=100, max_steps=5)
        env = RandomWalkEnv(config)

        result = run_episode(simple_policy, env, start=0.0, goal=1.0)

        # With only 5 steps, very unlikely to succeed
        # Check that episode ended due to steps limit
        assert result.steps <= 5

    def test_action_entropies_recorded(self, simple_policy, env):
        """Test that action entropies are recorded."""
        result = run_episode(simple_policy, env, start=0.0, goal=1.0)

        assert len(result.action_entropies) == len(result.actions)
        assert all(e >= 0 for e in result.action_entropies)


# =============================================================================
# Evaluate Policy Tests
# =============================================================================

class TestEvaluatePolicy:
    """Tests for evaluate_policy function."""

    def test_returns_evaluation_result(self, simple_policy):
        """Test that function returns EvaluationResult."""
        result = evaluate_policy(
            simple_policy,
            n_episodes=5,
            n_bins=10,
            max_steps=50,
            device="cpu",
        )

        assert isinstance(result, EvaluationResult)
        assert isinstance(result.metrics, PolicyMetrics)

    def test_correct_number_of_episodes(self, simple_policy):
        """Test that correct number of episodes are run."""
        result = evaluate_policy(
            simple_policy,
            n_episodes=10,
            n_bins=10,
            max_steps=50,
        )

        assert len(result.episodes) == 10

    def test_config_stored(self, simple_policy):
        """Test that config is stored in result."""
        result = evaluate_policy(
            simple_policy,
            n_episodes=5,
            start=0.2,
            goal=0.8,
            n_bins=50,
            max_steps=100,
        )

        assert result.config["n_episodes"] == 5
        assert result.config["start"] == 0.2
        assert result.config["goal"] == 0.8
        assert result.config["n_bins"] == 50
        assert result.config["max_steps"] == 100

    def test_reproducibility(self, simple_policy):
        """Test that same seed gives same results."""
        result1 = evaluate_policy(
            simple_policy, n_episodes=5, seed=42, n_bins=10, max_steps=50
        )
        result2 = evaluate_policy(
            simple_policy, n_episodes=5, seed=42, n_bins=10, max_steps=50
        )

        assert result1.metrics.success_rate == result2.metrics.success_rate
        assert result1.metrics.mean_steps == result2.metrics.mean_steps

    def test_deterministic_vs_stochastic(self, simple_policy):
        """Test deterministic vs stochastic action selection."""
        # Run with deterministic
        det_result = evaluate_policy(
            simple_policy, n_episodes=20, deterministic=True, seed=42, n_bins=10
        )

        # Run with stochastic
        stoch_result = evaluate_policy(
            simple_policy, n_episodes=20, deterministic=False, seed=42, n_bins=10
        )

        # Both should complete without error
        assert len(det_result.episodes) == 20
        assert len(stoch_result.episodes) == 20


# =============================================================================
# Generalization Tests
# =============================================================================

class TestEvaluateGeneralization:
    """Tests for evaluate_generalization function."""

    def test_start_states_evaluation(self, simple_policy):
        """Test evaluation with different start states."""
        results = evaluate_generalization(
            simple_policy,
            test_type="start_states",
            n_episodes_per_config=3,
            n_bins=10,
            max_steps=50,
            verbose=False,
        )

        assert len(results) == 5  # 5 different start states
        assert all(key.startswith("start_") for key in results.keys())

    def test_goal_states_evaluation(self, simple_policy):
        """Test evaluation with different goal states."""
        results = evaluate_generalization(
            simple_policy,
            test_type="goal_states",
            n_episodes_per_config=3,
            n_bins=10,
            max_steps=50,
            verbose=False,
        )

        assert len(results) == 5  # 5 different goal states
        assert all(key.startswith("goal_") for key in results.keys())

    def test_random_pairs_evaluation(self, simple_policy):
        """Test evaluation with random (start, goal) pairs."""
        results = evaluate_generalization(
            simple_policy,
            test_type="random_pairs",
            n_episodes_per_config=2,
            n_bins=10,
            max_steps=50,
            verbose=False,
        )

        assert len(results) == 50  # 50 random pairs
        assert all(key.startswith("random_") for key in results.keys())

    def test_unknown_test_type_raises(self, simple_policy):
        """Test that unknown test type raises error."""
        with pytest.raises(ValueError, match="Unknown test type"):
            evaluate_generalization(simple_policy, test_type="unknown")


class TestComputeGeneralizationSummary:
    """Tests for compute_generalization_summary function."""

    def test_computes_summary(self, simple_policy):
        """Test that summary is computed correctly."""
        results = evaluate_generalization(
            simple_policy,
            test_type="start_states",
            n_episodes_per_config=3,
            n_bins=10,
            max_steps=50,
            verbose=False,
        )

        summary = compute_generalization_summary(results)

        expected_keys = [
            "mean_success_rate",
            "std_success_rate",
            "min_success_rate",
            "max_success_rate",
            "mean_steps",
            "mean_path_efficiency",
            "n_configs",
        ]

        for key in expected_keys:
            assert key in summary

    def test_n_configs_correct(self, simple_policy):
        """Test that n_configs is correct."""
        results = evaluate_generalization(
            simple_policy,
            test_type="start_states",
            n_episodes_per_config=2,
            n_bins=10,
            max_steps=50,
            verbose=False,
        )

        summary = compute_generalization_summary(results)

        assert summary["n_configs"] == 5


# =============================================================================
# Full Evaluation Tests
# =============================================================================

class TestRunFullEvaluation:
    """Tests for run_full_evaluation function."""

    def test_returns_all_results(self, simple_policy):
        """Test that full evaluation returns all result types."""
        results = run_full_evaluation(
            simple_policy,
            n_bins=10,
            max_steps=50,
            n_episodes=5,
            verbose=False,
        )

        assert "standard" in results
        assert "generalization_start_states" in results
        assert "generalization_goal_states" in results
        assert "generalization_random_pairs" in results

    def test_standard_results_structure(self, simple_policy):
        """Test structure of standard results."""
        results = run_full_evaluation(
            simple_policy,
            n_episodes=5,
            n_bins=10,
            max_steps=50,
            verbose=False,
        )

        std = results["standard"]
        assert "success_rate" in std
        assert "mean_steps" in std
        assert "path_efficiency" in std


# =============================================================================
# Compare Policies Tests
# =============================================================================

class TestComparePolicies:
    """Tests for compare_policies function."""

    def test_compares_multiple_policies(self, simple_policy, interval_policy):
        """Test comparing multiple policies."""
        policies = {
            "single": simple_policy,
            "interval": interval_policy,
        }

        # Run a minimal comparison
        results = {}
        for name, policy in policies.items():
            result = evaluate_policy(policy, n_episodes=3, n_bins=10, max_steps=30)
            results[name] = {
                "standard": {
                    "success_rate": result.metrics.success_rate,
                    "mean_steps": result.metrics.mean_steps,
                }
            }

        assert "single" in results
        assert "interval" in results


# =============================================================================
# Save/Load Tests
# =============================================================================

class TestSaveLoadResults:
    """Tests for save/load functions."""

    def test_save_and_load(self, simple_policy):
        """Test saving and loading evaluation results."""
        result = evaluate_policy(
            simple_policy,
            n_episodes=5,
            n_bins=10,
            max_steps=50,
        )

        # Convert to dictionary format for saving
        results_dict = {
            "standard": {
                "success_rate": result.metrics.success_rate,
                "mean_steps": result.metrics.mean_steps,
                "path_efficiency": result.metrics.path_efficiency,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.json"

            save_evaluation_results(results_dict, str(filepath))
            loaded = load_evaluation_results(str(filepath))

            assert loaded["standard"]["success_rate"] == result.metrics.success_rate

    def test_save_creates_directory(self, simple_policy):
        """Test that save creates parent directories."""
        results = {"test": {"value": 0.5}}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "dir" / "results.json"

            save_evaluation_results(results, str(filepath))

            assert filepath.exists()

    def test_handles_special_values(self):
        """Test handling of special float values."""
        results = {
            "test": {
                "inf_value": float('inf'),
                "normal_value": 1.5,
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.json"

            save_evaluation_results(results, str(filepath))
            loaded = load_evaluation_results(str(filepath))

            # inf is converted to string "inf" when saved, then loaded back
            # The actual behavior depends on JSON library
            assert loaded["test"]["inf_value"] == "inf" or loaded["test"]["inf_value"] == float('inf')
            assert loaded["test"]["normal_value"] == 1.5

    def test_handles_numpy_types(self):
        """Test handling of numpy types."""
        results = {
            "test": {
                "np_float": np.float64(3.14),
                "np_int": np.int64(42),
                "np_array": np.array([1, 2, 3]),
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "results.json"

            save_evaluation_results(results, str(filepath))
            loaded = load_evaluation_results(str(filepath))

            assert loaded["test"]["np_float"] == 3.14
            assert loaded["test"]["np_int"] == 42
            assert loaded["test"]["np_array"] == [1, 2, 3]
