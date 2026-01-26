"""Tests for environment.py - 1D Discrete Random Walk Environment."""

import pytest
import numpy as np

from src.core import (
    EnvConfig,
    RandomWalkEnv,
    generate_random_trajectory,
    generate_dataset,
    reconstruct_actions,
)


# =============================================================================
# EnvConfig Tests
# =============================================================================

class TestEnvConfig:
    """Tests for EnvConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EnvConfig()
        assert config.n_bins == 10
        assert config.max_steps == 200

    def test_step_size_property(self):
        """Test step_size computation."""
        config = EnvConfig(n_bins=100, max_steps=2000)
        assert config.step_size == 0.01

        config = EnvConfig(n_bins=10, max_steps=200)
        assert config.step_size == 0.1

    def test_success_threshold_property(self):
        """Test success_threshold computation."""
        config = EnvConfig(n_bins=100, max_steps=2000)
        assert config.success_threshold == 0.005  # 0.5 / 100

        config = EnvConfig(n_bins=10, max_steps=200)
        assert config.success_threshold == 0.05  # 0.5 / 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EnvConfig(n_bins=50, max_steps=1000)
        assert config.n_bins == 50
        assert config.max_steps == 1000
        assert config.step_size == 0.02


# =============================================================================
# RandomWalkEnv Initialization Tests
# =============================================================================

class TestRandomWalkEnvInit:
    """Tests for RandomWalkEnv initialization."""

    def test_default_initialization(self):
        """Test environment with default config."""
        env = RandomWalkEnv()
        assert env.n_bins == 10
        assert env.state == 0.0
        assert env.goal == 1.0

    def test_custom_config_initialization(self):
        """Test environment with custom config."""
        config = EnvConfig(n_bins=50, max_steps=1000)
        env = RandomWalkEnv(config)
        assert env.n_bins == 50
        assert env.config.max_steps == 1000

    def test_action_constants(self):
        """Test action constant values."""
        assert RandomWalkEnv.ACTION_LEFT == 0
        assert RandomWalkEnv.ACTION_STAY == 1
        assert RandomWalkEnv.ACTION_RIGHT == 2
        assert RandomWalkEnv.NUM_ACTIONS == 3


# =============================================================================
# RandomWalkEnv Reset Tests
# =============================================================================

class TestRandomWalkEnvReset:
    """Tests for RandomWalkEnv reset functionality."""

    def test_reset_default(self):
        """Test reset with default values."""
        env = RandomWalkEnv()
        state = env.reset()
        assert state == 0.0
        assert env.goal == 1.0

    def test_reset_custom_start_goal(self):
        """Test reset with custom start and goal."""
        env = RandomWalkEnv()
        state = env.reset(start=0.5, goal=0.8)
        assert state == 0.5
        assert env.goal == 0.8

    def test_reset_snaps_to_grid(self):
        """Test that reset snaps values to grid."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)

        # 0.15 should snap to 0.2 (nearest grid point)
        # 0.85 snaps to 0.8 (round(0.85 * 10) / 10 = round(8.5) / 10 = 8 / 10 = 0.8)
        state = env.reset(start=0.15, goal=0.86)  # 0.86 -> 0.9
        assert state == 0.2
        assert env.goal == 0.9

    def test_reset_clears_step_count(self):
        """Test that reset clears step counter."""
        env = RandomWalkEnv()
        env.reset()
        env.step(2)  # Take a step
        env.reset()  # Reset
        # Step counter should be 0 after reset (tested via timeout)
        assert env._steps == 0


# =============================================================================
# RandomWalkEnv Step Tests
# =============================================================================

class TestRandomWalkEnvStep:
    """Tests for RandomWalkEnv step functionality."""

    def test_step_right(self):
        """Test stepping right increases state."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)
        env.reset(start=0.5, goal=1.0)

        state, done, info = env.step(RandomWalkEnv.ACTION_RIGHT)
        assert state == 0.6  # 0.5 + 0.1
        assert not done
        assert not info["success"]

    def test_step_left(self):
        """Test stepping left decreases state."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)
        env.reset(start=0.5, goal=1.0)

        state, done, info = env.step(RandomWalkEnv.ACTION_LEFT)
        assert state == 0.4  # 0.5 - 0.1
        assert not done

    def test_step_stay(self):
        """Test staying keeps state same."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)
        env.reset(start=0.5, goal=1.0)

        state, done, info = env.step(RandomWalkEnv.ACTION_STAY)
        assert state == 0.5
        assert not done

    def test_step_clips_at_lower_bound(self):
        """Test state is clipped at 0."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)
        env.reset(start=0.0, goal=1.0)

        state, done, info = env.step(RandomWalkEnv.ACTION_LEFT)
        assert state == 0.0  # Clipped at lower bound

    def test_step_clips_at_upper_bound(self):
        """Test state is clipped at 1."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)
        env.reset(start=1.0, goal=0.0)

        state, done, info = env.step(RandomWalkEnv.ACTION_RIGHT)
        assert state == 1.0  # Clipped at upper bound

    def test_success_detection(self):
        """Test success when reaching goal."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)
        env.reset(start=0.9, goal=1.0)

        state, done, info = env.step(RandomWalkEnv.ACTION_RIGHT)
        assert state == 1.0
        assert done
        assert info["success"]
        assert not info["timeout"]

    def test_timeout_detection(self):
        """Test timeout when max steps reached."""
        config = EnvConfig(n_bins=10, max_steps=5)
        env = RandomWalkEnv(config)
        env.reset(start=0.0, goal=1.0)

        # Take max_steps steps without reaching goal
        for i in range(5):
            state, done, info = env.step(RandomWalkEnv.ACTION_STAY)

        assert done
        assert info["timeout"]
        assert not info["success"]
        assert info["steps"] == 5

    def test_step_increments_counter(self):
        """Test that step increments the step counter."""
        env = RandomWalkEnv()
        env.reset()

        env.step(RandomWalkEnv.ACTION_RIGHT)
        assert env._steps == 1

        env.step(RandomWalkEnv.ACTION_RIGHT)
        assert env._steps == 2


# =============================================================================
# RandomWalkEnv Action Conversion Tests
# =============================================================================

class TestRandomWalkEnvActionConversion:
    """Tests for action-delta conversion methods."""

    def test_action_to_delta(self):
        """Test action_to_delta conversion."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)

        assert env.action_to_delta(0) == -0.1  # Left
        assert env.action_to_delta(1) == 0.0   # Stay
        assert env.action_to_delta(2) == 0.1   # Right

    def test_delta_to_action(self):
        """Test delta_to_action conversion."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)

        assert env.delta_to_action(-0.1) == 0  # Left
        assert env.delta_to_action(0.0) == 1   # Stay
        assert env.delta_to_action(0.1) == 2   # Right

    def test_delta_to_action_clipping(self):
        """Test delta_to_action clips large deltas."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)

        # Large positive delta should clip to right
        assert env.delta_to_action(0.5) == 2
        # Large negative delta should clip to left
        assert env.delta_to_action(-0.5) == 0

    def test_get_optimal_action_right(self):
        """Test optimal action when goal is to the right."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)

        action = env.get_optimal_action(0.0, 1.0)
        assert action == RandomWalkEnv.ACTION_RIGHT

    def test_get_optimal_action_left(self):
        """Test optimal action when goal is to the left."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)

        action = env.get_optimal_action(1.0, 0.0)
        assert action == RandomWalkEnv.ACTION_LEFT

    def test_get_optimal_action_stay(self):
        """Test optimal action when at goal."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)

        action = env.get_optimal_action(0.5, 0.5)
        assert action == RandomWalkEnv.ACTION_STAY


# =============================================================================
# RandomWalkEnv Render Tests
# =============================================================================

class TestRandomWalkEnvRender:
    """Tests for environment rendering."""

    def test_render_output_format(self):
        """Test render produces valid string output."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)
        env.reset(start=0.0, goal=1.0)

        output = env.render()
        assert isinstance(output, str)
        assert "state=" in output
        assert "goal=" in output

    def test_render_shows_position(self):
        """Test render shows agent position."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)
        env.reset(start=0.5, goal=1.0)

        output = env.render()
        # Position marker should be at index 5 (0.5 * 10)
        assert "X" in output or "*" in output


# =============================================================================
# Trajectory Generation Tests (n_bins=10 only for performance)
# =============================================================================

class TestGenerateRandomTrajectory:
    """Tests for generate_random_trajectory function."""

    def test_successful_trajectory(self):
        """Test generating a successful trajectory."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)

        result = generate_random_trajectory(
            env, max_length=200, start=0.0, goal=1.0, seed=42
        )

        assert result is not None
        states, actions = result
        assert len(states) > 0
        assert len(actions) == len(states) - 1
        assert states[0] == 0.0
        # Final state should be at or near goal
        assert abs(states[-1] - 1.0) < config.success_threshold

    def test_trajectory_respects_seed(self):
        """Test that seed produces reproducible results."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)

        result1 = generate_random_trajectory(
            env, max_length=200, start=0.0, goal=1.0, seed=42
        )
        result2 = generate_random_trajectory(
            env, max_length=200, start=0.0, goal=1.0, seed=42
        )

        assert result1 is not None and result2 is not None
        assert result1[0] == result2[0]  # Same states
        assert result1[1] == result2[1]  # Same actions

    def test_trajectory_fails_with_short_max_length(self):
        """Test trajectory returns None if max_length too short."""
        # Use n_bins=10 with max_steps=5 (way less than 20x)
        config = EnvConfig(n_bins=10, max_steps=5)
        env = RandomWalkEnv(config)

        # With only 5 steps, very unlikely to reach goal from 0 to 1
        # when n_bins=10 (need ~10 right steps minimum)
        result = generate_random_trajectory(
            env, max_length=5, start=0.0, goal=1.0, seed=42
        )

        # Should fail with such a short max_length
        assert result is None


# =============================================================================
# Dataset Generation Tests (n_bins=10 only for performance)
# =============================================================================

class TestGenerateDataset:
    """Tests for generate_dataset function."""

    def test_dataset_structure(self):
        """Test dataset has correct structure."""
        dataset = generate_dataset(
            n_trajectories=10,
            max_length=200,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        assert "trajectories" in dataset
        assert "actions" in dataset
        assert "metadata" in dataset

    def test_dataset_size(self):
        """Test dataset has requested number of trajectories."""
        dataset = generate_dataset(
            n_trajectories=20,
            max_length=200,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        assert len(dataset["trajectories"]) == 20
        assert len(dataset["actions"]) == 20

    def test_dataset_metadata(self):
        """Test dataset metadata is correct."""
        dataset = generate_dataset(
            n_trajectories=10,
            max_length=200,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        meta = dataset["metadata"]
        assert meta["n_trajectories"] == 10
        assert meta["max_length"] == 200
        assert meta["n_bins"] == 10
        assert meta["seed"] == 42
        assert "length_mean" in meta
        assert "success_rate" in meta

    def test_dataset_reproducibility(self):
        """Test dataset generation is reproducible."""
        dataset1 = generate_dataset(
            n_trajectories=5,
            max_length=200,
            n_bins=10,
            seed=42,
            verbose=False,
        )
        dataset2 = generate_dataset(
            n_trajectories=5,
            max_length=200,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        for i in range(5):
            assert dataset1["trajectories"][i] == dataset2["trajectories"][i]
            assert dataset1["actions"][i] == dataset2["actions"][i]


# =============================================================================
# Reconstruct Actions Tests
# =============================================================================

class TestReconstructActions:
    """Tests for reconstruct_actions function."""

    def test_reconstruct_monotonic_trajectory(self):
        """Test reconstructing actions from monotonic trajectory."""
        # Monotonically increasing trajectory
        trajectory = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        n_bins = 10

        actions = reconstruct_actions(trajectory, n_bins)

        assert len(actions) == 5
        assert all(a == RandomWalkEnv.ACTION_RIGHT for a in actions)

    def test_reconstruct_stationary_trajectory(self):
        """Test reconstructing actions from stationary trajectory."""
        trajectory = [0.5, 0.5, 0.5, 0.5]
        n_bins = 10

        actions = reconstruct_actions(trajectory, n_bins)

        assert len(actions) == 3
        assert all(a == RandomWalkEnv.ACTION_STAY for a in actions)

    def test_reconstruct_mixed_trajectory(self):
        """Test reconstructing actions from mixed trajectory."""
        trajectory = [0.5, 0.6, 0.6, 0.5]  # right, stay, left
        n_bins = 10

        actions = reconstruct_actions(trajectory, n_bins)

        assert len(actions) == 3
        assert actions[0] == RandomWalkEnv.ACTION_RIGHT
        assert actions[1] == RandomWalkEnv.ACTION_STAY
        assert actions[2] == RandomWalkEnv.ACTION_LEFT

    def test_reconstruct_roundtrip(self):
        """Test that reconstruct_actions roundtrips with env.step."""
        config = EnvConfig(n_bins=10, max_steps=200)
        env = RandomWalkEnv(config)

        # Generate a trajectory
        env.reset(start=0.0, goal=1.0)
        original_states = [env.state]
        original_actions = []

        for action in [2, 2, 1, 2, 0, 2]:  # Some sequence
            state, _, _ = env.step(action)
            original_states.append(state)
            original_actions.append(action)

        # Reconstruct actions from states
        reconstructed = reconstruct_actions(original_states, n_bins=10)

        assert reconstructed == original_actions
