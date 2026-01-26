"""1D Discrete Random Walk Environment for Goal-Conditioned Planning."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List

#TODO vectorize trajectory generation.

@dataclass
class EnvConfig:
    """Configuration for the 1D random walk environment."""

    n_bins: int = 10  # Number of discrete bins
    max_steps: int = 200  # Maximum steps per episode

    @property
    def step_size(self) -> float:
        """Size of one step (1/N)."""
        return 1.0 / self.n_bins

    @property
    def success_threshold(self) -> float:
        """Success if within half a bin of goal."""
        return 0.5 / self.n_bins


class RandomWalkEnv:
    """
    1D discrete random walk environment on [0, 1].

    State space: S = {0, 1/N, 2/N, ..., (N-1)/N, 1}
    Action space: A = {-1, 0, +1} (left, stay, right)
    Transition: s_{t+1} = clip(s_t + a_t / N, 0, 1)

    Args:
        config: Environment configuration
    """

    # Action mappings
    ACTION_LEFT = 0
    ACTION_STAY = 1
    ACTION_RIGHT = 2
    NUM_ACTIONS = 3

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self._state = 0.0
        self._steps = 0
        self._goal = 1.0

    @property
    def state(self) -> float:
        """Current state."""
        return self._state

    @property
    def goal(self) -> float:
        """Current goal state."""
        return self._goal

    @property
    def n_bins(self) -> int:
        """Number of discrete bins."""
        return self.config.n_bins

    def reset(self, start: float = 0.0, goal: float = 1.0) -> float:
        """
        Reset environment to initial state.

        Args:
            start: Starting state (default 0.0)
            goal: Goal state (default 1.0)

        Returns:
            Initial state
        """
        self._state = self._snap_to_grid(start)
        self._goal = self._snap_to_grid(goal)
        self._steps = 0
        return self._state

    def step(self, action: int) -> Tuple[float, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (0=left, 1=stay, 2=right)

        Returns:
            Tuple of (next_state, done, info)
        """
        # Convert action index to direction
        direction = action - 1  # {0, 1, 2} -> {-1, 0, +1}

        # Compute next state with clipping
        step_size = self.config.step_size
        next_state = self._state + direction * step_size
        next_state = np.clip(next_state, 0.0, 1.0)

        # Snap to grid to avoid floating point errors
        self._state = self._snap_to_grid(next_state)
        self._steps += 1

        # Check termination conditions
        success = self._is_success()
        timeout = self._steps >= self.config.max_steps
        done = success or timeout

        info = {
            "success": success,
            "timeout": timeout,
            "steps": self._steps,
        }

        return self._state, done, info

    def _snap_to_grid(self, value: float) -> float:
        """Snap value to nearest grid point."""
        n = self.config.n_bins
        return round(value * n) / n

    def _is_success(self) -> bool:
        """Check if current state is at goal."""
        return abs(self._state - self._goal) < self.config.success_threshold

    def action_to_delta(self, action: int) -> float:
        """Convert action index to state delta."""
        direction = action - 1
        return direction * self.config.step_size

    def delta_to_action(self, delta: float) -> int:
        """Convert state delta to action index."""
        direction = round(delta * self.config.n_bins)
        direction = np.clip(direction, -1, 1)
        return int(direction + 1)

    def get_optimal_action(self, state: float, goal: float) -> int:
        """Get optimal action to reach goal from state."""
        if goal > state + self.config.success_threshold:
            return self.ACTION_RIGHT
        elif goal < state - self.config.success_threshold:
            return self.ACTION_LEFT
        else:
            return self.ACTION_STAY

    def render(self) -> str:
        """Render current state as string."""
        n = self.config.n_bins
        pos = int(round(self._state * n))
        goal_pos = int(round(self._goal * n))

        # Create visualization string
        line = ["-"] * (n + 1)
        line[goal_pos] = "G"
        line[pos] = "X" if pos != goal_pos else "*"

        return "".join(line) + f" (state={self._state:.3f}, goal={self._goal:.3f})"


def generate_random_trajectory(
    env: RandomWalkEnv,
    max_length: int,
    start: float = 0.0,
    goal: float = 1.0,
    seed: Optional[int] = None,
) -> Optional[Tuple[List[float], List[int]]]:
    """
    Generate a single random walk trajectory from start to goal.

    Uses uniform random action sampling. Returns None if trajectory
    doesn't reach the goal within max_length steps.

    Args:
        env: RandomWalkEnv instance
        max_length: Maximum trajectory length
        start: Starting state
        goal: Goal state
        seed: Optional random seed

    Returns:
        Tuple of (states, actions) or None if unsuccessful
    """
    if seed is not None:
        np.random.seed(seed)

    state = env.reset(start=start, goal=goal)
    states = [state]
    actions = []

    for _ in range(max_length - 1):
        # Sample random action
        action = np.random.randint(0, env.NUM_ACTIONS)
        next_state, done, info = env.step(action)

        actions.append(action)
        states.append(next_state)

        if info["success"]:
            return states, actions

    # Failed to reach goal
    return None


def generate_dataset(
    n_trajectories: int,
    max_length: int,
    n_bins: int = 100,
    start: float = 0.0,
    goal: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Generate a dataset of successful random walk trajectories.

    Args:
        n_trajectories: Number of successful trajectories to generate
        max_length: Maximum length per trajectory
        n_bins: Number of discrete bins
        start: Starting state for all trajectories
        goal: Goal state for all trajectories
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary containing:
        - trajectories: List of state sequences
        - actions: List of action sequences
        - metadata: Dataset statistics
    """
    np.random.seed(seed)
    config = EnvConfig(n_bins=n_bins, max_steps=max_length)
    env = RandomWalkEnv(config)

    trajectories = []
    actions_list = []
    n_attempts = 0
    lengths = []

    if verbose:
        print(f"Generating {n_trajectories} trajectories (max_length={max_length})...")

    while len(trajectories) < n_trajectories:
        n_attempts += 1
        result = generate_random_trajectory(
            env, max_length=max_length, start=start, goal=goal
        )

        if result is not None:
            states, actions = result
            trajectories.append(states)
            actions_list.append(actions)
            lengths.append(len(states))

            if verbose and len(trajectories) % 1000 == 0:
                print(f"  Generated {len(trajectories)}/{n_trajectories} trajectories")

    # Compute statistics
    lengths = np.array(lengths)
    success_rate = n_trajectories / n_attempts

    metadata = {
        "n_trajectories": n_trajectories,
        "n_bins": n_bins,
        "max_length": max_length,
        "slack_factor": max_length / n_bins,
        "start": start,
        "goal": goal,
        "n_attempts": n_attempts,
        "success_rate": success_rate,
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()),
        "length_min": int(lengths.min()),
        "length_max": int(lengths.max()),
        "length_median": float(np.median(lengths)),
        "seed": seed,
    }

    if verbose:
        print(f"  Done! Success rate: {success_rate:.2%}")
        print(f"  Length stats: mean={metadata['length_mean']:.1f}, "
              f"std={metadata['length_std']:.1f}, "
              f"range=[{metadata['length_min']}, {metadata['length_max']}]")

    return {
        "trajectories": trajectories,
        "actions": actions_list,
        "metadata": metadata,
    }


def reconstruct_actions(trajectory: List[float], n_bins: int) -> List[int]:
    """
    Reconstruct actions from a trajectory of states.

    Args:
        trajectory: List of states
        n_bins: Number of bins (to compute step size)

    Returns:
        List of action indices
    """
    actions = []
    for i in range(len(trajectory) - 1):
        delta = trajectory[i + 1] - trajectory[i]
        direction = round(delta * n_bins)
        direction = np.clip(direction, -1, 1)
        actions.append(int(direction + 1))
    return actions


if __name__ == "__main__":
    # Test the environment
    print("Testing RandomWalkEnv...")

    config = EnvConfig(n_bins=10, max_steps=1000)
    env = RandomWalkEnv(config)

    # Test basic functionality
    state = env.reset()
    print(f"Initial state: {state}")
    print(env.render())

    # Take some steps
    for action in [2, 2, 2, 1, 2]:  # right, right, right, stay, right
        state, done, info = env.step(action)
        print(env.render())
        if done:
            print(f"Episode ended: {info}")
            break

    # Test trajectory generation
    print("\nTesting trajectory generation...")
    result = generate_random_trajectory(env, max_length=1000, start=0.0, goal=1.0, seed=42)
    if result:
        states, actions = result
        print(f"Generated trajectory with {len(states)} states")
        print(f"Start: {states[0]:.3f}, End: {states[-1]:.3f}")

    # Test dataset generation
    print("\nTesting dataset generation (small scale)...")
    import time

    start_time = time.time()
    dataset = generate_dataset(
        n_trajectories=100,
        max_length=10000,
        n_bins=100,
        seed=42,
        verbose=True,
    )
    end_time = time.time()
    print(f"Dataset generation took {end_time - start_time:.2f} seconds")
    print(f"Generated dataset with {len(dataset['trajectories'])} trajectories")
    print(f"Metadata: {dataset['metadata']}")
