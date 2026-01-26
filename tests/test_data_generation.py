"""Tests for data_generation.py - Dataset generation across regimes."""

import pytest
import numpy as np
import tempfile
import json
import pickle
from pathlib import Path

from src.core.data_generation import (
    DEFAULT_REGIMES,
    generate_regime_dataset,
    generate_all_regimes,
    save_dataset,
    load_dataset,
    load_all_regimes,
    analyze_trajectory_characteristics,
    _convert_for_json,
)


# =============================================================================
# Default Regimes Tests
# =============================================================================

class TestDefaultRegimes:
    """Tests for DEFAULT_REGIMES configuration."""

    def test_default_regimes_exist(self):
        """Test that all expected regimes exist."""
        expected = ["tight", "moderate", "loose", "very_loose"]
        for regime in expected:
            assert regime in DEFAULT_REGIMES

    def test_regime_has_max_length(self):
        """Test that each regime has max_length defined."""
        for name, config in DEFAULT_REGIMES.items():
            assert "max_length" in config
            assert isinstance(config["max_length"], int)
            assert config["max_length"] > 0

    def test_regime_slack_factors(self):
        """Test regime slack factors are increasing."""
        n_bins = 100  # Default
        slack_factors = []
        for name in ["tight", "moderate", "loose", "very_loose"]:
            slack = DEFAULT_REGIMES[name]["max_length"] / n_bins
            slack_factors.append(slack)

        # Each regime should have larger slack factor
        for i in range(len(slack_factors) - 1):
            assert slack_factors[i] < slack_factors[i + 1]

    def test_regime_has_description(self):
        """Test that each regime has a description."""
        for name, config in DEFAULT_REGIMES.items():
            assert "description" in config
            assert isinstance(config["description"], str)


# =============================================================================
# Generate Regime Dataset Tests
# =============================================================================

class TestGenerateRegimeDataset:
    """Tests for generate_regime_dataset function."""

    def test_basic_generation(self):
        """Test basic dataset generation."""
        dataset = generate_regime_dataset(
            regime_name="test",
            max_length=150,
            n_trajectories=10,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        assert "trajectories" in dataset
        assert "actions" in dataset
        assert "metadata" in dataset
        assert len(dataset["trajectories"]) == 10

    def test_metadata_contains_regime_name(self):
        """Test that regime name is added to metadata."""
        dataset = generate_regime_dataset(
            regime_name="custom_regime",
            max_length=150,
            n_trajectories=5,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        assert dataset["metadata"]["regime_name"] == "custom_regime"

    def test_reproducibility(self):
        """Test that same seed produces same dataset."""
        dataset1 = generate_regime_dataset(
            regime_name="test",
            max_length=150,
            n_trajectories=5,
            n_bins=10,
            seed=42,
            verbose=False,
        )
        dataset2 = generate_regime_dataset(
            regime_name="test",
            max_length=150,
            n_trajectories=5,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        for i in range(5):
            assert dataset1["trajectories"][i] == dataset2["trajectories"][i]

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different datasets."""
        dataset1 = generate_regime_dataset(
            regime_name="test",
            max_length=150,
            n_trajectories=5,
            n_bins=10,
            seed=42,
            verbose=False,
        )
        dataset2 = generate_regime_dataset(
            regime_name="test",
            max_length=150,
            n_trajectories=5,
            n_bins=10,
            seed=123,
            verbose=False,
        )

        # At least some trajectories should be different
        different = any(
            dataset1["trajectories"][i] != dataset2["trajectories"][i]
            for i in range(5)
        )
        assert different


# =============================================================================
# Generate All Regimes Tests
# =============================================================================

class TestGenerateAllRegimes:
    """Tests for generate_all_regimes function."""

    def test_generates_all_default_regimes(self):
        """Test that all default regimes are generated."""
        datasets = generate_all_regimes(
            n_trajectories=5,
            n_bins=10,
            verbose=False,
        )

        for regime in DEFAULT_REGIMES.keys():
            assert regime in datasets

    def test_custom_regimes(self):
        """Test generation with custom regimes."""
        custom_regimes = {
            "regime_a": {"max_length": 100},
            "regime_b": {"max_length": 200},
        }

        datasets = generate_all_regimes(
            regimes=custom_regimes,
            n_trajectories=5,
            n_bins=10,
            verbose=False,
        )

        assert "regime_a" in datasets
        assert "regime_b" in datasets
        assert len(datasets) == 2

    def test_different_seeds_per_regime(self):
        """Test that each regime gets a different seed."""
        datasets = generate_all_regimes(
            n_trajectories=5,
            n_bins=10,
            base_seed=42,
            verbose=False,
        )

        # Each regime should have different trajectories
        regimes = list(datasets.keys())
        if len(regimes) >= 2:
            traj1 = datasets[regimes[0]]["trajectories"]
            traj2 = datasets[regimes[1]]["trajectories"]
            # They should be different (different seeds)
            assert traj1 != traj2


# =============================================================================
# Save/Load Dataset Tests
# =============================================================================

class TestSaveLoadDataset:
    """Tests for dataset persistence."""

    def test_save_load_pickle(self):
        """Test saving and loading pickle format."""
        dataset = generate_regime_dataset(
            regime_name="test",
            max_length=150,
            n_trajectories=5,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = save_dataset(
                dataset, "test", tmpdir, save_format="pickle", verbose=False
            )

            loaded = load_dataset(filepath)

            assert loaded["trajectories"] == dataset["trajectories"]
            assert loaded["actions"] == dataset["actions"]
            assert loaded["metadata"]["regime_name"] == "test"

    def test_save_load_json(self):
        """Test saving and loading JSON format."""
        dataset = generate_regime_dataset(
            regime_name="test",
            max_length=150,
            n_trajectories=5,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = save_dataset(
                dataset, "test", tmpdir, save_format="json", verbose=False
            )

            loaded = load_dataset(filepath)

            assert loaded["trajectories"] == dataset["trajectories"]
            assert loaded["actions"] == dataset["actions"]

    def test_save_creates_directory(self):
        """Test that save_dataset creates output directory."""
        dataset = generate_regime_dataset(
            regime_name="test",
            max_length=150,
            n_trajectories=5,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "directory"
            filepath = save_dataset(
                dataset, "test", str(nested_dir), save_format="pickle", verbose=False
            )

            assert Path(filepath).exists()
            assert nested_dir.exists()

    def test_load_invalid_format_raises(self):
        """Test that loading unknown format raises error."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"dummy")
            filepath = f.name

        with pytest.raises(ValueError, match="Unknown file format"):
            load_dataset(filepath)

        Path(filepath).unlink()


# =============================================================================
# Load All Regimes Tests
# =============================================================================

class TestLoadAllRegimes:
    """Tests for load_all_regimes function."""

    def test_load_all_from_directory(self):
        """Test loading all datasets from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate and save some datasets
            datasets = generate_all_regimes(
                regimes={"regime_a": {"max_length": 150}, "regime_b": {"max_length": 200}},
                n_trajectories=5,
                n_bins=10,
                output_dir=tmpdir,
                verbose=False,
            )

            # Load them back
            loaded = load_all_regimes(tmpdir)

            assert "regime_a" in loaded
            assert "regime_b" in loaded

    def test_load_specific_regimes(self):
        """Test loading only specific regimes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate multiple regimes
            generate_all_regimes(
                regimes={
                    "regime_a": {"max_length": 150},
                    "regime_b": {"max_length": 200},
                    "regime_c": {"max_length": 250},
                },
                n_trajectories=5,
                n_bins=10,
                output_dir=tmpdir,
                verbose=False,
            )

            # Load only specific ones
            loaded = load_all_regimes(tmpdir, regimes=["regime_a", "regime_c"])

            assert "regime_a" in loaded
            assert "regime_c" in loaded
            assert "regime_b" not in loaded


# =============================================================================
# JSON Conversion Tests
# =============================================================================

class TestConvertForJson:
    """Tests for _convert_for_json function."""

    def test_converts_numpy_array(self):
        """Test conversion of numpy arrays."""
        arr = np.array([1, 2, 3])
        result = _convert_for_json(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_converts_numpy_int(self):
        """Test conversion of numpy integers."""
        val = np.int64(42)
        result = _convert_for_json(val)
        assert result == 42
        assert isinstance(result, int)

    def test_converts_numpy_float(self):
        """Test conversion of numpy floats."""
        val = np.float64(3.14)
        result = _convert_for_json(val)
        assert abs(result - 3.14) < 1e-6
        assert isinstance(result, float)

    def test_converts_nested_dict(self):
        """Test conversion of nested dictionaries."""
        data = {
            "array": np.array([1, 2]),
            "nested": {
                "value": np.float64(1.5),
            },
        }
        result = _convert_for_json(data)

        assert result["array"] == [1, 2]
        assert result["nested"]["value"] == 1.5

    def test_converts_list_of_numpy(self):
        """Test conversion of list containing numpy types."""
        data = [np.int32(1), np.float32(2.5), np.array([3, 4])]
        result = _convert_for_json(data)

        assert result[0] == 1
        assert abs(result[1] - 2.5) < 1e-5
        assert result[2] == [3, 4]


# =============================================================================
# Trajectory Analysis Tests
# =============================================================================

class TestAnalyzeTrajectoryCharacteristics:
    """Tests for analyze_trajectory_characteristics function."""

    def test_analysis_output_keys(self):
        """Test that analysis returns expected keys."""
        dataset = generate_regime_dataset(
            regime_name="test",
            max_length=150,
            n_trajectories=10,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        analysis = analyze_trajectory_characteristics(dataset)

        expected_keys = [
            "monotonicity_mean",
            "monotonicity_std",
            "backtrack_count_mean",
            "backtrack_count_std",
            "progress_rate_mean",
            "progress_rate_std",
            "optimal_length",
            "efficiency_mean",
        ]

        for key in expected_keys:
            assert key in analysis

    def test_monotonicity_in_valid_range(self):
        """Test that monotonicity is between 0 and 1."""
        dataset = generate_regime_dataset(
            regime_name="test",
            max_length=150,
            n_trajectories=10,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        analysis = analyze_trajectory_characteristics(dataset)

        assert 0.0 <= analysis["monotonicity_mean"] <= 1.0

    def test_tight_regime_more_monotonic(self):
        """Test that tight regime has higher monotonicity than loose."""
        tight_dataset = generate_regime_dataset(
            regime_name="tight",
            max_length=120,
            n_trajectories=50,
            n_bins=10,
            seed=42,
            verbose=False,
        )
        loose_dataset = generate_regime_dataset(
            regime_name="loose",
            max_length=200,
            n_trajectories=50,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        tight_analysis = analyze_trajectory_characteristics(tight_dataset)
        loose_analysis = analyze_trajectory_characteristics(loose_dataset)

        # Tight regime should be more monotonic (closer to 1)
        assert tight_analysis["monotonicity_mean"] > loose_analysis["monotonicity_mean"]

    def test_tight_regime_higher_efficiency(self):
        """Test that tight regime has higher efficiency."""
        tight_dataset = generate_regime_dataset(
            regime_name="tight",
            max_length=120,
            n_trajectories=50,
            n_bins=10,
            seed=42,
            verbose=False,
        )
        loose_dataset = generate_regime_dataset(
            regime_name="loose",
            max_length=200,
            n_trajectories=50,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        tight_analysis = analyze_trajectory_characteristics(tight_dataset)
        loose_analysis = analyze_trajectory_characteristics(loose_dataset)

        # Tight regime should be more efficient
        assert tight_analysis["efficiency_mean"] > loose_analysis["efficiency_mean"]

    def test_optimal_length_equals_n_bins(self):
        """Test that optimal length equals n_bins."""
        dataset = generate_regime_dataset(
            regime_name="test",
            max_length=150,
            n_trajectories=10,
            n_bins=10,
            seed=42,
            verbose=False,
        )

        analysis = analyze_trajectory_characteristics(dataset)

        assert analysis["optimal_length"] == 10
