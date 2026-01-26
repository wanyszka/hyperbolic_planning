# Hyperbolic Planning

Hyperbolic embeddings for hierarchical interval representations in goal-conditioned planning.

## Overview

This project explores the use of **hyperbolic geometry** (Poincaré ball model) for learning interval representations that capture hierarchical structure. The core hypothesis is that hyperbolic space naturally encodes containment relationships between intervals, which can improve goal-conditioned behavioral cloning.

### Key Features

- **Hyperbolic Interval Encoder**: Maps intervals `[a, b]` to the Poincaré ball using a hybrid Euclidean-Hyperbolic architecture
- **Contrastive Learning**: InfoNCE loss with hyperbolic distances for learning meaningful interval representations
- **Goal-Conditioned Policies**: GCBC (Goal-Conditioned Behavioral Cloning) using learned interval embeddings
- **1D Random Walk Environment**: Simple testbed for studying interval-based planning

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hyperbolic-planning.git
cd hyperbolic-planning

# Install dependencies
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- geoopt >= 0.5.0
- hypll >= 0.2.0
- numpy, matplotlib, seaborn, imageio

## Project Structure

```
hyperbolic_planning/
├── src/
│   ├── core/
│   │   ├── environment.py       # 1D random walk environment
│   │   ├── data_generation.py   # Trajectory data generation
│   │   └── datasets/            # Dataset classes
│   ├── models/
│   │   ├── encoders.py          # Euclidean & Hyperbolic interval encoders
│   │   ├── policies.py          # GCBC policy networks
│   │   └── utils.py             # Manifold utilities
│   ├── training/
│   │   ├── trainer.py           # Training loops
│   │   └── losses.py            # Contrastive loss functions
│   ├── evaluation/
│   │   ├── evaluator.py         # Policy evaluation
│   │   └── metrics.py           # Evaluation metrics
│   └── visualization/
│       ├── hyperbolic_viz.py    # Poincaré ball visualizations
│       └── experiment_viz.py    # Experiment result plots
├── scripts/
│   ├── generate_data.py         # Data generation script
│   ├── train_representations.py # Train interval encoders
│   ├── train_policies.py        # Train GCBC policies
│   ├── evaluate.py              # Evaluation script
│   └── run_all_experiments.py   # Full experiment pipeline
├── config/
│   └── experiment_config.yaml   # Experiment configuration
├── tests/                       # Unit tests
├── main.py                      # Quick start script
└── pyproject.toml               # Package configuration
```

## Quick Start

```python
from src.core.datasets import IntervalDataset
from src.models import HyperbolicIntervalEncoder
from src.training import train_model

# Create dataset
dataset = IntervalDataset(num_samples=1000, num_negatives=2, num_points=10)

# Create hyperbolic encoder
model = HyperbolicIntervalEncoder(embedding_dim=2, c=1.0)

# Train with contrastive learning
losses = train_model(model, dataset, num_epochs=200, batch_size=32, lr=0.001)
```

Or run the main script:

```bash
python main.py
```

## Architecture

### Hyperbolic Interval Encoder

The encoder uses a hybrid architecture:
1. **Euclidean layers** (2 layers): Initial feature extraction from interval `[a, b]`
2. **Exponential map**: Project to Poincaré ball
3. **Hyperbolic layers** (2 layers): Processing in hyperbolic space using Möbius operations

### Contrastive Learning

Training uses InfoNCE loss with hyperbolic distances:
- **Anchors**: Larger intervals (less specific)
- **Positives**: Contained sub-intervals (more specific)
- **Negatives**: Non-contained intervals

The hyperbolic distance naturally captures hierarchical relationships—contained intervals should be closer in hyperbolic space.

## Experiments

The experiment pipeline compares:

| Method | Description |
|--------|-------------|
| GCBC-Single | Direct (state, goal) → action mapping |
| GCBC-Interval (Euclidean) | Policy on Euclidean interval embeddings |
| GCBC-Interval (Hyperbolic) | Policy on hyperbolic interval embeddings |

### Data Regimes

Experiments use trajectory data with varying "slack factors":
- **Tight** (1.2×): Nearly monotonic paths
- **Moderate** (1.5×): Some backtracking
- **Loose** (2.0×): Significant wandering
- **Very Loose** (3.0×): Heavy backtracking

### Running Experiments

```bash
# Generate trajectory data
python scripts/generate_data.py

# Train representations
python scripts/train_representations.py

# Train policies
python scripts/train_policies.py

# Evaluate
python scripts/evaluate.py

# Or run the full pipeline
python scripts/run_all_experiments.py
```

## Configuration

Edit `config/experiment_config.yaml` to customize:
- Environment parameters
- Data generation settings
- Model architectures
- Training hyperparameters
- Evaluation settings

## Testing

```bash
pytest tests/
```

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hyperbolic_planning,
  title={Hyperbolic Planning: Interval Representations for Goal-Conditioned Learning},
  author={Wojtek},
  year={2025},
  url={https://github.com/yourusername/hyperbolic-planning}
}
```
