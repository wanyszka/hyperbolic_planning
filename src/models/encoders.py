"""Interval encoders: Euclidean and Hyperbolic variants."""

import torch
import torch.nn as nn
from typing import List, Optional, Literal
from hypll.manifolds.poincare_ball import PoincareBall, Curvature
from hypll.tensors.manifold_tensor import ManifoldTensor
import hypll.nn as hnn

from .utils import manifold_map


class EuclideanIntervalEncoder(nn.Module):
    """
    Encode intervals [a, b] to Euclidean space using MLP.

    Architecture: 4 hidden layers with ReLU activations

    Args:
        embedding_dim: Output embedding dimension
        hidden_sizes: List of hidden layer sizes (default: [128, 128, 128, 128])
        activation: Activation function ("relu", "tanh", "gelu")
    """

    def __init__(
        self,
        embedding_dim: int = 2,
        hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_sizes = hidden_sizes or [128, 128, 128, 128]
        self.manifold = None  # No manifold for Euclidean

        # Build MLP layers
        layers = []
        in_dim = 2  # Input: [start, end]

        for hidden_dim in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            in_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(in_dim, embedding_dim))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation module by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]

    def forward(self, intervals: torch.Tensor) -> torch.Tensor:
        """
        Encode intervals to Euclidean space.

        Args:
            intervals: Tensor of shape (batch, 2) containing [start, end] pairs

        Returns:
            Tensor of shape (batch, embedding_dim)
        """
        return self.network(intervals)


class HyperbolicIntervalEncoder(nn.Module):
    """
    Encode intervals [a, b] to Poincaré ball using hybrid Euclidean-Hyperbolic MLP.

    Architecture: 2 Euclidean layers -> exponential map -> 2 Hyperbolic layers

    Args:
        embedding_dim: Output dimension in hyperbolic space
        c: Curvature of the Poincaré ball (default 1.0)
        euc_width: Width of Euclidean hidden layers
        hyp_width: Width of hyperbolic hidden layers
    """

    def __init__(
        self,
        embedding_dim: int = 2,
        c: float = 1.0,
        euc_width: int = 128,
        hyp_width: int = 128,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Create manifold with fixed curvature
        curvature = Curvature(value=c, requires_grad=False)
        self.manifold = PoincareBall(c=curvature)

        # Euclidean layers (first 2 layers)
        self.euc_layers = nn.Sequential(
            nn.Linear(2, euc_width),
            nn.ReLU(),
            nn.Linear(euc_width, hyp_width),
        )

        # Hyperbolic layers (last 2 layers)
        self.hyp_layer1 = hnn.HLinear(
            in_features=hyp_width,
            out_features=hyp_width,
            manifold=self.manifold,
        )
        self.hyp_relu = hnn.HReLU(manifold=self.manifold)
        self.hyp_layer2 = hnn.HLinear(
            in_features=hyp_width,
            out_features=embedding_dim,
            manifold=self.manifold,
        )

    def forward(self, intervals: torch.Tensor) -> ManifoldTensor:
        """
        Encode intervals to hyperbolic space.

        Args:
            intervals: Tensor of shape (batch, 2) containing [start, end] pairs

        Returns:
            ManifoldTensor of embeddings in Poincaré ball
        """
        # Euclidean feature extraction
        x = self.euc_layers(intervals)

        # Map to hyperbolic space
        x = manifold_map(x, self.manifold)

        # Hyperbolic processing
        x = self.hyp_relu(self.hyp_layer1(x))
        x = self.hyp_layer2(x)

        return x

    def get_euclidean_embeddings(self, intervals: torch.Tensor) -> torch.Tensor:
        """
        Get the underlying Euclidean coordinates of hyperbolic embeddings.

        Args:
            intervals: Tensor of shape (batch, 2)

        Returns:
            Tensor of shape (batch, embedding_dim) - Euclidean coordinates in Poincaré ball
        """
        emb = self.forward(intervals)
        if isinstance(emb, ManifoldTensor):
            return emb.tensor
        return emb


def create_encoder(
    geometry: Literal["euclidean", "hyperbolic"],
    embedding_dim: int = 2,
    hidden_sizes: Optional[List[int]] = None,
    curvature: float = 1.0,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create an interval encoder.

    Args:
        geometry: "euclidean" or "hyperbolic"
        embedding_dim: Output embedding dimension
        hidden_sizes: Hidden layer sizes (for Euclidean)
        curvature: Curvature for hyperbolic space
        **kwargs: Additional arguments passed to encoder

    Returns:
        Encoder module
    """
    if geometry == "euclidean":
        return EuclideanIntervalEncoder(
            embedding_dim=embedding_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )
    elif geometry == "hyperbolic":
        # For hyperbolic, use the standard architecture
        euc_width = hidden_sizes[0] if hidden_sizes else 128
        hyp_width = hidden_sizes[-1] if hidden_sizes else 128
        return HyperbolicIntervalEncoder(
            embedding_dim=embedding_dim,
            c=curvature,
            euc_width=euc_width,
            hyp_width=hyp_width,
        )
    else:
        raise ValueError(f"Unknown geometry: {geometry}")
