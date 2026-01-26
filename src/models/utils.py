"""Model utilities for hyperbolic operations."""

import torch
from hypll.manifolds.poincare_ball import PoincareBall
from hypll.tensors import TangentTensor
from hypll.tensors.manifold_tensor import ManifoldTensor


def manifold_map(x: torch.Tensor, manifold: PoincareBall) -> ManifoldTensor:
    """
    Map Euclidean tensor to hyperbolic manifold via exponential map.

    Args:
        x: Euclidean tensor
        manifold: Poincaré ball manifold

    Returns:
        ManifoldTensor on the Poincaré ball
    """
    tangents = TangentTensor(x, man_dim=-1, manifold=manifold)
    return manifold.expmap(tangents)


def is_hyperbolic_model(model: torch.nn.Module) -> bool:
    """
    Check if a model uses hyperbolic embeddings.

    Args:
        model: PyTorch model

    Returns:
        True if model has a non-None manifold attribute
    """
    return hasattr(model, 'manifold') and model.manifold is not None


def extract_tensor(x) -> torch.Tensor:
    """
    Extract underlying tensor from ManifoldTensor or return as-is.

    Args:
        x: ManifoldTensor or torch.Tensor

    Returns:
        torch.Tensor
    """
    if isinstance(x, ManifoldTensor):
        return x.tensor
    return x


def get_model_geometry(model: torch.nn.Module) -> str:
    """
    Get the geometry type of a model.

    Args:
        model: PyTorch model

    Returns:
        "hyperbolic" or "euclidean"
    """
    if is_hyperbolic_model(model):
        return "hyperbolic"
    return "euclidean"
