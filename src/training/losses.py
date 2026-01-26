"""Loss functions for contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss using Euclidean distances.

    Args:
        anchor: Anchor embeddings (batch, dim)
        positive: Positive embeddings (batch, dim)
        negatives: Negative embeddings (batch, num_neg, dim)
        temperature: Temperature scaling factor

    Returns:
        Scalar loss tensor
    """
    pos_dist = torch.norm(anchor - positive, dim=-1)
    neg_dist = torch.norm(anchor.unsqueeze(1) - negatives, dim=-1)

    pos_sim = -pos_dist / temperature
    neg_sim = -neg_dist / temperature

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)


def hyperbolic_info_nce_loss(
    anchor,
    positive,
    negatives,
    manifold,
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss using hyperbolic distances.

    Args:
        anchor: Anchor embeddings (ManifoldTensor)
        positive: Positive embeddings (ManifoldTensor)
        negatives: Negative embeddings (ManifoldTensor with shape batch, num_neg, dim)
        manifold: Poincaré ball manifold
        temperature: Temperature scaling factor

    Returns:
        Scalar loss tensor
    """
    from hypll.tensors.manifold_tensor import ManifoldTensor

    num_neg = negatives.tensor.shape[1] if isinstance(negatives, ManifoldTensor) else negatives.shape[1]

    # Compute distances
    pos_dist = manifold.dist(x=anchor, y=positive)
    anchor_expanded = _expand_for_negatives(anchor, num_neg, manifold)
    neg_dist = manifold.dist(x=anchor_expanded, y=negatives)

    # Extract tensors and squeeze
    pos_dist = _to_tensor(pos_dist).squeeze(-1) if _to_tensor(pos_dist).dim() > 1 else _to_tensor(pos_dist)
    neg_dist = _to_tensor(neg_dist).squeeze(-1) if _to_tensor(neg_dist).dim() > 2 else _to_tensor(neg_dist)

    if torch.isnan(pos_dist).any() or torch.isnan(neg_dist).any():
        print("WARNING: NaN in distances")
        device = anchor.tensor.device if isinstance(anchor, ManifoldTensor) else anchor.device
        return torch.tensor(0.0, device=device, requires_grad=True)

    pos_sim = -pos_dist / temperature
    neg_sim = -neg_dist / temperature

    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    batch_size = anchor.tensor.shape[0] if isinstance(anchor, ManifoldTensor) else anchor.shape[0]
    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)


def _expand_for_negatives(anchor, num_neg: int, manifold):
    """Expand anchor for negative comparisons."""
    from hypll.tensors.manifold_tensor import ManifoldTensor

    if isinstance(anchor, ManifoldTensor):
        expanded = anchor.tensor.unsqueeze(1).expand(-1, num_neg, -1)
        return ManifoldTensor(expanded, manifold=manifold)
    return anchor.unsqueeze(1).expand(-1, num_neg, -1)


def _to_tensor(x):
    """Convert ManifoldTensor to regular tensor if needed."""
    from hypll.tensors.manifold_tensor import ManifoldTensor

    if isinstance(x, ManifoldTensor):
        return x.tensor
    return x
