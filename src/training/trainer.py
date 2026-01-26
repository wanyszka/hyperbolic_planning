"""Training module for interval encoders and policies."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Callable, Tuple, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from copy import deepcopy

from .losses import info_nce_loss, hyperbolic_info_nce_loss


@dataclass
class TrainingConfig:
    """Configuration for training."""

    num_epochs: int = 100
    batch_size: int = 256
    lr: float = 0.001
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    temperature: float = 0.1
    gradient_clip: float = 1.0
    early_stopping_patience: int = 20
    print_every: int = 5
    save_every: int = 10


@dataclass
class TrainingResult:
    """Result container for training."""

    train_losses: List[float]
    val_losses: List[float]
    best_epoch: int
    best_val_loss: float
    final_model_state: Dict


def train_model(
    model: nn.Module,
    dataset,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.0005,
    temperature: float = 0.5,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    verbose: bool = True,
):
    """
    Train an interval encoder model (backward compatible API).

    Automatically detects hyperbolic models and uses appropriate optimizer.

    Args:
        model: Encoder model (Euclidean or Hyperbolic)
        dataset: IntervalDataset instance
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        temperature: Temperature for InfoNCE loss
        weight_decay: Weight decay for optimizer
        device: Compute device (auto-detected if None)
        verbose: Print progress

    Returns:
        List of epoch losses
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    is_hyperbolic = hasattr(model, 'manifold') and model.manifold is not None
    optimizer = _create_optimizer(model, lr, weight_decay, is_hyperbolic)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if verbose:
        opt_name = "RiemannianAdam" if is_hyperbolic else "Adam"
        print(f"Training {'hyperbolic' if is_hyperbolic else 'Euclidean'} model with {opt_name}")

    losses = []
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = _train_encoder_epoch(model, dataloader, optimizer, temperature, device, is_hyperbolic)

        if epoch_loss is not None:
            losses.append(epoch_loss)
            scheduler.step()

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return losses


def train_encoder(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    config: Optional[TrainingConfig] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    callbacks: Optional[List[Callable]] = None,
    save_dir: Optional[str] = None,
) -> TrainingResult:
    """
    Train an interval encoder with validation and early stopping.

    Args:
        model: Encoder model
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        config: Training configuration
        device: Compute device
        verbose: Print progress
        callbacks: List of callback functions called each epoch
        save_dir: Directory to save checkpoints

    Returns:
        TrainingResult with losses and best model state
    """
    config = config or TrainingConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    is_hyperbolic = hasattr(model, 'manifold') and model.manifold is not None

    optimizer = _create_optimizer(model, config.lr, config.weight_decay, is_hyperbolic)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.lr_min
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )

    if verbose:
        opt_name = "RiemannianAdam" if is_hyperbolic else "Adam"
        print(f"Training {'hyperbolic' if is_hyperbolic else 'Euclidean'} encoder with {opt_name}")
        print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset) if val_dataset else 0}")

    # Training state
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = _train_encoder_epoch(
            model, train_loader, optimizer, config.temperature, device, is_hyperbolic
        )

        if train_loss is None:
            continue

        train_losses.append(train_loss)
        scheduler.step()

        # Validation
        val_loss = None
        if val_loader:
            model.eval()
            val_loss = _validate_encoder(
                model, val_loader, config.temperature, device, is_hyperbolic
            )
            val_losses.append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                patience_counter = 0
            else:
                patience_counter += 1

        # Callbacks
        if callbacks:
            for callback in callbacks:
                callback(epoch, train_loss, val_loss, model)

        # Logging
        if verbose and (epoch + 1) % config.print_every == 0:
            val_str = f", Val Loss: {val_loss:.4f}" if val_loss else ""
            print(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.4f}{val_str}")

        # Save checkpoint
        if save_dir and (epoch + 1) % config.save_every == 0:
            _save_checkpoint(model, optimizer, epoch, train_loss, save_dir)

        # Early stopping
        if patience_counter >= config.early_stopping_patience and val_loader:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model if validation was used
    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        best_model_state = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        best_epoch = len(train_losses) - 1
        best_val_loss = val_losses[-1] if val_losses else train_losses[-1]

    return TrainingResult(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        final_model_state=best_model_state,
    )


def train_policy(
    policy: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    config: Optional[TrainingConfig] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    save_dir: Optional[str] = None,
) -> TrainingResult:
    """
    Train a policy network using behavioral cloning.

    Args:
        policy: Policy network (GCBCSinglePolicy or GCBCIntervalPolicy)
        train_dataset: PolicyTrainingDataset
        val_dataset: Validation PolicyTrainingDataset
        config: Training configuration
        device: Compute device
        verbose: Print progress
        save_dir: Directory to save checkpoints

    Returns:
        TrainingResult with losses and best model state
    """
    config = config or TrainingConfig(num_epochs=50)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    policy = policy.to(device)

    # Only optimize policy parameters (encoder may be frozen)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.lr_min
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        )

    if verbose:
        n_trainable = sum(p.numel() for p in trainable_params)
        print(f"Training policy with {n_trainable:,} trainable parameters")
        print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset) if val_dataset else 0}")

    # Training state
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(config.num_epochs):
        # Training
        policy.train()
        train_loss, train_acc = _train_policy_epoch(policy, train_loader, optimizer, device)

        if train_loss is None:
            continue

        train_losses.append(train_loss)
        scheduler.step()

        # Validation
        val_loss = None
        val_acc = None
        if val_loader:
            policy.eval()
            val_loss, val_acc = _validate_policy(policy, val_loader, device)
            val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = deepcopy({k: v.cpu() for k, v in policy.state_dict().items()})
                patience_counter = 0
            else:
                patience_counter += 1

        # Logging
        if verbose and (epoch + 1) % config.print_every == 0:
            val_str = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}" if val_loss else ""
            print(f"Epoch {epoch+1}/{config.num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}{val_str}")

        # Save checkpoint
        if save_dir and (epoch + 1) % config.save_every == 0:
            _save_checkpoint(policy, optimizer, epoch, train_loss, save_dir, prefix="policy")

        # Early stopping
        if patience_counter >= config.early_stopping_patience and val_loader:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state:
        policy.load_state_dict(best_model_state)
    else:
        best_model_state = deepcopy({k: v.cpu() for k, v in policy.state_dict().items()})
        best_epoch = len(train_losses) - 1
        best_val_loss = val_losses[-1] if val_losses else train_losses[-1]

    return TrainingResult(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        final_model_state=best_model_state,
    )


def _create_optimizer(model: nn.Module, lr: float, weight_decay: float, is_hyperbolic: bool):
    """Create optimizer based on model type."""
    if is_hyperbolic:
        from hypll.optim import RiemannianAdam
        return RiemannianAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def _train_encoder_epoch(model, dataloader, optimizer, temperature, device, is_hyperbolic):
    """Train encoder for one epoch."""
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Handle both (anchor, positive, negatives) and (anchor, positive) formats
        if len(batch) == 3:
            anchor, positive, negatives = batch
            anchor = anchor.to(device)
            positive = positive.to(device)
            negatives = negatives.to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negatives_emb = _encode_negatives(model, negatives, is_hyperbolic)

            if is_hyperbolic:
                loss = hyperbolic_info_nce_loss(
                    anchor_emb, positive_emb, negatives_emb, model.manifold, temperature
                )
            else:
                loss = info_nce_loss(anchor_emb, positive_emb, negatives_emb, temperature)
        else:
            # In-batch contrastive: (anchor, positive) format
            anchor, positive = batch
            anchor = anchor.to(device)
            positive = positive.to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)

            # Use in-batch negatives
            if is_hyperbolic:
                loss = _in_batch_hyperbolic_loss(
                    anchor_emb, positive_emb, model.manifold, temperature
                )
            else:
                loss = _in_batch_euclidean_loss(anchor_emb, positive_emb, temperature)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else None


def _validate_encoder(model, dataloader, temperature, device, is_hyperbolic):
    """Validate encoder on a dataset."""
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                anchor, positive, negatives = batch
                anchor = anchor.to(device)
                positive = positive.to(device)
                negatives = negatives.to(device)

                anchor_emb = model(anchor)
                positive_emb = model(positive)
                negatives_emb = _encode_negatives(model, negatives, is_hyperbolic)

                if is_hyperbolic:
                    loss = hyperbolic_info_nce_loss(
                        anchor_emb, positive_emb, negatives_emb, model.manifold, temperature
                    )
                else:
                    loss = info_nce_loss(anchor_emb, positive_emb, negatives_emb, temperature)
            else:
                anchor, positive = batch
                anchor = anchor.to(device)
                positive = positive.to(device)

                anchor_emb = model(anchor)
                positive_emb = model(positive)

                if is_hyperbolic:
                    loss = _in_batch_hyperbolic_loss(
                        anchor_emb, positive_emb, model.manifold, temperature
                    )
                else:
                    loss = _in_batch_euclidean_loss(anchor_emb, positive_emb, temperature)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


def _train_policy_epoch(policy, dataloader, optimizer, device) -> Tuple[Optional[float], float]:
    """Train policy for one epoch. Returns (loss, accuracy)."""
    total_loss = 0.0
    correct = 0
    total = 0

    for states, goals, actions in dataloader:
        states = states.to(device)
        goals = goals.to(device)
        actions = actions.to(device)

        # Forward pass
        logits = policy(states, goals)
        loss = F.cross_entropy(logits, actions)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(actions)
        predictions = logits.argmax(dim=-1)
        correct += (predictions == actions).sum().item()
        total += len(actions)

    if total == 0:
        return None, 0.0

    return total_loss / total, correct / total


def _validate_policy(policy, dataloader, device) -> Tuple[float, float]:
    """Validate policy. Returns (loss, accuracy)."""
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for states, goals, actions in dataloader:
            states = states.to(device)
            goals = goals.to(device)
            actions = actions.to(device)

            logits = policy(states, goals)
            loss = F.cross_entropy(logits, actions)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * len(actions)
                predictions = logits.argmax(dim=-1)
                correct += (predictions == actions).sum().item()
                total += len(actions)

    if total == 0:
        return float('inf'), 0.0

    return total_loss / total, correct / total


def _encode_negatives(model, negatives, is_hyperbolic):
    """Encode and reshape negative samples."""
    batch_size, num_neg, _ = negatives.shape
    emb = model(negatives.view(-1, 2))

    if is_hyperbolic:
        from hypll.tensors.manifold_tensor import ManifoldTensor
        return ManifoldTensor(emb.tensor.view(batch_size, num_neg, -1), manifold=model.manifold)

    return emb.view(batch_size, num_neg, -1)


def _in_batch_euclidean_loss(anchor_emb, positive_emb, temperature):
    """Compute InfoNCE loss with in-batch negatives (Euclidean)."""
    batch_size = anchor_emb.shape[0]

    # Compute all pairwise distances
    # anchor_emb: (B, D), positive_emb: (B, D)
    # distances: (B, B) where [i,j] = ||anchor_i - positive_j||
    diff = anchor_emb.unsqueeze(1) - positive_emb.unsqueeze(0)  # (B, B, D)
    distances = torch.norm(diff, dim=-1)  # (B, B)

    # Convert to similarities (negative distance)
    similarities = -distances / temperature

    # Labels: diagonal elements are positives
    labels = torch.arange(batch_size, device=anchor_emb.device)

    return F.cross_entropy(similarities, labels)


def _in_batch_hyperbolic_loss(anchor_emb, positive_emb, manifold, temperature):
    """Compute InfoNCE loss with in-batch negatives (Hyperbolic)."""
    from hypll.tensors.manifold_tensor import ManifoldTensor

    batch_size = anchor_emb.tensor.shape[0]

    # Extract tensors
    anchor_tensor = anchor_emb.tensor  # (B, D)
    positive_tensor = positive_emb.tensor  # (B, D)

    # Compute pairwise hyperbolic distances
    # We need to compute d_H(anchor_i, positive_j) for all i, j
    distances = torch.zeros(batch_size, batch_size, device=anchor_tensor.device)

    for i in range(batch_size):
        anchor_i = ManifoldTensor(anchor_tensor[i:i+1].expand(batch_size, -1), manifold=manifold)
        dist_i = manifold.dist(anchor_i, positive_emb)
        if isinstance(dist_i, ManifoldTensor):
            distances[i] = dist_i.tensor.squeeze()
        else:
            distances[i] = dist_i.squeeze()

    # Convert to similarities
    similarities = -distances / temperature

    # Labels
    labels = torch.arange(batch_size, device=anchor_tensor.device)

    return F.cross_entropy(similarities, labels)


def _save_checkpoint(model, optimizer, epoch, loss, save_dir, prefix="encoder"):
    """Save training checkpoint."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    torch.save(checkpoint, save_path / f"{prefix}_epoch_{epoch+1}.pt")


def load_checkpoint(model, checkpoint_path, device="cpu"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('loss', 0)


if __name__ == "__main__":
    # Test the training functions
    print("Testing training module...")

    from src.models import EuclideanIntervalEncoder, GCBCSinglePolicy
    from src.core.datasets.trajectory_dataset import TrajectoryContrastiveDataset, PolicyTrainingDataset

    device = "cpu"

    # Create dummy trajectories
    np.random.seed(42)
    dummy_trajectories = []
    dummy_actions = []
    for _ in range(100):
        length = np.random.randint(50, 100)
        traj = np.linspace(0, 1, length).tolist()
        dummy_trajectories.append(traj)
        dummy_actions.append([2] * (length - 1))  # Always go right

    # Test encoder training
    print("\n1. Testing encoder training...")
    train_dataset = TrajectoryContrastiveDataset(
        trajectories=dummy_trajectories[:80],
        num_samples=500,
        n_negatives=5,
    )
    val_dataset = TrajectoryContrastiveDataset(
        trajectories=dummy_trajectories[80:],
        num_samples=100,
        n_negatives=5,
    )

    encoder = EuclideanIntervalEncoder(embedding_dim=2).to(device)
    training_config = TrainingConfig(num_epochs=10, batch_size=32, print_every=2)

    result = train_encoder(
        encoder, train_dataset, val_dataset,
        config=training_config, device=device, verbose=True
    )
    print(f"  Best epoch: {result.best_epoch}, Best val loss: {result.best_val_loss:.4f}")

    # Test policy training
    print("\n2. Testing policy training...")
    policy_train = PolicyTrainingDataset(
        trajectories=dummy_trajectories[:80],
        actions=dummy_actions[:80],
    )
    policy_val = PolicyTrainingDataset(
        trajectories=dummy_trajectories[80:],
        actions=dummy_actions[80:],
    )

    policy = GCBCSinglePolicy(n_actions=3).to(device)
    policy_config = TrainingConfig(num_epochs=10, batch_size=32, print_every=2)

    result = train_policy(
        policy, policy_train, policy_val,
        config=policy_config, device=device, verbose=True
    )
    print(f"  Best epoch: {result.best_epoch}, Best val loss: {result.best_val_loss:.4f}")

    print("\nAll tests passed!")
