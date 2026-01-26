"""Policy networks for goal-conditioned behavioral cloning."""

import torch
import torch.nn as nn
from typing import List, Optional
from hypll.tensors.manifold_tensor import ManifoldTensor


class PolicyNetwork(nn.Module):
    """
    Policy network for goal-conditioned behavioral cloning.

    Maps (state, goal) or interval embeddings to action logits.

    Args:
        input_dim: Input dimension (2 for GCBC-Single, embedding_dim for GCBC-Interval)
        n_actions: Number of discrete actions (default 3: left, stay, right)
        hidden_sizes: List of hidden layer sizes
        activation: Activation function name
    """

    def __init__(
        self,
        input_dim: int,
        n_actions: int = 3,
        hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_actions = n_actions
        self.hidden_sizes = hidden_sizes or [64, 64]

        # Build MLP
        layers = []
        in_dim = input_dim

        for hidden_dim in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, n_actions))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation module by name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        return activations.get(name, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Action logits of shape (batch, n_actions)
        """
        return self.network(x)

    def get_action(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Get action from input.

        Args:
            x: Input tensor
            deterministic: If True, return argmax; if False, sample from distribution

        Returns:
            Action indices
        """
        logits = self.forward(x)

        if deterministic:
            return logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class GCBCSinglePolicy(nn.Module):
    """
    GCBC-Single: Goal-conditioned policy with direct (state, goal) input.

    Input: concatenation of state s and goal g (dimension 2)
    """

    def __init__(
        self,
        n_actions: int = 3,
        hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
    ):
        super().__init__()
        self.policy = PolicyNetwork(
            input_dim=2,  # [state, goal]
            n_actions=n_actions,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Current state tensor (batch, 1) or (batch,)
            goal: Goal state tensor (batch, 1) or (batch,)

        Returns:
            Action logits (batch, n_actions)
        """
        # Ensure proper shape
        if state.dim() == 1:
            state = state.unsqueeze(1)
        if goal.dim() == 1:
            goal = goal.unsqueeze(1)

        x = torch.cat([state, goal], dim=-1)
        return self.policy(x)

    def get_action(
        self, state: torch.Tensor, goal: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        """Get action for given state and goal."""
        if state.dim() == 1:
            state = state.unsqueeze(1)
        if goal.dim() == 1:
            goal = goal.unsqueeze(1)

        x = torch.cat([state, goal], dim=-1)
        return self.policy.get_action(x, deterministic)


class GCBCIntervalPolicy(nn.Module):
    """
    GCBC-Interval: Goal-conditioned policy with learned interval embeddings.

    Input: φ(state, goal) from a pretrained encoder
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_actions: int = 3,
        hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        # Get embedding dimension from encoder
        embedding_dim = encoder.embedding_dim

        self.policy = PolicyNetwork(
            input_dim=embedding_dim,
            n_actions=n_actions,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Current state tensor (batch, 1) or (batch,)
            goal: Goal state tensor (batch, 1) or (batch,)

        Returns:
            Action logits (batch, n_actions)
        """
        # Ensure proper shape
        if state.dim() == 1:
            state = state.unsqueeze(1)
        if goal.dim() == 1:
            goal = goal.unsqueeze(1)

        # Create interval input for encoder
        intervals = torch.cat([state, goal], dim=-1)

        # Get embedding
        if self.freeze_encoder:
            with torch.no_grad():
                embedding = self.encoder(intervals)
        else:
            embedding = self.encoder(intervals)

        # Extract tensor if ManifoldTensor
        if isinstance(embedding, ManifoldTensor):
            embedding = embedding.tensor

        return self.policy(embedding)

    def get_action(
        self, state: torch.Tensor, goal: torch.Tensor, deterministic: bool = True
    ) -> torch.Tensor:
        """Get action for given state and goal."""
        logits = self.forward(state, goal)

        if deterministic:
            return logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
