"""
Mixture of Experts (MoE) implementation.
Based on recent MoE research including Switch Transformers and ExpertRAG (2025).
Uses top-k routing to select experts for each token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig


class Expert(nn.Module):
    """
    Single expert network (a simple FFN).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with top-k routing.

    For each token, the router selects top-k experts to process it.
    This allows for conditional computation and increased model capacity
    without proportional increase in computation cost.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.expert_capacity = config.expert_capacity
        self.d_model = config.d_model

        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(self.n_experts)
        ])

        # Router network (gating function)
        self.router = nn.Linear(config.d_model, self.n_experts)

    def forward(self, x):
        """
        Forward pass with top-k expert routing.

        Args:
            x: (batch, seq_len, d_model) - input tensor

        Returns:
            output: (batch, seq_len, d_model) - expert-mixed output
        """
        batch_size, seq_len, d_model = x.shape

        # Flatten batch and sequence dimensions for routing
        x_flat = x.view(-1, d_model)  # (batch*seq_len, d_model)

        # Router computes gating scores for each expert
        router_logits = self.router(x_flat)  # (batch*seq_len, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(
            router_probs,
            k=min(self.expert_capacity, self.n_experts),
            dim=-1
        )

        # Normalize selected expert probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process through selected experts
        for i in range(self.expert_capacity):
            # Get expert indices and probabilities for this position
            expert_idx = top_k_indices[:, i]  # (batch*seq_len,)
            expert_prob = top_k_probs[:, i].unsqueeze(-1)  # (batch*seq_len, 1)

            # Process through each expert
            for expert_id in range(self.n_experts):
                # Mask for tokens routed to this expert
                mask = (expert_idx == expert_id)

                if mask.any():
                    # Get tokens for this expert
                    expert_input = x_flat[mask]

                    # Process through expert
                    expert_output = self.experts[expert_id](expert_input)

                    # Weight by routing probability and accumulate
                    expert_prob_masked = expert_prob[mask]
                    output[mask] += expert_output * expert_prob_masked

        # Reshape back to original dimensions
        output = output.view(batch_size, seq_len, d_model)

        return output


class SwitchMoELayer(nn.Module):
    """
    Simplified Switch Transformer style MoE.
    Routes each token to exactly ONE expert (top-1 routing).
    More efficient than top-k routing.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.d_model = config.d_model

        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(self.n_experts)
        ])

        # Router network
        self.router = nn.Linear(config.d_model, self.n_experts)

    def forward(self, x):
        """
        Forward pass with top-1 (switch) routing.

        Args:
            x: (batch, seq_len, d_model) - input tensor

        Returns:
            output: (batch, seq_len, d_model) - expert output
        """
        batch_size, seq_len, d_model = x.shape

        # Flatten
        x_flat = x.view(-1, d_model)

        # Router computes gating scores
        router_logits = self.router(x_flat)

        # Select top-1 expert for each token
        expert_indices = torch.argmax(router_logits, dim=-1)  # (batch*seq_len,)

        # Get routing probabilities for load balancing loss
        router_probs = F.softmax(router_logits, dim=-1)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process through selected experts
        for expert_id in range(self.n_experts):
            # Mask for tokens routed to this expert
            mask = (expert_indices == expert_id)

            if mask.any():
                # Process tokens through this expert
                expert_input = x_flat[mask]
                expert_output = self.experts[expert_id](expert_input)
                output[mask] = expert_output

        # Reshape
        output = output.view(batch_size, seq_len, d_model)

        return output
