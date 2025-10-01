"""
Transformer Language Model implementation from scratch.
Based on "Attention is All You Need" (Vaswani et al., 2017)
with 2025 improvements including MoE and long-context support.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig
from .moe import MoELayer


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    Implements scaled dot-product attention across multiple heads.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_k

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape to (batch, n_heads, seq_len, d_k)
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # scores: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # output: (batch, n_heads, seq_len, d_k)
        output = torch.matmul(attn_weights, V)

        # Concatenate heads and project
        # (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position information.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)

        # Create positional encoding matrix
        pe = torch.zeros(config.max_seq_len, config.d_model)
        position = torch.arange(0, config.max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() *
                            -(math.log(10000.0) / config.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer decoder block with:
    - Multi-head self-attention
    - Feed-forward network or MoE
    - Layer normalization and residual connections
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)

        # Use MoE or standard FFN based on config
        if config.use_moe:
            self.ffn = MoELayer(config)
        else:
            self.ffn = FeedForward(config)

        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class TransformerLM(nn.Module):
    """
    Complete Transformer Language Model.
    Implements autoregressive language modeling with causal masking.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = PositionalEncoding(config)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Output projection
        self.norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

        # Causal mask for autoregressive modeling
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            )
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following best practices."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        """
        Forward pass.

        Args:
            input_ids: (batch, seq_len) - input token IDs
            targets: (batch, seq_len) - target token IDs for loss computation

        Returns:
            logits: (batch, seq_len, vocab_size) - output logits
            loss: scalar tensor (if targets provided)
        """
        batch_size, seq_len = input_ids.shape

        # Get causal mask for this sequence length
        mask = self.causal_mask[:, :, :seq_len, :seq_len]

        # Embeddings
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output projection
        x = self.norm(x)
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        Generate text autoregressively.

        Args:
            input_ids: (batch, seq_len) - prompt tokens
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k most likely tokens

        Returns:
            generated: (batch, seq_len + max_new_tokens) - generated sequence
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to max sequence length
            input_crop = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(input_crop)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
