"""
Model configuration based on 2025 LLM trends:
- Transformer architecture (Attention is All You Need)
- Mixture of Experts for efficiency
- Long-context support
"""

class ModelConfig:
    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 2048,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_moe: bool = True,
        n_experts: int = 4,
        expert_capacity: int = 2,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.use_moe = use_moe
        self.n_experts = n_experts
        self.expert_capacity = expert_capacity

        # Derived values
        self.d_k = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

    def __repr__(self):
        return f"ModelConfig(d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}, use_moe={self.use_moe})"
