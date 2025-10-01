#!/usr/bin/env python3
"""
Quick test script to verify model implementation.
Tests basic functionality without training.
"""

import torch
from src.model import ModelConfig, TransformerLM
from src.tokenizer import get_tokenizer


def test_model_creation():
    """Test model can be created."""
    print("Testing model creation...")
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        use_moe=False,
    )
    model = TransformerLM(config)
    print(f"✓ Model created successfully")
    print(f"  Parameters: {model.count_parameters():,}")
    return model


def test_forward_pass(model):
    """Test forward pass."""
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    target_ids = torch.randint(0, 1000, (batch_size, seq_len))

    logits, loss = model(input_ids, target_ids)

    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    assert logits.shape == (batch_size, seq_len, 1000), "Logits shape mismatch"
    assert loss.item() > 0, "Loss should be positive"


def test_generation(model):
    """Test text generation."""
    print("\nTesting generation...")
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    output = model.generate(input_ids, max_new_tokens=10, temperature=1.0)

    print(f"✓ Generation successful")
    print(f"  Input length: {input_ids.shape[1]}")
    print(f"  Output length: {output.shape[1]}")
    print(f"  Generated tokens: {output[0].tolist()}")

    assert output.shape[1] == input_ids.shape[1] + 10, "Output length mismatch"


def test_moe_model():
    """Test MoE model."""
    print("\nTesting MoE model...")
    config = ModelConfig(
        vocab_size=1000,
        max_seq_len=128,
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        use_moe=True,
        n_experts=4,
    )
    model = TransformerLM(config)
    print(f"✓ MoE model created")
    print(f"  Parameters: {model.count_parameters():,}")

    # Forward pass
    input_ids = torch.randint(0, 1000, (2, 32))
    target_ids = torch.randint(0, 1000, (2, 32))
    logits, loss = model(input_ids, target_ids)

    print(f"✓ MoE forward pass successful")
    print(f"  Loss: {loss.item():.4f}")


def test_tokenizer():
    """Test tokenizer."""
    print("\nTesting tokenizer...")
    tokenizer = get_tokenizer("tiktoken")

    text = "Hello, world! This is a test."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    print(f"✓ Tokenizer working")
    print(f"  Original: {text}")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: {decoded}")
    print(f"  Vocab size: {len(tokenizer)}")

    assert decoded == text, "Tokenization roundtrip failed"


def main():
    print("=" * 70)
    print("  Model Implementation Test")
    print("=" * 70)
    print()

    try:
        # Test basic model
        model = test_model_creation()
        test_forward_pass(model)
        test_generation(model)

        # Test MoE model
        test_moe_model()

        # Test tokenizer
        test_tokenizer()

        print("\n" + "=" * 70)
        print("  ✓ All tests passed!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Download data: python scripts/download_data.py --dataset tinystories")
        print("  2. Train model: python src/train.py --dataset tinystories --epochs 5")
        print("  3. Chat: python src/chat.py --checkpoint checkpoints/best_model.pt")
        print()

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
