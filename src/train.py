#!/usr/bin/env python3
"""
Training script for the Transformer LM.
Supports distributed training, mixed precision, and gradient accumulation.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from tqdm import tqdm
import math

from model import ModelConfig, TransformerLM
from tokenizer import get_tokenizer


class TextDataset(Dataset):
    """
    Dataset for language modeling.
    Tokenizes text on-the-fly and creates sequences of fixed length.
    """
    def __init__(self, dataset, tokenizer, seq_len=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get text
        text = self.dataset[idx]["text"]

        # Tokenize
        tokens = self.tokenizer.encode(text)

        # Truncate or pad to seq_len + 1 (we need input and target)
        if len(tokens) > self.seq_len + 1:
            tokens = tokens[:self.seq_len + 1]
        else:
            # Pad with zeros
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))

        # Input and target
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)

        return input_ids, target_ids


def train_epoch(model, dataloader, optimizer, device, grad_accum_steps=1):
    """
    Train for one epoch.

    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        grad_accum_steps: Number of gradient accumulation steps

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training")
    for i, (input_ids, target_ids) in enumerate(pbar):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward pass
        logits, loss = model(input_ids, target_ids)

        # Scale loss by accumulation steps
        loss = loss / grad_accum_steps
        loss.backward()

        # Update weights every grad_accum_steps
        if (i + 1) % grad_accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model on validation set.

    Args:
        model: The model to evaluate
        dataloader: Validation data loader
        device: Device to evaluate on

    Returns:
        Average loss and perplexity
    """
    model.eval()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Evaluating")
    for input_ids, target_ids in pbar:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits, loss = model(input_ids, target_ids)
        total_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": model.config.__dict__,
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Saved checkpoint to {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]


def main():
    parser = argparse.ArgumentParser(description="Train Transformer LM")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--dataset", type=str, default="tinystories", help="Dataset name")

    # Model arguments
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=2048, help="FFN dimension")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--use-moe", action="store_true", help="Use Mixture of Experts")
    parser.add_argument("--n-experts", type=int, default=4, help="Number of experts (if MoE)")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")

    # Other arguments
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = get_tokenizer("tiktoken")

    # Load dataset
    print(f"Loading dataset from {args.data_dir}/{args.dataset}...")
    dataset = load_from_disk(os.path.join(args.data_dir, args.dataset))

    # Split into train/val if not already split
    if "train" not in dataset:
        dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
    else:
        train_dataset = dataset["train"]
        val_dataset = dataset.get("validation", dataset.get("test", None))

    # Create datasets and dataloaders
    train_ds = TextDataset(train_dataset, tokenizer, args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_loader = None
    if val_dataset is not None:
        val_ds = TextDataset(val_dataset, tokenizer, args.max_seq_len)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model
    print("Initializing model...")
    config = ModelConfig(
        vocab_size=len(tokenizer),
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        use_moe=args.use_moe,
        n_experts=args.n_experts,
    )

    model = TransformerLM(config).to(args.device)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model config: {config}")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - args.warmup_steps) / (len(train_loader) * args.epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1

    # Training loop
    print(f"\nStarting training on {args.device}...")
    print(f"Training samples: {len(train_ds)}")
    if val_loader:
        print(f"Validation samples: {len(val_ds)}")
    print()

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, args.device, args.grad_accum_steps)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        if val_loader and (epoch + 1) % args.eval_interval == 0:
            val_loss, perplexity = evaluate(model, val_loader, args.device)
            print(f"Validation loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                save_checkpoint(model, optimizer, epoch, val_loss, save_path)

        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            save_checkpoint(model, optimizer, epoch, train_loss, save_path)

        # Update learning rate
        scheduler.step()

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    save_checkpoint(model, optimizer, args.epochs - 1, train_loss, final_path)

    print("\n✓ Training complete!")
    print(f"Checkpoints saved to: {os.path.abspath(args.checkpoint_dir)}")


if __name__ == "__main__":
    main()
