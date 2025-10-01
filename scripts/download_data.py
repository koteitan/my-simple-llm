#!/usr/bin/env python3
"""
Data downloading script for LLM training.
Downloads and prepares datasets from Hugging Face.

Supported datasets:
- openwebtext: OpenWebText corpus (GPT-2 training data replica)
- wikitext: WikiText-103 dataset
- tinystories: TinyStories dataset (smaller, good for testing)
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_openwebtext(data_dir: str, num_samples: int = None):
    """
    Download OpenWebText dataset.

    Args:
        data_dir: Directory to save the data
        num_samples: Maximum number of samples to download (None = all)
    """
    print("Downloading OpenWebText dataset...")
    dataset = load_dataset("openwebtext", split="train")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Save to disk
    save_path = os.path.join(data_dir, "openwebtext")
    dataset.save_to_disk(save_path)
    print(f"✓ Saved OpenWebText to {save_path}")
    print(f"  Total samples: {len(dataset)}")

    return dataset


def download_wikitext(data_dir: str):
    """
    Download WikiText-103 dataset.

    Args:
        data_dir: Directory to save the data
    """
    print("Downloading WikiText-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")

    # Save train/validation/test splits
    save_path = os.path.join(data_dir, "wikitext-103")
    dataset.save_to_disk(save_path)
    print(f"✓ Saved WikiText-103 to {save_path}")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])} samples")

    return dataset


def download_tinystories(data_dir: str, num_samples: int = None):
    """
    Download TinyStories dataset (smaller dataset for testing).

    Args:
        data_dir: Directory to save the data
        num_samples: Maximum number of samples to download
    """
    print("Downloading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    save_path = os.path.join(data_dir, "tinystories")
    dataset.save_to_disk(save_path)
    print(f"✓ Saved TinyStories to {save_path}")
    print(f"  Total samples: {len(dataset)}")

    return dataset


def download_custom_text(data_dir: str, file_path: str):
    """
    Process a custom text file.

    Args:
        data_dir: Directory to save processed data
        file_path: Path to input text file
    """
    print(f"Processing custom text file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into chunks (simple approach)
    chunk_size = 1000  # characters per chunk
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Create dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": chunks})

    save_path = os.path.join(data_dir, "custom")
    dataset.save_to_disk(save_path)
    print(f"✓ Saved custom dataset to {save_path}")
    print(f"  Total chunks: {len(dataset)}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Download training data for LLM")
    parser.add_argument(
        "--dataset",
        type=str,
        default="tinystories",
        choices=["openwebtext", "wikitext", "tinystories", "custom"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to save downloaded data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of samples to download (for testing)"
    )
    parser.add_argument(
        "--custom-file",
        type=str,
        default=None,
        help="Path to custom text file (if dataset=custom)"
    )

    args = parser.parse_args()

    # Create data directory
    os.makedirs(args.data_dir, exist_ok=True)

    # Download dataset
    if args.dataset == "openwebtext":
        download_openwebtext(args.data_dir, args.num_samples)
    elif args.dataset == "wikitext":
        download_wikitext(args.data_dir)
    elif args.dataset == "tinystories":
        download_tinystories(args.data_dir, args.num_samples)
    elif args.dataset == "custom":
        if not args.custom_file:
            print("Error: --custom-file is required when using dataset=custom")
            return
        download_custom_text(args.data_dir, args.custom_file)

    print("\n✓ Download complete!")
    print(f"Data saved to: {os.path.abspath(args.data_dir)}")
    print("\nNext steps:")
    print(f"  python src/train.py --data-dir {args.data_dir} --dataset {args.dataset}")


if __name__ == "__main__":
    main()
