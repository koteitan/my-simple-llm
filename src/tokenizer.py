"""
Tokenizer wrapper using tiktoken (OpenAI's BPE tokenizer).
Provides a simple interface for encoding and decoding text.
"""

import tiktoken
from typing import List, Union


class Tokenizer:
    """
    Wrapper around tiktoken for text tokenization.
    Uses GPT-2 encoding by default (50257 tokens).
    """
    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initialize tokenizer.

        Args:
            encoding_name: Name of the tiktoken encoding to use.
                          Options: 'gpt2', 'cl100k_base', 'p50k_base', etc.
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        return self.encoding.encode(text)

    def decode(self, token_ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token ID(s) to decode. Can be:
                      - List[int]: Single sequence
                      - List[List[int]]: Multiple sequences

        Returns:
            Decoded text string(s)
        """
        if not token_ids:
            return ""

        # Check if it's a batch (list of lists)
        if isinstance(token_ids[0], list):
            return [self.encoding.decode(ids) for ids in token_ids]
        else:
            return self.encoding.decode(token_ids)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts to token IDs.

        Args:
            texts: List of text strings

        Returns:
            List of token ID lists
        """
        return [self.encode(text) for text in texts]

    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size


class SimpleTokenizer:
    """
    Very simple character-level tokenizer for testing/demo purposes.
    Not recommended for actual training.
    """
    def __init__(self, chars: str = None):
        """
        Initialize character-level tokenizer.

        Args:
            chars: String of characters to include in vocabulary.
                  If None, uses basic ASCII printable characters.
        """
        if chars is None:
            # Basic ASCII printable characters
            chars = ''.join(chr(i) for i in range(32, 127))

        self.chars = sorted(list(set(chars)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, token_ids: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        """Decode token IDs to text."""
        if not token_ids:
            return ""

        # Check if it's a batch
        if isinstance(token_ids[0], list):
            return [''.join([self.idx_to_char.get(idx, '') for idx in ids]) for ids in token_ids]
        else:
            return ''.join([self.idx_to_char.get(idx, '') for idx in token_ids])

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(text) for text in texts]

    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size


def get_tokenizer(tokenizer_type: str = "tiktoken", **kwargs) -> Union[Tokenizer, SimpleTokenizer]:
    """
    Factory function to get a tokenizer.

    Args:
        tokenizer_type: Type of tokenizer ('tiktoken' or 'simple')
        **kwargs: Additional arguments for tokenizer initialization

    Returns:
        Tokenizer instance
    """
    if tokenizer_type == "tiktoken":
        return Tokenizer(**kwargs)
    elif tokenizer_type == "simple":
        return SimpleTokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = Tokenizer()
    text = "Hello, world! This is a test."

    print(f"Original text: {text}")
    encoded = tokenizer.encode(text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    print(f"Vocabulary size: {len(tokenizer)}")
