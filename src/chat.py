#!/usr/bin/env python3
"""
Interactive chat interface for the trained LLM.
Supports text generation with various sampling strategies.
"""

import argparse
import torch
from model import ModelConfig, TransformerLM
from tokenizer import get_tokenizer


class ChatBot:
    """
    Interactive chatbot using trained Transformer LM.
    """
    def __init__(self, checkpoint_path, device="cpu"):
        """
        Initialize chatbot.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device

        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Initialize model from config
        config_dict = checkpoint["config"]
        self.config = ModelConfig(**config_dict)

        self.model = TransformerLM(self.config).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Initialize tokenizer
        self.tokenizer = get_tokenizer("tiktoken")

        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Device: {device}")
        print()

    @torch.no_grad()
    def generate(self, prompt, max_tokens=100, temperature=0.8, top_k=50):
        """
        Generate text from prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (only sample from top k tokens)

        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Generate
        output_tensor = self.model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k
        )

        # Decode
        output_ids = output_tensor[0].tolist()
        generated_text = self.tokenizer.decode(output_ids)

        return generated_text

    def chat(self, temperature=0.8, top_k=50, max_tokens=100):
        """
        Interactive chat loop.

        Args:
            temperature: Sampling temperature
            top_k: Top-k sampling
            max_tokens: Maximum tokens to generate
        """
        print("=" * 70)
        print("  Interactive Chat - Type 'quit' or 'exit' to stop")
        print("=" * 70)
        print()
        print("Settings:")
        print(f"  Temperature: {temperature}")
        print(f"  Top-k: {top_k}")
        print(f"  Max tokens: {max_tokens}")
        print()
        print("Commands:")
        print("  /temp <value>   - Set temperature (0.1-2.0)")
        print("  /topk <value>   - Set top-k (1-100)")
        print("  /tokens <value> - Set max tokens")
        print("  /reset          - Reset conversation")
        print("  /help           - Show this help")
        print()

        conversation_history = ""

        while True:
            try:
                # Get user input
                user_input = input("\n\033[1;34mYou:\033[0m ").strip()

                if not user_input:
                    continue

                # Check for exit commands
                if user_input.lower() in ["quit", "exit"]:
                    print("\nGoodbye!")
                    break

                # Check for special commands
                if user_input.startswith("/"):
                    if user_input == "/help":
                        print("\nCommands:")
                        print("  /temp <value>   - Set temperature")
                        print("  /topk <value>   - Set top-k")
                        print("  /tokens <value> - Set max tokens")
                        print("  /reset          - Reset conversation")
                        print("  /help           - Show this help")
                        continue

                    elif user_input == "/reset":
                        conversation_history = ""
                        print("\n✓ Conversation reset")
                        continue

                    elif user_input.startswith("/temp "):
                        try:
                            temperature = float(user_input.split()[1])
                            temperature = max(0.1, min(2.0, temperature))
                            print(f"\n✓ Temperature set to {temperature}")
                        except:
                            print("\n✗ Invalid temperature value")
                        continue

                    elif user_input.startswith("/topk "):
                        try:
                            top_k = int(user_input.split()[1])
                            top_k = max(1, min(100, top_k))
                            print(f"\n✓ Top-k set to {top_k}")
                        except:
                            print("\n✗ Invalid top-k value")
                        continue

                    elif user_input.startswith("/tokens "):
                        try:
                            max_tokens = int(user_input.split()[1])
                            max_tokens = max(10, min(500, max_tokens))
                            print(f"\n✓ Max tokens set to {max_tokens}")
                        except:
                            print("\n✗ Invalid max tokens value")
                        continue

                    else:
                        print("\n✗ Unknown command. Type /help for available commands")
                        continue

                # Add to conversation history
                prompt = conversation_history + f"User: {user_input}\nAssistant:"

                # Generate response
                print("\n\033[1;32mAssistant:\033[0m ", end="", flush=True)
                generated_text = self.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k
                )

                # Extract just the assistant's response
                response = generated_text[len(prompt):].split("\n")[0].strip()

                # Handle empty responses
                if not response:
                    response = "[No response generated]"

                print(response)

                # Update conversation history
                conversation_history += f"User: {user_input}\nAssistant: {response}\n"

                # Truncate history if too long
                max_history_tokens = 1000
                history_tokens = self.tokenizer.encode(conversation_history)
                if len(history_tokens) > max_history_tokens:
                    # Keep only recent history
                    truncated_tokens = history_tokens[-max_history_tokens:]
                    conversation_history = self.tokenizer.decode(truncated_tokens)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
                continue

            except Exception as e:
                print(f"\n✗ Error: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with trained LLM")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt (non-interactive mode)"
    )

    args = parser.parse_args()

    # Initialize chatbot
    chatbot = ChatBot(args.checkpoint, args.device)

    # Single prompt mode or interactive mode
    if args.prompt:
        # Single prompt mode
        print(f"Prompt: {args.prompt}\n")
        response = chatbot.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        print(f"Generated:\n{response}")
    else:
        # Interactive mode
        chatbot.chat(
            temperature=args.temperature,
            top_k=args.top_k,
            max_tokens=args.max_tokens
        )


if __name__ == "__main__":
    main()
