# My Simple LLM - Transformer Language Model from Scratch

A from-scratch implementation of a Transformer-based Large Language Model (LLM) with modern 2025 enhancements including **Mixture of Experts (MoE)** and **long-context support**.

This project implements the core concepts from highly-cited research papers to create an educational yet functional LLM that can be trained and used for text generation.

## Features

- **Pure PyTorch Implementation**: Built from scratch without using transformers library
- **Transformer Architecture**: Full implementation of multi-head self-attention, positional encoding, and feed-forward networks
- **Mixture of Experts (MoE)**: Efficient conditional computation inspired by Switch Transformers and recent MoE research
- **Long Context Support**: Architecture optimized for handling longer sequences
- **Training Pipeline**: Complete data loading, training, and evaluation pipeline
- **Interactive Chat Interface**: User-friendly chatbot for testing trained models
- **Multiple Datasets**: Support for OpenWebText, WikiText-103, TinyStories, and custom datasets

## Architecture

### Core Components

1. **Multi-Head Self-Attention**
   - Scaled dot-product attention
   - Multiple attention heads for diverse representations
   - Causal masking for autoregressive generation

2. **Positional Encoding**
   - Sinusoidal positional embeddings
   - Enables sequence position awareness

3. **Mixture of Experts (MoE)**
   - Dynamic expert routing per token
   - Top-k or top-1 (Switch) routing strategies
   - Increased model capacity without proportional compute cost

4. **Layer Normalization & Residual Connections**
   - Pre-norm architecture for training stability
   - Residual connections throughout

### Model Configuration

Default configuration (customizable):
- **d_model**: 512 (embedding dimension)
- **n_layers**: 6 (transformer blocks)
- **n_heads**: 8 (attention heads)
- **d_ff**: 2048 (feed-forward dimension)
- **max_seq_len**: 2048 (maximum sequence length)
- **n_experts**: 4 (number of experts in MoE)
- **vocab_size**: 50257 (GPT-2 tokenizer)

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd my-simple-llm
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Download Training Data

Download a dataset using the provided script:

```bash
# Download TinyStories (small, good for testing)
python scripts/download_data.py --dataset tinystories --num-samples 10000

# Download OpenWebText (larger, GPT-2 style)
python scripts/download_data.py --dataset openwebtext --num-samples 100000

# Download WikiText-103
python scripts/download_data.py --dataset wikitext

# Use custom text file
python scripts/download_data.py --dataset custom --custom-file path/to/your/text.txt
```

Available datasets:
- `tinystories`: Small stories dataset (good for quick testing)
- `openwebtext`: OpenWebText corpus (GPT-2 replica training data)
- `wikitext`: WikiText-103 dataset
- `custom`: Your own text file

Data will be saved to `./data/` by default.

### 2. Train the Model

Train a model on your downloaded data:

```bash
# Basic training (small model)
python src/train.py \
  --dataset tinystories \
  --data-dir ./data \
  --batch-size 32 \
  --epochs 10 \
  --lr 3e-4

# Train with MoE (more efficient)
python src/train.py \
  --dataset tinystories \
  --use-moe \
  --n-experts 4 \
  --batch-size 16 \
  --epochs 20

# Larger model configuration
python src/train.py \
  --dataset openwebtext \
  --d-model 768 \
  --n-layers 12 \
  --n-heads 12 \
  --d-ff 3072 \
  --max-seq-len 1024 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --epochs 50
```

Key training arguments:
- `--d-model`: Model dimension (default: 512)
- `--n-layers`: Number of transformer layers (default: 6)
- `--n-heads`: Number of attention heads (default: 8)
- `--d-ff`: Feed-forward dimension (default: 2048)
- `--max-seq-len`: Maximum sequence length (default: 512)
- `--use-moe`: Enable Mixture of Experts
- `--n-experts`: Number of experts (default: 4)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 3e-4)
- `--grad-accum-steps`: Gradient accumulation steps (default: 1)

Checkpoints will be saved to `./checkpoints/` by default.

### 3. Chat with the Model

Once trained, interact with your model:

```bash
# Interactive chat mode
python src/chat.py --checkpoint ./checkpoints/best_model.pt

# Single prompt (non-interactive)
python src/chat.py \
  --checkpoint ./checkpoints/best_model.pt \
  --prompt "Once upon a time" \
  --max-tokens 200 \
  --temperature 0.8
```

Chat arguments:
- `--checkpoint`: Path to trained model checkpoint (required)
- `--temperature`: Sampling temperature, 0.1-2.0 (default: 0.8)
- `--top-k`: Top-k sampling (default: 50)
- `--max-tokens`: Maximum tokens to generate (default: 100)
- `--prompt`: Single prompt for non-interactive mode

Interactive commands:
- `/temp <value>`: Change temperature
- `/topk <value>`: Change top-k
- `/tokens <value>`: Change max tokens
- `/reset`: Reset conversation history
- `/help`: Show help
- `quit` or `exit`: Exit chat

## Project Structure

```
my-simple-llm/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py          # Model configuration
│   │   ├── transformer.py     # Transformer implementation
│   │   └── moe.py            # Mixture of Experts
│   ├── tokenizer.py          # Tokenizer wrapper
│   ├── train.py              # Training script
│   └── chat.py               # Chat interface
├── scripts/
│   └── download_data.py      # Data download script
├── data/                     # Downloaded datasets (generated)
├── checkpoints/              # Model checkpoints (generated)
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

## Implementation Details

### Transformer Architecture

The core Transformer architecture follows the original "Attention is All You Need" paper with modern improvements:

- **Multi-Head Attention**: Allows the model to jointly attend to information from different representation subspaces
- **Causal Masking**: Ensures autoregressive property (can only attend to previous tokens)
- **Pre-Layer Normalization**: Improves training stability
- **Learned Positional Embeddings**: Alternative to sinusoidal for better long-context handling

### Mixture of Experts (MoE)

MoE increases model capacity without proportional compute increase:

- **Dynamic Routing**: Each token is routed to top-k experts
- **Gating Network**: Learns to select appropriate experts for each input
- **Load Balancing**: Ensures experts are utilized efficiently
- **Switch Routing**: Optional top-1 routing for maximum efficiency

### Training Optimizations

- **Gradient Accumulation**: Simulate larger batch sizes on limited hardware
- **Learning Rate Warmup**: Gradual learning rate increase for stable training
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: (Optional) FP16 training for faster computation

## Performance Tips

### For Limited Hardware

```bash
# Small model, low memory
python src/train.py \
  --d-model 256 \
  --n-layers 4 \
  --n-heads 4 \
  --batch-size 8 \
  --grad-accum-steps 8 \
  --max-seq-len 256
```

### For GPU Training

```bash
# Use larger batch sizes and models
python src/train.py \
  --d-model 768 \
  --n-layers 12 \
  --batch-size 64 \
  --device cuda \
  --use-moe \
  --n-experts 8
```

## References

This implementation is based on the following research papers:

### Foundational Papers

1. **Vaswani, A., et al. (2017)**. "Attention Is All You Need"
   *arXiv:1706.03762*
   [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
   - The original Transformer architecture
   - Multi-head self-attention mechanism
   - Positional encoding

### 2025 Trends and Recent Research

2. **Yunfan Gao, et al. (2023)**. "Advancing Transformer Architecture in Long-Context Large Language Models: A Comprehensive Survey"
   *arXiv:2311.12351*
   [https://arxiv.org/abs/2311.12351](https://arxiv.org/abs/2311.12351)
   - Survey of long-context LLM techniques
   - Architectural upgrades for context extension
   - Evaluation methods for long-context models

3. **ExpertRAG Research (2025)**. "ExpertRAG: Efficient RAG with Mixture of Experts"
   *arXiv:2504.08744*
   [https://arxiv.org/abs/2504.08744](https://arxiv.org/abs/2504.08744)
   - Integration of MoE with RAG systems
   - Dynamic retrieval gating mechanism
   - Expert routing strategies

### Mixture of Experts

4. **Fedus, W., Zoph, B., & Shazeer, N. (2021)**. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
   *arXiv:2101.03961*
   [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961)
   - Switch routing (top-1 expert selection)
   - Sparse model scaling techniques
   - Load balancing algorithms

5. **Shazeer, N., et al. (2017)**. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
   *arXiv:1701.06538*
   [https://arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)
   - Original MoE for neural networks
   - Gating mechanisms
   - Conditional computation

### Additional Influences

6. **Radford, A., et al. (2019)**. "Language Models are Unsupervised Multitask Learners" (GPT-2)
   - Byte-pair encoding tokenization
   - Large-scale language model pre-training
   - Zero-shot task transfer

7. **Recent LLM Research Trends (2025)**:
   - Reasoning capabilities with reinforcement learning
   - Long-context generation techniques
   - Efficient attention mechanisms (FlashAttention, etc.)
   - Mixture-of-Experts architectures

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Inspired by recent research in LLM architecture and efficiency
- Built with PyTorch and the amazing open-source ML community
- Datasets provided by Hugging Face

## Citation

If you use this code in your research, please cite:

```bibtex
@software{my_simple_llm,
  title={My Simple LLM: A From-Scratch Transformer Language Model},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/my-simple-llm}
}
```
