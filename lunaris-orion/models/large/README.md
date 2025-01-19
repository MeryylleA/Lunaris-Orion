# Large Model

A powerful transformer-based model for generating pixel art from text descriptions, optimized for multi-GPU training on NVIDIA H100s.

## Architecture

- **Model Size**: ~300M parameters
- **Architecture**: Transformer with advanced features
  - Embedding Dimension: 1024
  - Attention Heads: 16
  - Layers: 24
  - FFN Dimension: 4096
  - Max Sequence Length: 2048

## Advanced Features

1. **Attention Mechanisms**:
   - Flash Attention V2
   - Rotary Position Embeddings (RoPE)
   - Sliding Window Attention
   - Parallel Attention/FFN blocks

2. **Optimizations**:
   - Mixed Precision Training (bfloat16)
   - Gradient Checkpointing
   - Smart Batching
   - ZeRO Stage 2 Optimization

3. **Training Features**:
   - Distributed Training (Multi-GPU)
   - LMDB Dataset Caching
   - Advanced Augmentation
   - Robust Checkpointing

## Requirements

- 2x NVIDIA H100 GPUs (80GB)
- CUDA 11.8+
- PyTorch 2.0+
- Flash Attention 2.0
- NCCL for distributed training

## Usage

1. **Training**:
   ```bash
   # Full training
   python retrain.py --epochs 1000 --data-dir data

   # Development/testing
   python retrain.py --dev --max-samples 1000
   ```

2. **Inference**:
   ```bash
   python inference.py --prompt "your text prompt" --output output.png
   ```

3. **Analyze Training**:
   ```bash
   python analyze_checkpoints.py --run-dir runs/latest
   ```

## Model Performance

- Training Time: ~48 hours on 2x H100
- Memory Usage: ~60GB per GPU
- Throughput: ~100 images/second
- FLOPs: ~2.5T per forward pass

## Directory Structure

```
large/
├── config.py          # Model and training configuration
├── model.py           # Model architecture implementation
├── dataset.py         # Dataset with LMDB caching
├── retrain.py         # Distributed training script
├── inference.py       # Inference and generation
└── analyze_checkpoints.py  # Training analysis tools
```

## Improvements over Mini

1. **Architecture**:
   - 4x larger embedding dimension
   - 4x more attention heads
   - 3x more layers
   - Advanced attention mechanisms

2. **Training**:
   - Multi-GPU support
   - Better data handling
   - More robust checkpointing
   - Enhanced monitoring

3. **Features**:
   - Sliding window attention for longer sequences
   - Parallel attention/FFN computation
   - SwiGLU activations
   - Advanced position embeddings

## License

This project is licensed under the MIT License - see the LICENSE file for details. 