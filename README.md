# Lunaris-Orion 🌌

A hybrid VAE-based pixel art generation system with semantic understanding and quality assessment.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.0.4-blue.svg)](CHANGELOG.md)

## Overview

Lunaris-Orion is a sophisticated pixel art generation system that combines a Variational Autoencoder (VAE) with a Teacher model for quality assessment and semantic understanding. The system is designed to learn and generate high-quality pixel art while maintaining semantic consistency and artistic style.

### Key Features

- 🎨 High-quality pixel art generation
- 🧠 Semantic understanding and preservation
- 📊 Built-in quality assessment
- 💫 Hybrid training system
- 🚀 Mixed precision training support
- 📈 Comprehensive logging and visualization
- 💾 Checkpoint management and training resumption
- 🖥️ CPU training support (v0.0.4)
- 📊 Dynamic memory optimization (v0.0.4)
- 📉 Automatic batch size adjustment (v0.0.4)

## What's New in v0.0.4

- **CPU Support**: Train on systems without GPUs (slower but functional)
- **Memory Optimization**: Dynamic batch size adjustment and memory tracking
- **Better Monitoring**: Progress bars and detailed memory statistics
- **Enhanced Stability**: Improved error handling and recovery
- **Performance**: Reduced memory usage and better OOM handling

## Installation

```bash
# Clone the repository
git clone https://github.com/MeryylleA/Lunaris-Orion.git
cd lunaris-orion

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. **Prepare your data:**
   ```bash
   data/
   ├── sprites_001.npy    # Image arrays (N, 128, 128, 3)
   ├── sprites_002.npy
   ├── labels_001.csv     # Corresponding labels
   └── labels_002.csv
   ```

2. **Basic training:**
   ```bash
   python train_hybrid.py \
       --data_dir ./data \
       --output_dir ./output \
       --mixed_precision \
       --batch_size 32 \
       --num_workers 4
   ```

3. **Memory-efficient training (new in v0.0.4):**
   ```bash
   python train_hybrid.py \
       --data_dir ./data \
       --output_dir ./output \
       --mixed_precision \
       --memory_efficient \
       --batch_size 32 \
       --num_workers 4
   ```

4. **CPU training (new in v0.0.4):**
   ```bash
   python train_hybrid.py \
       --data_dir ./data \
       --output_dir ./output \
       --force_cpu \
       --batch_size 8 \
       --num_workers 2
   ```

5. **Resume training:**
   ```bash
   python train_hybrid.py \
       --data_dir ./data \
       --output_dir ./output \
       --resume_from ./output/checkpoints/best.pt
   ```

## Hardware Configurations

### High-End GPUs (A100, H100, L40S)
```bash
python train_hybrid.py \
    --data_dir ./data \
    --output_dir ./output \
    --mixed_precision \
    --memory_efficient \
    --batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_workers 8 \
    --chunk_size 128 \
    --vae_lr 3e-4 \
    --teacher_lr 2e-4 \
    --latent_dim 512 \
    --embedding_dim 256 \
    --feature_dim 512
```

### Mid-Range GPUs (RTX 3090, 4090)
```bash
python train_hybrid.py \
    --data_dir ./data \
    --output_dir ./output \
    --mixed_precision \
    --memory_efficient \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --num_workers 4 \
    --chunk_size 64 \
    --vae_lr 2e-4 \
    --teacher_lr 1e-4 \
    --latent_dim 384 \
    --embedding_dim 192 \
    --feature_dim 384
```

### Lower-End GPUs (RTX 3060, 2080)
```bash
python train_hybrid.py \
    --data_dir ./data \
    --output_dir ./output \
    --mixed_precision \
    --memory_efficient \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --num_workers 2 \
    --chunk_size 32 \
    --vae_lr 1e-4 \
    --teacher_lr 5e-5 \
    --latent_dim 256 \
    --embedding_dim 128 \
    --feature_dim 256
```

### CPU Training (New in v0.0.4)
```bash
python train_hybrid.py \
    --data_dir ./data \
    --output_dir ./output \
    --force_cpu \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --num_workers 2 \
    --chunk_size 32 \
    --vae_lr 5e-5 \
    --teacher_lr 2e-5 \
    --latent_dim 256 \
    --embedding_dim 128 \
    --feature_dim 256
```

## Memory Management (New in v0.0.4)

The new version includes several memory optimization features:

- **Dynamic Batch Size**: Automatically adjusts batch size if OOM errors occur
- **Memory Tracking**: Monitors and logs GPU memory usage
- **Efficient Data Loading**: Optimized DataLoader settings
- **Gradient Accumulation**: Better memory efficiency during training
- **Automatic Cleanup**: Regular memory cleanup between batches

Enable these features with the `--memory_efficient` flag.

## Project Structure

```
lunaris-orion/
├── data/                  # Dataset directory
├── docs/                  # Documentation
├── output/               # Training outputs
│   ├── checkpoints/     # Model checkpoints
│   ├── tensorboard/     # Training logs
│   └── eval_samples/    # Generated samples
├── models/              # Model definitions
├── utils/               # Utility functions
└── examples/            # Example scripts
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lunaris_orion,
  author = {Moon Cloud Services},
  title = {Lunaris-Orion: Hybrid VAE-based Pixel Art Generation},
  year = {2025},
  publisher = {Moon Cloud Services},
  url = {https://github.com/MeryylleA/Lunaris-Orion}
}
```

## Acknowledgments

- Thanks to the PyTorch team for their excellent framework
- Special thanks to all contributors and users of the project
