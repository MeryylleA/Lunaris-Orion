# Lunaris-Orion 🌌

A hybrid VAE-based pixel art generation system with semantic understanding and quality assessment.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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

3. **Resume training:**
   ```bash
   python train_hybrid.py \
       --data_dir ./data \
       --output_dir ./output \
       --resume_from ./output/checkpoints/best.pt
   ```

## Hardware Recommendations

### High-End GPUs (A100, H100, L40S)
```bash
python train_hybrid.py \
    --data_dir ./data \
    --output_dir ./output \
    --mixed_precision \
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

## Project Structure

```
lunaris-orion/
├── data/                  # Dataset directory
├── docs/                  # Documentation
├── output/               # Training outputs
│   ├── checkpoints/     # Model checkpoints
│   ├── eval_samples/    # Generated samples
│   └── tensorboard/     # Training logs
├── lunar_generate.py     # VAE model
├── lunar_evaluator.py    # Teacher model
├── train_hybrid.py       # Training script
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Documentation

Detailed documentation is available in the [docs](docs/) directory:
- [Architecture Overview](docs/architecture.md)
- [Model Details](docs/models.md)
- 
## Training Outputs

The system generates several types of outputs during training:

1. **Checkpoints** (`output/checkpoints/`):
   - `latest.pt`: Most recent training state
   - `best.pt`: Best performing model

2. **Evaluation Samples** (`output/eval_samples/`):
   - Side-by-side comparisons of original and generated images
   - Quality and semantic scores
   - Generated every N steps (configurable)

3. **TensorBoard Logs** (`output/tensorboard/`):
   - Loss curves
   - Quality metrics
   - Learning rates
   - Memory usage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

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
