# Lunaris-Orion: Advanced Pixel Art VAE Training System

A sophisticated deep learning system for training Variational Autoencoders (VAE) specialized in pixel art generation and manipulation, now supporting 128×128 resolution with enhanced color fidelity.

## Features

- **Enhanced VAE Architecture**
  - Optimized for 128×128 pixel art
  - Advanced color quantization system
  - Improved edge preservation mechanisms
  - Configurable latent space dimension
  - Progressive resolution scaling
  - Enhanced skip connections for detail preservation

- **Robust Training System**
  - Secure checkpoint management with exception handling
  - Advanced logging system with detailed error tracking
  - Mixed precision training support
  - PyTorch 2.0+ optimizations with `torch.compile`
  - Automatic memory management and CUDA optimization
  - Dynamic learning rate scheduling with extended warmup
  - Enhanced early stopping with configurable parameters
  - Progressive color quantization during training
  - Improved loss function for pixel art fidelity

- **Professional Project Structure**
```
output/
├── logs/
│   ├── training/           # Detailed training logs with timestamps
│   └── errors/            # Separate error logs with stack traces
├── checkpoints/
│   ├── best/              # Best performing model checkpoints
│   ├── periodic/          # Regular interval checkpoints
│   └── interrupt/         # Safe interrupt checkpoints
├── outputs/
│   ├── samples/           # Generated samples by epoch
│   ├── metrics/           # Detailed training metrics
│   ├── progress/          # Visual progress tracking
│   └── reports/           # Comprehensive training reports
└── tensorboard/           # TensorBoard logs
```

## New Features and Improvements

### Enhanced Color Management
- Discrete color quantization during training
- Color palette preservation mechanisms
- Improved edge detection and preservation
- Advanced pixel-perfect reconstruction

### Robust Error Handling
- Comprehensive exception logging with stack traces
- Safe checkpoint saving and loading
- Graceful training interruption handling
- Automatic memory cleanup and optimization

### Advanced Training Metrics
- Color fidelity tracking
- Edge preservation metrics
- Pixel accuracy measurements
- Detailed progress visualization

## Training Requirements

### Hardware Requirements

#### Recommended Specifications
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 16GB+ system memory
- Storage: 100GB+ free space for datasets and checkpoints

#### Minimum Specifications
- GPU: NVIDIA GPU with 4GB VRAM
- RAM: 8GB system memory
- Storage: 50GB free space

### Software Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU support)
- Additional dependencies:
  ```
  torch>=2.0.0
  torchvision>=0.15.0
  numpy>=1.24.0
  tensorboard>=2.14.0
  tqdm>=4.65.0
  pillow>=10.0.0
  psutil>=5.9.0
  ```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MeryylleA/Lunaris-Orion.git
cd Lunaris-Orion
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Guide

### Data Preparation

1. Prepare your dataset:
   - Images should be 128×128 pixels
   - Save sprites as .npy file
   - Create corresponding labels.csv
   - Place in your data directory

2. Directory structure:
```
data/
├── sprites.npy    # Shape: (N, 128, 128, 3)
└── labels.csv     # Corresponding labels
```

### Training Configurations

#### High-End GPU (16GB+ VRAM)
```bash
python lunarispixel/train_lunar_core.py \
    --data_dir data \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.0003 \
    --latent_dim 128 \
    --kl_weight 0.00001 \
    --patience 20 \
    --min_delta 0.001 \
    --compile \
    --mixed_precision
```

#### Mid-Range GPU (8GB VRAM)
```bash
python lunarispixel/train_lunar_core.py \
    --data_dir data \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 0.0003 \
    --latent_dim 128 \
    --kl_weight 0.00001 \
    --patience 20 \
    --min_delta 0.001 \
    --mixed_precision
```

#### Entry-Level GPU (4GB VRAM)
```bash
python lunarispixel/train_lunar_core.py \
    --data_dir data \
    --batch_size 8 \
    --epochs 100 \
    --learning_rate 0.0003 \
    --latent_dim 64 \
    --kl_weight 0.00001 \
    --patience 20 \
    --min_delta 0.001
```

### Key Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| batch_size | Batch size | 32 | Adjust based on VRAM |
| epochs | Training epochs | 100 | Increase for better results |
| learning_rate | Initial learning rate | 0.0003 | Optimized for stability |
| latent_dim | Latent space dimension | 128 | Trade-off between detail and memory |
| kl_weight | KL divergence weight | 0.00001 | Controls latent space regularization |
| patience | Early stopping patience | 20 | More epochs before stopping |
| min_delta | Min improvement threshold | 0.001 | Reduced sensitivity |
| compile | Use torch.compile | False | For modern GPUs |
| mixed_precision | Use mixed precision | False | Recommended for GPUs |

### Training Monitoring

1. Real-time logs:
   - Training progress with detailed metrics
   - Exception logging with stack traces
   - Hardware utilization monitoring
   - Checkpoint saving status

2. TensorBoard monitoring:
```bash
tensorboard --logdir path/to/output/tensorboard
```

3. Visual progress:
   - Check `outputs/progress/` for image comparisons
   - Monitor `outputs/metrics/` for detailed metrics
   - Review `logs/training/` for complete logs

### Best Practices

1. **Memory Management**
   - Start with smaller batch sizes
   - Monitor GPU memory usage
   - Enable mixed precision for larger models

2. **Training Stability**
   - Use the provided warmup period
   - Monitor the KL divergence
   - Check color quantization effects

3. **Checkpointing**
   - Regular checkpoints are saved automatically
   - Interrupt training safely with Ctrl+C
   - Best models are preserved separately

4. **Error Handling**
   - Check error logs for detailed traces
   - Monitor validation metrics
   - Use safe cleanup procedures

## Results and Evaluation

The model now generates high-quality 128×128 pixel art while maintaining:
- Sharp pixel edges
- Consistent color palettes
- Detailed feature preservation
- Stable training progression
- Controlled latent space distribution

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

```bibtex
@software{lunaris_orion,
  title = {Lunaris-Orion: Advanced Pixel Art VAE with Enhanced Color Fidelity},
  year = {2025},
  author = {Moon Cloud Services},
  url = {https://github.com/MeryylleA/Lunaris-Orion}
}
```

## Acknowledgments

- Thanks to the PyTorch team for their excellent framework
- Inspired by various VAE architectures and color quantization techniques
- Special thanks to the pixel art community for dataset contributions
