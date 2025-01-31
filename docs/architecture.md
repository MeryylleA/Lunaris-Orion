# LunarCore Architecture Documentation

This document provides a detailed explanation of the LunarCore VAE architecture, its components, and guidelines for modifications and improvements. The architecture has been updated to support 128×128 resolution pixel art images.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Component Details](#component-details)
3. [Training Process](#training-process)
4. [Modification Guidelines](#modification-guidelines)
5. [Implementation Tips](#implementation-tips)
6. [Common Issues](#common-issues)

## Architecture Overview

The LunarCore VAE is designed specifically for high-resolution pixel art processing (128×128), with several key architectural decisions:

### Design Philosophy
- Preserve pixel-perfect details through multi-scale skip connections
- Capture global patterns with strategically placed self-attention
- Maintain color consistency and fine details with proper normalization
- Balance reconstruction quality with latent space regularization
- Support high-resolution (128×128) pixel art while maintaining computational efficiency

### Core Components
```
Input (128×128×3)
    ↓
Encoder (Progressive downsampling + Self-attention)
    128×128 → 64×64 → 32×32 → 16×16 → 8×8
    ↓
Latent Space (128-dim)
    ↓
Decoder (Progressive upsampling + Skip connections)
    8×8 → 16×16 → 32×32 → 64×64 → 128×128
    ↓
Output (128×128×3)
```

## Component Details

### 1. Encoder

#### Input Processing
```python
# Initial convolution (maintains 128×128 resolution)
Conv2D(in_channels=3, out_channels=64, kernel_size=3, padding=1)
BatchNorm2D(64)
LeakyReLU(0.2)
ResBlock(64)  # Preserve initial details
```

#### Downsampling Blocks
The encoder uses four downsampling blocks:

**Block 1: 128×128 → 64×64**
```python
# Downsample
Conv2D(64, 128, kernel_size=4, stride=2, padding=1)
BatchNorm2D(128)
LeakyReLU(0.2)
ResBlock(128)
SelfAttention(128)  # Capture global patterns at 64×64
```

**Block 2: 64×64 → 32×32**
```python
Conv2D(128, 256, kernel_size=4, stride=2, padding=1)
BatchNorm2D(256)
LeakyReLU(0.2)
ResBlock(256)
SelfAttention(256)  # Capture global patterns at 32×32
```

**Block 3: 32×32 → 16×16**
```python
Conv2D(256, 512, kernel_size=4, stride=2, padding=1)
BatchNorm2D(512)
LeakyReLU(0.2)
ResBlock(512)
```

**Block 4: 16×16 → 8×8**
```python
Conv2D(512, 512, kernel_size=4, stride=2, padding=1)
BatchNorm2D(512)
LeakyReLU(0.2)
ResBlock(512)
```

#### Skip Connections
Skip connections are stored at multiple resolutions:
- Level 1: 128×128 (64 channels)
- Level 2: 64×64 (128 channels)
- Level 3: 32×32 (256 channels)
- Level 4: 16×16 (512 channels)

#### Latent Projection
```python
Flatten(512 * 8 * 8)  # 32,768-dimensional vector
Linear(32768, 128)    # Mean
Linear(32768, 128)    # LogVar
```

### 2. Latent Space

The latent space remains 128-dimensional and uses the reparametrization trick:
```python
def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std
```

### 3. Decoder

#### Initial Projection
```python
Linear(128, 512 * 8 * 8)
Reshape(-1, 512, 8, 8)
```

#### Upsampling Blocks

**Block 1: 8×8 → 16×16**
```python
ConvTranspose2D(512, 512, kernel_size=4, stride=2, padding=1)
BatchNorm2D(512)
LeakyReLU(0.2)
# Concatenate with skip connection (512 + 512 = 1024 channels)
ResBlock(1024)
Conv2D(1024, 512, kernel_size=1)  # Channel reduction
```

**Block 2: 16×16 → 32×32**
```python
ConvTranspose2D(512, 256, kernel_size=4, stride=2, padding=1)
BatchNorm2D(256)
LeakyReLU(0.2)
# Concatenate with skip connection (256 + 256 = 512 channels)
ResBlock(512)
Conv2D(512, 256, kernel_size=1)  # Channel reduction
```

**Block 3: 32×32 → 64×64**
```python
ConvTranspose2D(256, 128, kernel_size=4, stride=2, padding=1)
BatchNorm2D(128)
LeakyReLU(0.2)
# Concatenate with skip connection (128 + 128 = 256 channels)
ResBlock(256)
Conv2D(256, 128, kernel_size=1)  # Channel reduction
```

**Block 4: 64×64 → 128×128**
```python
ConvTranspose2D(128, 64, kernel_size=4, stride=2, padding=1)
BatchNorm2D(64)
LeakyReLU(0.2)
# Concatenate with skip connection (64 + 64 = 128 channels)
ResBlock(128)
Conv2D(128, 64, kernel_size=1)  # Channel reduction
```

#### Output Processing
```python
Conv2D(64, 3, kernel_size=3, padding=1)
Tanh()  # Normalize to [-1, 1]
```

## Training Process

### Loss Function
The model uses a weighted combination of reconstruction loss and KL divergence:
```python
reconstruction_loss = MSE(output, target)  # For 128×128 images
kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
total_loss = reconstruction_loss + kl_weight * kl_loss

# KL weight annealing for 128×128 training
kl_weight = min(1.0, current_epoch / 50) * base_kl_weight
```

### Optimization
- Optimizer: AdamW with weight decay 1e-4
- Learning rate: 0.001 with warmup and cosine decay
- Mixed precision training for efficient 128×128 processing
- Gradient clipping at 0.5 to prevent instability
- Batch size adjusted based on available GPU memory

## Modification Guidelines

### High-Resolution Specific Improvements

1. **Memory Optimization**
```python
# Example: Gradient checkpointing for memory efficiency
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.block1, x)
    x = checkpoint(self.block2, x)
```

2. **Feature Preservation**
```python
# Example: Adding residual connections in upsampling
class UpsampleBlock(nn.Module):
    def forward(self, x, skip):
        up = self.upsample(x)
        combined = torch.cat([up, skip], dim=1)
        return self.process(combined) + up
```

### Architecture Improvements

1. **Enhanced Skip Connections**
- Feature refinement before concatenation
- Adaptive skip connection weighting
- Multi-scale feature fusion

2. **Advanced Attention Mechanisms**
- Window attention for efficiency
- Cross-resolution attention
- Channel attention in skip connections

## Implementation Tips

1. **128×128 Specific Optimizations**
- Use torch.compile() for faster processing
- Implement efficient memory management
- Optimize data loading for large images

2. **Training Stability**
- Progressive resolution increase during training
- Careful learning rate scheduling
- Monitor and adjust KL weight annealing

## Common Issues

1. **High-Resolution Specific Problems**
- Memory management for large batches
- Balancing detail preservation with compression
- Training stability at higher resolutions

2. **Quality Considerations**
- Fine detail preservation at 128×128
- Color consistency across scales
- Artifact prevention in upsampling

## Future Improvements

Potential areas for enhancement:
1. Support for even higher resolutions (256×256, 512×512)
2. Conditional generation with text prompts
3. Style-based generation and control
4. Progressive growing training support
5. Adaptive resolution handling
6. Real-time inference optimization
7. Enhanced detail preservation techniques
8. Multi-scale discriminator integration 