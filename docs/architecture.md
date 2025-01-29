# LunarCore Architecture Documentation

This document provides a detailed explanation of the LunarCore VAE architecture, its components, and guidelines for modifications and improvements.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Component Details](#component-details)
3. [Training Process](#training-process)
4. [Modification Guidelines](#modification-guidelines)
5. [Implementation Tips](#implementation-tips)
6. [Common Issues](#common-issues)

## Architecture Overview

The LunarCore VAE is designed specifically for pixel art processing, with several key architectural decisions:

### Design Philosophy
- Preserve pixel-perfect details through skip connections
- Capture global patterns with self-attention
- Maintain color consistency with proper normalization
- Balance reconstruction quality with latent space regularization

### Core Components
```
Input (16×16×3)
    ↓
Encoder (Progressive downsampling + Self-attention)
    ↓
Latent Space (128-dim)
    ↓
Decoder (Progressive upsampling + Self-attention)
    ↓
Output (16×16×3)
```

## Component Details

### 1. Encoder

#### Input Processing
```python
# Initial convolution
Conv2D(in_channels=3, out_channels=64, kernel_size=3, padding=1)
BatchNorm2D(64)
LeakyReLU(0.2)
```

#### Downsampling Blocks
Each block consists of:
```python
# Downsample
Conv2D(in_channels=C_in, out_channels=C_out, kernel_size=4, stride=2, padding=1)
BatchNorm2D(C_out)
LeakyReLU(0.2)

# ResBlock
ResBlock(
    Conv2D(C_out, C_out, kernel_size=3, padding=1)
    BatchNorm2D(C_out)
    Dropout(0.1)
    LeakyReLU(0.2)
)

# Self-Attention (at 8×8 and 4×4 resolutions)
SelfAttention(
    query_conv = Conv2D(C_out, C_out//8, kernel_size=1)
    key_conv = Conv2D(C_out, C_out//8, kernel_size=1)
    value_conv = Conv2D(C_out, C_out, kernel_size=1)
)
```

#### Latent Projection
```python
Flatten()
Linear(2048, 128)  # Mean
Linear(2048, 128)  # LogVar
```

### 2. Latent Space

The latent space uses the reparametrization trick:
```python
def reparameterize(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std
```

### 3. Decoder

#### Latent Processing
```python
Linear(128, 2048)
Reshape(512, 2, 2)
```

#### Upsampling Blocks
Each block consists of:
```python
# Upsample
ConvTranspose2D(C_in, C_out, kernel_size=4, stride=2, padding=1)
BatchNorm2D(C_out)
LeakyReLU(0.2)

# ResBlock
ResBlock(
    Conv2D(C_out*2, C_out*2, kernel_size=3, padding=1)  # *2 due to skip connection
    BatchNorm2D(C_out*2)
    Dropout(0.2)
    LeakyReLU(0.2)
)

# Self-Attention (at 4×4 and 8×8 resolutions)
SelfAttention(
    query_conv = Conv2D(C_out, C_out//8, kernel_size=1)
    key_conv = Conv2D(C_out, C_out//8, kernel_size=1)
    value_conv = Conv2D(C_out, C_out, kernel_size=1)
)
```

#### Output Processing
```python
Conv2D(64, 3, kernel_size=3, padding=1)
Tanh()
```

## Training Process

### Loss Function
The model uses a combination of reconstruction loss and KL divergence:
```python
reconstruction_loss = MSE(output, target)
kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
total_loss = reconstruction_loss + kl_weight * kl_loss
```

### Optimization
- Optimizer: AdamW with weight decay 1e-5
- Learning rate: 0.001 with StepLR scheduler
- Mixed precision training for efficiency
- Gradient clipping to prevent exploding gradients

## Modification Guidelines

### Adding New Features

1. **Increasing Model Capacity**
```python
# Example: Adding more channels
self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)  # 64 -> 128
```

2. **Adding More Attention Layers**
```python
# Add attention after any conv layer
x = self.conv(x)
x = self.attention(x)  # New attention layer
```

3. **Modifying ResBlocks**
```python
# Example: Adding squeeze-and-excitation
class ResBlockSE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.se = SqueezeExcitation(channels)
        # ... rest of ResBlock implementation
```

### Architecture Improvements

1. **Latent Space Modifications**
- Increase dimensionality for more capacity
- Add additional regularization
- Implement hierarchical latent spaces

2. **Skip Connection Variations**
- Add feature transformation before skip connection
- Implement dense skip connections
- Add attention to skip connections

3. **Loss Function Enhancements**
- Add perceptual loss
- Implement feature matching
- Add adversarial loss

## Implementation Tips

1. **Memory Optimization**
- Use gradient checkpointing for large models
- Implement efficient attention mechanisms
- Optimize batch size based on available memory

2. **Training Stability**
- Start with small kl_weight and increase gradually
- Use learning rate warmup
- Monitor gradient norms

3. **Performance Improvements**
- Use torch.compile() for faster training
- Implement efficient data loading
- Utilize GPU profiling tools

## Common Issues

1. **Training Problems**
- Posterior collapse: Increase kl_weight
- Blurry outputs: Check reconstruction loss weight
- Mode collapse: Adjust latent space regularization

2. **Memory Issues**
- OOM errors: Reduce batch size or model size
- Slow training: Check data loading bottlenecks
- GPU utilization: Monitor with nvidia-smi

3. **Quality Issues**
- Loss of details: Adjust skip connections
- Color inconsistency: Check normalization
- Artifacts: Adjust attention mechanisms

## Future Improvements

Potential areas for enhancement:
1. Implement conditional generation
2. Add style transfer capabilities
3. Integrate with diffusion models
4. Implement interactive editing features
5. Add support for different resolutions 