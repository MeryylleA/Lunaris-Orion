# Model Details

## LunarCoreVAE

The LunarCoreVAE is a Variational Autoencoder specifically designed for pixel art generation.

### Architecture

```
Input Image (128×128×3)
       ↓
┌─── Encoder ───┐
│ Conv2d + ReLU │
│ Conv2d + ReLU │
│ Conv2d + ReLU │
│ Conv2d + ReLU │
└───────────────┘
       ↓
┌─── Latent Space ───┐
│    mu, logvar      │
│    sampling        │
└──────────────────┘
       ↓
┌─── Decoder ───┐
│ ConvTranspose2d │
│ ConvTranspose2d │
│ ConvTranspose2d │
│ ConvTranspose2d │
└───────────────┘
       ↓
Output Image (128×128×3)
```

### Key Components

1. **Encoder**
   - Convolutional layers for feature extraction
   - Downsampling to compress information
   - Output: mean (mu) and log variance (logvar)

2. **Latent Space**
   - Dimension: Configurable (default: 256)
   - Reparameterization trick for backpropagation
   - KL divergence regularization

3. **Decoder**
   - Transposed convolutions for upsampling
   - Progressive resolution increase
   - Tanh activation for [-1, 1] output range

### Loss Functions

1. **Reconstruction Loss**
   ```python
   recon_loss = F.mse_loss(recon, images, reduction='mean')
   ```

2. **KL Divergence Loss**
   ```python
   kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
   ```

## LunarMoETeacher

The Teacher model uses a Mixture of Experts architecture for quality assessment and semantic understanding.

### Architecture

```
Input Image (128×128×3)
       ↓
┌─── Feature Extractor ───┐
│     ResNet Backbone     │
└──────────────────────┘
       ↓
┌─── Expert Gate ───┐
│  Routing Network  │
└─────────────────┘
       ↓
┌─── Expert Modules ───┐
│  Expert 1  Expert 2  │
│  Expert 3  Expert 4  │
└──────────────────┘
       ↓
┌─── Output Heads ───┐
│ Quality Assessment │
│ Semantic Matching  │
└─────────────────┘
```

### Key Components

1. **Feature Extractor**
   - Modified ResNet backbone
   - Adapted for pixel art features
   - Output: High-dimensional feature maps

2. **Expert Gate**
   - Learned routing mechanism
   - Soft assignment of inputs to experts
   - Dynamic expert selection

3. **Expert Modules**
   - Specialized neural networks
   - Each expert focuses on different aspects
   - Number of experts: Configurable (default: 4)

4. **Output Heads**
   - Quality score prediction
   - Semantic embedding generation
   - Multi-task learning approach

### Loss Functions

1. **Quality Assessment Loss**
   ```python
   quality_loss = -torch.mean(quality_scores)
   ```

2. **Semantic Matching Loss**
   ```python
   semantic_loss = F.mse_loss(pred_embedding, target_embedding)
   ```

## Model Configuration

### VAE Parameters

```python
vae_config = {
    'latent_dim': 256,      # Latent space dimension
    'channels': [64, 128, 256, 512],  # Channel progression
    'kernel_size': 3,       # Convolution kernel size
    'stride': 2,            # Downsampling/upsampling factor
    'padding': 1,           # Padding for same size
}
```

### Teacher Parameters

```python
teacher_config = {
    'num_experts': 4,       # Number of expert modules
    'feature_dim': 256,     # Feature dimension
    'embedding_dim': 64,    # Semantic embedding size
    'hidden_dim': 512,      # Expert network size
}
```

## Memory Requirements

Memory usage scales with model parameters and batch size:

1. **VAE Memory**
   - Parameters: ~35M
   - Activation memory: ~batch_size * 128 * 128 * 4
   - Gradient memory: ~parameters * 4

2. **Teacher Memory**
   - Parameters: ~16M
   - Activation memory: ~batch_size * 256 * 4
   - Expert memory: ~num_experts * hidden_dim * 4

## Optimization Guidelines

1. **VAE Training**
   - Start with lower learning rates (1e-4)
   - Gradually increase KL weight
   - Monitor reconstruction quality

2. **Teacher Training**
   - Higher learning rates possible (2e-4)
   - Balance quality and semantic losses
   - Regular expert utilization checks

3. **Joint Training**
   - Alternate between models if needed
   - Adjust reward scaling carefully
   - Monitor both losses for stability

## Extension and Modification

### Adding New Features

1. **VAE Modifications**
   ```python
   class CustomVAE(LunarCoreVAE):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           # Add custom layers
           self.custom_layer = nn.Sequential(...)
   ```

2. **Teacher Modifications**
   ```python
   class CustomTeacher(LunarMoETeacher):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           # Add custom experts
           self.new_expert = ExpertModule(...)
   ```

### Custom Loss Functions

```python
def custom_loss(recon, target, mu, logvar):
    # Standard losses
    recon_loss = F.mse_loss(recon, target)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Custom loss components
    style_loss = compute_style_loss(recon, target)
    
    return recon_loss + kl_loss + style_loss
```

# Model Implementation Details

## VAE Implementation

### Network Architecture

```python
class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),    # 64x64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 32x32x128
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # 16x16x256
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), # 8x8x512
            nn.ReLU(),
        )
        
        # Latent projections
        self.fc_mu = nn.Linear(8*8*512, latent_dim)
        self.fc_var = nn.Linear(8*8*512, latent_dim)
        
        # Decoder layers
        self.decoder_input = nn.Linear(latent_dim, 8*8*512)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 16x16x256
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 32x32x128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64x64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),    # 128x128x3
            nn.Tanh()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 512, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

### Loss Functions

```python
def vae_loss(recon_x, x, mu, logvar, kl_weight=0.01):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
```

## Quality Assessment Implementation

### Network Architecture

```python
class QualityAssessor(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        # Feature extraction
        self.backbone = nn.Sequential(
            ResNetBlock(3, 64),      # 128x128x64
            ResNetBlock(64, 128),    # 64x64x128
            ResNetBlock(128, 256),   # 32x32x256
            ResNetBlock(256, 512),   # 16x16x512
            nn.AdaptiveAvgPool2d(1)  # 512x1x1
        )
        
        # Quality scoring
        self.quality_head = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        quality = self.quality_head(features)
        return quality

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out
```

### Quality Loss

```python
def quality_loss(quality_scores, target_quality=0.8):
    return F.binary_cross_entropy(quality_scores, 
                                torch.ones_like(quality_scores) * target_quality)
```

## Training Integration

### Combined Training Step

```python
def training_step(batch, vae, quality_net, optimizers, scaler):
    vae_opt, quality_opt = optimizers
    
    # Zero gradients
    vae_opt.zero_grad()
    quality_opt.zero_grad()
    
    with torch.cuda.amp.autocast():
        # VAE forward pass
        recon_x, mu, logvar = vae(batch)
        
        # Quality assessment
        quality_scores = quality_net(recon_x)
        
        # Compute losses
        vae_total_loss, recon_loss, kl_loss = vae_loss(
            recon_x, batch, mu, logvar
        )
        qual_loss = quality_loss(quality_scores)
        
        # Combined loss
        total_loss = vae_total_loss + qual_loss
    
    # Backward pass with gradient scaling
    scaler.scale(total_loss).backward()
    
    # Optimize
    scaler.step(vae_opt)
    scaler.step(quality_opt)
    scaler.update()
    
    return {
        'loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'kl_loss': kl_loss.item(),
        'quality_loss': qual_loss.item(),
        'quality_score': quality_scores.mean().item()
    }
```

## Memory Optimization

### Gradient Checkpointing

```python
def enable_checkpointing(model):
    # Enable gradient checkpointing for all eligible layers
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            module.register_forward_hook(lambda m, _, output: output.requires_grad_(True))
    
    def checkpoint_forward(self, x):
        return torch.utils.checkpoint.checkpoint(self.forward_impl, x)
    
    model.forward = types.MethodType(checkpoint_forward, model)
```

### Mixed Precision Setup

```python
def setup_mixed_precision():
    # Initialize gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Optimize CUDA operations
    torch.backends.cudnn.benchmark = True
    
    # Set default tensor type
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    return scaler
```

## Model Configuration

### Default Configuration

```python
model_config = {
    # VAE parameters
    'latent_dim': 256,
    'encoder_channels': [64, 128, 256, 512],
    'decoder_channels': [512, 256, 128, 64],
    'activation': 'relu',
    
    # Quality assessment parameters
    'feature_dim': 256,
    'quality_threshold': 0.8,
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 1e-4,
    'kl_weight': 0.01,
    'quality_weight': 1.0,
    
    # Optimization
    'mixed_precision': True,
    'gradient_checkpointing': False,
    'weight_decay': 1e-6
}
```

## Extension Guidelines

### Custom Encoder Example

```python
class CustomEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.Conv2d(256, 512, 3, 2, 1)
        ])
        
        self.attention = SelfAttention(512)
        self.fc_mu = nn.Linear(512 * 8 * 8, latent_dim)
        self.fc_var = nn.Linear(512 * 8 * 8, latent_dim)
    
    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        
        return self.fc_mu(x), self.fc_var(x)
```

### Custom Quality Metric Example

```python
class CustomQualityMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        
        self.quality_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.quality_head(features)
``` 