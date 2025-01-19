"""
Mini model implementation for Lunaris Orion.
Optimized for NVIDIA H100 GPU with flash attention and memory optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    """Multi-head attention with improved memory efficiency."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.num_heads = config.get('num_heads', 8)
        self.head_dim = config['embedding_dim'] // self.num_heads
        self.embedding_dim = config['embedding_dim']
        
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(f"embedding_dim {self.embedding_dim} must be divisible by num_heads {self.num_heads}")
            
        # Unified projection matrix for better efficiency
        self.qkv_proj = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.scale = self.head_dim ** -0.5
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize QKV projection
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
            
        # Initialize output projection
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = q.shape[0]
        
        # Unified QKV projection
        qkv = self.qkv_proj(q)
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(batch_size, -1, self.embedding_dim)
        
        # Output projection
        output = self.out_proj(context)
        return output

class TransformerBlock(nn.Module):
    """Optimized transformer block with better gradient flow."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config['embedding_dim'], config['ff_dim']),
            nn.GELU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(config['ff_dim'], config['embedding_dim'])
        )
        self.norm1 = nn.LayerNorm(config['embedding_dim'])
        self.norm2 = nn.LayerNorm(config['embedding_dim'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        
        # Initialize feed forward layers
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize feed forward weights for better training."""
        for module in self.feed_forward.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def enable_flash_attention(self):
        """Enable flash attention in the attention layer."""
        self.attention.enable_flash_attention()
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture for better training stability
        attended = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attended)
        
        # Feed forward with residual
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class PixelArtBlock(nn.Module):
    """Bloco especializado para geração de pixel art."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.conv(x)

class MiniModel(nn.Module):
    """
    Mini model for pixel art generation.
    Optimized for H100 GPU with improved architecture and memory efficiency.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        try:
            self.config = self._validate_config(config)
            
            # Text embedding layers with better initialization and error checks
            self.token_embedding = self._create_token_embedding()
            self.position_embedding = self._create_position_embedding()
            
            # Transformer layers with gradient checkpointing for memory efficiency
            self.transformer_blocks = self._create_transformer_blocks()
            
            # Image generation layers with improved architecture
            self.to_initial = self._create_initial_layers()
            self.pixel_art_generator = self._create_pixel_art_generator()
            
            # Improved dropout and normalization
            self.dropout = nn.Dropout(self.config['dropout_rate'])
            self.layer_norm = nn.LayerNorm(self.config['embedding_dim'])
            
            # Enhanced color palette system
            self.setup_color_palette()
            
            # Initialize weights with improved method
            self._init_weights()
            
            # Enable gradient checkpointing by default
            self.enable_gradient_checkpointing()
            
        except Exception as e:
            logging.error(f"Error initializing MiniModel: {str(e)}")
            raise
    
    def _validate_config(self, config):
        """Validate and set default values for config."""
        required_keys = [
            'vocab_size', 'embedding_dim', 'num_layers', 'image_size',
            'max_sequence_length', 'dropout_rate'
        ]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Set default values if not provided
        defaults = {
            'ff_dim': config['embedding_dim'] * 4,
            'num_heads': 8,
            'dropout_rate': 0.1,
            'palette_size': 16
        }
        
        return {**defaults, **config}
    
    def _create_token_embedding(self):
        """Create token embedding with proper initialization."""
        embedding = nn.Embedding(
            self.config['vocab_size'],
            self.config['embedding_dim'],
            padding_idx=0  # Add padding token
        )
        nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        return embedding
    
    def _create_position_embedding(self):
        """Create position embedding with proper initialization."""
        pos_embedding = nn.Parameter(
            torch.zeros(1, self.config['max_sequence_length'], self.config['embedding_dim'])
        )
        nn.init.normal_(pos_embedding, mean=0.0, std=0.02)
        return pos_embedding
    
    def _create_transformer_blocks(self):
        """Create transformer blocks with gradient checkpointing."""
        return nn.ModuleList([
            TransformerBlock(self.config)
            for _ in range(self.config['num_layers'])
        ])
    
    def _create_initial_layers(self):
        """Create initial layers with improved architecture."""
        return nn.Sequential(
            nn.LayerNorm(self.config['embedding_dim']),
            nn.Linear(self.config['embedding_dim'], 8 * 8 * self.config['embedding_dim']),
            nn.GELU(),
            nn.Dropout(self.config['dropout_rate'])
        )
    
    def _create_pixel_art_generator(self):
        """Create improved pixel art generator pipeline."""
        hidden_dim = self.config['embedding_dim']
        return nn.Sequential(
            PixelArtBlock(hidden_dim, hidden_dim),
            nn.Upsample(scale_factor=2, mode='nearest'),
            PixelArtBlock(hidden_dim, hidden_dim // 2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            PixelArtBlock(hidden_dim // 2, hidden_dim // 4),
            nn.Upsample(scale_factor=2, mode='nearest'),
            PixelArtBlock(hidden_dim // 4, 3),
            nn.Tanh()
        )
    
    def setup_color_palette(self):
        """Setup improved color palette system."""
        palette_size = self.config.get('palette_size', 16)
        self.color_palette = nn.Parameter(torch.randn(palette_size, 3))
        self.palette_attention = nn.Linear(self.config['embedding_dim'], palette_size)
        
        # Initialize with improved color distribution
        nn.init.uniform_(self.color_palette, -1, 1)
        self.color_palette.data = torch.clamp(self.color_palette.data, -1, 1)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        for block in self.transformer_blocks:
            block.gradient_checkpointing = True
    
    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        try:
            if tokens.dim() != 2:
                raise ValueError(f"Expected 2D input tensor, got {tokens.dim()}D")
                
            batch_size, seq_length = tokens.shape
            if seq_length > self.config['max_sequence_length']:
                raise ValueError(f"Sequence length {seq_length} exceeds maximum {self.config['max_sequence_length']}")
            
            # Text encoding with improved stability
            x = self.token_embedding(tokens) * math.sqrt(self.config['embedding_dim'])
            x = x + self.position_embedding[:, :seq_length, :]
            x = self.dropout(x)
            
            # Transformer processing with gradient checkpointing
            attention_mask = self._create_attention_mask(seq_length) if mask is None else mask
            
            for block in self.transformer_blocks:
                if getattr(block, 'gradient_checkpointing', False):
                    x = torch.utils.checkpoint.checkpoint(block, x, attention_mask)
                else:
                    x = block(x, attention_mask)
            
            # Global pooling with improved attention
            x = self.layer_norm(x)
            attention_weights = self._compute_attention_weights(x)
            x = torch.matmul(attention_weights, x).mean(dim=1)
            
            # Generate image with enhanced pipeline
            features = self.to_initial(x)
            features = features.view(batch_size, -1, 8, 8)
            
            images = self.pixel_art_generator(features)
            
            # Apply improved color palette
            images = self._apply_color_palette(images, x)
            
            return images
            
        except Exception as e:
            logging.error(f"Error in forward pass: {str(e)}")
            raise
    
    def _create_attention_mask(self, seq_length: int) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_length, seq_length, dtype=torch.bool),
            diagonal=1
        )
        return mask.unsqueeze(0)  # [1, seq_length, seq_length]
    
    def _compute_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention weights with improved numerical stability."""
        attention_logits = torch.matmul(x, x.transpose(-2, -1))
        attention_logits = attention_logits / math.sqrt(self.config['embedding_dim'])
        
        # Apply masking for numerical stability
        max_value = torch.max(attention_logits, dim=-1, keepdim=True)[0]
        exp_weights = torch.exp(attention_logits - max_value)
        
        return exp_weights / (exp_weights.sum(dim=-1, keepdim=True) + 1e-9)
    
    def _apply_color_palette(self, images: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Apply color palette with improved quantization."""
        try:
            # Generate custom palette for each image
            palette_weights = F.softmax(self.palette_attention(features), dim=-1)
            custom_palette = torch.matmul(palette_weights, self.color_palette)
            
            # Reshape for efficient computation
            images = images.permute(0, 2, 3, 1)  # [B, H, W, C]
            
            # Compute color distances with improved numerical stability
            distances = torch.norm(
                images.unsqueeze(-2) - custom_palette.view(-1, 1, 1, self.config['palette_size'], 3),
                dim=-1
            )
            
            # Find nearest colors
            color_indices = distances.argmin(dim=-1)
            
            # Apply palette colors
            quantized = custom_palette.view(-1, self.config['palette_size'], 3)[
                torch.arange(images.size(0)).view(-1, 1, 1),
                color_indices
            ]
            
            # Return to original format
            quantized = quantized.permute(0, 3, 1, 2)
            
            # Normalize to [0, 1] range
            quantized = (quantized + 1) / 2
            quantized = torch.clamp(quantized, 0, 1)
            
            return quantized
            
        except Exception as e:
            logging.error(f"Error in color palette application: {str(e)}")
            raise
    
    def generate(self, tokens: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
        """Generate images with improved stability and quality."""
        try:
            if not 0 < temperature <= 1.0:
                raise ValueError(f"Temperature must be in (0, 1], got {temperature}")
            
            with torch.no_grad():
                # Basic generation
                images = self.forward(tokens)
                
                # Apply temperature
                images = self._apply_temperature(images, temperature)
                
                # Enhance pixel art characteristics
                images = self.enhance_pixel_art(images)
                
                return images
                
        except Exception as e:
            logging.error(f"Error in image generation: {str(e)}")
            raise
    
    def _apply_temperature(self, images: torch.Tensor, temperature: float) -> torch.Tensor:
        """Apply temperature with improved stability."""
        # Scale logits by temperature
        scaled = images / temperature
        # Apply softmax for smooth color transitions
        weights = F.softmax(scaled.reshape(*scaled.shape[:2], -1), dim=-1)
        return weights.reshape(images.shape)
    
    def enhance_pixel_art(self, images: torch.Tensor) -> torch.Tensor:
        """Enhance pixel art characteristics."""
        # Quantize colors
        images = self.quantize_pixels(images, num_colors=16)
        
        # Enhance contrast
        images = self.normalize_contrast(images)
        
        # Apply pixelation effect
        images = self.pixelate(images, block_size=2)
        
        # Final cleanup
        images = torch.clamp(images, 0, 1)
        
        return images

    def quantize_pixels(self, images: torch.Tensor, num_colors: int = 32) -> torch.Tensor:
        """Quantize the image to use fewer colors, making it more pixel-art like."""
        # Scale to [0, num_colors-1]
        images = images * (num_colors - 1)
        # Round to nearest integer
        images = torch.round(images)
        # Scale back to [0, 1]
        images = images / (num_colors - 1)
        return images

    def normalize_contrast(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize contrast to use full range."""
        batch_size, channels, height, width = images.shape
        for b in range(batch_size):
            for c in range(channels):
                # Get min and max for this channel
                min_val = images[b, c].min()
                max_val = images[b, c].max()
                if max_val > min_val:  # Avoid division by zero
                    # Normalize to [0, 1]
                    images[b, c] = (images[b, c] - min_val) / (max_val - min_val)
        return images

    def pixelate(self, images: torch.Tensor, block_size: int = 4) -> torch.Tensor:
        """Create pixel art effect by averaging blocks of pixels."""
        batch_size, channels, height, width = images.shape
        
        # Ensure dimensions are divisible by block_size
        new_height = height - (height % block_size)
        new_width = width - (width % block_size)
        
        # Reshape to group pixels into blocks
        images = images[:, :, :new_height, :new_width]
        blocked = images.view(batch_size, channels, 
                            new_height // block_size, block_size,
                            new_width // block_size, block_size)
        
        # Average each block
        blocked = blocked.mean([3, 5])
        
        # Repeat values to restore original size
        images = blocked.repeat_interleave(block_size, dim=2).repeat_interleave(block_size, dim=3)
        
        return images

    def _init_weights(self):
        """Initialize weights with improved scaling."""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        
        # Initialize color palette
        nn.init.uniform_(self.color_palette, -1, 1)
        self.color_palette.data = torch.clamp(self.color_palette.data, -1, 1)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        # Initialize transformer blocks specially
        for block in self.transformer_blocks:
            # Initialize attention weights
            if hasattr(block.attention, 'qkv_proj'):
                nn.init.xavier_uniform_(block.attention.qkv_proj.weight)
                nn.init.xavier_uniform_(block.attention.out_proj.weight)
            
            # Initialize feed forward weights
            for module in block.feed_forward.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                        
        # Initialize pixel art generator specially
        for module in self.pixel_art_generator.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.InstanceNorm2d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias) 