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

class MultiHeadAttention(nn.Module):
    """Multi-head attention with flash attention support."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = config['embedding_dim']
        self.num_heads = config['num_heads']
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)  # Remove bias for better performance
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
        self.dropout = nn.Dropout(config['dropout_rate'])
        
        # Initialize with scaled weights
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize attention weights with scaled xavier uniform."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.shape[0]
        
        # Fused QKV projection for better performance
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape and transpose in one operation
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with flash attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch 2.0 flash attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Fallback to regular attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.embed_dim)
        
        return self.out_proj(attn_output)

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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture for better training stability
        attended = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attended)
        
        # Feed forward with residual
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class MiniModel(nn.Module):
    """
    Mini model for pixel art generation.
    Optimized for H100 GPU with improved architecture and memory efficiency.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Text embedding layers with better initialization
        self.token_embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.position_embedding = nn.Parameter(torch.zeros(1, config['max_sequence_length'], config['embedding_dim']))
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['num_layers'])
        ])
        
        # Image generation layers with improved capacity
        img_size = config['image_size']
        self.to_pixels = nn.Sequential(
            nn.LayerNorm(config['embedding_dim']),  # Normalize before generation
            nn.Linear(config['embedding_dim'], img_size * img_size * 8),
            nn.GELU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(img_size * img_size * 8, img_size * img_size * 4),
            nn.GELU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(img_size * img_size * 4, img_size * img_size * 3),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.layer_norm = nn.LayerNorm(config['embedding_dim'])
        
        # Initialize weights with better scaling
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with improved scaling."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        
        # Initialize image generation layers
        for module in self.to_pixels.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = tokens.shape
        
        # Combine embeddings with better gradient flow
        x = self.token_embedding(tokens) * math.sqrt(self.config['embedding_dim'])
        x = x + self.position_embedding[:, :seq_length, :]
        x = self.dropout(x)
        
        # Apply transformer blocks with residual connections
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Global pooling with attention weights
        x = self.layer_norm(x)
        attention_weights = F.softmax(torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.config['embedding_dim']), dim=-1)
        x = torch.matmul(attention_weights, x).mean(dim=1)
        
        # Generate pixel values with improved scaling
        pixels = self.to_pixels(x)
        images = pixels.reshape(batch_size, 3, self.config['image_size'], self.config['image_size'])
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        return images

    def generate(self, prompt_tokens: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
        """Generate a pixel art image with controlled randomness."""
        self.eval()
        with torch.no_grad():
            if temperature > 0:
                noise = torch.randn_like(self.position_embedding) * temperature
                position_embedding = self.position_embedding + noise
            else:
                position_embedding = self.position_embedding
            
            # Custom forward pass for generation
            batch_size, seq_length = prompt_tokens.shape
            x = self.token_embedding(prompt_tokens) * math.sqrt(self.config['embedding_dim'])
            x = x + position_embedding[:, :seq_length, :]
            x = self.dropout(x)
            
            for block in self.transformer_blocks:
                x = block(x)
            
            x = self.layer_norm(x)
            attention_weights = F.softmax(torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.config['embedding_dim']), dim=-1)
            x = torch.matmul(attention_weights, x).mean(dim=1)
            
            pixels = self.to_pixels(x)
            images = pixels.reshape(batch_size, 3, self.config['image_size'], self.config['image_size'])
            images = (images + 1) / 2
            images = torch.clamp(images, 0, 1)
            
        return images 