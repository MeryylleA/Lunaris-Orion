"""
Mini model implementation for Lunaris Orion.
A lightweight transformer-based model for pixel art generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.embed_dim = config['embedding_dim']
        self.num_heads = config['num_heads']
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.dropout = nn.Dropout(config['dropout_rate'])
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.shape[0]
        
        # Linear projections and reshape
        q = self.q_proj(query).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)
        
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    """Feed-forward network with residual connection."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.fc1 = nn.Linear(config['embedding_dim'], config['ff_dim'])
        self.fc2 = nn.Linear(config['ff_dim'], config['embedding_dim'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward network."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.norm1 = nn.LayerNorm(config['embedding_dim'])
        self.norm2 = nn.LayerNorm(config['embedding_dim'])
        self.dropout = nn.Dropout(config['dropout_rate'])
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection and layer normalization
        attn_output = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x

class MiniModel(nn.Module):
    """
    Mini model for pixel art generation.
    A lightweight transformer-based model optimized for speed and efficiency.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Text embedding layers
        self.token_embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.position_embedding = nn.Parameter(torch.zeros(1, config['max_sequence_length'], config['embedding_dim']))
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['num_layers'])
        ])
        
        # Image generation layers
        self.to_pixels = nn.Sequential(
            nn.Linear(config['embedding_dim'], config['image_size'] * config['image_size'] * 4),
            nn.GELU(),
            nn.Linear(config['image_size'] * config['image_size'] * 4, config['image_size'] * config['image_size'] * 3)
        )
        
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.layer_norm = nn.LayerNorm(config['embedding_dim'])
        
    def forward(self, 
                tokens: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get batch size and sequence length
        batch_size, seq_length = tokens.shape
        
        # Combine token and position embeddings
        x = self.token_embedding(tokens) + self.position_embedding[:, :seq_length, :]
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Apply final layer norm
        x = self.layer_norm(x)
        
        # Generate pixel values (batch_size, seq_length, image_size * image_size * 3)
        pixels = self.to_pixels(x)
        
        # Reshape to image format (batch_size, 3, image_size, image_size)
        images = pixels.reshape(batch_size, -1, 3, self.config['image_size'], self.config['image_size'])
        
        # Take the last token's output as the final image
        images = images[:, -1]  # Shape: (batch_size, 3, image_size, image_size)
        
        # Normalize pixel values to [0, 1]
        images = torch.sigmoid(images)
        
        return images

    def generate(self, prompt_tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generate a pixel art image from a text prompt.
        
        Args:
            prompt_tokens: Tensor of token IDs (batch_size, seq_length)
            **kwargs: Additional generation parameters
            
        Returns:
            Tensor of generated images (batch_size, 3, image_size, image_size)
        """
        self.eval()
        with torch.no_grad():
            images = self.forward(prompt_tokens)
        return images 
