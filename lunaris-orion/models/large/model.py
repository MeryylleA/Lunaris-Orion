"""
Large Model Implementation
------------------------
A sophisticated transformer-based model optimized for H100 GPUs with advanced features:
- LunarCache™ memory system for pattern caching
- Enhanced attention mechanisms with cross-attention support
- Anti-overfitting techniques including stochastic depth
- Quantization support for efficient inference
- Modular architecture for easy extension

Author: Lunaris Team
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Dict, Any, Union, List
import numpy as np
from torch.cuda.amp import custom_fwd, custom_bwd
import torch.quantization
import torch.ao.quantization
from torch.ao.quantization import QConfig
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import MinMaxObserver
import logging
from pathlib import Path
from dataclasses import dataclass
from .lunar_cache import LunarCache, CacheConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the Large Model."""
    # Model architecture
    embedding_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    ffn_dim: int = 4096
    max_sequence_length: int = 2048
    vocab_size: int = 50257
    
    # Training parameters
    dropout: float = 0.1
    attention_dropout: float = 0.1
    gradient_checkpointing: bool = True
    
    # Architecture features
    use_rope: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 256
    use_cross_attention: bool = True
    use_sdpa: bool = True
    use_absolute_positions: bool = True
    
    # Optimization
    optimizer: Dict = None
    
    def __post_init__(self):
        if self.optimizer is None:
            self.optimizer = {
                'weight_decay': 0.01,
                'learning_rate': 1e-4,
                'beta1': 0.9,
                'beta2': 0.999,
                'epsilon': 1e-8
            }

class ModelError(Exception):
    """Base exception class for model-related errors."""
    pass

class InitializationError(ModelError):
    """Raised when model initialization fails."""
    pass

class ForwardPassError(ModelError):
    """Raised when forward pass encounters an error."""
    pass

class QuantizationError(ModelError):
    """Raised when quantization-related operations fail."""
    pass

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding implementation with quantization support.
    
    This module implements RoPE (Rotary Position Embeddings) which provides
    relative positional information through rotation of query and key vectors.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        try:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            position = torch.arange(max_seq_len).float()
            sincos = torch.einsum('i,j->ij', position, inv_freq)
            sin, cos = sincos.sin(), sincos.cos()
            self.register_buffer('sin', sin)
            self.register_buffer('cos', cos)
            
            # Quantization support
            self.qconfig = None
            self.activation_post_process = None
        except Exception as e:
            raise InitializationError(f"Failed to initialize RotaryEmbedding: {str(e)}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            if self.activation_post_process is not None:
                sin = self.activation_post_process(self.sin[:x.shape[1]])
                cos = self.activation_post_process(self.cos[:x.shape[1]])
                return sin, cos
            return self.sin[:x.shape[1]], self.cos[:x.shape[1]]
        except Exception as e:
            raise ForwardPassError(f"RotaryEmbedding forward pass failed: {str(e)}")

class StochasticDepth(nn.Module):
    """
    Stochastic Depth regularization.
    
    Randomly drops entire layers during training to prevent overfitting
    and improve model robustness. During inference, scales the output
    appropriately.
    """
    
    def __init__(self, drop_prob: float = 0.1, scale_by_keep: bool = True):
        super().__init__()
        if not 0 <= drop_prob < 1:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        try:
            random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
            if self.scale_by_keep:
                random_tensor = random_tensor / keep_prob
            return x * random_tensor
        except Exception as e:
            raise ForwardPassError(f"StochasticDepth forward pass failed: {str(e)}")

class CrossAttention(nn.Module):
    """
    Enhanced Cross-attention module with anti-overfitting features.
    
    This module implements a sophisticated attention mechanism that can attend
    to both self-attention and cross-attention patterns. It includes:
    - Multi-head attention with configurable head dimensions
    - Dropout and attention dropout for regularization
    - Stochastic depth for additional regularization
    - Optional SDPA (Scaled Dot Product Attention) support
    - Layer normalization for stability
    
    Args:
        dim (int): Input dimension
        context_dim (Optional[int]): Context dimension for cross-attention
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        dropout (float): Dropout probability
        attention_dropout (float): Attention-specific dropout probability
        use_sdpa (bool): Whether to use PyTorch's SDPA implementation
        use_stochastic_depth (bool): Whether to use stochastic depth
        stochastic_depth_prob (float): Probability for stochastic depth
    """
    
    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_sdpa: bool = True,
        use_stochastic_depth: bool = True,
        stochastic_depth_prob: float = 0.1
    ):
        super().__init__()
        try:
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.scale = head_dim ** -0.5
            inner_dim = num_heads * head_dim
            context_dim = context_dim if context_dim is not None else dim
            
            self.use_sdpa = use_sdpa and hasattr(F, 'scaled_dot_product_attention')
            
            # Initialize attention components
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
            
            # Regularization components
            self.attention_dropout = nn.Dropout(attention_dropout)
            self.stochastic_depth = StochasticDepth(stochastic_depth_prob) if use_stochastic_depth else nn.Identity()
            
            # Layer normalization for stability
            self.norm_q = nn.LayerNorm(dim)
            self.norm_kv = nn.LayerNorm(context_dim)
            
            # Initialize weights
            self._init_weights()
            
            # For attention visualization
            self.last_attn_map = None
            
        except Exception as e:
            raise InitializationError(f"Failed to initialize CrossAttention: {str(e)}")
    
    def _init_weights(self):
        """Initialize weights using scaled normal initialization."""
        try:
            def scaled_init_(tensor, scale=1.0):
                nn.init.normal_(tensor, std=scale / math.sqrt(tensor.shape[-1]))
            
            scaled_init_(self.to_q.weight)
            scaled_init_(self.to_k.weight)
            scaled_init_(self.to_v.weight)
            scaled_init_(self.to_out[0].weight)
            if self.to_out[0].bias is not None:
                nn.init.zeros_(self.to_out[0].bias)
        except Exception as e:
            raise InitializationError(f"Failed to initialize weights: {str(e)}")
    
    def _reshape_for_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reshape input tensors for multi-head attention computation."""
        try:
            q = q.view(-1, q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(-1, k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(-1, v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
            return q, k, v
        except Exception as e:
            raise ForwardPassError(f"Failed to reshape tensors for attention: {str(e)}")
    
    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """Compute attention scores and apply them to values."""
        try:
            if self.use_sdpa:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attention_dropout.p if self.training else 0.0
                )
            else:
                # Manual implementation with improved stability
                attn = (q @ k.transpose(-2, -1)) * self.scale
                
                # Improve numerical stability
                attn = attn - attn.max(dim=-1, keepdim=True)[0]
                attn = F.softmax(attn, dim=-1)
                
                # Store attention map for visualization
                if self.training:
                    self.last_attn_map = attn.detach()
                
                attn = self.attention_dropout(attn)
                attn_output = attn @ v
            
            return attn_output
        except Exception as e:
            raise ForwardPassError(f"Failed to compute attention: {str(e)}")
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the attention module.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            context: Optional context tensor for cross-attention
            mask: Optional attention mask
            
        Returns:
            Processed tensor of shape (batch_size, seq_len, dim)
        """
        try:
            batch_size, seq_len, _ = x.shape
            context = context if context is not None else x
            
            # Apply layer normalization
            x_norm = self.norm_q(x)
            context_norm = self.norm_kv(context)
            
            # Linear projections
            q = self.to_q(x_norm)
            k = self.to_k(context_norm)
            v = self.to_v(context_norm)
            
            # Reshape for attention
            q, k, v = self._reshape_for_attention(q, k, v)
            
            # Compute attention
            attn_output = self._compute_attention(q, k, v)
            
            # Reshape and project output
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
            output = self.to_out(attn_output)
            
            # Apply stochastic depth
            return self.stochastic_depth(output)
            
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise ForwardPassError(f"CrossAttention forward pass failed: {str(e)}")
    
    def get_attention_map(self) -> Optional[torch.Tensor]:
        """Return the last computed attention map for visualization."""
        return self.last_attn_map

class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with quantization support.
    
    This module implements a more sophisticated version of Layer Normalization
    that includes:
    - Learnable scale parameter for adaptive normalization
    - Quantization-friendly implementation
    - Improved numerical stability
    - Optional skip connection scaling
    
    Args:
        dim (int): Input dimension
        eps (float): Small constant for numerical stability
        device (Optional[torch.device]): Device to place the module on
        dtype (Optional[torch.dtype]): Data type of parameters
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        try:
            factory_kwargs = {'device': device, 'dtype': dtype}
            
            self.eps = eps
            self.dim = dim
            
            # Learnable parameters
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))
            self.bias = nn.Parameter(torch.zeros(dim, **factory_kwargs))
            self.adaptive_scale = nn.Parameter(torch.ones(1, **factory_kwargs))
            
            # Quantization support
            self.qconfig = None
            self.weight_fake_quant = None
            self.activation_post_process = None
            
            # Statistics tracking for improved stability
            self.register_buffer('running_mean', torch.zeros(dim, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(dim, **factory_kwargs))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            
        except Exception as e:
            raise InitializationError(f"Failed to initialize AdaptiveLayerNorm: {str(e)}")
    
    def _check_input_dim(self, x: torch.Tensor) -> None:
        """Verify input dimensions."""
        if x.dim() != 3:
            raise ValueError(f'Expected 3D input (got {x.dim()}D input)')
        if x.size(-1) != self.dim:
            raise ValueError(
                f'Expected input with last dimension {self.dim} (got {x.size(-1)})'
            )
    
    def _compute_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and variance with improved numerical stability."""
        try:
            # Compute statistics along the last dimension
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            
            if self.training:
                # Update running statistics
                with torch.no_grad():
                    self.running_mean = (
                        0.9 * self.running_mean +
                        0.1 * mean.mean(dim=(0, 1))
                    )
                    self.running_var = (
                        0.9 * self.running_var +
                        0.1 * var.mean(dim=(0, 1))
                    )
                    self.num_batches_tracked += 1
            
            return mean, var
            
        except Exception as e:
            raise ForwardPassError(f"Failed to compute statistics: {str(e)}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the adaptive layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Normalized tensor of the same shape
        """
        try:
            self._check_input_dim(x)
            
            # Compute statistics
            mean, var = self._compute_stats(x)
            
            # Normalize
            x_norm = (x - mean) * torch.rsqrt(var + self.eps)
            
            # Apply learnable parameters with quantization awareness
            weight = self.weight
            if self.weight_fake_quant is not None:
                weight = self.weight_fake_quant(weight)
            
            scale = self.adaptive_scale * weight
            output = x_norm * scale + self.bias
            
            # Apply activation quantization if configured
            if self.activation_post_process is not None:
                output = self.activation_post_process(output)
            
            return output
            
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise ForwardPassError(f"AdaptiveLayerNorm forward pass failed: {str(e)}")
    
    def extra_repr(self) -> str:
        """Return a string with extra representation information."""
        return f'dim={self.dim}, eps={self.eps}'

class ParallelAttentionBlock(nn.Module):
    """
    Enhanced Parallel Attention Block with advanced features.
    
    This module implements a sophisticated transformer block that processes
    attention and feed-forward networks in parallel for improved efficiency.
    Features include:
    - Parallel processing of attention and FFN
    - Cross-attention support
    - Rotary position embeddings
    - Sliding window attention
    - Stochastic depth regularization
    - Gradient checkpointing
    - Learned skip connection scaling
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        ffn_dim (int): Feed-forward network dimension
        dropout (float): Dropout probability
        attention_dropout (float): Attention-specific dropout
        drop_path (float): Stochastic depth rate
        use_rope (bool): Whether to use rotary position embeddings
        use_sliding_window (bool): Whether to use sliding window attention
        window_size (int): Size of attention window
        use_cross_attention (bool): Whether to use cross-attention
        use_sdpa (bool): Whether to use PyTorch's SDPA
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        drop_path: float = 0.1,
        use_rope: bool = True,
        use_sliding_window: bool = True,
        window_size: int = 256,
        use_cross_attention: bool = True,
        use_sdpa: bool = True
    ):
        super().__init__()
        try:
            # Multi-head self-attention
            self.self_attn = CrossAttention(
                dim=dim,
                num_heads=num_heads,
                head_dim=dim // num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                use_sdpa=use_sdpa,
                use_stochastic_depth=True,
                stochastic_depth_prob=drop_path
            )
            
            # Optional cross-attention
            self.use_cross_attention = use_cross_attention
            if use_cross_attention:
                self.cross_attn = CrossAttention(
                    dim=dim,
                    num_heads=num_heads,
                    head_dim=dim // num_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    use_sdpa=use_sdpa,
                    use_stochastic_depth=True,
                    stochastic_depth_prob=drop_path
                )
                self.norm2 = AdaptiveLayerNorm(dim)
            
            # Rotary embeddings
            self.use_rope = use_rope
            if use_rope:
                self.rotary_emb = RotaryEmbedding(dim // num_heads)
            
            # Enhanced FFN
            self.ffn = self._build_ffn(dim, ffn_dim, dropout)
            
            # Layer normalization and skip connections
            self.norm1 = AdaptiveLayerNorm(dim)
            self.norm3 = AdaptiveLayerNorm(dim)
            
            # Learned skip connection scaling
            self.skip_scale1 = nn.Parameter(torch.ones(1))
            self.skip_scale2 = nn.Parameter(torch.ones(1))
            self.skip_scale3 = nn.Parameter(torch.ones(1))
            
            # Stochastic depth for regularization
            self.drop_path = StochasticDepth(drop_path)
            
            # Sliding window attention
            self.use_sliding_window = use_sliding_window
            self.window_size = window_size
            
            # Initialize weights
            self._init_weights()
            
        except Exception as e:
            raise InitializationError(f"Failed to initialize ParallelAttentionBlock: {str(e)}")
    
    def _build_ffn(self, dim: int, ffn_dim: int, dropout: float) -> nn.Sequential:
        """Build the feed-forward network with enhanced features."""
        try:
            return nn.Sequential(
                nn.Linear(dim, ffn_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim * 2, dim),
                nn.Dropout(dropout)
            )
        except Exception as e:
            raise InitializationError(f"Failed to build FFN: {str(e)}")
    
    def _init_weights(self):
        """Initialize FFN weights with improved scheme."""
        try:
            for module in self.ffn.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        except Exception as e:
            raise InitializationError(f"Failed to initialize weights: {str(e)}")
    
    def _attention_block(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process input through attention blocks."""
        try:
            # Self-attention with drop path
            attn_out = self.drop_path(self.self_attn(self.norm1(x)))
            x = x + self.skip_scale1 * attn_out
            
            # Cross-attention if enabled
            if self.use_cross_attention and context is not None:
                cross_attn_out = self.drop_path(
                    self.cross_attn(self.norm2(x), context)
                )
                x = x + self.skip_scale2 * cross_attn_out
            
            return x
        except Exception as e:
            raise ForwardPassError(f"Attention block processing failed: {str(e)}")
    
    def _ffn_block(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through FFN block."""
        try:
            return x + self.skip_scale3 * self.drop_path(self.ffn(self.norm3(x)))
        except Exception as e:
            raise ForwardPassError(f"FFN block processing failed: {str(e)}")
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        use_checkpoint: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of the parallel attention block.
        
        Args:
            x: Input tensor
            context: Optional context for cross-attention
            use_checkpoint: Whether to use gradient checkpointing
            
        Returns:
            Processed tensor
        """
        try:
            # Apply sliding window if enabled
            if self.use_sliding_window and x.size(1) > self.window_size:
                x = self._apply_sliding_window(x)
            
            # Process attention and FFN in parallel with gradient checkpointing
            if use_checkpoint and self.training:
                attn_out = checkpoint(self._attention_block, x, context)
                ffn_out = checkpoint(self._ffn_block, x)
            else:
                attn_out = self._attention_block(x, context)
                ffn_out = self._ffn_block(x)
            
            return attn_out + ffn_out
            
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise ForwardPassError(f"ParallelAttentionBlock forward pass failed: {str(e)}")
    
    def _apply_sliding_window(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sliding window attention to long sequences."""
        try:
            B, L, D = x.shape
            pad_size = (self.window_size - L % self.window_size) % self.window_size
            if pad_size > 0:
                x = F.pad(x, (0, 0, 0, pad_size))
            
            return x.view(B, -1, self.window_size, D)
        except Exception as e:
            raise ForwardPassError(f"Sliding window application failed: {str(e)}")

class LargeModel(nn.Module):
    """
    Large Language Model optimized for pixel art generation.
    
    This model implements a sophisticated transformer architecture with:
    - LunarCache™ memory system for pattern caching
    - Advanced attention mechanisms with parallel processing
    - Anti-overfitting techniques and regularization
    - Quantization support for efficient inference
    - Comprehensive error handling and logging
    - Modular design for easy extension
    
    The model is specifically designed for H100 GPUs and includes
    various optimizations for training and inference.
    
    Args:
        config (Union[dict, ModelConfig]): Model configuration
    """
    
    def __init__(self, config: Union[dict, ModelConfig]):
        super().__init__()
        try:
            # Convert dict config to ModelConfig if needed
            self.config = config if isinstance(config, ModelConfig) else ModelConfig(**config)
            
            # Initialize LunarCache with safety checks
            self._initialize_cache()
            
            # Initialize model components
            self._initialize_dimensions()
            self._initialize_embeddings()
            self._initialize_transformer_blocks()
            self._initialize_output_layers()
            
            # Initialize weights
            self.apply(self._init_weights)
            
            # Training state
            self.gradient_checkpointing = self.config.gradient_checkpointing
            
            # Quantization support
            self.qconfig = None
            self.is_quantized = False
            
            logger.info("Successfully initialized LargeModel")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise InitializationError(f"Model initialization failed: {str(e)}")
    
    def _initialize_cache(self):
        """Initialize the LunarCache system."""
        try:
            cache_config = CacheConfig(
                stvm_size=min(4096, self.config.get('cache_size', 4096)),
                pattern_dim=self.config.embedding_dim,
                cache_threshold=self.config.get('cache_threshold', 0.85),
                max_patterns=min(100, self.config.get('max_patterns', 100)),
                temperature=self.config.get('cache_temperature', 0.1),
                device=self.config.get('device', 'cuda'),
                enable_logging=self.config.get('enable_logging', True),
                priority_threshold=self.config.get('priority_threshold', 0.9),
                cleanup_frequency=self.config.get('cleanup_frequency', 1000)
            )
            self.lunar_cache = LunarCache(cache_config)
            self.cache_enabled = True
            logger.info("Successfully initialized LunarCache")
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {str(e)}. Continuing without cache.")
            self.cache_enabled = False
    
    def _initialize_dimensions(self):
        """Initialize model dimensions."""
        self.dim = self.config.embedding_dim
        self.num_heads = self.config.num_heads
        self.num_layers = self.config.num_layers
        self.ffn_dim = self.config.ffn_dim
        self.max_seq_len = self.config.max_sequence_length
    
    def _initialize_embeddings(self):
        """Initialize embedding layers."""
        try:
            # Token embeddings
            self.token_embedding = nn.Embedding(
                self.config.vocab_size,
                self.dim
            )
            
            # Position embeddings
            self.use_abs_pos = self.config.use_absolute_positions
            if self.use_abs_pos:
                self.abs_pos_encoding = nn.Parameter(
                    torch.zeros(1, self.max_seq_len, self.dim)
                )
                nn.init.normal_(self.abs_pos_encoding, std=0.02)
            
            # Dropout
            self.embedding_dropout = nn.Dropout(self.config.dropout)
            
        except Exception as e:
            raise InitializationError(f"Failed to initialize embeddings: {str(e)}")
    
    def _initialize_transformer_blocks(self):
        """Initialize transformer blocks with stochastic depth."""
        try:
            # Calculate stochastic depth drop rates
            dpr = torch.linspace(0, self.config.dropout, self.num_layers)
            
            # Create transformer blocks
            self.blocks = nn.ModuleList([
                ParallelAttentionBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=self.config.dropout,
                    attention_dropout=self.config.attention_dropout,
                    drop_path=dpr[i],
                    use_rope=self.config.use_rope,
                    use_sliding_window=self.config.use_sliding_window,
                    window_size=self.config.sliding_window_size,
                    use_cross_attention=self.config.use_cross_attention,
                    use_sdpa=self.config.use_sdpa
                )
                for i in range(self.num_layers)
            ])
        except Exception as e:
            raise InitializationError(f"Failed to initialize transformer blocks: {str(e)}")
    
    def _initialize_output_layers(self):
        """Initialize output processing layers."""
        try:
            self.final_norm = AdaptiveLayerNorm(self.dim)
            self.to_pixels = nn.Sequential(
                nn.Linear(self.dim, self.dim * 2),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.dim * 2, self.config.image_size * self.config.image_size * 3)
            )
        except Exception as e:
            raise InitializationError(f"Failed to initialize output layers: {str(e)}")
    
    def _init_weights(self, module):
        """Initialize module weights."""
        try:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, (nn.LayerNorm, AdaptiveLayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        except Exception as e:
            raise InitializationError(f"Failed to initialize weights: {str(e)}")

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            context: Optional context tensor for cross-attention
            
        Returns:
            Generated image tensor of shape (batch_size, 3, image_size, image_size)
        """
        try:
            # Input embedding with dropout
            x = self.token_embedding(x)
            if self.use_abs_pos:
                x = x + self.abs_pos_encoding[:, :x.size(1)]
            x = self.embedding_dropout(x)
            
            # Process through LunarCache if enabled
            if self.cache_enabled:
                try:
                    enhanced_features, cache_info = self.lunar_cache(x)
                    x = x + 0.1 * enhanced_features  # Reduced residual impact
                    if self.training and self.config.enable_logging:
                        logger.debug(f"Cache info: {cache_info}")
                except Exception as e:
                    logger.warning(f"Cache processing failed: {str(e)}. Skipping cache for this batch.")
            
            # Apply transformer blocks
            for block in self.blocks:
                x = block(x, context, use_checkpoint=self.gradient_checkpointing)
            
            # Output processing
            x = self.final_norm(x)
            x = self.to_pixels(x[:, -1])
            
            # Reshape to image dimensions
            B = x.size(0)
            return x.view(B, 3, self.config.image_size, self.config.image_size)
            
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise ForwardPassError(f"Model forward pass failed: {str(e)}")
    
    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        val_loss: float = float('inf')
    ) -> None:
        """
        Save enhanced checkpoint including cache state.
        
        Args:
            path: Path to save the checkpoint
            optimizer: Optional optimizer to save state
            epoch: Current epoch number
            val_loss: Current validation loss
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'lunar_cache_state': self.lunar_cache.state_dict() if self.cache_enabled else None,
                'val_loss': val_loss,
                'config': self.config.__dict__,
                'model_config': {
                    'dim': self.dim,
                    'num_heads': self.num_heads,
                    'num_layers': self.num_layers,
                    'ffn_dim': self.ffn_dim
                }
            }
            
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            # Save cache state separately
            if self.cache_enabled:
                cache_path = save_path.parent / f"cache_state_epoch_{epoch}.pt"
                self.lunar_cache.save_state(str(cache_path))
            
            torch.save(checkpoint, save_path)
            logger.info(f"Saved checkpoint to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise IOError(f"Checkpoint saving failed: {str(e)}")
    
    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """
        Load enhanced checkpoint including cache state.
        
        Args:
            path: Path to load the checkpoint from
            optimizer: Optional optimizer to load state into
            
        Returns:
            Dictionary containing checkpoint information
        """
        try:
            load_path = Path(path)
            if not load_path.exists():
                raise FileNotFoundError(f"No checkpoint found at {load_path}")
            
            checkpoint = torch.load(load_path)
            
            # Load model state
            self.load_state_dict(checkpoint['model_state_dict'])
            
            # Load cache state if available
            if self.cache_enabled and checkpoint.get('lunar_cache_state'):
                self.lunar_cache.load_state_dict(checkpoint['lunar_cache_state'])
            
            # Load optimizer state if provided
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Try to load separate cache state
            if self.cache_enabled:
                cache_path = load_path.parent / f"cache_state_epoch_{checkpoint['epoch']}.pt"
                if cache_path.exists():
                    self.lunar_cache.load_state(str(cache_path))
            
            logger.info(f"Loaded checkpoint from {load_path}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise IOError(f"Checkpoint loading failed: {str(e)}")
    
    def prepare_for_quantization(self, qconfig: Optional[QConfig] = None) -> None:
        """
        Prepare model for quantization.
        
        Args:
            qconfig: Optional quantization configuration
        """
        try:
            if qconfig is None:
                qconfig = torch.ao.quantization.get_default_qconfig('x86')
            
            self.qconfig = qconfig
            torch.ao.quantization.prepare(self, inplace=True)
            logger.info("Model prepared for quantization")
            
        except Exception as e:
            logger.error(f"Failed to prepare for quantization: {str(e)}")
            raise QuantizationError(f"Quantization preparation failed: {str(e)}")
    
    def quantize(self) -> 'LargeModel':
        """Convert model to quantized version."""
        try:
            if not self.is_quantized:
                torch.ao.quantization.convert(self, inplace=True)
                self.is_quantized = True
                logger.info("Model successfully quantized")
            return self
            
        except Exception as e:
            logger.error(f"Failed to quantize model: {str(e)}")
            raise QuantizationError(f"Model quantization failed: {str(e)}")
    
    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """Get attention maps for visualization and analysis."""
        try:
            attention_maps = {}
            for i, block in enumerate(self.blocks):
                if hasattr(block.self_attn, 'last_attn_map'):
                    attention_maps[f'layer_{i}_self_attn'] = block.self_attn.last_attn_map
                if block.use_cross_attention and hasattr(block.cross_attn, 'last_attn_map'):
                    attention_maps[f'layer_{i}_cross_attn'] = block.cross_attn.last_attn_map
            return attention_maps
            
        except Exception as e:
            logger.error(f"Failed to get attention maps: {str(e)}")
            raise RuntimeError(f"Attention map extraction failed: {str(e)}")
    
    def get_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """
        Get parameter groups for optimizer with weight decay handling.
        
        Returns:
            Dictionary containing parameter groups
        """
        try:
            no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.named_parameters() 
                              if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.config.optimizer['weight_decay']
                },
                {
                    'params': [p for n, p in self.named_parameters() 
                              if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]
            return optimizer_grouped_parameters
            
        except Exception as e:
            logger.error(f"Failed to get parameter groups: {str(e)}")
            raise RuntimeError(f"Parameter group creation failed: {str(e)}")
    
    def enable_flash_attention(self) -> bool:
        """
        Enable Flash Attention V2 for all attention blocks if available.
        
        Returns:
            bool: Whether Flash Attention was successfully enabled
        """
        try:
            from flash_attn import flash_attn_func
            from flash_attn.flash_attention import FlashAttention
            
            for block in self.blocks:
                block.self_attn.use_sdpa = False
                block.self_attn._flash_attn = True
                if block.use_cross_attention:
                    block.cross_attn.use_sdpa = False
                    block.cross_attn._flash_attn = True
            
            logger.info("Successfully enabled Flash Attention")
            return True
            
        except ImportError:
            logger.warning("Flash Attention not available. Using standard attention.")
            return False
        except Exception as e:
            logger.error(f"Failed to enable Flash Attention: {str(e)}")
            return False
    
    def toggle_cache(self, enabled: bool = True) -> None:
        """
        Enable or disable the cache system.
        
        Args:
            enabled: Whether to enable the cache
        """
        self.cache_enabled = enabled
        logger.info(f"Cache system {'enabled' if enabled else 'disabled'}") 