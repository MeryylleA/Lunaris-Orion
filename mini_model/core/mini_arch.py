"""
Arquitetura principal do modelo Mini para pixel art 16x16.
Otimizado para eficiência e qualidade em baixa resolução.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any, List
import yaml
from pathlib import Path
from torchvision import models
import numpy as np
import math
import torch.utils.checkpoint as checkpoint
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import spectral_norm
from einops import rearrange, repeat

# Configurações de otimização
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Desabilitar otimizações que requerem compilador
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.jit.enable = False
torch._dynamo.config.dynamic_shapes = False
torch._dynamo.config.cache_size_limit = 0

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Flash Attention não disponível, usando atenção padrão")

try:
    import transformer_engine.pytorch as te
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("Transformer Engine não disponível, usando camadas padrão")

# Função de checkpoint customizada
def checkpoint_wrapper(module, *args, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs)
    return checkpoint.checkpoint(custom_forward, *args, **kwargs)

class PositionalEmbedding(nn.Module):
    """Embedding posicional otimizado para sequências 2D."""
    
    def __init__(self, dim: int, max_size: int = 32):
        super().__init__()
        self.d_model = dim
        self.max_size = max_size
        
        # Pré-computar embeddings para eficiência
        position = torch.arange(0, max_size * max_size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(1, max_size * max_size, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            # Calcular posições reais ao invés de usar max_size pré-definido
            position = torch.stack(torch.meshgrid(
                torch.arange(H, device=x.device),
                torch.arange(W, device=x.device)
            ), dim=-1).float().view(1, H*W, 2)
            
            # Usar senoides para coordenadas separadas
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * 
                               (-math.log(10000.0) / self.d_model))
            pe = torch.zeros(1, H*W, self.d_model, device=x.device)
            pe[..., 0::2] = torch.sin(position[..., 0:1] * div_term)
            pe[..., 1::2] = torch.cos(position[..., 1:2] * div_term)
            return pe.expand(B, -1, -1)
        return self.pe[:, :x.size(1), :].expand(x.size(0), -1, -1)

class AdaIN(nn.Module):
    """AdaIN otimizado com FP16 e TF32."""
    def __init__(self, dim: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(dim, affine=False)
        self.style = nn.Sequential(
            nn.Linear(style_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2)
        )
        
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        style = self.style(style)
        scale, bias = style.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return normalized * (1 + scale) + bias

class ColorQuantizer(nn.Module):
    """Quantizador otimizado com cache de paleta."""
    def __init__(self, num_colors: int = 16):
        super().__init__()
        self.num_colors = num_colors
        # Inicialização mais estável com cores normalizadas
        self.colors = torch.rand(num_colors, 3) 
        self.register_buffer('palette', self.colors)
    
    def forward(self, x: torch.Tensor, temperature: float = 0.01) -> torch.Tensor:
        # Otimizar para CPU usando MPS se disponível
        if not x.is_cuda and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3).to('mps')
            with torch.cuda.amp.autocast(enabled=False):
                distances = torch.cdist(x_flat, self.palette)
                logits = -distances / temperature
                weights = F.softmax(logits, dim=-1)
                
                indices = weights.argmax(dim=-1)
                hard_weights = F.one_hot(indices, num_classes=self.num_colors).float()
                quantized = torch.matmul(hard_weights, self.palette)
            
            out = x_flat + (quantized - x_flat).detach()
            # Retornar ao shape original
            return out.reshape(x.shape[0], x.shape[2], x.shape[3], 3).permute(0, 3, 1, 2)
        else:
            # Preservar shape original
            original_shape = x.shape
            
            # Adicionar ruído controlado para melhor treinamento
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3)
            x_flat = x_flat + torch.randn_like(x_flat) * 0.01
            
            with torch.cuda.amp.autocast(enabled=False):
                distances = torch.cdist(x_flat, self.palette)
                logits = -distances / temperature
                weights = F.softmax(logits, dim=-1)
                
                indices = weights.argmax(dim=-1)
                hard_weights = F.one_hot(indices, num_classes=self.num_colors).float()
                quantized = torch.matmul(hard_weights, self.palette)
            
            out = x_flat + (quantized - x_flat).detach()
            # Retornar ao shape original
            return out.reshape(original_shape[0], original_shape[2], original_shape[3], 3).permute(0, 3, 1, 2)

class FlashAttention(nn.Module):
    """Atenção otimizada com suporte a Flash Attention."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim deve ser divisível por num_heads"
        
        # Simplificar inicialização das camadas
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Ajuste na inicialização para melhor convergência
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(self.head_dim))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(self.head_dim))
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, x: torch.Tensor, return_attention: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Adicionar fallback para CPU
        if not x.is_cuda and not FLASH_ATTN_AVAILABLE:
            return self._fallback_attention(x)
        
        B, L, D = x.shape
        H = self.num_heads
        
        # Usar checkpoint para economizar memória
        def attention_forward(q, k, v):
            if FLASH_ATTN_AVAILABLE and x.is_cuda:
                output = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0)
                attn = None
            else:
                scale = 1.0 / math.sqrt(self.head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                
                attn = F.softmax(scores, dim=-1)
                if self.dropout > 0 and self.training:
                    attn = F.dropout(attn, p=self.dropout)
                
                output = torch.matmul(attn, v)
            return output, attn
        
        q = self.q_proj(x).view(B, L, H, -1).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, -1).transpose(1, 2)
        
        if self.training:
            output, attn = checkpoint.checkpoint(attention_forward, q, k, v)
        else:
            output, attn = attention_forward(q, k, v)
            
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.out_proj(output)
        
        return output, attn

    def _fallback_attention(self, x):
        """Implementação alternativa otimizada para CPU"""
        # Código de atenção padrão sem flash
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn

class SparseAttention(nn.Module):
    """Implementação de atenção esparsa para eficiência."""
    def __init__(self, dim, num_heads=8, dropout=0.0, causal=False, window_size=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.causal = causal
        
        self.q_proj = spectral_norm(nn.Linear(dim, dim))
        self.k_proj = spectral_norm(nn.Linear(dim, dim))
        self.v_proj = spectral_norm(nn.Linear(dim, dim))
        self.out_proj = spectral_norm(nn.Linear(dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        H = self.num_heads

        # Project and reshape
        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=H)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=H)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=H)

        # Split into windows
        q = rearrange(q, 'b h (w n) d -> b h w n d', w=self.window_size)
        k = rearrange(k, 'b h (w n) d -> b h w n d', w=self.window_size)
        v = rearrange(v, 'b h (w n) d -> b h w n d', w=self.window_size)

        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            dots = dots.masked_fill(mask == 0, float('-inf'))

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.triu(torch.ones(i, j), j - i + 1).bool()
            dots = dots.masked_fill(mask, float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = rearrange(out, 'b h w n d -> b (w n) (h d)')
        out = self.out_proj(out)

        return out, attn

class TransformerBlock(nn.Module):
    """Bloco transformer otimizado com self-attention e cross-attention."""
    
    def __init__(self, dim, num_heads=8, ff_dim=None, dropout=0.0, 
                 use_sparse=True, window_size=16, use_alibi=True, use_checkpoint=True):
        # Desativar recursos avançados em CPU
        if not torch.cuda.is_available():
            use_sparse = False
            use_alibi = False
            use_checkpoint = False
            
        super().__init__()
        self.use_sparse = use_sparse
        self.use_alibi = use_alibi
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        if use_sparse:
            self.attention = SparseAttention(dim, num_heads, dropout, window_size=window_size)
        else:
            self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        
        ff_dim = ff_dim or dim * 4
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),  # Remover weight_norm para CPU
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim)  # Remover weight_norm para CPU
        )
        
        self.dropout = nn.Dropout(dropout)
        self.gamma1 = nn.Parameter(torch.ones(1, 1, dim) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(1, 1, dim) * 0.1)
        
        if use_alibi:
            self.register_buffer("alibi", self._build_alibi_tensor(num_heads))

    def _build_alibi_tensor(self, num_heads):
        """Build ALiBi (Attention with Linear Biases) position encoding."""
        slopes = torch.Tensor(self._get_slopes(num_heads))
        # Inicializar com tamanho máximo fixo para evitar problemas de redimensionamento
        max_pos = 1024  # Tamanho máximo da sequência
        pos = torch.arange(max_pos, device=slopes.device)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        
        # Criar tensor ALiBi com broadcast correto
        alibi = slopes.unsqueeze(1).unsqueeze(1) * rel_pos.unsqueeze(0)
        return alibi.contiguous()  # Garantir layout de memória contíguo

    def _get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + \
                   get_slopes_power_of_2(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    def forward(self, x, mask=None):
        # Pre-norm
        normed = self.norm1(x)
        
        # Self-attention com fallback para CPU
        if self.use_sparse and torch.cuda.is_available():
            attended, self.attn = self.attention(normed, mask=mask)
        else:
            # Usar batch_first=True para evitar transposes desnecessários
            attended, self.attn = self.attention(normed, normed, normed, 
                                   attn_mask=mask, need_weights=True)
            
        # Add ALiBi bias if enabled (apenas em GPU)
        if self.use_alibi and not self.use_sparse and torch.cuda.is_available():
            seq_len = x.shape[1]
            alibi = self.alibi[:, :seq_len, :seq_len].to(x.device)
            alibi = alibi.expand(x.shape[0], -1, -1, -1)
            
            # Reshape para adicionar alibi
            attended = attended.reshape(x.shape[0], seq_len, self.num_heads, self.head_dim)
            attended = attended.permute(0, 2, 1, 3)
            attended = attended + alibi
            attended = attended.permute(0, 2, 1, 3).contiguous()
            attended = attended.reshape(x.shape[0], seq_len, -1)
        
        # Residual connection com dropout
        x = x + self.dropout(self.gamma1 * attended)
        
        # Feed-forward com otimização de memória
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(self.gamma2 * ff_out)
        
        return x

class DecoderBlock(nn.Module):
    """Bloco do decoder otimizado."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        use_spectral=True,
        dropout=0.0
    ):
        super().__init__()
        padding = kernel_size // 2
        
        # Projeção inicial para ajustar dimensões
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        
        # Ramo principal com dimensões corretas
        main_layers = []
        if use_spectral:
            main_layers.extend([
                nn.utils.spectral_norm(nn.Conv2d(
                    out_channels, out_channels,  # Usando out_channels como entrada
                    kernel_size, padding=padding
                )),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
                nn.Dropout2d(dropout),
                nn.utils.spectral_norm(nn.Conv2d(
                    out_channels, out_channels,
                    kernel_size, padding=padding
                )),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
                nn.Dropout2d(dropout)
            ])
        else:
            main_layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
                nn.Dropout2d(dropout),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
                nn.Dropout2d(dropout)
            ])
        self.main = nn.Sequential(*main_layers)
        
        # Ramo de gate com dimensões corretas
        gate_layers = []
        if use_spectral:
            gate_layers.extend([
                nn.utils.spectral_norm(nn.Conv2d(
                    out_channels, out_channels,  # Usando out_channels como entrada
                    kernel_size, padding=padding
                )),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.Sigmoid()
            ])
        else:
            gate_layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.Sigmoid()
            ])
        self.gate = nn.Sequential(*gate_layers)
        
        # Fator de escala para skip connection
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, skip=None):
        # Projetar entrada para a dimensão correta
        x = self.proj(x)
        
        # Adicionar skip connection se fornecida
        if skip is not None:
            if skip.shape[1] != x.shape[1]:
                skip = self.proj(skip)
            x = x + skip
        
        # Caminho principal da convolução
        identity = x
        out = self.main(x)
        
        # Aplicar gate
        gate = self.gate(x)
        out = out * gate
        
        # Conexão residual
        return identity + self.gamma * out

class Pixel16Generator(nn.Module):
    """Gerador de pixel art 16x16 otimizado."""
    
    def __init__(self, config):
        super().__init__()
        
        # Parâmetros do modelo
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.embedding_dim = config["embedding_dim"]
        self.num_heads = config["num_heads"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.style_dim = 5  # Fixado em 5 para corresponder ao número de labels
        self.use_checkpoint = config.get("use_checkpoint", True)
        
        # Patch embedding com normalização
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, self.embedding_dim, self.patch_size, stride=self.patch_size),
            nn.GroupNorm(8, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Projeção do estilo
        self.style_projection = nn.Sequential(
            nn.Linear(self.style_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Embeddings posicionais
        seq_len = (self.image_size // self.patch_size) ** 2
        self.pos_embed = PositionalEmbedding(self.embedding_dim, seq_len)
        
        # Blocos transformer
        self.transformers = nn.ModuleList([
            TransformerBlock(
                dim=self.embedding_dim,
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
                use_checkpoint=self.use_checkpoint
            )
            for _ in range(self.num_layers)
        ])
        
        # Blocos do decoder
        decoder_dims = [
            self.embedding_dim,  # Primeira camada: embedding_dim (1024)
            self.embedding_dim // 2,  # Segunda camada: 512
            self.embedding_dim // 4,  # Terceira camada: 256
            3  # Camada final: 3 canais RGB
        ]
        self.decoders = nn.ModuleList([
            DecoderBlock(
                in_channels=decoder_dims[i],
                out_channels=decoder_dims[i+1],
                dropout=self.dropout_rate
            )
            for i in range(len(decoder_dims)-1)
        ])
        
        # Quantizador de cores
        self.quantizer = ColorQuantizer(num_colors=16)
        
        # Inicialização dos pesos
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_normal_(m.weight, a=0.1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
                
    def forward(self, x, style):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Add positional embedding
        x = x + self.pos_embed(x)
        
        # Project and add style
        style = self.style_projection(style)  # Projetar para dimensão do embedding
        style = style.unsqueeze(1)  # [B, 1, D]
        style = style.expand(-1, x.size(1), -1)  # Expandir para o mesmo tamanho de x
        x = x + style
        
        # Apply transformer blocks with checkpointing
        attention_maps = []
        for block in self.transformers:
            if self.training:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
            if hasattr(block, 'attn'):
                attention_maps.append(block.attn)
                
        # Reshape for decoder
        h = w = self.image_size // self.patch_size
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        # Apply decoder blocks with skip connections
        skip_features = []
        for i, decoder in enumerate(self.decoders):
            if i > 0:  # Pular a primeira iteração pois não tem skip connection anterior
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
                # Projetar skip connection para a dimensão correta antes de passar para o decoder
                skip = skip_features[-1]
                if skip.shape[1] != decoder.proj.out_channels:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
                x = decoder(x, skip)
            else:
                x = decoder(x)
            skip_features.append(x)
        
        # Final processing
        x = self.quantizer(x)
        return x, attention_maps

class PixelArtLoss(nn.Module):
    """Função de perda especializada para pixel art."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Usar VGG16 sem pesos pré-treinados
        try:
            self.vgg = models.vgg16(weights=None).features[:8].eval()
        except:
            # Fallback para versões antigas do torchvision
            self.vgg = models.vgg16(pretrained=False).features[:8].eval()
            
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        # Registrar médias e desvios para normalização
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Quantizador de cores
        self.quantizer = ColorQuantizer()
        
        # Pesos das losses do config
        self.content_weight = config.get("content_weight", 0.5)
        self.style_weight = config.get("style_weight", 0.8)
        self.pixel_weight = config.get("pixel_weight", 0.3)
        self.quant_weight = config.get("quant_weight", 1.0)
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = config.get("early_stopping_patience", 10)
        self.steps_without_improvement = 0
        
        # Tamanho fixo para features da VGG
        self.vgg_size = (32, 32)
        
        # Tamanho da imagem alvo
        self.target_size = (16, 16)  # Tamanho do pixel art
    
    def normalize(self, x):
        """Normaliza as imagens para o formato esperado pela VGG."""
        # Garantir que a entrada tenha 3 canais
        if x.size(1) != 3:
            # Se tiver mais que 3 canais, pegar apenas os 3 primeiros
            x = x[:, :3, :, :]
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
    
    def gram_matrix(self, x):
        """Calcula a matriz de Gram para loss de estilo."""
        B, C, H, W = x.shape
        features = x.view(B, C, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(C * H * W)
    
    def get_vgg_features(self, x):
        """Extrai features da VGG com redimensionamento adequado."""
        # Redimensionar para tamanho fixo antes de extrair features
        x = F.interpolate(x, size=self.vgg_size, mode='bilinear', align_corners=True)
        return self.vgg(x)
    
    def forward(self, x, target):
        """
        Args:
            x: Imagem gerada (output do modelo)
            target: Imagem alvo (ground truth)
        """
        # Desempacotar a saída do modelo se for uma tupla
        if isinstance(x, tuple):
            x = x[0]  # Pegar apenas a imagem gerada, ignorar attention maps
            
        # Garantir que as imagens estejam no formato correto
        x = x.clamp(0, 1)  # Garantir valores entre 0 e 1
        target = target.clamp(0, 1)
        
        # Garantir que x e target tenham 3 canais e o mesmo tamanho
        if x.size(1) != 3:
            x = x[:, :3, :, :]
        if target.size(1) != 3:
            target = target[:, :3, :, :]
            
        # Redimensionar x para o mesmo tamanho do target se necessário
        if x.shape[-2:] != target.shape[-2:]:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)
        
        # Normalizar as imagens para a VGG
        x_norm = self.normalize(x)
        target_norm = self.normalize(target)
        
        # Extrair features da VGG com redimensionamento
        x_features = self.get_vgg_features(x_norm)
        target_features = self.get_vgg_features(target_norm)
        
        # Loss de conteúdo
        content_loss = F.mse_loss(x_features, target_features)
        
        # Loss de estilo
        x_gram = self.gram_matrix(x_features)
        target_gram = self.gram_matrix(target_features)
        style_loss = F.mse_loss(x_gram, target_gram)
        
        # Loss de pixels (garantir mesmo tamanho)
        pixel_loss = F.l1_loss(x, target)
        
        # Loss de quantização
        quant_x = self.quantizer(x)
        quant_loss = F.mse_loss(x, quant_x.detach())
        
        # Loss total
        total_loss = (
            self.content_weight * content_loss +
            self.style_weight * style_loss +
            self.pixel_weight * pixel_loss +
            self.quant_weight * quant_loss
        )
        
        return {
            'loss': total_loss,
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'pixel_loss': pixel_loss.item(),
            'quant_loss': quant_loss.item()
        } 