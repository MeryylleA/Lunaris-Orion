"""
Arquitetura principal do modelo Mini para pixel art 16x16.
Otimizado para eficiência e qualidade em baixa resolução.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import yaml
from pathlib import Path

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

class PositionalEmbedding(nn.Module):
    """Embeddings posicionais com suporte a diferentes precisões."""
    def __init__(self, dim: int, max_seq_len: int = 256):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        
        pe = torch.zeros(1, max_seq_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            x + pe: [B, N, D]
        """
        return x + self.pe[:, :x.size(1)]

class AdaIN(nn.Module):
    """Versão simplificada do AdaIN para 16x16."""
    def __init__(self, dim: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(dim, affine=False)
        self.style = nn.Linear(style_dim, dim * 2)  # Para scale e bias
        
        # Inicialização mais forte
        nn.init.xavier_uniform_(self.style.weight, gain=2.0)
        nn.init.zeros_(self.style.bias)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        style = self.style(style)
        scale, bias = style.chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return normalized * (1 + scale) + bias

class ColorQuantizer(nn.Module):
    """Quantizador otimizado para paleta de 16 cores."""
    def __init__(self):
        super().__init__()
        # Paleta fixa de 16 cores (mesma do dataset)
        colors = [
            [0, 0, 0],      # Preto
            [255, 255, 255], # Branco
            [255, 0, 0],     # Vermelho
            [0, 255, 0],     # Verde
            [0, 0, 255],     # Azul
            [255, 255, 0],   # Amarelo
            [255, 0, 255],   # Magenta
            [0, 255, 255],   # Ciano
            [128, 128, 128], # Cinza
            [128, 0, 0],     # Vermelho escuro
            [0, 128, 0],     # Verde escuro
            [0, 0, 128],     # Azul escuro
            [128, 128, 0],   # Marrom
            [128, 0, 128],   # Roxo
            [0, 128, 128],   # Turquesa
            [192, 192, 192]  # Cinza claro
        ]
        palette = torch.tensor(colors, dtype=torch.float32) / 255.0
        self.register_buffer('palette', palette * 2 - 1)  # Normalizar para [-1, 1]
        
        # MLP simplificado para processamento de cores
        self.color_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    def forward(self, x: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3)
        
        # Processar cores
        processed = self.color_mlp(x_flat)
        
        # Calcular distâncias e softmax
        distances = torch.cdist(processed, self.palette)
        logits = -distances / temperature
        weights = F.softmax(logits, dim=-1)
        
        # Quantização com straight-through estimator
        quantized = torch.matmul(weights, self.palette)
        indices = weights.argmax(dim=-1)
        hard_weights = F.one_hot(indices, num_classes=16).float()
        quantized_hard = torch.matmul(hard_weights, self.palette)
        
        # Straight-through trick
        out = (quantized_hard - quantized).detach() + quantized
        return out.reshape(b, h, w, 3).permute(0, 3, 1, 2)

class FlashAttention(nn.Module):
    """Atenção otimizada com suporte a FP8."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Usar Transformer Engine se disponível
        if TE_AVAILABLE:
            self.qkv = te.Linear(dim, dim * 3, bias=True)
            self.proj = te.Linear(dim, dim, bias=True)
        else:
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if FLASH_ATTN_AVAILABLE and x.is_cuda:
            # Usar Flash Attention se disponível
            out = flash_attn_func(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        else:
            # Fallback para atenção padrão
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
        
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class DecoderBlock(nn.Module):
    """Bloco do decoder otimizado."""
    def __init__(self, in_channels: int, out_channels: int, style_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 
                             kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, out_channels)
        )
        self.adain = AdaIN(out_channels, style_dim)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.adain(x, style)
        gate = self.gate(x)
        return self.activation(x * gate)

class Pixel16Generator(nn.Module):
    """Gerador de pixel art 16x16 com transformer otimizado."""
    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["model"]
        
        # Embeddings
        self.patch_embed = nn.Conv2d(3, self.config["embedding_dim"],
                                   kernel_size=self.config["patch_size"],
                                   stride=self.config["patch_size"])
        
        # Positional embedding
        num_patches = (16 // self.config["patch_size"]) ** 2
        self.pos_embed = PositionalEmbedding(self.config["embedding_dim"], num_patches + 1)
        
        # Processamento de estilo
        self.style_embed = nn.Linear(self.config["style_dim"], 
                                   self.config["embedding_dim"])
        
        # Transformer blocks com suporte a FP8
        if TE_AVAILABLE:
            self.blocks = nn.ModuleList([
                te.TransformerLayer(
                    hidden_size=self.config["embedding_dim"],
                    ffn_hidden_size=self.config["ff_dim"],
                    num_attention_heads=self.config["num_heads"],
                    attention_dropout=self.config["attention_dropout"],
                    hidden_dropout=self.config["dropout_rate"],
                    fp8_format="e4m3",
                    fp8_margin=0,
                    fp8_interval=1
                ) for _ in range(self.config["num_layers"])
            ])
        else:
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    FlashAttention(self.config["embedding_dim"],
                                 self.config["num_heads"],
                                 self.config["attention_dropout"]),
                    nn.LayerNorm(self.config["embedding_dim"]),
                    nn.Linear(self.config["embedding_dim"], self.config["ff_dim"]),
                    nn.GELU(),
                    nn.Dropout(self.config["dropout_rate"]),
                    nn.Linear(self.config["ff_dim"], self.config["embedding_dim"]),
                    nn.Dropout(self.config["dropout_rate"])
                ) for _ in range(self.config["num_layers"])
            ])
        
        # Decoder otimizado
        self.decoder = nn.ModuleList([
            DecoderBlock(self.config["embedding_dim"], 64, self.config["style_dim"]),
            DecoderBlock(64, 32, self.config["style_dim"]),
            nn.Sequential(
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.Tanh()
            )
        ])
        
        # Quantizador de cores (apenas na saída)
        self.quantizer = ColorQuantizer()
        
        # Compilar modelo se configurado
        if self.config["use_dynamic_scaling"]:
            self.compile(mode=self.config.get("compile_mode", "max-autotune"))
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # Extrair patches
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Processar estilo
        style = self.style_embed(style)
        style = style.unsqueeze(1)  # [B, 1, C]
        
        # Adicionar estilo e posição aos tokens
        x = torch.cat([style, x], dim=1)
        x = self.pos_embed(x)
        
        # Aplicar transformer blocks
        for block in self.blocks:
            if TE_AVAILABLE:
                x = block(x)[0]  # TE retorna tupla (output, attention_weights)
            else:
                x = x + block(x)  # Residual
        
        # Remover token de estilo
        x = x[:, 1:]
        
        # Reshape para spatial
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Decodificar
        for layer in self.decoder[:-1]:
            x = layer(x, style.squeeze(1))
        x = self.decoder[-1](x)
        
        # Quantizar apenas na saída
        x = self.quantizer(x)
        
        return x

class PixelArtLoss(nn.Module):
    """Função de perda adaptada para 16x16."""
    def __init__(self):
        super().__init__()
        # Pesos das perdas
        self.lambda_pixel = 1.0
        self.lambda_edge = 0.5
        self.lambda_palette = 0.1
    
    def edge_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Perda para preservar bordas nítidas."""
        # Kernels Sobel
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                             device=x.device).float().unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                             device=x.device).float().unsqueeze(0).unsqueeze(0)
        
        # Aplicar kernels
        grad_x = F.conv2d(x, sobel_x.expand(3, 1, 3, 3), padding=1, groups=3)
        grad_y = F.conv2d(x, sobel_y.expand(3, 1, 3, 3), padding=1, groups=3)
        
        # Calcular magnitude do gradiente
        grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        
        # Encorajar bordas nítidas
        return -torch.mean(grad_mag)
    
    def palette_loss(self, x: torch.Tensor, quantizer: ColorQuantizer) -> torch.Tensor:
        """Perda para encorajar uso da paleta."""
        # Quantizar e calcular diferença
        x_quant = quantizer(x)
        return F.mse_loss(x, x_quant.detach())
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
               quantizer: ColorQuantizer) -> Dict[str, torch.Tensor]:
        # Perda L1 (melhor que L2 para preservar cores)
        pixel_loss = F.l1_loss(pred, target)
        
        # Perda de bordas
        edge_loss = self.edge_loss(pred)
        
        # Perda de paleta
        palette_loss = self.palette_loss(pred, quantizer)
        
        # Combinar perdas
        total_loss = (self.lambda_pixel * pixel_loss +
                     self.lambda_edge * edge_loss +
                     self.lambda_palette * palette_loss)
        
        return {
            'loss': total_loss,
            'pixel_loss': pixel_loss.item(),
            'edge_loss': edge_loss.item(),
            'palette_loss': palette_loss.item()
        } 