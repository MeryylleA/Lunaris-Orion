"""
Lunar Core: Um modelo encoder-decoder para geração de pixel art 16x16
usando JAX e Flax.
"""

from typing import Sequence, Tuple, Optional, Dict
import flax.linen as nn
import jax
import jax.numpy as jnp
from .text_encoder import TextEncoder

class ResidualBlock(nn.Module):
    """Bloco Residual para o Lunar Core."""
    filters: int
    norm: bool = True

    @nn.compact
    def __call__(self, x, training: bool = True):
        residual = x
        y = nn.Conv(self.filters, kernel_size=(3, 3), padding="SAME")(x)
        if self.norm:
            y = nn.BatchNorm(use_running_average=not training)(y)
        y = nn.relu(y)
        
        y = nn.Conv(self.filters, kernel_size=(3, 3), padding="SAME")(y)
        if self.norm:
            y = nn.BatchNorm(use_running_average=not training)(y)
        
        if residual.shape != y.shape:
            residual = nn.Conv(self.filters, kernel_size=(1, 1))(residual)
        
        return nn.relu(residual + y)

class Encoder(nn.Module):
    """Encoder do Lunar Core."""
    latent_dim: int
    filters: Sequence[int]
    num_residual_blocks: int = 2

    @nn.compact
    def __call__(self, x, training: bool = True):
        # Camada inicial de convolução
        x = nn.Conv(self.filters[0], kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        # Blocos residuais com downsampling
        for filters in self.filters:
            # Blocos residuais
            for _ in range(self.num_residual_blocks):
                x = ResidualBlock(filters)(x, training)
            
            # Downsampling
            x = nn.Conv(filters, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)

        # Projeção final para o espaço latente
        x = x.reshape((x.shape[0], -1))
        mean = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        
        return mean, logvar

class Decoder(nn.Module):
    """Decoder do Lunar Core."""
    output_shape: Tuple[int, int, int]
    filters: Sequence[int]
    num_residual_blocks: int = 2

    @nn.compact
    def __call__(self, z, training: bool = True):
        # Calcula as dimensões iniciais
        h = self.output_shape[0] // (2 ** len(self.filters))
        w = self.output_shape[1] // (2 ** len(self.filters))
        initial_filters = self.filters[-1]

        # Projeção inicial do espaço latente
        x = nn.Dense(h * w * initial_filters)(z)
        x = x.reshape((-1, h, w, initial_filters))
        
        # Blocos residuais com upsampling
        for filters in reversed(self.filters):
            # Blocos residuais
            for _ in range(self.num_residual_blocks):
                x = ResidualBlock(filters)(x, training)
            
            # Upsampling
            x = jax.image.resize(x, 
                               shape=(x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]),
                               method="nearest")
            x = nn.Conv(filters, kernel_size=(3, 3), padding="SAME")(x)
            x = nn.BatchNorm(use_running_average=not training)(x)
            x = nn.relu(x)

        # Camada final
        x = nn.Conv(self.output_shape[-1], kernel_size=(3, 3), padding="SAME")(x)
        return nn.tanh(x)

class LatentFusion(nn.Module):
    """Módulo para fundir os espaços latentes de imagem e texto."""
    latent_dim: int
    fusion_type: str = 'concat'  # 'concat', 'add', ou 'gate'
    
    @nn.compact
    def __call__(self, image_latent, text_latent):
        if self.fusion_type == 'concat':
            # Concatena e projeta de volta para o tamanho original
            fused = jnp.concatenate([image_latent, text_latent], axis=-1)
            return nn.Dense(self.latent_dim)(fused)
        
        elif self.fusion_type == 'add':
            # Soma direta (requer mesmas dimensões)
            return image_latent + text_latent
        
        elif self.fusion_type == 'gate':
            # Mecanismo de gate para controlar a influência do texto
            gate = nn.sigmoid(nn.Dense(self.latent_dim)(text_latent))
            return image_latent * gate + text_latent * (1 - gate)
        
        else:
            raise ValueError(f"Tipo de fusão desconhecido: {self.fusion_type}")

class LunarCore(nn.Module):
    """Modelo completo Lunar Core com suporte a prompts."""
    latent_dim: int
    filters: Sequence[int]
    num_residual_blocks: int = 2
    input_shape: Tuple[int, int, int] = (16, 16, 3)
    use_text: bool = False  # Flag para habilitar/desabilitar suporte a texto
    fusion_type: str = 'concat'  # Tipo de fusão dos espaços latentes
    text_encoder_config: Optional[Dict] = None  # Configuração do encoder de texto

    @property
    def default_text_encoder_config(self):
        return {
            'hidden_dim': 1024,
            'intermediate_dim': 512,
            'num_layers': 3,
            'num_heads': 8,
            'dropout_rate': 0.1
        }

    def setup(self):
        self.encoder = Encoder(
            latent_dim=self.latent_dim,
            filters=self.filters,
            num_residual_blocks=self.num_residual_blocks
        )
        
        if self.use_text:
            config = self.text_encoder_config or self.default_text_encoder_config
            
            self.text_encoder = TextEncoder(
                output_dim=self.latent_dim,
                **config
            )
            
            self.latent_fusion = LatentFusion(
                latent_dim=self.latent_dim,
                fusion_type=self.fusion_type
            )
        
        self.decoder = Decoder(
            output_shape=self.input_shape,
            filters=self.filters,
            num_residual_blocks=self.num_residual_blocks
        )

    def encode_image(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Codifica uma imagem no espaço latente."""
        return self.encoder(x, training)
    
    def encode_text(self, tokens: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """Codifica texto no espaço latente."""
        if not self.use_text:
            raise ValueError("Modelo não configurado para usar texto")
        return self.text_encoder(tokens, training)
    
    def fuse_latents(self, image_latent: jnp.ndarray, text_latent: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Funde os espaços latentes de imagem e texto."""
        if not self.use_text or text_latent is None:
            return image_latent
        return self.latent_fusion(image_latent, text_latent)

    def __call__(self, x: jnp.ndarray, tokens: Optional[jnp.ndarray] = None, 
                 training: bool = True, rngs: Optional[Dict] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Codifica imagem
        mean, logvar = self.encode_image(x, training)
        
        if training:
            # Reparametrização durante o treinamento
            std = jnp.exp(0.5 * logvar)
            if rngs is None:
                rng = self.make_rng('params')
            else:
                rng = rngs['params']
            eps = jax.random.normal(rng, mean.shape)
            z = mean + eps * std
        else:
            z = mean
        
        # Processa texto se disponível
        if self.use_text and tokens is not None:
            text_latent = self.encode_text(tokens, training)
            z = self.fuse_latents(z, text_latent)
            
        # Decodifica
        reconstruction = self.decoder(z, training)
        return reconstruction, mean, logvar

    def generate(self, z: jnp.ndarray, tokens: Optional[jnp.ndarray] = None, 
                training: bool = False) -> jnp.ndarray:
        """Gera imagens a partir do espaço latente e opcionalmente texto."""
        if self.use_text and tokens is not None:
            text_latent = self.encode_text(tokens, training)
            z = self.fuse_latents(z, text_latent)
        return self.decoder(z, training) 