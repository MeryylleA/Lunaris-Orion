"""
Módulo do encoder de texto para o Lunar Core.
Implementa um encoder simples mas robusto para processar prompts.
"""

from typing import Sequence
import jax.numpy as jnp
import flax.linen as nn

class TextEncoder(nn.Module):
    """
    Encoder de texto simples para processar prompts.
    Usa uma arquitetura transformer simplificada.
    """
    output_dim: int
    hidden_dim: int = 1024  # Atualizado para corresponder ao config
    intermediate_dim: int = 512  # Nova dimensão intermediária
    num_layers: int = 3
    num_heads: int = 8
    vocab_size: int = 50257  # Tamanho do vocabulário GPT-2
    max_length: int = 128
    dropout_rate: float = 0.1
    
    def setup(self):
        # Embedding de tokens
        self.token_embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.intermediate_dim  # Usa dimensão intermediária
        )
        
        # Embedding posicional
        self.position_embedding = nn.Embed(
            num_embeddings=self.max_length,
            features=self.intermediate_dim  # Usa dimensão intermediária
        )
        
        # Camadas de atenção
        self.transformer_blocks = [
            TransformerBlock(
                hidden_dim=self.intermediate_dim,  # Usa dimensão intermediária
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            ) for _ in range(self.num_layers)
        ]
        
        # Projeção intermediária para hidden_dim
        self.intermediate_projection = nn.Dense(self.hidden_dim)
        
        # Projeção final para o espaço latente
        self.final_projection = nn.Dense(self.output_dim)
        
    def __call__(self, tokens, training: bool = False):
        # Gera embedding posicional
        positions = jnp.arange(tokens.shape[1])[None, :]
        position_embeddings = self.position_embedding(positions)
        
        # Combina embeddings
        x = self.token_embedding(tokens) + position_embeddings
        
        # Passa pelos blocos transformer
        for block in self.transformer_blocks:
            x = block(x, training=training)
            
        # Pooling global (média)
        x = jnp.mean(x, axis=1)
        
        # Projeção intermediária
        x = self.intermediate_projection(x)
        
        # Projeção final
        return self.final_projection(x)

class TransformerBlock(nn.Module):
    """
    Bloco básico do transformer.
    """
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    
    def setup(self):
        # Multi-head attention
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )
        
        # Feed-forward
        self.dense1 = nn.Dense(self.hidden_dim * 4)
        self.dense2 = nn.Dense(self.hidden_dim)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def __call__(self, x, training: bool = False):
        # Self-attention com residual
        normalized = self.ln1(x)
        # MultiHeadDotProductAttention espera query, key, value
        attn_output = self.attention(
            inputs_q=normalized,
            inputs_kv=normalized,
            deterministic=not training
        )
        x = x + self.dropout(attn_output, deterministic=not training)
        
        # Feed-forward com residual
        normalized = self.ln2(x)
        
        # Feed-forward network
        h = self.dense1(normalized)
        h = nn.gelu(h)
        h = self.dropout(h, deterministic=not training)
        h = self.dense2(h)
        
        return x + h

class TextEncoderWithPooling(nn.Module):
    """
    Wrapper do encoder de texto com diferentes estratégias de pooling.
    """
    output_dim: int
    hidden_dim: int = 512
    num_layers: int = 3
    pooling_strategy: str = 'mean'  # 'mean', 'max', ou 'cls'
    
    def setup(self):
        self.encoder = TextEncoder(
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        
    def __call__(self, tokens, training: bool = False):
        # Codifica o texto
        encoded = self.encoder(tokens, training)
        
        # Aplica a estratégia de pooling
        if self.pooling_strategy == 'mean':
            return jnp.mean(encoded, axis=1)
        elif self.pooling_strategy == 'max':
            return jnp.max(encoded, axis=1)
        elif self.pooling_strategy == 'cls':
            return encoded[:, 0]  # Retorna o token [CLS]
        else:
            raise ValueError(f"Estratégia de pooling desconhecida: {self.pooling_strategy}") 