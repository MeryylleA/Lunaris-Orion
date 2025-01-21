"""
Dataset para imagens pixel art 16x16.
Otimizado para trabalhar com sprites.npy, labels.csv e sprites_labels.npy.
"""

import os
from typing import Optional, Tuple, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
from pathlib import Path
import yaml

class Pixel16Dataset(Dataset):
    """Dataset para imagens pixel art 16x16 com dados pré-processados."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = "train",
                 train_ratio: float = 0.9,
                 transform: Optional[T.Compose] = None,
                 use_cache: bool = True):
        """
        Args:
            data_dir: Diretório contendo sprites.npy, labels.csv e sprites_labels.npy
            split: 'train' ou 'val'
            train_ratio: Proporção dos dados para treino
            transform: Transformações adicionais
            use_cache: Se deve usar cache em memória
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.use_cache = use_cache
        
        # Carregar dados
        self.sprites = np.load(self.data_dir / "sprites.npy")
        self.labels_df = pd.read_csv(self.data_dir / "labels.csv")
        self.sprites_labels = np.load(self.data_dir / "sprites_labels.npy")
        
        # Verificar formatos
        assert len(self.sprites.shape) == 4, "sprites.npy deve ter formato (N, H, W, C)"
        assert self.sprites.shape[1:3] == (16, 16), "Sprites devem ser 16x16"
        assert len(self.sprites) == len(self.sprites_labels), "Número de sprites e labels deve ser igual"
        
        # Dividir em treino/validação
        num_samples = len(self.sprites)
        indices = np.random.permutation(num_samples)
        train_size = int(train_ratio * num_samples)
        
        if split == "train":
            self.indices = indices[:train_size]
        else:
            self.indices = indices[train_size:]
            
        # Cache para transformações
        self.cache: Dict[int, torch.Tensor] = {}
        
        # Logging de estatísticas
        print(f"\nDataset {split}:")
        print(f"Total de imagens: {len(self.indices)}")
        print(f"Dimensões dos sprites: {self.sprites.shape}")
        print(f"Dimensões dos labels: {self.sprites_labels.shape}")
    
    def _process_sprite(self, idx: int) -> torch.Tensor:
        """Processa um sprite individual."""
        # Verificar cache
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
            
        # Carregar e converter para tensor
        sprite = self.sprites[idx]
        
        # Converter para float e normalizar para [-1, 1]
        if sprite.dtype == np.uint8:
            sprite = sprite.astype(np.float32) / 127.5 - 1
        
        # Converter para tensor e ajustar canais
        sprite = torch.from_numpy(sprite).float()
        if sprite.shape[-1] == 3:  # HWC para CHW
            sprite = sprite.permute(2, 0, 1)
        
        # Aplicar transformações adicionais
        if self.transform is not None:
            sprite = self.transform(sprite)
        
        # Atualizar cache
        if self.use_cache:
            self.cache[idx] = sprite
            
        return sprite
    
    def _get_style_label(self, idx: int) -> torch.Tensor:
        """Retorna o label de estilo para um sprite."""
        label = self.sprites_labels[idx]
        return torch.from_numpy(label).float()
    
    def get_sprite_info(self, idx: int) -> Dict:
        """Retorna informações do sprite do dataframe de labels."""
        return self.labels_df.iloc[idx].to_dict()
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna um par (sprite, style_label)."""
        real_idx = self.indices[idx]
        
        # Carregar e processar sprite
        sprite = self._process_sprite(real_idx)
        
        # Carregar label de estilo
        style_label = self._get_style_label(real_idx)
        
        return sprite, style_label
    
    def get_all_sprites(self) -> torch.Tensor:
        """Retorna todos os sprites do split atual como um único tensor."""
        sprites = []
        for idx in self.indices:
            sprites.append(self._process_sprite(idx))
        return torch.stack(sprites)
    
    def get_all_labels(self) -> torch.Tensor:
        """Retorna todos os labels do split atual como um único tensor."""
        return torch.from_numpy(self.sprites_labels[self.indices]).float()
    
    def clear_cache(self):
        """Limpa o cache de sprites processados."""
        self.cache.clear() 