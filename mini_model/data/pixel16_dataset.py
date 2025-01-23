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
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        use_cache: bool = True
    ):
        """
        Args:
            data_dir: Diretório com os dados
            split: 'train', 'val' ou 'test'
            train_ratio: Proporção dos dados para treino
            val_ratio: Proporção dos dados para validação
            use_cache: Se True, mantém dados em memória
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_cache = use_cache
        
        # Carregar dados
        sprites_path = self.data_dir / "sprites.npy"
        labels_path = self.data_dir / "sprites_labels.npy"
        
        if not sprites_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"Arquivos não encontrados em {data_dir}"
            )
        
        # Carregar arrays
        self.sprites = np.load(sprites_path)
        self.labels = np.load(labels_path)
        
        # Normalizar sprites para [-1, 1]
        self.sprites = self.sprites.astype(np.float32) / 127.5 - 1
        
        # Calcular índices para cada split
        total_samples = len(self.sprites)
        indices = np.random.permutation(total_samples)
        
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = total_samples - train_size - val_size
        
        if split == "train":
            self.indices = indices[:train_size]
        elif split == "val":
            self.indices = indices[train_size:train_size + val_size]
        else:  # test
            self.indices = indices[train_size + val_size:]
        
        # Cache para acesso rápido
        self._cache = {}
        
        print(f"\nDataset {split}:")
        print(f"Total de imagens: {len(self.indices)}")
        print(f"Dimensões dos sprites: {self.sprites.shape}")
        print(f"Dimensões dos labels: {self.labels.shape}")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        
        if self.use_cache and real_idx in self._cache:
            return self._cache[real_idx]
        
        # Carregar e processar dados
        sprite = self.sprites[real_idx]
        label = self.labels[real_idx]
        
        # Converter para tensores
        sprite = torch.from_numpy(sprite).permute(2, 0, 1)  # HWC -> CHW
        label = torch.from_numpy(label).float()
        
        if self.use_cache:
            self._cache[real_idx] = (sprite, label)
        
        return sprite, label
    
    def get_sprite_info(self, idx: int) -> Dict:
        """Retorna informações do sprite do dataframe de labels."""
        return self.labels_df.iloc[idx].to_dict()
    
    def get_all_sprites(self) -> torch.Tensor:
        """Retorna todos os sprites do split atual como um único tensor."""
        sprites = []
        for idx in self.indices:
            sprites.append(self.sprites[idx])
        return torch.stack(sprites)
    
    def get_all_labels(self) -> torch.Tensor:
        """Retorna todos os labels do split atual como um único tensor."""
        return torch.from_numpy(self.labels[self.indices]).float()
    
    def clear_cache(self):
        """Limpa o cache de sprites processados."""
        self._cache.clear() 