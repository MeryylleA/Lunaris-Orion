"""
Módulo para carregar e pré-processar o dataset de pixel art 16x16.
"""

from typing import Tuple, Dict, Optional, List
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from .prompt_manager import PromptManager
import logging

logger = logging.getLogger(__name__)

class PixelArtDataset:
    def __init__(self, config: Dict):
        """
        Inicializa o dataset de pixel art.
        
        Args:
            config: Dicionário com configurações do dataset
        """
        self.config = config
        self.sprites = None
        self.labels = None
        self.sprite_labels = None
        self.prompt_manager = None
        self._validate_config()
        
    def _validate_config(self):
        """Valida a configuração do dataset."""
        required_paths = ['dataset_path', 'labels_path', 'sprite_labels_path']
        for path in required_paths:
            if path not in self.config:
                raise ValueError(f"Configuração ausente: {path}")
            
            file_path = Path(self.config[path])
            if not file_path.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
                
        # Valida formato dos arquivos
        try:
            sprites = np.load(self.config['dataset_path'], mmap_mode='r')
            if len(sprites.shape) != 4 or sprites.shape[1:] != (16, 16, 3):
                raise ValueError(f"Formato inválido de sprites: {sprites.shape}, esperado (N, 16, 16, 3)")
                
            sprite_labels = np.load(self.config['sprite_labels_path'], mmap_mode='r')
            if len(sprite_labels) != len(sprites):
                raise ValueError(f"Número de labels ({len(sprite_labels)}) não corresponde ao número de sprites ({len(sprites)})")
                
            labels_df = pd.read_csv(self.config['labels_path'])
            if 'label_id' not in labels_df.columns or 'description' not in labels_df.columns:
                raise ValueError("CSV de labels deve conter colunas 'label_id' e 'description'")
                
        except Exception as e:
            logger.error(f"Erro na validação dos arquivos: {str(e)}")
            raise
            
    def _validate_and_process_sprites(self, sprites: np.ndarray) -> jnp.ndarray:
        """
        Valida e processa os sprites.
        
        Args:
            sprites: Array numpy com os sprites
            
        Returns:
            Array JAX processado
        """
        # Verifica valores
        if sprites.min() < 0 or sprites.max() > 255:
            raise ValueError("Valores dos sprites devem estar entre 0 e 255")
            
        # Converte para float32 e normaliza para [-1, 1]
        sprites = sprites.astype(np.float32)
        sprites = sprites / 127.5 - 1.0
        
        # Verifica NaN/Inf
        if np.any(np.isnan(sprites)) or np.any(np.isinf(sprites)):
            raise ValueError("Detectados valores NaN/Inf nos sprites")
            
        return jnp.array(sprites)
        
    def _validate_and_process_labels(self, sprite_labels: np.ndarray, 
                                   labels_df: pd.DataFrame) -> Tuple[jnp.ndarray, pd.DataFrame]:
        """
        Valida e processa as labels.
        
        Args:
            sprite_labels: Array numpy com as labels dos sprites
            labels_df: DataFrame com as descrições das labels
            
        Returns:
            Tuple com array JAX de labels e DataFrame processado
        """
        # Verifica valores únicos nas labels
        unique_labels = np.unique(sprite_labels)
        if not all(label in labels_df['label_id'].values for label in unique_labels):
            raise ValueError("Algumas labels dos sprites não existem no CSV de labels")
            
        # Converte para int32
        sprite_labels = sprite_labels.astype(np.int32)
        
        # Verifica dimensões
        if len(sprite_labels.shape) > 2:
            sprite_labels = sprite_labels[..., 0]  # Pega apenas primeira dimensão se multidimensional
            
        return jnp.array(sprite_labels), labels_df
        
    def load_data(self) -> Tuple[jnp.ndarray, jnp.ndarray, pd.DataFrame]:
        """
        Carrega os dados do dataset.
        
        Returns:
            Tuple contendo sprites, labels dos sprites e DataFrame com as labels
        """
        try:
            # Carrega sprites
            logger.info("Carregando sprites...")
            sprites = np.load(self.config['dataset_path'])
            self.sprites = self._validate_and_process_sprites(sprites)
            logger.info(f"Carregados {len(sprites)} sprites")
            
            # Carrega labels
            logger.info("Carregando labels...")
            sprite_labels = np.load(self.config['sprite_labels_path'])
            labels_df = pd.read_csv(self.config['labels_path'])
            self.sprite_labels, self.labels = self._validate_and_process_labels(sprite_labels, labels_df)
            logger.info(f"Carregadas {len(labels_df)} classes únicas")
            
            # Inicializa prompt manager se necessário
            if 'prompt_dataset' in self.config:
                logger.info("Inicializando gerenciador de prompts...")
                self.prompt_manager = PromptManager(self.config)
                success = self.prompt_manager.download_and_prepare(
                    self.config['prompt_dataset'].get('subset', '2k_random_1k')
                )
                if success:
                    logger.info(f"Carregados {self.prompt_manager.get_prompt_count()} prompts")
                else:
                    logger.warning("Falha ao carregar prompts. Continuando sem suporte a prompts.")
            
            return self.sprites, self.sprite_labels, self.labels
            
        except Exception as e:
            logger.error(f"Erro ao carregar dataset: {str(e)}")
            raise
            
    def _prepare_batch(self, batch: Dict) -> Dict:
        """
        Prepara um batch para treinamento.
        
        Args:
            batch: Dicionário com dados do batch
            
        Returns:
            Batch processado
        """
        # Garante tipos corretos
        batch = {
            'image': tf.cast(batch['image'], tf.float32),
            'label': tf.cast(batch['label'], tf.int32)
        }
        
        # Adiciona prompt se disponível
        if self.prompt_manager and self.prompt_manager.get_prompt_count() > 0:
            prompts = self.prompt_manager.get_random_prompts(batch['image'].shape[0])
            batch['prompt'] = tf.convert_to_tensor(prompts, dtype=tf.string)
            
        return batch
        
    def create_dataset(self, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        """
        Cria um tf.data.Dataset para treinamento.
        
        Args:
            batch_size: Tamanho do batch
            shuffle: Se deve embaralhar os dados
            
        Returns:
            Dataset do TensorFlow
        """
        if self.sprites is None:
            self.load_data()
            
        # Cria dataset base
        dataset = tf.data.Dataset.from_tensor_slices({
            'image': self.sprites,
            'label': self.sprite_labels
        })
        
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=min(self.config['shuffle_buffer'], len(self.sprites)),
                reshuffle_each_iteration=True
            )
            
        # Aplica batch e processamento
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.map(self._prepare_batch)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    def split_train_val(self, val_split: float = 0.1) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Divide o dataset em treino e validação.
        
        Args:
            val_split: Proporção do dataset para validação
            
        Returns:
            Datasets de treino e validação
        """
        if self.sprites is None:
            self.load_data()
            
        # Calcula índices para divisão
        num_samples = len(self.sprites)
        num_val = int(num_samples * val_split)
        
        # Usa seed fixa para reprodutibilidade
        rng = np.random.RandomState(42)
        indices = rng.permutation(num_samples)
        
        train_idx = indices[num_val:]
        val_idx = indices[:num_val]
        
        # Divide os dados
        train_sprites = self.sprites[train_idx]
        train_labels = self.sprite_labels[train_idx]
        
        val_sprites = self.sprites[val_idx]
        val_labels = self.sprite_labels[val_idx]
        
        # Cria datasets
        train_ds = tf.data.Dataset.from_tensor_slices({
            'image': train_sprites,
            'label': train_labels
        })
        
        val_ds = tf.data.Dataset.from_tensor_slices({
            'image': val_sprites,
            'label': val_labels
        })
        
        # Configura os datasets
        batch_size = self.config.get('batch_size', 32)
        
        # Aplica shuffle apenas no treino
        train_ds = train_ds.shuffle(
            buffer_size=min(self.config['shuffle_buffer'], len(train_sprites)),
            reshuffle_each_iteration=True
        )
        
        # Aplica batch e processamento
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
        val_ds = val_ds.batch(batch_size, drop_remainder=True)
        
        train_ds = train_ds.map(self._prepare_batch)
        val_ds = val_ds.map(self._prepare_batch)
        
        # Prefetch para melhor performance
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        
        logger.info(f"Dataset dividido em {len(train_sprites)} amostras de treino e {len(val_sprites)} de validação")
        
        return train_ds, val_ds 