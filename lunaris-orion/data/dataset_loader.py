"""
Módulo para carregar e pré-processar o dataset de pixel art 16x16.
"""

from typing import Tuple, Dict, Optional
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
        self.prompt_manager = PromptManager(config)
        
    def load_data(self) -> Tuple[jnp.ndarray, jnp.ndarray, pd.DataFrame]:
        """
        Carrega os dados do dataset.
        
        Returns:
            Tuple contendo sprites, labels dos sprites e DataFrame com as labels
        """
        # Carrega os arquivos
        try:
            sprites = np.load(self.config['dataset_path'])
            sprite_labels = np.load(self.config['sprite_labels_path'])
            labels_df = pd.read_csv(self.config['labels_path'])
        except Exception as e:
            logger.error(f"Erro ao carregar arquivos do dataset: {e}")
            raise
        
        # Normaliza os sprites para [-1, 1]
        sprites = sprites.astype(np.float32) / 127.5 - 1.0
        
        # Converte para arrays JAX
        self.sprites = jnp.array(sprites)
        self.sprite_labels = jnp.array(sprite_labels)
        self.labels = labels_df
        
        # Carrega prompts se configurado
        if 'prompt_dataset' in self.config:
            logger.info("Iniciando carregamento dos prompts...")
            success = self.prompt_manager.download_and_prepare(
                self.config['prompt_dataset'].get('subset', '2k_random_1k')
            )
            if success:
                logger.info(f"Carregados {self.prompt_manager.get_prompt_count()} prompts")
            else:
                logger.warning("Falha ao carregar prompts. Continuando sem suporte a prompts.")
        
        return self.sprites, self.sprite_labels, self.labels
    
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
            
        # Cria dataset do TensorFlow
        dataset = tf.data.Dataset.from_tensor_slices((self.sprites, self.sprite_labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.config['shuffle_buffer'])
            
        # Se temos prompts disponíveis, adiciona ao dataset
        if self.prompt_manager and self.prompt_manager.get_prompt_count() > 0:
            def add_random_prompt(image, label):
                prompt = tf.py_function(
                    lambda: self.prompt_manager.get_random_prompts(1)[0],
                    [], tf.string
                )
                return {'image': image, 'label': label, 'prompt': prompt}
            
            dataset = dataset.map(add_random_prompt)
            
        dataset = dataset.batch(batch_size, drop_remainder=True)
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
        indices = np.random.permutation(num_samples)
        
        train_idx = indices[num_val:]
        val_idx = indices[:num_val]
        
        # Divide os dados
        train_sprites = self.sprites[train_idx]
        train_labels = self.sprite_labels[train_idx]
        
        val_sprites = self.sprites[val_idx]
        val_labels = self.sprite_labels[val_idx]
        
        # Ajusta o formato das labels antes de criar o dataset
        if len(train_labels.shape) > 2:
            train_labels = train_labels[..., 0]  # Pega apenas a primeira dimensão
        if len(val_labels.shape) > 2:
            val_labels = val_labels[..., 0]  # Pega apenas a primeira dimensão
            
        # Converte para o tipo correto
        train_sprites = train_sprites.astype(np.float32)
        train_labels = train_labels.astype(np.int32)
        val_sprites = val_sprites.astype(np.float32)
        val_labels = val_labels.astype(np.int32)
        
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
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
        val_ds = val_ds.batch(batch_size, drop_remainder=True)
        
        # Função para converter batch para numpy
        def convert_batch_to_numpy(batch):
            images = batch['image'].numpy()
            labels = batch['label'].numpy()
            
            # Garante o formato correto
            if len(images.shape) == 3:
                images = images[None, ...]
            
            return {
                'image': tf.convert_to_tensor(images, dtype=tf.float32),
                'label': tf.convert_to_tensor(labels, dtype=tf.int32)
            }
        
        # Aplica a conversão e define shapes
        def convert_and_set_shape(batch):
            result = tf.py_function(
                func=convert_batch_to_numpy,
                inp=[batch],
                Tout=[tf.float32, tf.int32]
            )
            result[0].set_shape([batch_size, 16, 16, 3])
            result[1].set_shape([batch_size])
            return {'image': result[0], 'label': result[1]}
        
        # Aplica a conversão
        train_ds = train_ds.map(convert_and_set_shape)
        val_ds = val_ds.map(convert_and_set_shape)
        
        return train_ds, val_ds 