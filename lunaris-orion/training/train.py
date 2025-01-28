"""
Script principal de treinamento do Lunar Core.
"""

import os
from typing import Dict, Tuple, Optional
import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state
import wandb
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import logging
from jax.experimental import enable_x64

# Habilita suporte a x64 para maior precisão em cálculos críticos
jax.config.update("jax_enable_x64", True)

from models.lunar_core import LunarCore
from data.dataset_loader import PixelArtDataset

logger = logging.getLogger(__name__)

class TrainState(train_state.TrainState):
    batch_stats: Dict
    key: jnp.ndarray

def create_train_state(rng: jax.random.PRNGKey, config: dict, model_config: Optional[dict] = None, checkpoint_path: Optional[str] = None) -> TrainState:
    """
    Cria o estado inicial do treinamento.
    
    Args:
        rng: Chave PRNG do JAX
        config: Configuração do treinamento
        model_config: Configuração específica do modelo (opcional)
        checkpoint_path: Caminho para carregar checkpoint (opcional)
    """
    # Usa model_config se fornecido, senão usa config['model']
    model_config = model_config or config['model']
    
    # Inicializa modelo
    model = LunarCore(
        latent_dim=model_config['latent_dim'],
        filters=model_config['filters'],
        num_residual_blocks=model_config.get('num_residual_blocks', 2),
        input_shape=model_config['input_shape'],
        use_text=model_config.get('use_text', False),
        fusion_type=model_config.get('latent_fusion', {}).get('type', 'concat'),
        text_encoder_config=model_config.get('text_encoder_config')
    )
    
    # Cria variáveis
    rng, init_rng = jax.random.split(rng)
    input_shape = (1,) + tuple(model_config['input_shape'])
    variables = model.init(
        {'params': init_rng}, 
        jnp.ones(input_shape, dtype=jnp.float32),
        tokens=None if not model_config.get('use_text') else jnp.ones((1, 128), dtype=jnp.int32),
        training=True
    )
    
    # Configura otimizador
    learning_rate = config['training']['learning_rate']
    if config['training'].get('transfer_learning', {}).get('enabled', False):
        learning_rate *= config['training']['transfer_learning'].get('learning_rate_factor', 0.5)
    
    tx = optax.adam(
        learning_rate=learning_rate,
        b1=config['training'].get('beta1', 0.9),
        b2=config['training'].get('beta2', 0.999)
    )
    
    # Cria estado inicial
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables.get('batch_stats'),
        key=rng
    )
    
    # Carrega checkpoint se fornecido
    if checkpoint_path is not None:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = flax.serialization.from_bytes(state, f.read())
            state = state.replace(
                step=checkpoint['step'],
                params=checkpoint['params'],
                opt_state=checkpoint['opt_state'],
                batch_stats=checkpoint['batch_stats'],
                key=checkpoint['key']
            )
    
    return state

def compute_metrics(batch: Dict, state: TrainState, key: jnp.ndarray, training_mode: str = 'pixel_art') -> Dict:
    """
    Calcula as métricas de treinamento.
    
    Args:
        batch: Batch de dados (imagens e opcionalmente prompts)
        state: Estado do modelo
        key: Chave PRNG
        training_mode: 'pixel_art' ou 'prompt'
    """
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    
    # Prepara inputs
    images = batch['image'].astype(jnp.float16)
    tokens = batch.get('prompt') if training_mode == 'prompt' else None
    
    output = state.apply_fn(
        variables,
        images,
        tokens=tokens,
        training=False,
        rngs={'params': key}
    )
    reconstruction, mean, logvar = output
    
    # Converte para float32 para cálculos de métricas
    images = images.astype(jnp.float32)
    reconstruction = reconstruction.astype(jnp.float32)
    mean = mean.astype(jnp.float32)
    logvar = logvar.astype(jnp.float32)
    
    # Perda de reconstrução
    recon_loss = jnp.mean((images - reconstruction) ** 2)
    
    # Perda KL
    kl_loss = -0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar))
    
    # Perda total
    total_loss = recon_loss + kl_loss
    
    return {
        'loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss
    }

def _train_step_impl(state: TrainState, batch: Dict, training_mode: str = 'pixel_art') -> Tuple[TrainState, Dict]:
    """Implementação interna do passo de treinamento."""
    rng, new_rng = jax.random.split(state.key)
    
    def loss_fn(params):
        variables = {'params': params, 'batch_stats': state.batch_stats}
        
        # Prepara inputs
        images = batch['image'].astype(jnp.float16)
        tokens = batch.get('prompt') if training_mode == 'prompt' else None
        
        output, new_model_state = state.apply_fn(
            variables,
            images,
            tokens=tokens,
            training=True,
            rngs={'params': rng},
            mutable=['batch_stats']
        )
        reconstruction, mean, logvar = output
        
        # Converte para float32 para cálculos de perda
        images_f32 = images.astype(jnp.float32)
        reconstruction_f32 = reconstruction.astype(jnp.float32)
        mean_f32 = mean.astype(jnp.float32)
        logvar_f32 = logvar.astype(jnp.float32)
        
        recon_loss = jnp.mean((images_f32 - reconstruction_f32) ** 2)
        kl_loss = -0.5 * jnp.mean(1 + logvar_f32 - mean_f32**2 - jnp.exp(logvar_f32))
        total_loss = recon_loss + kl_loss
        
        return total_loss, (reconstruction_f32, mean_f32, logvar_f32, new_model_state)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params)
    reconstruction, mean, logvar, new_model_state = aux
    
    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_model_state['batch_stats'],
        key=new_rng
    )
    
    metrics = {
        'loss': loss,
        'recon_loss': jnp.mean((batch['image'].astype(jnp.float32) - reconstruction) ** 2),
        'kl_loss': -0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar))
    }
    
    return state, metrics

# Cria versões jit-compiladas para cada modo de treinamento
_train_step_pixel_art = jax.jit(_train_step_impl)
_train_step_prompt = jax.jit(lambda state, batch: _train_step_impl(state, batch, training_mode='prompt'))

def train_step(state: TrainState, batch: Dict, training_mode: str = 'pixel_art') -> Tuple[TrainState, Dict]:
    """
    Executa um passo de treinamento.
    
    Args:
        state: Estado atual do modelo
        batch: Batch de dados
        training_mode: 'pixel_art' ou 'prompt'
    """
    if training_mode == 'prompt':
        return _train_step_prompt(state, batch)
    else:
        return _train_step_pixel_art(state, batch)

def save_sample_images(reconstruction: jnp.ndarray, original: jnp.ndarray,
                      step: int, output_dir: str, prompt: str = None):
    """
    Salva imagens de exemplo durante o treinamento.
    """
    # Converte de [-1, 1] para [0, 1]
    reconstruction = (reconstruction + 1) / 2
    original = (original + 1) / 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(original[0])
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(reconstruction[0])
    title = 'Reconstrução'
    if prompt:
        title += f'\nPrompt: {prompt}'
    ax2.set_title(title)
    ax2.axis('off')
    
    plt.savefig(os.path.join(output_dir, f'sample_{step}.png'))
    plt.close()

def validate_checkpoint(config: Dict) -> bool:
    """
    Valida se existe um checkpoint válido para treinamento com prompts.
    """
    checkpoint_dir = Path(config['logging']['checkpoint_dir'])
    if not checkpoint_dir.exists():
        return False
    
    checkpoints = list(checkpoint_dir.glob('checkpoint_*.pkl'))
    if not checkpoints:
        return False
    
    # Verifica o último checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    try:
        with open(latest_checkpoint, 'rb') as f:
            _ = f.read()  # Tenta ler o arquivo
        return True
    except:
        return False

def main():
    # Carrega configuração
    print("Carregando configuração...")
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Verifica modo de treinamento
    training_mode = config['training'].get('mode', 'pixel_art')
    if training_mode == 'prompt' and not validate_checkpoint(config):
        logger.error("Treinamento com prompts requer um checkpoint válido do treinamento de pixel art!")
        return
    
    # Inicializa W&B
    print("Inicializando Weights & Biases...")
    wandb.init(
        project=config['logging']['wandb_project'],
        config=config,
        name=f"lunar_core_{training_mode}"
    )
    
    # Cria diretórios necessários
    print("Criando diretórios...")
    output_dir = Path(config['logging']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializa dataset
    print("Carregando dataset...")
    dataset = PixelArtDataset(config['data'])
    train_ds, val_ds = dataset.split_train_val(config['data']['validation_split'])
    
    # Configura os datasets
    train_ds = train_ds.batch(config['training']['batch_size'], drop_remainder=True)
    val_ds = val_ds.batch(config['training']['batch_size'], drop_remainder=True)
    
    # Conta o número total de batches
    num_train_batches = sum(1 for _ in train_ds)
    num_val_batches = sum(1 for _ in val_ds)
    
    print(f"Número de batches de treino: {num_train_batches}")
    print(f"Número de batches de validação: {num_val_batches}")
    
    # Cria estado de treinamento
    rng = jax.random.PRNGKey(0)
    checkpoint_path = None
    if training_mode == 'prompt':
        checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        latest_checkpoint = max(checkpoint_dir.glob('checkpoint_*.pkl'), 
                              key=lambda x: int(x.stem.split('_')[1]))
        checkpoint_path = str(latest_checkpoint)
    
    state = create_train_state(rng, config, checkpoint_path=checkpoint_path)
    print("Modelo inicializado com sucesso!")
    
    # Loop de treinamento
    print(f"\nIniciando treinamento no modo: {training_mode}")
    for epoch in range(config['training']['num_epochs']):
        # Treino
        train_metrics = []
        train_iterator = tqdm(train_ds, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch in train_iterator:
            state, metrics = train_step(state, batch, training_mode)
            train_metrics.append(metrics)
            
            # Atualiza a barra de progresso
            train_iterator.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'recon': f"{metrics['recon_loss']:.4f}",
                'kl': f"{metrics['kl_loss']:.4f}"
            })
        
        # Calcula médias do treino
        avg_train_loss = np.mean([m['loss'] for m in train_metrics])
        avg_train_recon = np.mean([m['recon_loss'] for m in train_metrics])
        avg_train_kl = np.mean([m['kl_loss'] for m in train_metrics])
        
        # Validação
        if (epoch + 1) % config['training']['eval_interval'] == 0:
            val_metrics = []
            for batch in val_ds:
                metrics = compute_metrics(batch, state, state.key, training_mode)
                val_metrics.append(metrics)
            
            # Calcula médias da validação
            avg_val_loss = np.mean([m['loss'] for m in val_metrics])
            avg_val_recon = np.mean([m['recon_loss'] for m in val_metrics])
            avg_val_kl = np.mean([m['kl_loss'] for m in val_metrics])
            
            # Loga métricas
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': avg_train_loss,
                'train/recon_loss': avg_train_recon,
                'train/kl_loss': avg_train_kl,
                'val/loss': avg_val_loss,
                'val/recon_loss': avg_val_recon,
                'val/kl_loss': avg_val_kl
            })
            
            # Gera e salva amostras
            if (epoch + 1) % config['logging']['sample_interval'] == 0:
                for batch in val_ds.take(1):
                    reconstruction, _, _ = state.apply_fn(
                        {'params': state.params, 'batch_stats': state.batch_stats},
                        batch['image'],
                        tokens=batch.get('prompt') if training_mode == 'prompt' else None,
                        training=False
                    )
                    
                    prompt = None
                    if training_mode == 'prompt' and 'prompt' in batch:
                        prompt = batch['prompt'][0].numpy().decode()
                    
                    save_sample_images(
                        reconstruction,
                        batch['image'],
                        epoch + 1,
                        output_dir,
                        prompt
                    )
        
        # Salva checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint = {
                'step': state.step,
                'params': state.params,
                'opt_state': state.opt_state,
                'batch_stats': state.batch_stats,
                'key': state.key
            }
            
            checkpoint_dir = Path(config['logging']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"checkpoint_{epoch+1}.pkl"
            
            with open(checkpoint_path, 'wb') as f:
                f.write(flax.serialization.to_bytes(checkpoint))
            print(f"\nCheckpoint salvo em: {checkpoint_path}")

if __name__ == "__main__":
    main() 