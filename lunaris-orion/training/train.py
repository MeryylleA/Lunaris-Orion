"""
Script principal de treinamento do Lunar Core.
"""

import os
from typing import Dict, Tuple, Optional, List, Any
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

# Configurações de otimização
jax.config.update('jax_default_matmul_precision', 'bfloat16')  # Mixed precision
jax.config.update('jax_enable_x64', True)  # Precisão aumentada quando necessário
jax.config.update('jax_debug_nans', True)  # Detecta NaNs durante o treinamento

# Cache de compilação JIT
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
if not os.path.exists('.jax_cache'):
    os.makedirs('.jax_cache')
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'  # Controle de memória

from models.lunar_core import LunarCore
from data.dataset_loader import PixelArtDataset

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Gerenciador de checkpoints com validação e recuperação automática."""
    
    def __init__(self, config: Dict[str, Any]):
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.max_checkpoints = config.get('checkpointing', {}).get('max_checkpoints', 5)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        
    def save_checkpoint(self, state: 'TrainState', epoch: int, metrics: Dict[str, float]) -> Optional[Path]:
        """
        Salva checkpoint com validação de integridade.
        
        Args:
            state: Estado atual do modelo
            epoch: Época atual
            metrics: Métricas do treinamento
            
        Returns:
            Path do checkpoint salvo ou None se falhou
        """
        try:
            checkpoint = {
                'step': state.step,
                'params': state.params,
                'opt_state': state.opt_state,
                'batch_stats': state.batch_stats,
                'key': state.key,
                'metrics': metrics,
                'epoch': epoch
            }
            
            # Gera nome único com timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            checkpoint_path = self.checkpoint_dir / f"checkpoint_e{epoch}_{timestamp}.pkl"
            
            # Salva com verificação de integridade
            with open(checkpoint_path, 'wb') as f:
                checkpoint_bytes = flax.serialization.to_bytes(checkpoint)
                f.write(checkpoint_bytes)
                
            # Verifica integridade
            self._validate_checkpoint(checkpoint_path)
            
            # Mantém número máximo de checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"Checkpoint salvo com sucesso em: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Erro ao salvar checkpoint: {str(e)}")
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            return None
    
    def load_last_valid_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Carrega o último checkpoint válido.
        
        Returns:
            Dados do checkpoint ou None se não encontrado
        """
        try:
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
            if not checkpoints:
                return None
                
            # Tenta carregar checkpoints do mais recente ao mais antigo
            for checkpoint_path in reversed(checkpoints):
                try:
                    if self._validate_checkpoint(checkpoint_path):
                        with open(checkpoint_path, 'rb') as f:
                            return flax.serialization.from_bytes({}, f.read())
                except Exception as e:
                    logger.warning(f"Checkpoint {checkpoint_path} inválido: {str(e)}")
                    continue
                    
            return None
            
        except Exception as e:
            logger.error(f"Erro ao carregar checkpoints: {str(e)}")
            return None
    
    def _validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """Valida integridade do checkpoint."""
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = flax.serialization.from_bytes({}, f.read())
                
            # Verifica campos obrigatórios
            required_fields = ['step', 'params', 'opt_state', 'batch_stats', 'key']
            for field in required_fields:
                if field not in checkpoint_data:
                    raise ValueError(f"Campo obrigatório ausente: {field}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Erro na validação do checkpoint {checkpoint_path}: {str(e)}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """Remove checkpoints antigos mantendo apenas os mais recentes."""
        try:
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
            if len(checkpoints) > self.max_checkpoints:
                for checkpoint in checkpoints[:-self.max_checkpoints]:
                    checkpoint.unlink()
                    logger.info(f"Checkpoint antigo removido: {checkpoint}")
        except Exception as e:
            logger.error(f"Erro ao limpar checkpoints antigos: {str(e)}")

def validate_inputs(batch: Dict[str, Any], training_mode: str) -> bool:
    """
    Valida inputs do treinamento.
    
    Args:
        batch: Batch de dados
        training_mode: Modo de treinamento
        
    Returns:
        True se válido, False caso contrário
    """
    try:
        # Verifica campos obrigatórios
        if 'image' not in batch:
            logger.error("Batch não contém campo 'image'")
            return False
            
        # Verifica tipos e formatos
        if not isinstance(batch['image'], (np.ndarray, jnp.ndarray)):
            logger.error("Campo 'image' deve ser um array")
            return False
            
        # Verifica valores NaN/Inf
        if jnp.any(jnp.isnan(batch['image'])) or jnp.any(jnp.isinf(batch['image'])):
            logger.error("Detectados valores NaN/Inf nas imagens")
            return False
            
        # Validações específicas do modo prompt
        if training_mode == 'prompt':
            if 'prompt' not in batch:
                logger.error("Modo prompt requer campo 'prompt' no batch")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Erro na validação de inputs: {str(e)}")
        return False

class TrainState(train_state.TrainState):
    batch_stats: Dict
    key: jnp.ndarray

def create_train_state(rng: jax.random.PRNGKey, config: dict, model_config: Optional[dict] = None, checkpoint_path: Optional[str] = None) -> TrainState:
    """
    Cria o estado inicial do treinamento com suporte a transfer learning.
    
    Args:
        rng: Chave PRNG do JAX
        config: Configuração do treinamento
        model_config: Configuração específica do modelo (opcional)
        checkpoint_path: Caminho para carregar checkpoint (opcional)
    """
    # Usa model_config se fornecido, senão usa config['model']
    model_config = model_config or config['model']
    
    # Verifica configuração de transfer learning
    transfer_config = model_config.get('transfer_learning', {})
    is_transfer = transfer_config.get('enabled', False)
    
    # Configura mixed precision
    use_mixed_precision = config['training'].get('mixed_precision', True)
    if use_mixed_precision:
        policy = jax.experimental.enable_x64(False)
        dtype = jnp.bfloat16
        logger.info("Usando mixed precision training (bfloat16)")
    else:
        dtype = jnp.float32
        logger.info("Usando precisão padrão (float32)")
    
    # Se transfer learning está ativado, usa checkpoint especificado
    if is_transfer and not checkpoint_path:
        transfer_checkpoint = transfer_config.get('checkpoint_path')
        if transfer_checkpoint:
            checkpoint_path = transfer_checkpoint
            logger.info(f"Usando checkpoint para transfer learning: {checkpoint_path}")
        else:
            logger.warning("Transfer learning ativado mas nenhum checkpoint especificado")
    
    # Inicializa modelo com dtype apropriado
    model = LunarCore(
        latent_dim=model_config['latent_dim'],
        filters=model_config['filters'],
        num_residual_blocks=model_config.get('num_residual_blocks', 2),
        input_shape=model_config['input_shape'],
        use_text=model_config.get('use_text', False),
        fusion_type=model_config.get('latent_fusion', {}).get('type', 'concat'),
        text_encoder_config=model_config.get('text_encoder_config'),
        dtype=dtype
    )
    
    # Cria variáveis com dtype apropriado
    rng, init_rng = jax.random.split(rng)
    input_shape = (1,) + tuple(model_config['input_shape'])
    variables = model.init(
        {'params': init_rng}, 
        jnp.ones(input_shape, dtype=dtype),
        tokens=None if not model_config.get('use_text') else jnp.ones((1, 128), dtype=jnp.int32),
        training=True
    )
    
    # Configura otimizador com suporte a mixed precision
    learning_rate = config['training']['learning_rate']
    if is_transfer:
        lr_factor = transfer_config.get('learning_rate_factor', 0.5)
        learning_rate *= lr_factor
        logger.info(f"Taxa de aprendizado ajustada para transfer learning: {learning_rate}")
    
    # Configura otimizador com mixed precision
    if use_mixed_precision:
        tx = optax.chain(
            optax.scale_by_adam(b1=config['training'].get('beta1', 0.9),
                              b2=config['training'].get('beta2', 0.999)),
            optax.scale_by_schedule(lambda step: learning_rate),
            optax.scale(1.0)  # Loss scaling para estabilidade numérica
        )
    else:
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
    
    # Carrega e valida checkpoint se fornecido
    if checkpoint_path is not None:
        try:
            logger.info(f"Carregando checkpoint: {checkpoint_path}")
            with open(checkpoint_path, 'rb') as f:
                checkpoint = flax.serialization.from_bytes(state, f.read())
                
            # Valida estrutura do checkpoint
            required_fields = ['step', 'params', 'opt_state', 'batch_stats', 'key']
            for field in required_fields:
                if field not in checkpoint:
                    raise ValueError(f"Campo obrigatório ausente no checkpoint: {field}")
            
            # Valida compatibilidade dos parâmetros
            for key in checkpoint['params']:
                if key not in state.params:
                    raise ValueError(f"Parâmetro incompatível no checkpoint: {key}")
                if checkpoint['params'][key].shape != state.params[key].shape:
                    raise ValueError(f"Shape incompatível para parâmetro {key}")
            
            # Converte parâmetros para dtype apropriado se necessário
            if use_mixed_precision:
                checkpoint['params'] = jax.tree_map(
                    lambda x: x.astype(dtype) if x.dtype != jnp.int32 else x,
                    checkpoint['params']
                )
            
            # Atualiza estado
            state = state.replace(
                step=checkpoint['step'],
                params=checkpoint['params'],
                opt_state=checkpoint['opt_state'],
                batch_stats=checkpoint['batch_stats'],
                key=checkpoint['key']
            )
            
            logger.info(f"Checkpoint carregado com sucesso. Step inicial: {state.step}")
            
        except Exception as e:
            if is_transfer:
                logger.error(f"Erro fatal ao carregar checkpoint para transfer learning: {str(e)}")
                raise
            else:
                logger.warning(f"Erro ao carregar checkpoint, iniciando do zero: {str(e)}")
    
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
    Executa um passo de treinamento com validação e tratamento de erros.
    
    Args:
        state: Estado atual do modelo
        batch: Batch de dados
        training_mode: 'pixel_art' ou 'prompt'
        
    Returns:
        Tuple[TrainState, Dict]: Novo estado e métricas ou estado anterior e métricas de erro
    """
    try:
        # Validação de inputs
        if not validate_inputs(batch, training_mode):
            error_metrics = {
                'loss': float('inf'),
                'recon_loss': float('inf'),
                'kl_loss': float('inf'),
                'error': 'Inputs inválidos'
            }
            return state, error_metrics
            
        # Executa passo de treinamento com tratamento de erros
        try:
            if training_mode == 'prompt':
                new_state, metrics = _train_step_prompt(state, batch)
            else:
                new_state, metrics = _train_step_pixel_art(state, batch)
                
            # Validação pós-treino
            if any(jnp.isnan(v) or jnp.isinf(v) for v in metrics.values()):
                logger.error("Detectados valores NaN/Inf nas métricas")
                return state, {
                    'loss': float('inf'),
                    'recon_loss': float('inf'),
                    'kl_loss': float('inf'),
                    'error': 'NaN/Inf nas métricas'
                }
                
            return new_state, metrics
            
        except Exception as e:
            logger.error(f"Erro durante passo de treinamento: {str(e)}")
            return state, {
                'loss': float('inf'),
                'recon_loss': float('inf'),
                'kl_loss': float('inf'),
                'error': f'Erro no treino: {str(e)}'
            }
            
    except Exception as e:
        logger.error(f"Erro fatal no passo de treinamento: {str(e)}")
        return state, {
            'loss': float('inf'),
            'recon_loss': float('inf'),
            'kl_loss': float('inf'),
            'error': f'Erro fatal: {str(e)}'
        }

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

def save_generated_samples(state: TrainState, num_samples: int, step: int, 
                         output_dir: str, training_mode: str = 'pixel_art',
                         prompts: Optional[List[str]] = None):
    """
    Gera e salva amostras geradas pelo modelo.
    
    Args:
        state: Estado atual do modelo
        num_samples: Número de amostras para gerar
        step: Passo atual do treinamento
        output_dir: Diretório para salvar as imagens
        training_mode: Modo de treinamento ('pixel_art' ou 'prompt')
        prompts: Lista de prompts para geração condicional (opcional)
    """
    try:
        # Cria diretório para amostras geradas
        samples_dir = Path(output_dir) / 'generated_samples'
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Gera latents aleatórios
        rng = jax.random.PRNGKey(int(time.time()))
        z = jax.random.normal(rng, shape=(num_samples, state.params['encoder']['mean']['kernel'].shape[1]))
        
        # Prepara tokens se necessário
        tokens = None
        if training_mode == 'prompt' and prompts:
            # Aqui você precisaria implementar a tokenização dos prompts
            # Por enquanto, vamos usar tokens dummy
            tokens = jnp.ones((num_samples, 128), dtype=jnp.int32)
        
        # Gera imagens
        samples = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats},
            z,
            tokens=tokens,
            method='generate',
            training=False
        )
        
        # Converte de [-1, 1] para [0, 1]
        samples = (samples + 1) / 2
        
        # Salva cada amostra individualmente
        for i in range(num_samples):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(samples[i])
            ax.axis('off')
            
            title = f'Geração {i+1}'
            if prompts and i < len(prompts):
                title += f'\nPrompt: {prompts[i]}'
            ax.set_title(title)
            
            plt.savefig(samples_dir / f'generated_{step}_sample_{i+1}.png')
            plt.close()
            
        logger.info(f"Salvas {num_samples} amostras geradas no passo {step}")
        
    except Exception as e:
        logger.error(f"Erro ao gerar/salvar amostras: {str(e)}")

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

def optimize_batch_size(num_samples: int, min_batch: int = 8, max_batch: int = 128) -> int:
    """
    Calcula o batch size ótimo baseado no número de amostras e memória disponível.
    
    Args:
        num_samples: Número total de amostras no dataset
        min_batch: Tamanho mínimo do batch
        max_batch: Tamanho máximo do batch
    
    Returns:
        Tamanho ótimo do batch
    """
    try:
        # Detecta memória disponível
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            # Se GPU disponível, usa 75% da memória para determinar batch size
            mem_available = 0.75 * jax.device_memory_info(gpu_devices[0])['available']
        else:
            # Se CPU, usa um valor conservador
            mem_available = 1024 * 1024 * 1024  # 1GB
        
        # Calcula batch size baseado na memória (estimativa aproximada)
        estimated_sample_size = 256 * 256 * 4  # Tamanho estimado de uma amostra
        max_possible_batch = int(mem_available / (estimated_sample_size * 2))  # Fator 2 para segurança
        
        # Ajusta para limites definidos
        optimal_batch = min(max(min_batch, max_possible_batch), max_batch)
        
        # Ajusta para ser potência de 2 para melhor performance
        optimal_batch = 2 ** int(np.log2(optimal_batch))
        
        logger.info(f"Batch size ótimo calculado: {optimal_batch}")
        return optimal_batch
        
    except Exception as e:
        logger.warning(f"Erro ao calcular batch size ótimo: {str(e)}. Usando valor padrão: {min_batch}")
        return min_batch

def create_optimized_dataset(dataset, batch_size: int, is_training: bool = True):
    """
    Cria um dataset otimizado com prefetch e cache.
    
    Args:
        dataset: Dataset base
        batch_size: Tamanho do batch
        is_training: Se True, aplica shuffle e repeat
    """
    # Aplica cache para evitar recarregamento desnecessário
    dataset = dataset.cache()
    
    if is_training:
        # Shuffle com buffer grande o suficiente
        dataset = dataset.shuffle(buffer_size=10000, seed=42)
        dataset = dataset.repeat()
    
    # Batch com drop_remainder para evitar problemas com últimos batches incompletos
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Prefetch próximo batch usando JAX
    dataset = dataset.prefetch(2)  # Prefetch 2 batches
    
    return dataset

def main():
    try:
        # Carrega configuração
        print("Carregando configuração...")
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Inicializa logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        
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
        
        # Inicializa gerenciador de checkpoints
        checkpoint_manager = CheckpointManager(config)
        
        # Inicializa dataset com tratamento de erros
        try:
            print("Carregando dataset...")
            dataset = PixelArtDataset(config['data'])
            train_ds, val_ds = dataset.split_train_val(config['data']['validation_split'])
            
            # Calcula batch size ótimo
            num_samples = len(train_ds)
            optimal_batch_size = optimize_batch_size(num_samples)
            config['training']['batch_size'] = optimal_batch_size
            
            # Cria datasets otimizados
            train_ds = create_optimized_dataset(train_ds, optimal_batch_size, is_training=True)
            val_ds = create_optimized_dataset(val_ds, optimal_batch_size, is_training=False)
            
        except Exception as e:
            logger.error(f"Erro ao carregar dataset: {str(e)}")
            return
        
        # Conta o número total de batches
        num_train_batches = sum(1 for _ in train_ds)
        num_val_batches = sum(1 for _ in val_ds)
        
        print(f"Número de batches de treino: {num_train_batches}")
        print(f"Número de batches de validação: {num_val_batches}")
        
        # Inicializa estado com recuperação automática
        try:
            rng = jax.random.PRNGKey(0)
            checkpoint_path = None
            
            # Tenta carregar último checkpoint válido
            if config['training'].get('auto_resume', True):
                last_checkpoint = checkpoint_manager.load_last_valid_checkpoint()
                if last_checkpoint is not None:
                    logger.info("Recuperando de checkpoint anterior...")
                    checkpoint_path = str(last_checkpoint)
                    
            # Se não houver checkpoint válido e for modo prompt, erro
            elif training_mode == 'prompt':
                checkpoint_dir = Path(config['logging']['checkpoint_dir'])
                latest_checkpoint = max(checkpoint_dir.glob('checkpoint_*.pkl'), 
                                     key=lambda x: int(x.stem.split('_')[1]))
                checkpoint_path = str(latest_checkpoint)
            
            state = create_train_state(rng, config, checkpoint_path=checkpoint_path)
            print("Modelo inicializado com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo: {str(e)}")
            return
        
        # Configurações de recuperação
        max_recovery_attempts = config.get('training', {}).get('max_recovery_attempts', 3)
        recovery_attempts = 0
        
        # Loop de treinamento
        print(f"\nIniciando treinamento no modo: {training_mode}")
        try:
            for epoch in range(config['training']['num_epochs']):
                epoch_start_time = time.time()
                
                # Treino
                train_metrics = []
                train_iterator = tqdm(train_ds, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
                
                for batch_idx, batch in enumerate(train_iterator):
                    try:
                        state, metrics = train_step(state, batch, training_mode)
                        
                        # Verifica se houve erro no treino
                        if 'error' in metrics:
                            logger.warning(f"Erro no batch {batch_idx}: {metrics['error']}")
                            recovery_attempts += 1
                            
                            if recovery_attempts > max_recovery_attempts:
                                raise RuntimeError("Número máximo de tentativas de recuperação excedido")
                                
                            # Tenta recuperar do último checkpoint
                            last_checkpoint = checkpoint_manager.load_last_valid_checkpoint()
                            if last_checkpoint is not None:
                                state = create_train_state(rng, config, checkpoint_path=str(last_checkpoint))
                                logger.info("Recuperado de checkpoint anterior")
                                continue
                                
                        else:
                            recovery_attempts = 0  # Reset contador se sucesso
                            train_metrics.append(metrics)
                        
                        # Atualiza a barra de progresso
                        train_iterator.set_postfix({
                            'loss': f"{metrics['loss']:.4f}",
                            'recon': f"{metrics['recon_loss']:.4f}",
                            'kl': f"{metrics['kl_loss']:.4f}"
                        })
                        
                    except Exception as e:
                        logger.error(f"Erro no batch {batch_idx}: {str(e)}")
                        recovery_attempts += 1
                        if recovery_attempts > max_recovery_attempts:
                            raise RuntimeError("Número máximo de tentativas de recuperação excedido")
                        continue
                
                # Calcula médias do treino (ignora batches com erro)
                if train_metrics:
                    avg_train_loss = np.mean([m['loss'] for m in train_metrics])
                    avg_train_recon = np.mean([m['recon_loss'] for m in train_metrics])
                    avg_train_kl = np.mean([m['kl_loss'] for m in train_metrics])
                    
                    # Validação
                    if (epoch + 1) % config['training']['eval_interval'] == 0:
                        val_metrics = []
                        for batch in val_ds:
                            try:
                                metrics = compute_metrics(batch, state, state.key, training_mode)
                                if 'error' not in metrics:
                                    val_metrics.append(metrics)
                            except Exception as e:
                                logger.error(f"Erro na validação: {str(e)}")
                                continue
                        
                        # Calcula médias da validação
                        if val_metrics:
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
                        try:
                            # Salva amostras de reconstrução
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
                                
                                # Loga reconstruções no W&B
                                if config['logging'].get('save_generations', True):
                                    wandb.log({
                                        'samples/reconstruction': wandb.Image(
                                            os.path.join(output_dir, f'sample_{epoch+1}.png'),
                                            caption=f"Reconstrução (Época {epoch+1})"
                                        )
                                    })
                            
                            # Gera e salva amostras se habilitado
                            if config['training'].get('generation', {}).get('enabled', False):
                                if (epoch + 1) % config['training']['generation']['interval'] == 0:
                                    num_samples = config['training']['generation']['num_samples']
                                    prompts = None
                                    if training_mode == 'prompt':
                                        prompts = config['training']['generation'].get('prompts', [])[:num_samples]
                                    
                                    save_generated_samples(
                                        state,
                                        num_samples,
                                        epoch + 1,
                                        output_dir,
                                        training_mode,
                                        prompts
                                    )
                                    
                                    # Loga gerações no W&B
                                    if config['logging'].get('save_generations', True):
                                        samples_dir = Path(output_dir) / 'generated_samples'
                                        for i, sample_path in enumerate(samples_dir.glob(f'generated_{epoch+1}_sample_*.png')):
                                            caption = f"Geração {i+1} (Época {epoch+1})"
                                            if prompts and i < len(prompts):
                                                caption += f"\nPrompt: {prompts[i]}"
                                            
                                            wandb.log({
                                                f'samples/generation_{i+1}': wandb.Image(
                                                    str(sample_path),
                                                    caption=caption
                                                )
                                            })
                                    
                        except Exception as e:
                            logger.error(f"Erro ao salvar amostras: {str(e)}")
                    
                    # Salva checkpoint
                    if (epoch + 1) % config['training']['save_interval'] == 0:
                        checkpoint_path = checkpoint_manager.save_checkpoint(
                            state, 
                            epoch + 1,
                            {
                                'loss': avg_train_loss,
                                'recon_loss': avg_train_recon,
                                'kl_loss': avg_train_kl
                            }
                        )
                        if checkpoint_path:
                            print(f"\nCheckpoint salvo em: {checkpoint_path}")
                
                # Monitora tempo de época
                epoch_time = time.time() - epoch_start_time
                if epoch_time > config.get('monitoring', {}).get('max_epoch_time', 3600):
                    logger.warning(f"Época {epoch+1} demorou mais que o esperado: {epoch_time:.2f}s")
        
        except KeyboardInterrupt:
            logger.info("Treinamento interrompido pelo usuário")
            # Salva checkpoint final
            checkpoint_manager.save_checkpoint(
                state,
                epoch + 1,
                {
                    'loss': avg_train_loss,
                    'recon_loss': avg_train_recon,
                    'kl_loss': avg_train_kl
                }
            )
            
        except Exception as e:
            logger.error(f"Erro fatal durante treinamento: {str(e)}")
            # Tenta salvar checkpoint de emergência
            checkpoint_manager.save_checkpoint(
                state,
                epoch + 1,
                {
                    'loss': float('inf'),
                    'recon_loss': float('inf'),
                    'kl_loss': float('inf'),
                    'error': str(e)
                }
            )
            raise
            
    except Exception as e:
        logger.error(f"Erro fatal na inicialização: {str(e)}")
        raise

if __name__ == "__main__":
    main() 