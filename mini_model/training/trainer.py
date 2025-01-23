"""
Sistema de treinamento otimizado para H100 com suporte a FP8, mixed precision e DDP.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.nccl as nccl
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torch.cuda.memory as memory_utils
import wandb
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
from rich.console import Console
import warnings
from tqdm import tqdm
import copy
import torch.optim as optim
from collections import defaultdict
import itertools
import traceback
from datetime import timedelta
import optuna
from optuna.distributions import FloatDistribution, CategoricalDistribution
import random
import albumentations as A
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity

# Desabilitar otimizações que requerem compilador
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.jit.enable = False
torch._dynamo.config.dynamic_shapes = False
torch._dynamo.config.cache_size_limit = 0

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False
    print("Transformer Engine não disponível, usando PyTorch padrão")

from ..core.mini_arch import Pixel16Generator, PixelArtLoss
from ..data.pixel16_dataset import Pixel16Dataset
from ..utils.monitoring import ResourceMonitor
from ..utils.checkpointing import CheckpointManager
from ..utils.visualization import VisualizationManager

class GPUManager:
    """Gerenciador de GPUs."""
    
    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """Coleta informações detalhadas das GPUs."""
        gpus = []
        
        if not torch.cuda.is_available():
            return gpus
            
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024 * 1024)  # Converter para MB
            allocated_memory = torch.cuda.memory_allocated(i) / (1024 * 1024)
            free_memory = total_memory - allocated_memory
            
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory": total_memory,
                "free_memory": free_memory,
                "compute_capability": f"{props.major}.{props.minor}"
            })
        
        return gpus
    
    @staticmethod
    def select_best_gpus(num_gpus: Optional[int] = None) -> List[int]:
        """Seleciona as melhores GPUs disponíveis."""
        if not torch.cuda.is_available():
            return []
            
        gpus = GPUManager.get_gpu_info()
        if not gpus:
            return []
            
        # Ordenar por capacidade de computação e memória livre
        gpus.sort(
            key=lambda x: (
                float(x["compute_capability"]),
                x["free_memory"]
            ),
            reverse=True
        )
        
        if num_gpus is None:
            num_gpus = len(gpus)
        
        return [gpu["index"] for gpu in gpus[:num_gpus]]
    
    @staticmethod
    def optimize_gpu_settings():
        """Otimiza configurações de GPU."""
        if not torch.cuda.is_available():
            return
            
        # Configurar cuDNN
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        
        # Habilitar TF32 em GPUs Ampere+
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Configurar fração de memória
        torch.cuda.set_per_process_memory_fraction(0.95)

class MultiGPUTrainer:
    """Treinador otimizado para múltiplas GPUs."""
    
    def __init__(
        self,
        config_path: str,
        data_dir: str,
        output_dir: str,
        local_rank: int = 0,
        world_size: int = 1,
        gpu_ids: Optional[List[int]] = None,
        dev_mode: bool = False
    ):
        """Inicializa o treinador.
        
        Args:
            config_path: Caminho para arquivo de configuração
            data_dir: Diretório com dados de treinamento
            output_dir: Diretório para salvar outputs
            local_rank: Rank local do processo
            world_size: Número total de processos
            gpu_ids: Lista de IDs das GPUs a usar
            dev_mode: Se True, usa CPU e configurações reduzidas
        """
        # Configurações básicas
        self.config_path = config_path
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.local_rank = local_rank
        self.world_size = world_size
        self.gpu_ids = gpu_ids
        self.dev_mode = dev_mode
        self.is_main_process = (local_rank == 0)
        
        # Carregar configuração
        self.config = self._load_config()
        
        # Configurar device
        self._setup_device()
        
        # Configurar distribuição
        if not dev_mode and world_size > 1:
            self._setup_distributed()
        
        # Configurar modelo e otimizador
        self._setup_model()
        self._setup_data()
        self._setup_optimization()
        
        # Configurar logging e checkpointing
        if self.is_main_process:
            self._setup_logging()
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=self.output_dir / "checkpoints",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                save_best_only=True
            )
            
        # Monitor de recursos
        self.monitor = ResourceMonitor()
        if self.is_main_process:
            self.monitor.start()
            
        # Visualização
        if self.is_main_process:
            self.viz = VisualizationManager(str(self.output_dir))
            
        # Estado do treinamento
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.steps_without_improvement = 0
        
        # Configurações de logging
        self.log_interval = self.config["logging"].get("log_interval", 100)
        self.save_interval = self.config["logging"].get("save_interval", 1000)
        self.vis_interval = self.config["logging"].get("vis_interval", 100)
        
        # Automatic Mixed Precision
        self.scaler = GradScaler()
        
    def _load_config(self) -> Dict[str, Any]:
        """Carrega e valida configuração."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
            
        # Validar seções obrigatórias
        required_sections = ["model", "training", "optimization", "logging"]
        for section in required_sections:
            if section not in config:
                config[section] = {}
                
        # Configurações padrão para modo dev
        if self.dev_mode:
            config["training"]["batch_size"] = 8
            config["training"]["num_epochs"] = 2
            config["training"]["num_workers"] = 0
            
        return config
        
    def _setup_device(self):
        """Configura o dispositivo de execução."""
        if self.dev_mode:
            self.device = torch.device("cpu")
            return
            
        if not torch.cuda.is_available():
            print("\nCUDA não disponível - usando CPU")
            self.device = torch.device("cpu")
            return
            
        if self.gpu_ids:
            self.device = torch.device(f"cuda:{self.gpu_ids[self.local_rank]}")
        else:
            self.device = torch.device("cuda")
            
        # Otimizar configurações
        GPUManager.optimize_gpu_settings()
        
    def _setup_distributed(self):
        """Configura treinamento distribuído."""
        # Inicializar processo
        torch.cuda.set_device(self.device)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.local_rank
        )
        
    def _setup_model(self):
        """Inicializa modelo e função de perda."""
        # Criar modelo
        self.model = Pixel16Generator(self.config["model"])
        self.model = self.model.to(self.device)
        
        # Distribuir modelo se necessário
        if not self.dev_mode and self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index],
                output_device=self.device.index
            )
            
        # Função de perda
        loss_config = {
            "content_weight": self.config["losses"]["content_weight"],
            "style_weight": self.config["losses"]["style_weight"],
            "pixel_weight": self.config["losses"]["pixel_weight"],
            "quant_weight": self.config["losses"]["quant_weight"],
            "temperature_init": self.config["losses"]["temperature_init"],
            "temperature_min": self.config["losses"]["temperature_min"],
            "temperature_max": self.config["losses"]["temperature_max"]
        }
        self.criterion = PixelArtLoss(loss_config).to(self.device)
        
    def _setup_data(self):
        """Configura dataloaders."""
        # Dataset de treino
        self.train_dataset = Pixel16Dataset(
            data_dir=self.data_dir,
            split="train"
        )
        
        # Dataset de validação
        self.val_dataset = Pixel16Dataset(
            data_dir=self.data_dir,
            split="val"
        )
        
        # Dataset de teste
        self.test_dataset = Pixel16Dataset(
            data_dir=self.data_dir,
            split="test"
        )
        
        # Configurar samplers
        train_sampler = (
            DistributedSampler(self.train_dataset)
            if not self.dev_mode and self.world_size > 1
            else None
        )
        
        # Configurar dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=(train_sampler is None),
            num_workers=self.config["training"].get("num_workers", 4),
            pin_memory=True,
            sampler=train_sampler
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"].get("num_workers", 4),
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"].get("num_workers", 4),
            pin_memory=True
        )
        
        if self.is_main_process:
            print(f"\nDataset carregado:")
            print(f"- Treino: {len(self.train_dataset)} imagens")
            print(f"- Validação: {len(self.val_dataset)} imagens")
            print(f"- Teste: {len(self.test_dataset)} imagens")
            
            # Mostrar dimensões
            sprites, labels = next(iter(self.train_loader))
            print(f"\nDimensões:")
            print(f"- Sprites: {tuple(sprites.shape)}")
            print(f"- Labels: {tuple(labels.shape)}")
        
    def _setup_optimization(self):
        """Configura otimizador e scheduler."""
        # Otimizador
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.config["training"]["learning_rate"]),
            weight_decay=float(self.config["training"]["weight_decay"])
        )
        
        # Learning rate scheduler
        scheduler_config = self.config["training"].get("scheduler", {})
        if scheduler_config.get("type") == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["training"]["num_epochs"],
                eta_min=float(scheduler_config.get("min_lr", 1e-6))
            )
        else:
            self.scheduler = None
            
    def _setup_logging(self):
        """Configura logging com W&B."""
        if self.config["logging"].get("use_wandb", False):
            wandb.init(
                project=self.config["logging"].get("wandb_project", "mini-pixel-art"),
                config=self.config,
                dir=str(self.output_dir / "logs" / "wandb")
            )
            
    def train_epoch(self):
        """Treina por uma época."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Configurar tqdm
        train_iter = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            disable=not self.is_main_process
        )
        
        for batch_idx, (sprites, style_labels) in enumerate(train_iter):
            # Mover para device
            sprites = sprites.to(self.device)
            style_labels = style_labels.to(self.device)
            
            # Forward pass com mixed precision
            with autocast(dtype=torch.float16):
                output = self.model(sprites, style_labels)
                loss_dict = self.criterion(output, sprites)
                loss = loss_dict["loss"]
                
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if "gradient_clip_val" in self.config["training"]:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["gradient_clip_val"]
                )
                
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Atualizar métricas
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if self.is_main_process and batch_idx % self.log_interval == 0:
                # Coletar métricas
                current_lr = self.optimizer.param_groups[0]["lr"]
                
                # Estatísticas do sistema
                stats = self.monitor.get_stats_summary()
                
                # Métricas para logging
                metrics = {
                    "train/loss": loss_dict["loss"],
                    "train/content_loss": loss_dict["content_loss"],
                    "train/style_loss": loss_dict["style_loss"],
                    "train/pixel_loss": loss_dict["pixel_loss"],
                    "train/learning_rate": current_lr,
                    "train/epoch": self.current_epoch
                }
                
                # Adicionar métricas do sistema
                if "cpu" in stats and "ram" in stats:
                    metrics.update({
                        "system/cpu_percent": stats["cpu"]["mean"],
                        "system/ram_percent": stats["ram"]["mean"]
                    })
                
                if torch.cuda.is_available() and "gpu_memory" in stats and "gpu_utilization" in stats:
                    metrics.update({
                        "system/gpu_memory": stats["gpu_memory"][0]["mean"],
                        "system/gpu_utilization": stats["gpu_utilization"][0]["mean"]
                    })
                
                # Logging
                if self.config["logging"].get("use_wandb", False):
                    wandb.log(metrics, step=self.global_step)
                    
                # Atualizar barra de progresso
                train_iter.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.2e}"
                })
                
            # Visualização
            if self.is_main_process and batch_idx % self.vis_interval == 0:
                with torch.no_grad():
                    self.viz.save_image_grid(
                        real_images=sprites[:8].cpu(),
                        generated_images=output[0][:8].cpu() if isinstance(output, tuple) else output[:8].cpu(),
                        style_labels=style_labels[:8].cpu(),
                        phase="train",
                        epoch=self.current_epoch,
                        batch_idx=batch_idx
                    )
                    
                    # Visualizar mapas de atenção apenas se estiverem disponíveis e configurados
                    if self.config["training"].get("visualize_attention", False) and isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], list) and len(output[1]) > 0:
                        attention_maps = torch.stack([att[:8].cpu() for att in output[1]], dim=0)
                        self.viz.visualize_attention(
                            attention_maps=attention_maps,
                            images=sprites[:8].cpu(),
                            epoch=self.current_epoch,
                            phase="train"
                        )
                    
            self.global_step += 1
            
        # Retornar loss média
        return total_loss / num_batches
        
    def validate(self):
        """Valida o modelo."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for sprites, style_labels in self.val_loader:
                # Mover para device
                sprites = sprites.to(self.device)
                style_labels = style_labels.to(self.device)
                
                # Forward pass
                with autocast(dtype=torch.float16):
                    output = self.model(sprites, style_labels)
                    loss_dict = self.criterion(output, sprites)
                    loss = loss_dict["loss"]
                    
                total_loss += loss.item()
                num_batches += 1
                
        val_loss = total_loss / num_batches
        
        # Early stopping
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.steps_without_improvement = 0
            
            # Salvar melhor modelo
            if self.is_main_process:
                self.checkpoint_manager.save(
                    {
                        "epoch": self.current_epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                        "scaler_state_dict": self.scaler.state_dict(),
                        "val_loss": val_loss
                    },
                    is_best=True
                )
        else:
            self.steps_without_improvement += 1
            
        # Logging
        if self.is_main_process:
            metrics = {
                "val/loss": val_loss,
                "val/best_loss": self.best_val_loss,
                "val/steps_without_improvement": self.steps_without_improvement
            }
            
            if self.config["logging"].get("use_wandb", False):
                wandb.log(metrics, step=self.global_step)
                
        return val_loss
        
    def test(self):
        """Avalia o modelo no conjunto de teste."""
        self.model.eval()
        total_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for sprites, style_labels in self.test_loader:
                # Mover para device
                sprites = sprites.to(self.device)
                style_labels = style_labels.to(self.device)
                
                # Forward pass
                with autocast(dtype=torch.float16):
                    output = self.model(sprites, style_labels)
                    loss_dict = self.criterion(output, sprites)
                    
                # Atualizar métricas
                for k, v in loss_dict.items():
                    total_metrics[k] += v.item()
                num_batches += 1
                
        # Calcular médias
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
        
        # Logging
        if self.is_main_process:
            metrics = {f"test/{k}": v for k, v in avg_metrics.items()}
            
            if self.config["logging"].get("use_wandb", False):
                wandb.log(metrics, step=self.global_step)
                
            print("\nResultados do teste:")
            for k, v in avg_metrics.items():
                print(f"- {k}: {v:.4f}")
                
        return avg_metrics
        
    def train(self):
        """Treina o modelo."""
        try:
            for epoch in range(self.current_epoch, self.config["training"]["num_epochs"]):
                try:
                    self._train_epoch(epoch)
                    self._validate(epoch)
                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        self._handle_oom()
                    else:
                        raise
        except Exception as e:
            self._emergency_save()
            raise

    def _handle_oom(self):
        torch.cuda.empty_cache()
        self.config["training"]["batch_size"] = max(1, self.config["training"]["batch_size"] // 2)
        print(f"OOM detectado! Reduzindo batch size para {self.config['training']['batch_size']}")

    def _train_epoch(self, epoch: int):
        self.current_epoch = epoch
        train_loss = self.train_epoch()
        if self.is_main_process:
            print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")

    def _validate(self, epoch: int):
        val_loss = self.validate()
        if self.is_main_process:
            print(f"\nEpoch {epoch}: Val Loss = {val_loss:.4f}")

    def _emergency_save(self):
        if self.is_main_process:
            print("\nTreinamento interrompido! Salvando estado atual...")
            self.checkpoint_manager.save(
                epoch=self.current_epoch,
                val_loss=self.best_val_loss
            )
            self.monitor.stop()
            if self.config["logging"].get("use_wandb", False):
                wandb.finish()

    def _log_grad_stats(self, step: int):
        grads = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grads.append((name, grad_norm))
        
        # Identificar gradientes problemáticos
        top_grads = sorted(grads, key=lambda x: x[1], reverse=True)[:5]
        for name, norm in top_grads:
            wandb.log({f"gradients/{name}": norm}, step=step)

    def _validate_batch(self, batch):
        images, labels = batch
        # Verificar valores válidos
        if torch.isnan(images).any() or torch.isinf(images).any():
            raise ValueError("Dados inválidos detectados no batch!")
        
        # Verificar faixa de valores
        if (images.min() < -1.0) or (images.max() > 1.0):
            raise ValueError("Valores de pixel fora do intervalo [-1, 1]")
        
        return images, labels

class CheckpointManager:
    def __init__(self, 
                 checkpoint_dir: str,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 max_retries: int = 3,
                 backup_interval: timedelta = timedelta(hours=1),
                 save_best_only: bool = False,
                 max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checksum_enabled = True
        self.max_retries = max_retries
        self.backup_interval = backup_interval
        self.save_best_only = save_best_only
        self.max_checkpoints = max_checkpoints
        self.best_val_loss = float('inf')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, epoch: int, val_loss: float) -> None:
        """Salva checkpoint."""
        # Verificar se deve salvar
        if self.save_best_only and val_loss >= self.best_val_loss:
            return
            
        # Atualizar melhor loss
        self.best_val_loss = min(val_loss, self.best_val_loss)
            
        # Criar checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Salvar checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, str(checkpoint_path))
    
    def _verify_checkpoint(self, path: Path) -> bool:
        try:
            checkpoint = torch.load(path, map_location='cpu')
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            return True
        except:
            return False

class ExperimentManager:
    def __init__(self):
        self.hyperparams = {}
        self.metrics = defaultdict(list)
        self.artifacts = {}
        self._setup_optimization_space()
    
    def _setup_optimization_space(self):
        self.search_space = {
            'learning_rate': FloatDistribution(1e-6, 1e-3, log=True),
            'batch_size': CategoricalDistribution([32, 64, 128]),
            'optimizer': CategoricalDistribution(['AdamW', 'Lion', 'RMSprop'])
        }
    
    def run_sweep(self, num_trials=50):
        study = optuna.create_study(direction='minimize')
        study.optimize(self._objective, n_trials=num_trials)
        self._save_best_params(study.best_params)

class AdaptiveAugmenter:
    def __init__(self, initial_strength=0.5):
        self.current_strength = initial_strength
        self.performance_history = []
        self.augmentation_pool = [
            A.RandomBrightnessContrast(p=0.7),
            A.HueSaturationValue(p=0.7),
            A.PixelDropout(p=0.5),
            A.GridDistortion(p=0.3),
            A.ElasticTransform(p=0.3),
            A.ChannelShuffle(p=0.2)
        ]
        
    def adapt_augmentations(self, val_loss_trend):
        """Adapta força das augmentações baseado na performance"""
        if self._is_overfitting(val_loss_trend):
            self.current_strength = min(1.0, self.current_strength * 1.2)
        elif self._is_underfitting(val_loss_trend):
            self.current_strength = max(0.1, self.current_strength * 0.8)
            
        return self._create_transform_pipeline()
    
    def _create_transform_pipeline(self):
        return A.Compose([
            aug(p=self.current_strength) for aug in self.augmentation_pool
        ])

class PerformanceProfiler:
    def __init__(self):
        self.memory_peaks = []
        self.throughput_history = []
        self.bottlenecks = defaultdict(list)
        
    def profile_iteration(self, batch_size, step_time, memory_used):
        """Analisa performance por iteração"""
        throughput = batch_size / step_time
        self.throughput_history.append(throughput)
        self.memory_peaks.append(memory_used)
        
        # Detectar gargalos
        if step_time > self.avg_step_time * 1.5:
            self.bottlenecks['slow_iteration'].append({
                'step': len(self.throughput_history),
                'time': step_time,
                'memory': memory_used
            })
            
    def optimize_parameters(self):
        """Ajusta parâmetros baseado no profiling"""
        avg_throughput = np.mean(self.throughput_history[-100:])
        memory_headroom = 1 - (np.mean(self.memory_peaks) / torch.cuda.max_memory_allocated())
        
        return {
            'batch_size': self.suggest_batch_size(memory_headroom),
            'num_workers': self.suggest_num_workers(avg_throughput),
            'gradient_accumulation': self.suggest_grad_accum()
        } 

class RobustTrainingManager:
    def __init__(self):
        self.checkpoints = []
        self.error_log = defaultdict(list)
        self.recovery_states = {}
        
    def monitor_training(self, step, loss, grads, memory):
        """Monitora estado do treinamento"""
        # Detectar anomalias
        if self._detect_anomaly(loss, grads):
            self._trigger_recovery_protocol()
            
    def _detect_anomaly(self, loss, grads):
        """Detecta problemas no treinamento"""
        return (
            torch.isnan(loss) or
            torch.isinf(loss) or
            any(torch.isnan(g).any() for g in grads if g is not None) or
            any(torch.isinf(g).any() for g in grads if g is not None)
        )
        
    def _trigger_recovery_protocol(self):
        """Protocolo de recuperação"""
        # 1. Salvar estado atual
        self._save_emergency_checkpoint()
        
        # 2. Reduzir learning rate
        self.optimizer.param_groups[0]['lr'] *= 0.5
        
        # 3. Carregar último checkpoint bom
        self._load_last_good_checkpoint()
        
        # 4. Reiniciar acumuladores do otimizador
        self.optimizer.zero_grad(set_to_none=True)

class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.running_stats = {}
        
    def update(self, outputs, targets, loss_dict):
        """Atualiza métricas detalhadas"""
        # Métricas básicas
        self.metrics['loss'].append(loss_dict['loss'].item())
        
        # Métricas de qualidade
        self._compute_quality_metrics(outputs, targets)
        
        # Métricas de estilo
        self._compute_style_metrics(outputs, targets)
        
        # Métricas de performance
        self._compute_performance_metrics()
        
    def _compute_quality_metrics(self, outputs, targets):
        """Calcula métricas de qualidade de imagem"""
        with torch.no_grad():
            # PSNR
            psnr = -10 * torch.log10(F.mse_loss(outputs, targets))
            self.metrics['psnr'].append(psnr.item())
            
            # SSIM
            ssim = structural_similarity(outputs, targets)
            self.metrics['ssim'].append(ssim.item())
            
            # Edge Preservation
            edge_score = self._compute_edge_preservation(outputs, targets)
            self.metrics['edge_score'].append(edge_score)

class MultiObjectiveOptimizer:
    def __init__(self):
        self.objectives = {
            'quality': {'weight': 0.4, 'metric': 'psnr'},
            'style': {'weight': 0.3, 'metric': 'style_loss'},
            'performance': {'weight': 0.3, 'metric': 'throughput'}
        }
        
    def compute_weighted_loss(self, metrics):
        """Calcula loss ponderada multi-objetivo"""
        total_loss = 0
        for obj_name, obj_config in self.objectives.items():
            metric_value = metrics[obj_config['metric']]
            total_loss += obj_config['weight'] * metric_value
        return total_loss
    
    def update_weights(self, validation_metrics):
        """Atualiza pesos dos objetivos dinamicamente"""
        performance = self._evaluate_objectives(validation_metrics)
        self._adjust_weights(performance) 

class AdvancedLogger:
    def __init__(self):
        self.log_dir = Path("logs")
        self.fig_dir = self.log_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        
    def log_training_state(self, epoch, step, metrics, samples):
        """Log detalhado do estado do treinamento"""
        # Log métricas
        wandb.log({
            f"train/{k}": v for k, v in metrics.items()
        }, step=step)
        
        # Visualizações
        if step % 100 == 0:
            self._create_visualizations(samples, step)
            
    def _create_visualizations(self, samples, step):
        """Cria visualizações detalhadas"""
        # Grid de amostras
        fig = self._create_sample_grid(samples)
        wandb.log({"samples": wandb.Image(fig)}, step=step)
        
        # Mapa de atenção
        if hasattr(samples, 'attention_maps'):
            att_fig = self._visualize_attention(samples.attention_maps)
            wandb.log({"attention": wandb.Image(att_fig)}, step=step) 

class EnhancedTrainer(MultiGPUTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = PerformanceProfiler()
        self.augmenter = AdaptiveAugmenter()
        self.recovery = RobustTrainingManager()
        self.metrics = MetricsTracker()
        self.multi_obj = MultiObjectiveOptimizer()
        self.logger = AdvancedLogger()
        
    def train_epoch(self):
        for batch in self.train_loader:
            # Profile performance
            with self.profiler.track():
                loss = self.training_step(batch)
                
            # Monitor e recupera de problemas
            self.recovery.monitor_training(
                self.global_step, loss,
                self.get_gradients(),
                torch.cuda.memory_allocated()
            )
            
            # Adapta augmentações
            if self.global_step % 100 == 0:
                self.augmenter.adapt_augmentations(
                    self.metrics.get_val_loss_trend()
                )