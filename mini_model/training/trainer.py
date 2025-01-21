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

class GPUManager:
    """Gerenciador inteligente de GPUs."""
    
    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """Coleta informações detalhadas das GPUs."""
        gpus = []
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory = torch.cuda.get_device_memory_usage(i)
            
            gpus.append({
                "index": i,
                "name": props.name,
                "total_memory": props.total_memory,
                "free_memory": memory[1],
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
                "max_threads_per_block": props.max_threads_per_block,
                "is_integrated": props.is_integrated,
                "can_map_host_memory": props.can_map_host_memory
            })
        
        return gpus
    
    @staticmethod
    def select_best_gpus(num_gpus: int = None) -> List[int]:
        """Seleciona as melhores GPUs disponíveis."""
        gpus = GPUManager.get_gpu_info()
        
        # Filtrar GPUs integradas
        gpus = [g for g in gpus if not g["is_integrated"]]
        
        # Ordenar por capacidade computacional e memória livre
        gpus.sort(key=lambda x: (
            x["compute_capability"],
            x["free_memory"],
            x["multi_processor_count"]
        ), reverse=True)
        
        # Selecionar número de GPUs
        if num_gpus is None:
            num_gpus = len(gpus)
        
        return [g["index"] for g in gpus[:num_gpus]]
    
    @staticmethod
    def optimize_gpu_settings():
        """Otimiza configurações das GPUs."""
        # Habilitar TF32 se disponível (Ampere+)
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Otimizar cuDNN
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Configurar alocação de memória
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.memory_stats(True)
        
        # Configurar política de alocação
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.95)

class Trainer:
    """Treinador otimizado para H100."""
    
    def __init__(
        self,
        config_path: str,
        data_dir: str,
        output_dir: str,
        local_rank: int = 0,
        world_size: int = 1
    ):
        """
        Args:
            config_path: Caminho para arquivo de configuração
            data_dir: Diretório com dados de treinamento
            output_dir: Diretório para salvar checkpoints e logs
            local_rank: Rank local do processo
            world_size: Número total de processos
        """
        self.config = self._load_config(config_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.local_rank = local_rank
        self.world_size = world_size
        
        # Configurar ambiente distribuído
        if world_size > 1:
            dist.init_process_group("nccl")
            torch.cuda.set_device(local_rank)
        
        self.device = torch.device(f"cuda:{local_rank}")
        self.is_main_process = local_rank == 0
        
        # Inicializar componentes
        self._setup_model()
        self._setup_data()
        self._setup_optimization()
        self._setup_monitoring()
        
        # Logging
        if self.is_main_process:
            self.console = Console()
            self._setup_wandb()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carrega e valida configuração."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Validar configuração
        required_keys = ["model", "training", "optimization"]
        for key in required_keys:
            assert key in config, f"Configuração deve conter {key}"
        
        return config
    
    def _setup_model(self):
        """Inicializa modelo e função de perda."""
        # Criar modelo
        self.model = Pixel16Generator(self.config["model"])
        self.model.to(self.device)
        
        # Configurar FP8 se disponível
        if TE_AVAILABLE:
            fp8_recipe = recipe.DelayedScaling(
                margin=0,
                interval=1,
                fp8_format=recipe.Format.E4M3
            )
            self.model = te.train.fp8_autocast(
                self.model,
                fp8_recipe=fp8_recipe
            )
        
        # Distribuir modelo se necessário
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
        
        # Função de perda
        self.criterion = PixelArtLoss().to(self.device)
    
    def _setup_data(self):
        """Configura dataloaders."""
        # Datasets
        self.train_dataset = Pixel16Dataset(
            data_dir=self.data_dir,
            split="train",
            train_ratio=0.9,
            use_cache=True
        )
        
        self.val_dataset = Pixel16Dataset(
            data_dir=self.data_dir,
            split="val",
            train_ratio=0.9,
            use_cache=True
        )
        
        # Samplers distribuídos
        train_sampler = (
            DistributedSampler(self.train_dataset)
            if self.world_size > 1 else None
        )
        
        val_sampler = (
            DistributedSampler(self.val_dataset, shuffle=False)
            if self.world_size > 1 else None
        )
        
        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["batch_size"] * 2,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True
        )
    
    def _setup_optimization(self):
        """Configura otimizador e schedulers."""
        # Otimizador
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["training"]["num_epochs"],
            eta_min=self.config["training"]["min_lr"]
        )
        
        # Gradient scaler para mixed precision
        self.scaler = GradScaler()
        
        # Checkpoint manager
        if self.is_main_process:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=self.output_dir / "checkpoints",
                backup_dir=self.output_dir / "backups",
                max_checkpoints=5
            )
    
    def _setup_monitoring(self):
        """Configura monitoramento de recursos."""
        if self.is_main_process:
            self.monitor = ResourceMonitor(
                check_interval=1.0,
                memory_threshold=90.0
            )
            
            def alert_callback(message: str, stats):
                self.console.print(f"[red]{message}[/red]")
                if wandb.run is not None:
                    wandb.alert(
                        title="Recurso Crítico",
                        text=message,
                        level=wandb.AlertLevel.WARN
                    )
            
            self.monitor.add_alert_callback(alert_callback)
            self.monitor.start()
    
    def _setup_wandb(self):
        """Inicializa W&B para logging."""
        wandb.init(
            project=self.config["logging"]["wandb_project"],
            name=self.config["logging"]["experiment_name"],
            config=self.config,
            resume=True
        )
    
    def train_epoch(self, epoch: int) -> float:
        """Treina um epoch completo."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task(
                f"Epoch {epoch}", total=num_batches
            )
            
            for batch_idx, (sprites, style_labels) in enumerate(self.train_loader):
                sprites = sprites.to(self.device)
                style_labels = style_labels.to(self.device)
                
                # Forward pass com mixed precision
                with autocast(dtype=torch.float16):
                    output = self.model(sprites, style_labels)
                    loss_dict = self.criterion(output, sprites, self.model.module.quantizer)
                    loss = loss_dict["loss"]
                
                # Backward pass com gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["gradient_clip_val"]
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Logging
                if self.is_main_process and batch_idx % self.config["logging"]["log_interval"] == 0:
                    self._log_training_step(epoch, batch_idx, loss_dict)
                
                total_loss += loss.item()
                progress.update(task, advance=1)
        
        # Sincronizar loss entre processos
        if self.world_size > 1:
            dist.all_reduce(torch.tensor([total_loss]).to(self.device))
            total_loss /= self.world_size
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Valida o modelo."""
        self.model.eval()
        total_loss = 0
        
        for sprites, style_labels in self.val_loader:
            sprites = sprites.to(self.device)
            style_labels = style_labels.to(self.device)
            
            with autocast(dtype=torch.float16):
                output = self.model(sprites, style_labels)
                loss_dict = self.criterion(output, sprites, self.model.module.quantizer)
                loss = loss_dict["loss"]
            
            total_loss += loss.item()
        
        # Sincronizar loss
        if self.world_size > 1:
            dist.all_reduce(torch.tensor([total_loss]).to(self.device))
            total_loss /= self.world_size
        
        return total_loss / len(self.val_loader)
    
    def _log_training_step(self, epoch: int, batch_idx: int, loss_dict: Dict[str, float]):
        """Registra métricas de treinamento."""
        # Calcular LR atual
        current_lr = self.optimizer.param_groups[0]["lr"]
        
        # Logging no W&B
        metrics = {
            "train/loss": loss_dict["loss"],
            "train/pixel_loss": loss_dict["pixel_loss"],
            "train/edge_loss": loss_dict["edge_loss"],
            "train/palette_loss": loss_dict["palette_loss"],
            "train/learning_rate": current_lr,
            "train/epoch": epoch
        }
        
        # Adicionar métricas de recursos
        if self.monitor:
            stats = self.monitor.get_stats_summary()
            metrics.update({
                "system/gpu_memory": stats["gpu_memory"][0]["mean"],
                "system/gpu_utilization": stats["gpu_utilization"][0]["mean"],
                "system/cpu_percent": stats["cpu"]["mean"],
                "system/ram_percent": stats["ram"]["mean"]
            })
        
        wandb.log(metrics)
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Salva checkpoint do modelo."""
        if not self.is_main_process:
            return
            
        self.checkpoint_manager.save(
            model=self.model.module if self.world_size > 1 else self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            metadata={
                "val_loss": val_loss,
                "learning_rate": self.optimizer.param_groups[0]["lr"]
            }
        )
    
    def train(self):
        """Loop principal de treinamento."""
        num_epochs = self.config["training"]["num_epochs"]
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            # Atualizar samplers distribuídos
            if self.world_size > 1:
                self.train_loader.sampler.set_epoch(epoch)
            
            # Treinar
            train_loss = self.train_epoch(epoch)
            
            # Validar
            val_loss = self.validate()
            
            # Atualizar scheduler
            self.scheduler.step()
            
            # Logging e checkpoint
            if self.is_main_process:
                self.console.print(f"\nEpoch {epoch}:")
                self.console.print(f"Train Loss: {train_loss:.4f}")
                self.console.print(f"Val Loss: {val_loss:.4f}")
                
                wandb.log({
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/loss": val_loss
                })
                
                # Salvar melhor modelo
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)
        
        # Limpeza final
        if self.is_main_process:
            self.monitor.stop()
            wandb.finish()
        
        if self.world_size > 1:
            dist.destroy_process_group()

class MultiGPUTrainer(Trainer):
    """Treinador otimizado para múltiplas GPUs."""
    
    def __init__(
        self,
        config_path: str,
        data_dir: str,
        output_dir: str,
        local_rank: int = 0,
        world_size: int = 1,
        gpu_ids: List[int] = None
    ):
        """
        Args:
            config_path: Caminho para arquivo de configuração
            data_dir: Diretório com dados de treinamento
            output_dir: Diretório para salvar checkpoints e logs
            local_rank: Rank local do processo
            world_size: Número total de processos
            gpu_ids: IDs das GPUs a serem usadas
        """
        # Otimizar GPUs
        GPUManager.optimize_gpu_settings()
        
        # Selecionar GPUs
        if gpu_ids is None:
            gpu_ids = GPUManager.select_best_gpus(world_size)
        
        self.gpu_ids = gpu_ids
        super().__init__(config_path, data_dir, output_dir, local_rank, world_size)
    
    def _setup_distributed(self):
        """Configura ambiente distribuído otimizado."""
        if self.world_size > 1:
            # Configurar backend NCCL
            os.environ["NCCL_DEBUG"] = "INFO"
            os.environ["NCCL_SOCKET_IFNAME"] = "^lo,docker"
            
            # Inicializar processo grupo
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=self.world_size,
                rank=self.local_rank
            )
            
            # Configurar GPU local
            torch.cuda.set_device(self.gpu_ids[self.local_rank])
            
            # Sincronizar processos
            dist.barrier()
    
    def _setup_model(self):
        """Configura modelo com otimizações para multi-GPU."""
        # Criar modelo
        self.model = Pixel16Generator(self.config["model"])
        
        # Mover para GPU
        device = torch.device(f"cuda:{self.gpu_ids[self.local_rank]}")
        self.model.to(device)
        
        # Configurar FP8 se disponível
        if TE_AVAILABLE:
            fp8_recipe = recipe.DelayedScaling(
                margin=0,
                interval=1,
                fp8_format=recipe.Format.E4M3
            )
            self.model = te.train.fp8_autocast(
                self.model,
                fp8_recipe=fp8_recipe
            )
        
        # Configurar DDP com otimizações
        if self.world_size > 1:
            # Otimizar comunicação
            ddp_kwargs = {
                "device_ids": [self.gpu_ids[self.local_rank]],
                "output_device": self.gpu_ids[self.local_rank],
                "broadcast_buffers": False,
                "bucket_cap_mb": 25,
                "gradient_as_bucket_view": True,
                "static_graph": True
            }
            
            self.model = DDP(self.model, **ddp_kwargs)
        
        # Função de perda
        self.criterion = PixelArtLoss().to(device)
    
    def _setup_data(self):
        """Configura dataloaders com otimizações para multi-GPU."""
        # Datasets
        self.train_dataset = Pixel16Dataset(
            data_dir=self.data_dir,
            split="train",
            train_ratio=0.9,
            use_cache=True
        )
        
        self.val_dataset = Pixel16Dataset(
            data_dir=self.data_dir,
            split="val",
            train_ratio=0.9,
            use_cache=True
        )
        
        # Samplers distribuídos otimizados
        train_sampler = None
        val_sampler = None
        
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True,
                seed=42
            )
            
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False
            )
        
        # Dataloaders otimizados
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["batch_size"] * 2,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

def train_distributed(
    config_path: str,
    data_dir: str,
    output_dir: str,
    world_size: int,
    gpu_ids: List[int] = None
):
    """Inicia treinamento distribuído otimizado."""
    # Selecionar melhores GPUs
    if gpu_ids is None:
        gpu_ids = GPUManager.select_best_gpus(world_size)
    
    # Configurar variáveis de ambiente
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    # Iniciar processos
    torch.multiprocessing.spawn(
        _train_worker,
        args=(config_path, data_dir, output_dir, world_size, gpu_ids),
        nprocs=world_size,
        join=True
    )

def _train_worker(
    local_rank: int,
    config_path: str,
    data_dir: str,
    output_dir: str,
    world_size: int,
    gpu_ids: List[int]
):
    """Worker para treinamento distribuído."""
    trainer = MultiGPUTrainer(
        config_path=config_path,
        data_dir=data_dir,
        output_dir=output_dir,
        local_rank=local_rank,
        world_size=world_size,
        gpu_ids=gpu_ids
    )
    trainer.train() 