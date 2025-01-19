"""
Distributed training script for the Large model optimized for 2x NVIDIA H100 GPUs.
Enhanced with robust error handling, anti-overfitting, and advanced logging.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import logging
import argparse
from pathlib import Path
import time
from datetime import datetime
import json
import signal
import sys
from typing import Optional, Dict, Any
import GPUtil
import numpy as np
from torch.utils.data import DataLoader
import wandb
from torch.utils.tensorboard import SummaryWriter
import traceback
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from rich.logging import RichHandler
import psutil

from model import LargeModel
from dataset import LargeDataset
from config import config

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    # Enable cuDNN benchmarking and TF32
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def cleanup_distributed():
    """Cleanup distributed training resources."""
    dist.destroy_process_group()

class RobustTrainingManager:
    """Enhanced training manager with robust error handling and monitoring."""
    
    def __init__(self, args: argparse.Namespace, rank: int):
        self.args = args
        self.rank = rank
        self.run_dir = self._setup_run_dir()
        self.best_loss = float('inf')
        self.last_save_time = time.time()
        self.training_start_time = time.time()
        self.early_stopping_counter = 0
        self.best_metrics = {}
        self.setup_logging()
        self.setup_monitoring()
        self.register_handlers()
        
    def setup_logging(self):
        """Setup enhanced logging with rich formatting and file output."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                RichHandler(rich_tracebacks=True),
                logging.FileHandler(self.run_dir / 'training.log')
            ]
        )
        self.logger = logging.getLogger('LargeTraining')
        
    def setup_monitoring(self):
        """Setup monitoring tools including W&B and TensorBoard."""
        if self.rank == 0:
            # Initialize Weights & Biases
            wandb.init(
                project="large-pixel-art",
                config=config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                dir=str(self.run_dir)
            )
            
            # Initialize TensorBoard
            self.writer = SummaryWriter(self.run_dir / 'tensorboard')
            
            # Initialize metrics history
            self.metrics_history = {
                'train_loss': [],
                'val_loss': [],
                'learning_rate': [],
                'gpu_memory': [],
                'gpu_utilization': [],
                'parameter_norm': [],
                'gradient_norm': []
            }
    
    def register_handlers(self):
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle interruption signals gracefully."""
        self.logger.warning(f"Received interrupt signal {signum}. Performing cleanup...")
        self.save_final_state()
        sys.exit(0)
    
    def _setup_run_dir(self) -> Path:
        """Setup directory for this training run."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path('runs') / timestamp
        if self.rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'checkpoints').mkdir(exist_ok=True)
            (run_dir / 'backups').mkdir(exist_ok=True)
        return run_dir
    
    def log_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                   val_metrics: Dict[str, float], lr: float, epoch_time: float):
        """Enhanced metric logging with detailed statistics."""
        if self.rank != 0:
            return
            
        # Get GPU stats
        gpu = GPUtil.getGPUs()[self.rank]
        gpu_memory = gpu.memoryUsed / gpu.memoryTotal
        gpu_util = gpu.load
        
        # Get CPU stats
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        
        # Calculate parameter and gradient norms
        param_norm = self._calculate_parameter_norm()
        grad_norm = self._calculate_gradient_norm()
        
        # Update metrics history
        metrics = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'learning_rate': lr,
            'epoch_time': epoch_time,
            'gpu_memory': gpu_memory,
            'gpu_utilization': gpu_util,
            'cpu_percent': cpu_percent,
            'ram_percent': ram_percent,
            'parameter_norm': param_norm,
            'gradient_norm': grad_norm
        }
        
        # Log to W&B
        wandb.log(metrics)
        
        # Log to TensorBoard
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)
        
        # Update metrics history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        # Save metrics to JSON
        with open(self.run_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Log to console with rich formatting
        self.logger.info(
            f"Epoch {epoch:4d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.2f}s | "
            f"GPU Mem: {gpu_memory*100:.1f}% | "
            f"GPU Util: {gpu_util*100:.1f}%"
        )
        
        # Check for best model
        if val_metrics['loss'] < self.best_loss:
            self.best_loss = val_metrics['loss']
            self.early_stopping_counter = 0
            self.best_metrics = metrics.copy()
        else:
            self.early_stopping_counter += 1
    
    def _calculate_parameter_norm(self) -> float:
        """Calculate total parameter norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.data.norm().item() ** 2
        return total_norm ** 0.5
    
    def _calculate_gradient_norm(self) -> float:
        """Calculate total gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm().item() ** 2
        return total_norm ** 0.5
    
    def should_stop_early(self) -> bool:
        """Check if training should stop early."""
        return (self.early_stopping_counter >= config['training']['early_stopping_patience'] or
                self._check_divergence() or
                self._check_resource_limits())
    
    def _check_divergence(self) -> bool:
        """Check if training is diverging."""
        if len(self.metrics_history['val_loss']) > 10:
            recent_losses = self.metrics_history['val_loss'][-10:]
            if all(x > y for x, y in zip(recent_losses, recent_losses[1:])):
                self.logger.warning("Training is diverging - validation loss increasing consistently")
                return True
        return False
    
    def _check_resource_limits(self) -> bool:
        """Check if we're hitting resource limits."""
        gpu = GPUtil.getGPUs()[self.rank]
        if gpu.memoryUsed / gpu.memoryTotal > 0.95:
            self.logger.warning("GPU memory usage too high (>95%)")
            return True
        if psutil.virtual_memory().percent > 95:
            self.logger.warning("System memory usage too high (>95%)")
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       scaler: GradScaler, epoch: int, val_loss: float,
                       is_best: bool = False):
        """Save enhanced checkpoint with additional metadata."""
        if self.rank != 0:
            return
            
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': val_loss,
            'best_loss': self.best_loss,
            'metrics_history': self.metrics_history,
            'training_duration': time.time() - self.training_start_time,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'gpu_info': {
                'name': torch.cuda.get_device_name(0),
                'memory_allocated': torch.cuda.memory_allocated(0),
                'memory_reserved': torch.cuda.memory_reserved(0)
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = self.run_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = self.run_dir / 'checkpoints' / 'best.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved new best model with val_loss: {val_loss:.4f}")
        
        # Periodic backup
        current_time = time.time()
        if current_time - self.last_save_time > config['training']['backup_freq_hours'] * 3600:
            backup_path = self.run_dir / 'backups' / f'backup_epoch_{epoch}.pt'
            torch.save(checkpoint, backup_path)
            self.last_save_time = current_time
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints keeping only the last N."""
        checkpoint_dir = self.run_dir / 'checkpoints'
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        keep_last = config['training']['keep_last_n_checkpoints']
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
    def save_final_state(self):
        """Save final training state with detailed summary."""
        if self.rank != 0:
            return
            
        final_state = {
            'training_duration': time.time() - self.training_start_time,
            'best_metrics': self.best_metrics,
            'final_metrics': self.metrics_history,
            'early_stopping_counter': self.early_stopping_counter,
            'timestamp': datetime.now().isoformat(),
            'gpu_stats': {
                'memory_allocated': torch.cuda.memory_allocated(0),
                'memory_reserved': torch.cuda.memory_reserved(0),
                'max_memory_allocated': torch.cuda.max_memory_allocated(0)
            }
        }
        
        # Save final state
        with open(self.run_dir / 'final_state.json', 'w') as f:
            json.dump(final_state, f, indent=2)
        
        # Close monitoring tools
        wandb.finish()
        self.writer.close()
        
        self.logger.info("Training completed - saved final state")

def train_epoch(epoch: int, model: nn.Module, train_loader: DataLoader,
               optimizer: optim.Optimizer, criterion: nn.Module,
               scaler: GradScaler, device: torch.device) -> Dict[str, float]:
    """Enhanced training loop with robust error handling and monitoring."""
    model.train()
    total_loss = 0
    batch_count = 0
    
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn()
    )
    
    try:
        with progress:
            task = progress.add_task(f"Epoch {epoch}", total=len(train_loader))
            
            for batch_idx, (images, prompts) in enumerate(train_loader):
                images = images.to(device)
                prompts = prompts.to(device)
                
                # Forward pass with gradient scaling
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(prompts)
                    loss = criterion(outputs, images)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if config['model']['gradient_clipping']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['model']['gradient_clipping']
                    )
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Update metrics
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress
                progress.update(task, advance=1)
                
                # Log batch metrics
                if batch_idx % config['training']['log_every_n_steps'] == 0:
                    logging.debug(
                        f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f}"
                    )
    
    except Exception as e:
        logging.error(f"Error in training loop: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    
    return {
        'loss': total_loss / batch_count if batch_count > 0 else float('inf'),
        'batch_count': batch_count
    }

def validate(model: nn.Module, val_loader: DataLoader,
            criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """Enhanced validation with detailed metrics."""
    model.eval()
    total_loss = 0
    batch_count = 0
    
    try:
        with torch.inference_mode():
            for images, prompts in val_loader:
                images = images.to(device)
                prompts = prompts.to(device)
                
                # Forward pass
                outputs = model(prompts)
                loss = criterion(outputs, images)
                
                # Update metrics
                total_loss += loss.item()
                batch_count += 1
    
    except Exception as e:
        logging.error(f"Error in validation loop: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    
    return {
        'loss': total_loss / batch_count if batch_count > 0 else float('inf'),
        'batch_count': batch_count
    }

def main(rank: int, world_size: int, args: argparse.Namespace):
    """Enhanced main training function with robust error handling."""
    try:
        # Setup distributed training
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        
        # Initialize training manager
        manager = RobustTrainingManager(args, rank)
        
        # Create model and move to device
        model = LargeModel(config['model'])
        model.to(device)
        model = DDP(model, device_ids=[rank])
        
        # Setup datasets and dataloaders
        train_dataset = LargeDataset(
            root_dir=args.data_dir,
            image_size=config['model']['image_size'],
            max_samples=args.max_samples if args.dev else None
        )
        
        val_dataset = LargeDataset(
            root_dir=args.data_dir,
            image_size=config['model']['image_size'],
            max_samples=args.max_samples if args.dev else None,
            use_augmentation=False
        )
        
        # Create distributed samplers
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['model']['batch_size_per_gpu'],
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['model']['batch_size_per_gpu'],
            sampler=val_sampler,
            num_workers=8,
            pin_memory=True
        )
        
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            **config['model']['optimizer']
        )
        scaler = GradScaler()
        
        # Training loop
        try:
            for epoch in range(args.epochs):
                epoch_start = time.time()
                
                # Set epoch for samplers
                train_sampler.set_epoch(epoch)
                val_sampler.set_epoch(epoch)
                
                # Training phase
                train_metrics = train_epoch(
                    epoch, model, train_loader,
                    optimizer, criterion, scaler, device
                )
                
                # Validation phase
                val_metrics = validate(model, val_loader, criterion, device)
                
                # Log metrics and save checkpoints on main process
                if rank == 0:
                    epoch_time = time.time() - epoch_start
                    manager.log_metrics(
                        epoch,
                        train_metrics,
                        val_metrics,
                        optimizer.param_groups[0]['lr'],
                        epoch_time
                    )
                    
                    is_best = val_metrics['loss'] < manager.best_loss
                    manager.save_checkpoint(
                        model, optimizer, scaler,
                        epoch, val_metrics['loss'], is_best
                    )
                
                # Check for early stopping
                if manager.should_stop_early():
                    logging.info("Early stopping triggered")
                    break
                
                # Synchronize processes
                dist.barrier()
        
        except Exception as e:
            logging.error(f"Training interrupted by error: {str(e)}")
            logging.error(traceback.format_exc())
            if rank == 0:
                manager.save_final_state()
            raise
        
        finally:
            if rank == 0:
                manager.save_final_state()
            cleanup_distributed()
    
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Large model with distributed training')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--max-samples', type=int, default=1000)
    args = parser.parse_args()
    
    try:
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(
            main,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        logging.error(f"Failed to start training: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1) 