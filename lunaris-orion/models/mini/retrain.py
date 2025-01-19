"""
Retraining script for the Mini model optimized for NVIDIA H100 GPU.
Includes optimizations for high memory (350GB RAM) and multi-core CPU (26 vCPUs).
Enhanced with robust features for long-duration training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os
import GPUtil
from torch.utils.data.distributed import DistributedSampler
import time
import json
import shutil
from datetime import datetime
import signal
import sys
from collections import deque
import numpy as np
import torch.nn.functional as F

from model import MiniModel
from dataset import LocalPixelArtDataset, get_train_transforms, get_val_transforms
from config import config

def setup_logging():
    """Set up logging configuration."""
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with colors"""
        grey = "\x1b[38;21m"
        blue = "\x1b[38;5;39m"
        yellow = "\x1b[38;5;226m"
        red = "\x1b[38;5;196m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        def __init__(self, fmt):
            super().__init__()
            self.fmt = fmt
            self.FORMATS = {
                logging.DEBUG: self.grey + self.fmt + self.reset,
                logging.INFO: self.blue + self.fmt + self.reset,
                logging.WARNING: self.yellow + self.fmt + self.reset,
                logging.ERROR: self.red + self.fmt + self.reset,
                logging.CRITICAL: self.bold_red + self.fmt + self.reset
            }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('retrain.log')

    # Create formatters
    console_fmt = '%(asctime)s | %(levelname)-8s | %(message)s'
    file_fmt = '%(asctime)s | %(levelname)-8s | %(message)s'

    # Set formatters
    console_handler.setFormatter(ColoredFormatter(console_fmt))
    file_handler.setFormatter(logging.Formatter(file_fmt))

    # Get logger and set level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def compute_loss(outputs, targets, palette_weights=None):
    """Compute combined loss with pixel art specific terms."""
    # MSE e L1 loss básicos
    mse_loss = F.mse_loss(outputs, targets)
    l1_loss = F.l1_loss(outputs, targets)
    
    # Perda de estrutura de pixels
    pixel_loss = F.mse_loss(
        outputs[:, :, ::2, ::2],
        outputs[:, :, 1::2, 1::2]
    )
    
    # Perda de paleta de cores (se disponível)
    palette_loss = 0
    if palette_weights is not None:
        # Encoraja o uso distribuído da paleta
        entropy = -(palette_weights * torch.log(palette_weights + 1e-10)).sum(-1).mean()
        palette_loss = -0.1 * entropy  # Maximizar entropia
    
    # Perda de bordas para preservar detalhes de pixel art
    edge_loss = F.l1_loss(
        outputs[:, :, 1:] - outputs[:, :, :-1],
        targets[:, :, 1:] - targets[:, :, :-1]
    ) + F.l1_loss(
        outputs[:, :, :, 1:] - outputs[:, :, :, :-1],
        targets[:, :, :, 1:] - targets[:, :, :, :-1]
    )
    
    # Combinar todas as perdas
    total_loss = mse_loss + 0.1 * l1_loss + 0.05 * pixel_loss + 0.1 * edge_loss + palette_loss
    
    return total_loss, {
        'mse': mse_loss.item(),
        'l1': l1_loss.item(),
        'pixel': pixel_loss.item(),
        'edge': edge_loss.item(),
        'palette': palette_loss if isinstance(palette_loss, float) else palette_loss.item()
    }

def train_epoch(model, dataloader, optimizer, device, scaler, epoch, scheduler=None):
    """Train for one epoch with improved error handling and monitoring."""
    model.train()
    total_loss = 0
    metrics = {'mse': 0, 'l1': 0, 'pixel': 0, 'edge': 0, 'palette': 0}
    
    # Enable tensor cores and autocast
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Monitor memory usage
    initial_memory = torch.cuda.memory_allocated()
    peak_memory = 0
    
    # Create progress bar
    pbar = tqdm(
        dataloader,
        desc=f'Epoch {epoch:3d}',
        bar_format='{l_bar}{bar:30}{r_bar}',
        dynamic_ncols=True
    )
    
    # Gradient accumulation setup
    accum_steps = 4
    optimizer.zero_grad()
    
    # Error tracking
    error_count = 0
    max_errors = 5
    
    try:
        for batch_idx, (prompts, images) in enumerate(pbar):
            try:
                # Move data to device with error handling
                prompts = prompts.to(device, non_blocking=True)
                images = images.to(device, non_blocking=True)
                
                # Check for invalid values
                if torch.isnan(prompts).any() or torch.isnan(images).any():
                    raise ValueError("NaN values detected in input data")
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda'):
                    outputs = model(prompts)
                    if hasattr(model, 'palette_attention'):
                        palette_weights = F.softmax(model.palette_attention(outputs), dim=-1)
                    else:
                        palette_weights = None
                    
                    loss, batch_metrics = compute_loss(outputs, images, palette_weights)
                    loss = loss / accum_steps
                
                # Backward pass with gradient accumulation
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accum_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Optimizer step with error handling
                    try:
                        scaler.step(optimizer)
                        scaler.update()
                    except RuntimeError as e:
                        logging.warning(f"Optimizer step failed: {str(e)}")
                        scaler.update()
                    
                    optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
                
                # Update metrics
                total_loss += loss.item() * accum_steps
                for k, v in batch_metrics.items():
                    metrics[k] += v
                
                # Monitor memory
                current_memory = torch.cuda.memory_allocated()
                peak_memory = max(peak_memory, current_memory)
                
                # Update progress bar with detailed metrics
                avg_loss = total_loss / (batch_idx + 1)
                gpu_info = GPUtil.getGPUs()[0]
                
                metrics_str = f'loss={avg_loss:.4f}'
                for k, v in metrics.items():
                    metrics_str += f' {k}={v/(batch_idx+1):.4f}'
                
                memory_used = current_memory / 1024**3
                memory_peak = peak_memory / 1024**3
                
                pbar.set_postfix_str(
                    f'{metrics_str} | '
                    f'mem={memory_used:.1f}GB '
                    f'peak={memory_peak:.1f}GB '
                    f'util={gpu_info.load*100:.0f}%'
                )
                
            except Exception as e:
                error_count += 1
                logging.error(f"Error in batch {batch_idx}: {str(e)}")
                if error_count >= max_errors:
                    raise RuntimeError(f"Too many errors ({error_count}), stopping training")
                continue
        
        # Memory cleanup
        torch.cuda.empty_cache()
        
        # Normalize metrics
        metrics = {k: v/len(dataloader) for k, v in metrics.items()}
        
        # Log memory usage
        memory_stats = {
            'initial_memory': initial_memory / 1024**3,
            'peak_memory': peak_memory / 1024**3,
            'final_memory': torch.cuda.memory_allocated() / 1024**3
        }
        logging.info(f"Memory stats (GB): {memory_stats}")
        
        return total_loss / len(dataloader), metrics
        
    except Exception as e:
        logging.error(f"Fatal error in training epoch: {str(e)}")
        raise

def validate(model, dataloader, device):
    """Validate the model with improved error handling."""
    model.eval()
    total_loss = 0
    metrics = {'mse': 0, 'l1': 0, 'pixel': 0, 'edge': 0, 'palette': 0}
    
    pbar = tqdm(
        dataloader,
        desc='Validation',
        bar_format='{l_bar}{bar:30}{r_bar}',
        dynamic_ncols=True
    )
    
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for batch_idx, (prompts, images) in enumerate(pbar):
                    try:
                        prompts = prompts.to(device, non_blocking=True)
                        images = images.to(device, non_blocking=True)
                        
                        outputs = model(prompts)
                        loss, batch_metrics = compute_loss(outputs, images)
                        
                        total_loss += loss.item()
                        for k, v in batch_metrics.items():
                            metrics[k] += v
                        
                        avg_loss = total_loss / (batch_idx + 1)
                        metrics_str = f'loss={avg_loss:.4f}'
                        for k, v in metrics.items():
                            metrics_str += f' {k}={v/(batch_idx+1):.4f}'
                        
                        pbar.set_postfix_str(metrics_str)
                        
                    except Exception as e:
                        logging.error(f"Error in validation batch {batch_idx}: {str(e)}")
                        continue
        
        metrics = {k: v/len(dataloader) for k, v in metrics.items()}
        return total_loss / len(dataloader), metrics
        
    except Exception as e:
        logging.error(f"Fatal error in validation: {str(e)}")
        raise

class TrainingManager:
    """Manages the training process with advanced features."""
    
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        self.last_backup_time = time.time()
        self.moving_avg_window = deque(maxlen=50)
        self.setup_paths()
        self.setup_signal_handlers()
        
        # Novas features
        self.loss_trend = deque(maxlen=100)  # Para early stopping adaptativo
        self.lr_history = []  # Para tracking do learning rate
        self.setup_monitoring()
    
    def setup_paths(self):
        """Setup directory structure for training artifacts."""
        self.run_dir = Path(f'runs/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.backup_dir = self.run_dir / 'backups'
        self.log_dir = self.run_dir / 'logs'
        
        for dir_path in [self.run_dir, self.checkpoint_dir, self.backup_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_interrupt)
    
    def handle_interrupt(self, signum, frame):
        """Handle interruption signals gracefully."""
        logging.warning("Received interrupt signal. Performing graceful shutdown...")
        self.save_final_state()
        sys.exit(0)
    
    def save_final_state(self):
        """Save final state before shutdown."""
        if hasattr(self, 'model'):
            self.save_checkpoint(
                self.model, self.optimizer, self.scaler, 
                self.current_epoch, self.current_val_loss,
                is_final=True
            )
    
    def setup_monitoring(self):
        """Setup monitoring and logging features."""
        self.metrics_history = {
            'train_loss': [], 'val_loss': [],
            'mse': [], 'l1': [], 'pixel': [], 'edge': [], 'palette': [],
            'learning_rate': [], 'gpu_memory': [], 'gpu_util': []
        }
        
        # Criar arquivo de log detalhado
        self.metrics_log = self.log_dir / 'detailed_metrics.jsonl'
    
    def should_stop_early(self):
        """Early stopping adaptativo baseado na tendência da loss."""
        if len(self.loss_trend) < 50:
            return False
            
        # Calcular tendência usando regressão linear
        x = np.arange(len(self.loss_trend))
        y = np.array(self.loss_trend)
        slope = np.polyfit(x, y, 1)[0]
        
        # Se a tendência for positiva por muito tempo, parar
        if slope > 0 and self.patience_counter > 10:
            return True
            
        return False
    
    def update_metrics(self, epoch, train_metrics, val_metrics, lr):
        """Update and log detailed metrics."""
        metrics = {
            'epoch': epoch,
            'timestamp': time.time(),
            'train': train_metrics,
            'val': val_metrics,
            'learning_rate': lr,
            'gpu_info': {
                'memory': GPUtil.getGPUs()[0].memoryUsed,
                'utilization': GPUtil.getGPUs()[0].load
            }
        }
        
        # Append to log file
        with open(self.metrics_log, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        # Update history
        for k, v in train_metrics.items():
            if k in self.metrics_history:
                self.metrics_history[k].append(v)
        
        self.metrics_history['learning_rate'].append(lr)
        self.metrics_history['gpu_memory'].append(metrics['gpu_info']['memory'])
        self.metrics_history['gpu_util'].append(metrics['gpu_info']['utilization'])
    
    def save_checkpoint(self, model, optimizer, scaler, epoch, val_loss, train_metrics=None, is_best=False, is_final=False):
        """Enhanced checkpoint saving with detailed metrics."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': val_loss,
            'timestamp': time.time(),
            'training_duration': time.time() - self.start_time,
            'training_history': self.training_history,
            'metrics_history': self.metrics_history,
            'args': vars(self.args),
            'train_metrics': train_metrics,
            'gpu_info': {
                'name': torch.cuda.get_device_name(0),
                'memory_allocated': torch.cuda.memory_allocated(0),
                'memory_reserved': torch.cuda.memory_reserved(0)
            }
        }
        
        # Save checkpoint
        suffix = '_final' if is_final else f'_epoch_{epoch}'
        checkpoint_path = self.checkpoint_dir / f'checkpoint{suffix}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logging.info(f'New best model saved! (val_loss: {val_loss:.4f})')
            
            # Save best metrics separately
            best_metrics = {
                'epoch': epoch,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'timestamp': time.time()
            }
            with open(self.log_dir / 'best_metrics.json', 'w') as f:
                json.dump(best_metrics, f, indent=2)
        
        # Backup handling
        if time.time() - self.last_backup_time > self.args.backup_interval:
            backup_path = self.backup_dir / f'backup_epoch_{epoch}.pt'
            shutil.copy(checkpoint_path, backup_path)
            self.last_backup_time = time.time()
            self.clean_old_backups()
            
    def clean_old_backups(self, keep_last_n=5):
        """Clean old backup files, keeping only the last N."""
        backup_files = sorted(self.backup_dir.glob('backup_epoch_*.pt'))
        if len(backup_files) > keep_last_n:
            for file in backup_files[:-keep_last_n]:
                file.unlink()

def main():
    try:
        parser = argparse.ArgumentParser(description='Train the Mini model')
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=1000)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--backup-interval', type=int, default=3600)
        parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
        parser.add_argument('--dataset-dir', type=str, default='dataset/16x16', help='Path to dataset directory')
        args = parser.parse_args()
        
        # Setup logging and manager
        setup_logging()
        manager = TrainingManager(args)
        
        # Setup device and verify CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training")
        
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        
        # Log system info
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Create model with error handling
        try:
            model = MiniModel(config['model']).to(device)
            model = torch.compile(model)  # Use torch.compile for better performance
            logging.info("Model created and compiled successfully")
        except Exception as e:
            logging.error(f"Error creating model: {str(e)}")
            raise
        
        # Create datasets with error handling
        try:
            train_dataset = LocalPixelArtDataset(
                root_dir=args.dataset_dir,
                split='train',
                transform=get_train_transforms()
            )
            
            val_dataset = LocalPixelArtDataset(
                root_dir=args.dataset_dir,
                split='val',
                transform=get_val_transforms()
            )
        except Exception as e:
            logging.error(f"Error creating datasets: {str(e)}")
            raise
        
        # Setup data loaders
        num_workers = min(8, os.cpu_count() or 1)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        logging.info(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples")
        
        # Setup optimizer and scheduler
        try:
            from lion_pytorch import Lion
            optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=0.01)
            logging.info("Using Lion optimizer")
        except ImportError:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            logging.info("Using AdamW optimizer (Lion not available)")
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1000,
            num_training_steps=args.epochs * len(train_loader)
        )
        
        scaler = GradScaler()
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            try:
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                logging.info(f'Resumed from epoch {start_epoch}')
            except Exception as e:
                logging.error(f"Error loading checkpoint: {str(e)}")
                raise
        
        # Training loop with error handling
        try:
            for epoch in range(start_epoch, args.epochs):
                # Training phase
                train_loss, train_metrics = train_epoch(
                    model, train_loader, optimizer, device, scaler, epoch, scheduler
                )
                
                # Validation phase
                val_loss, val_metrics = validate(model, val_loader, device)
                
                # Update metrics and check for early stopping
                manager.update_metrics(epoch, train_metrics, val_metrics, scheduler.get_last_lr()[0])
                
                is_best = val_loss < manager.best_loss
                manager.best_loss = min(val_loss, manager.best_loss)
                
                # Save checkpoint
                manager.save_checkpoint(
                    model, optimizer, scaler, epoch, val_loss,
                    train_metrics=train_metrics,
                    is_best=is_best
                )
                
                # Check for early stopping
                if manager.should_stop_early():
                    logging.info("Early stopping triggered")
                    break
                    
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            manager.save_final_state()
            raise
            
        finally:
            manager.save_final_state()
            logging.info("Training completed")
            
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main() 