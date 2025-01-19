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

from model import MiniModel
from dataset import DiffusionDBDataset
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

def train_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch):
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0
    
    # Enable tensor cores for mixed precision training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create progress bar with better formatting
    pbar = tqdm(
        dataloader,
        desc=f'Epoch {epoch:3d}',
        bar_format='{l_bar}{bar:30}{r_bar}',
        dynamic_ncols=True
    )
    
    for batch_idx, (prompts, images) in enumerate(pbar):
        prompts = prompts.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Use mixed precision training with updated syntax
        with torch.amp.autocast('cuda'):
            outputs = model(prompts)
            mse_loss = criterion(outputs, images)
            l1_loss = torch.abs(outputs - images).mean()
            loss = mse_loss + 0.1 * l1_loss
        
        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        
        # Unscale before gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar with cleaner metrics
        gpu_info = GPUtil.getGPUs()[0]
        pbar.set_postfix(
            loss=f'{avg_loss:.4f}',
            mem=f'{gpu_info.memoryUsed/1024:.1f}GB',
            util=f'{gpu_info.load*100:.0f}%'
        )
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate the model with mixed precision."""
    model.eval()
    total_loss = 0
    
    pbar = tqdm(
        dataloader,
        desc='Validation',
        bar_format='{l_bar}{bar:30}{r_bar}',
        dynamic_ncols=True
    )
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            for batch_idx, (prompts, images) in enumerate(pbar):
                prompts = prompts.to(device, non_blocking=True)
                images = images.to(device, non_blocking=True)
                
                outputs = model(prompts)
                mse_loss = criterion(outputs, images)
                l1_loss = torch.abs(outputs - images).mean()
                loss = mse_loss + 0.1 * l1_loss
                
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix(loss=f'{avg_loss:.4f}')
    
    return total_loss / len(dataloader)

class TrainingManager:
    """Manages the training process with advanced features."""
    
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        self.last_backup_time = time.time()
        self.moving_avg_window = deque(maxlen=50)  # For loss smoothing
        self.setup_paths()
        self.setup_signal_handlers()
        
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
    
    def save_checkpoint(self, model, optimizer, scaler, epoch, val_loss, is_best=False, is_final=False):
        """Enhanced checkpoint saving with metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': val_loss,
            'timestamp': time.time(),
            'training_duration': time.time() - self.start_time,
            'training_history': self.training_history,
            'args': vars(self.args),
            'gpu_info': {
                'name': torch.cuda.get_device_name(0),
                'memory_allocated': torch.cuda.memory_allocated(0),
                'memory_reserved': torch.cuda.memory_reserved(0)
            }
        }
        
        # Save regular checkpoint
        suffix = '_final' if is_final else f'_epoch_{epoch}'
        checkpoint_path = self.checkpoint_dir / f'checkpoint{suffix}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if applicable
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logging.info(f'New best model saved! (val_loss: {val_loss:.4f})')
        
        # Create periodic backup
        if time.time() - self.last_backup_time > self.args.backup_interval:
            backup_path = self.backup_dir / f'backup_epoch_{epoch}.pt'
            shutil.copy(checkpoint_path, backup_path)
            self.last_backup_time = time.time()
            
            # Clean old backups if needed
            self.clean_old_backups()
    
    def clean_old_backups(self, keep_last_n=5):
        """Clean old backup files, keeping only the last N."""
        backup_files = sorted(self.backup_dir.glob('backup_epoch_*.pt'))
        if len(backup_files) > keep_last_n:
            for file in backup_files[:-keep_last_n]:
                file.unlink()
    
    def should_stop_early(self, val_loss):
        """Check if training should stop early."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        
        self.patience_counter += 1
        if self.patience_counter >= self.args.patience:
            logging.info(f'Early stopping triggered after {self.patience_counter} epochs without improvement')
            return True
        return False
    
    def update_history(self, metrics):
        """Update and save training history."""
        # Convert any tensor values to Python native types
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = value.item()  # Convert tensor to Python number
            else:
                processed_metrics[key] = value
        
        # Append processed metrics to history
        self.training_history.append(processed_metrics)
        
        # Save updated history
        history_file = self.log_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def log_epoch_metrics(self, train_loss, val_loss, lr):
        """Log epoch metrics with enhanced formatting."""
        self.moving_avg_window.append(val_loss)
        moving_avg = sum(self.moving_avg_window) / len(self.moving_avg_window)
        
        # Calculate timings
        total_duration = time.time() - self.start_time
        epoch_duration = time.time() - getattr(self, 'last_epoch_time', self.start_time)
        self.last_epoch_time = time.time()
        
        # Calculate ETA
        epochs_remaining = self.args.epochs - (self.current_epoch + 1)
        eta_seconds = epoch_duration * epochs_remaining
        eta_hours = eta_seconds / 3600
        
        # Store raw numeric values for history
        raw_metrics = {
            'epoch': self.current_epoch + 1,
            'total_epochs': self.args.epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'moving_avg': moving_avg,
            'learning_rate': lr,
            'duration_hours': total_duration / 3600,
            'epoch_duration_min': epoch_duration / 60,
            'eta_hours': eta_hours
        }
        
        # Format metrics for display
        display_metrics = {
            'Epoch': f'{raw_metrics["epoch"]}/{raw_metrics["total_epochs"]}',
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Moving Avg': f'{moving_avg:.4f}',
            'LR': f'{lr:.6f}',
            'Time': f'{raw_metrics["duration_hours"]:.1f}h ({raw_metrics["epoch_duration_min"]:.1f}m/epoch, ETA: {eta_hours:.1f}h)'
        }
        
        # Get and store GPU metrics
        gpu_info = GPUtil.getGPUs()[0]
        raw_metrics.update({
            'gpu_utilization': gpu_info.load * 100,
            'gpu_memory_used': gpu_info.memoryUsed / 1024,
            'gpu_memory_total': gpu_info.memoryTotal / 1024,
            'gpu_temperature': gpu_info.temperature
        })
        
        # Format GPU metrics for display
        gpu_display = {
            'GPU Util': f'{raw_metrics["gpu_utilization"]:.1f}%',
            'GPU Mem': f'{raw_metrics["gpu_memory_used"]:.1f}/{raw_metrics["gpu_memory_total"]:.1f}GB',
            'GPU Temp': f'{raw_metrics["gpu_temperature"]}°C'
        }
        
        # Combine all display metrics
        all_display_metrics = {**display_metrics, **gpu_display}
        
        # Log the formatted metrics
        logging.info('=' * 40)
        logging.info(' | '.join(f'{k}: {v}' for k, v in all_display_metrics.items()))
        
        # Update history with raw numeric values
        self.update_history(raw_metrics)

def main():
    parser = argparse.ArgumentParser(description='Retrain Mini model with robust features')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--dev', action='store_true', help='Run in dev mode')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--backup-interval', type=int, default=3600, help='Backup interval in seconds')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint file')
    args = parser.parse_args()
    
    # Initialize training manager
    manager = TrainingManager(args)
    
    # Setup logging and device
    setup_logging()
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    logging.info(f'Using device: {device} - CUDA {torch.version.cuda}')
    logging.info(f'Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    # Create model with compile and Flash Attention
    model = MiniModel(config['model'])
    model.enable_flash_attention()  # Enable Flash Attention if available
    model = torch.compile(model, mode="reduce-overhead")
    model = model.to(device)
    logging.info('Model created and compiled with Flash Attention')
    
    # Create datasets and dataloaders
    train_dataset = DiffusionDBDataset(
        root_dir=args.data_dir,
        image_size=config['model']['image_size'],
        dev_mode=args.dev,
        dev_samples=1000 if args.dev else None
    )
    
    # Calculate optimal number of workers based on CPU cores
    num_workers = min(26, os.cpu_count() or 1)  # Cap at 26 vCPUs
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_dataset = DiffusionDBDataset(
        root_dir=args.data_dir,
        image_size=config['model']['image_size'],
        dev_mode=args.dev,
        dev_samples=100 if args.dev else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Setup training with mixed precision
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        # Convert to tensor for cosine calculation
        progress = torch.tensor((epoch - warmup_epochs) / (args.epochs - warmup_epochs))
        return 0.5 * (1 + torch.cos(torch.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler)
        logging.info(f'Resumed from checkpoint at epoch {start_epoch}')
    
    # Training loop
    try:
        for epoch in range(start_epoch, args.epochs):
            manager.current_epoch = epoch
            
            # Training phase
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch)
            
            # Validation phase
            val_loss = validate(model, val_loader, criterion, device)
            manager.current_val_loss = val_loss
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Log metrics
            manager.log_epoch_metrics(train_loss, val_loss, current_lr)
            
            # Save checkpoint
            is_best = val_loss < manager.best_loss
            manager.save_checkpoint(model, optimizer, scaler, epoch, val_loss, is_best)
            
            # Check for early stopping
            if manager.should_stop_early(val_loss):
                logging.info('Early stopping triggered')
                break
                
    except Exception as e:
        logging.error(f'Training interrupted by error: {str(e)}')
        manager.save_final_state()
        raise
    
    finally:
        manager.save_final_state()
        logging.info('Training completed successfully')

if __name__ == '__main__':
    main() 