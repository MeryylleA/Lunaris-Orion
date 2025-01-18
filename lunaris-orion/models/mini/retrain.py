"""
Retraining script for the Mini model optimized for NVIDIA H100 GPU.
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

from model import MiniModel
from dataset import DiffusionDBDataset
from config import config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('retrain.log'),
            logging.StreamHandler()
        ]
    )

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    
    with tqdm(dataloader, desc='Training') as pbar:
        for batch_idx, (prompts, images) in enumerate(pbar):
            prompts, images = prompts.to(device, non_blocking=True), images.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast():
                outputs = model(prompts)
                loss = criterion(outputs, images)
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad(), autocast():
        with tqdm(dataloader, desc='Validation') as pbar:
            for batch_idx, (prompts, images) in enumerate(pbar):
                prompts, images = prompts.to(device, non_blocking=True), images.to(device, non_blocking=True)
                outputs = model(prompts)
                loss = criterion(outputs, images)
                
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / len(dataloader)

def save_checkpoint(model, optimizer, scaler, epoch, loss, is_best=False):
    """Save a checkpoint."""
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'config': config
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(state, checkpoint_path)
    logging.info(f'Saved checkpoint to {checkpoint_path}')
    
    # Save best model if needed
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(state, best_path)
        logging.info(f'Saved best model to {best_path}')

def main():
    parser = argparse.ArgumentParser(description='Retrain Mini model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--dev', action='store_true', help='Run in dev mode')
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    logging.info(f'Using device: {device} - CUDA {torch.version.cuda}')
    
    # Create model with compile
    model = MiniModel(config['model'])
    model = torch.compile(model)  # Using torch.compile for PyTorch 2.0+ optimizations
    model = model.to(device)
    logging.info('Model created and compiled')
    
    # Create datasets
    train_dataset = DiffusionDBDataset(
        root_dir='data',
        image_size=config['model']['image_size'],
        dev_mode=args.dev
    )
    
    val_dataset = DiffusionDBDataset(
        root_dir='data',
        image_size=config['model']['image_size'],
        dev_mode=args.dev
    )
    
    # Create dataloaders optimized for H100
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=14,  # Half of CPU cores
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=14,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    # Setup training with mixed precision
    scaler = GradScaler()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(args.epochs):
        logging.info(f'Starting epoch {epoch + 1}/{args.epochs}')
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        logging.info(f'Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.6f}')
        
        # Save checkpoint
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        
        save_checkpoint(model, optimizer, scaler, epoch, val_loss, is_best)
    
    logging.info('Training completed')

if __name__ == '__main__':
    main() 