#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_hybrid.py

Hybrid training script for the Lunaris project. This script trains both the 
generator (LunarCoreVAE) and the evaluator/teacher (LunarMoETeacher) models using
a labeled dataset of 128×128 pixel art. It automatically creates output directories,
logs training progress, cleans up old checkpoints and CUDA memory, and saves metrics 
and sample images.
"""

import os
# Set PYTORCH_CUDA_ALLOC_CONF to use expandable_segments to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import logging
import argparse
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import signal
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.amp.autocast_mode import autocast
from contextlib import nullcontext
from PIL import Image, ImageDraw, ImageFont
import time
import torch.multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Set multiprocessing start method to 'spawn'
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

# Import your models (assumed to be implemented in the respective files)
from lunar_generate import LunarisCoreVAE
from lunar_evaluator import LunarMoETeacher

# --------------------------
# Logging Setup
# --------------------------
def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging with detailed formatting and file logging."""
    log_file = output_dir / 'training.log'
    logger = logging.getLogger("TrainHybrid")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

# --------------------------
# Dataset Class
# --------------------------
class PixelArtDataset(Dataset):
    """
    Dataset for pixel art using multiple sprites.npy and labels.csv files.
    
    Expects a data directory containing:
      - Multiple sprites_*.npy files: NumPy arrays of shape (N, 128, 128, 3)
      - Multiple labels_*.csv files: CSV files with columns: filename, category, prompt, seed, pixel_size, guidance_scale, pag_scale, num_steps
    """
    def __init__(self, data_dir: str, teacher_model=None):
        self.data_dir = Path(data_dir)
        self.sprites_files = sorted(self.data_dir.glob("sprites*.npy"))
        self.labels_files = sorted(self.data_dir.glob("labels*.csv"))
        self.teacher_model = teacher_model
        
        if not self.sprites_files or not self.labels_files:
            raise ValueError(f"No sprites or labels files found in {data_dir}")
        
        logging.info("Loading sprite data...")
        # Memory-mapped loading of sprite data
        self.sprites = []
        total_size = 0
        for sprite_file in self.sprites_files:
            # Use memory mapping for large arrays
            sprite_data = np.load(sprite_file, mmap_mode='r')
            if sprite_data.shape[1:] != (128, 128, 3):
                raise ValueError(f"Expected 128x128x3 images in {sprite_file}, got {sprite_data.shape[1:]}")
            self.sprites.append(sprite_data)
            total_size += len(sprite_data)
            logging.info(f"Loaded {sprite_file.name} with {len(sprite_data)} images")
        
        # Keep track of cumulative sizes for indexing
        self.cumulative_sizes = np.cumsum([0] + [len(x) for x in self.sprites])
        logging.info(f"Total sprites available: {total_size}")
        
        logging.info("Loading label data...")
        # Load labels in chunks to save memory
        self.labels_df = []
        chunk_size = 10000  # Adjust based on available RAM
        for labels_file in self.labels_files:
            for chunk in pd.read_csv(labels_file, chunksize=chunk_size):
                self.labels_df.append(chunk)
                logging.info(f"Loaded chunk from {labels_file.name} with {len(chunk)} entries")
        self.labels_df = pd.concat(self.labels_df, ignore_index=True)
        
        # Verify data integrity
        total_sprites = sum(len(x) for x in self.sprites)
        assert len(self.labels_df) == total_sprites, \
            f"Mismatch between total sprites ({total_sprites}) and labels ({len(self.labels_df)})"
        
        logging.info("Dataset initialization completed")
    
    def _get_sprite_index(self, idx):
        """Get the correct sprite array and local index for the given global index"""
        file_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        local_idx = idx - self.cumulative_sizes[file_idx]
        return file_idx, local_idx
    
    def _get_prompt_embedding(self, image_tensor):
        """Generate prompt embedding on-the-fly for a single image"""
        if self.teacher_model is None:
            return None
        
        # Move operations to CPU first, then to GPU only in the main process
        with torch.no_grad():
            self.teacher_model.eval()
            # Keep tensor on CPU
            image_batch = image_tensor.unsqueeze(0)
            # Get embedding (teacher model should handle device placement)
            teacher_out = self.teacher_model(image_batch)
            embedding = teacher_out['prompt_embedding']
            self.teacher_model.train()
            return embedding[0]  # Remove batch dimension
    
    def __len__(self):
        return sum(len(x) for x in self.sprites)
    
    def __getitem__(self, idx):
        # Get correct sprite array and local index
        file_idx, local_idx = self._get_sprite_index(idx)
        
        # Load and normalize image
        image = self.sprites[file_idx][local_idx].astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Get corresponding row from the labels
        row = self.labels_df.iloc[idx]
        metadata = {
            'filename': row['filename'],
            'category': row['category'],
            'prompt': row['prompt'],
            'seed': row['seed'],
            'pixel_size': row['pixel_size'],
            'guidance_scale': row['guidance_scale'],
            'pag_scale': row['pag_scale'],
            'num_steps': row['num_steps']
        }
        
        # Generate prompt embedding on-the-fly if teacher model is available
        if self.teacher_model is not None:
            metadata['prompt_embedding'] = self._get_prompt_embedding(image)
        
        return {'image': image, 'metadata': metadata}

# --------------------------
# Early Stopping
# --------------------------
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if self.best_loss is None:
            self.best_loss = validation_loss
        elif validation_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = validation_loss
            self.counter = 0

# --------------------------
# Training Manager
# --------------------------
class TrainingManager:
    """
    Manages the hybrid training process for the generator (LunarCoreVAE) and teacher (LunarMoETeacher) models.
    
    This class handles:
      - Creating output directories (checkpoints, tensorboard logs, sample images)
      - Loading the dataset from sprites.npy and labels.csv
      - Initializing models, optimizers, and learning rate schedulers
      - Running the training loop with mixed precision and error handling
      - Saving checkpoints and logging metrics to TensorBoard and the console
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amp_dtype = torch.float16 if args.mixed_precision else torch.float32
        self.use_amp = args.mixed_precision
        
        # Setup directories and logging
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.eval_samples_dir = self.output_dir / 'eval_samples'
        self.eval_samples_dir.mkdir(exist_ok=True)
        self.logger = setup_logging(self.output_dir)
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'tensorboard'))
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(patience=args.early_stopping_patience)
        
        # Initialize models, optimizers, and data
        self._setup_models()
        self._setup_optimizers()
        self._setup_data()
        
        # Load checkpoint if specified
        if args.resume_from:
            if not self._load_checkpoint(args.resume_from):
                self.logger.warning("Training will start from scratch due to checkpoint loading failure")
        
        # Initialize RL components
        self.baseline = None  # Moving average of rewards
        self.reward_scale = args.reward_scale
        self.semantic_weight = args.semantic_weight
        self.baseline_momentum = args.baseline_momentum
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Training state
        if not hasattr(self, 'global_step'):  # Only set if not loaded from checkpoint
            self.global_step = 0
        if not hasattr(self, 'best_loss'):  # Only set if not loaded from checkpoint
            self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self._last_eval_save = 0
        
        # Enable cuDNN autotuner
        cudnn.benchmark = True
        
        # Setup signal handling
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            self.logger.error("CUDA is required for training")
            sys.exit(1)
    
    def _setup_models(self):
        """Initialize the generator and teacher models."""
        self.logger.info("Initializing models...")
        try:
            # Clear CUDA cache before model initialization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
                self.logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Initialize VAE
            self.logger.info("Initializing VAE...")
            self.vae = LunarisCoreVAE(
                latent_dim=self.args.latent_dim
            ).to(self.device)
            
            # Initialize Teacher
            self.logger.info("Initializing Teacher model...")
            self.teacher = LunarMoETeacher(
                num_experts=self.args.num_experts,
                feature_dim=self.args.feature_dim,
                embedding_dim=self.args.embedding_dim
            ).to(self.device)
            
            # Configure gradient checkpointing after initialization
            if hasattr(self.vae, 'enable_gradient_checkpointing'):
                self.vae.enable_gradient_checkpointing(use_reentrant=False)
            if hasattr(self.teacher, 'enable_gradient_checkpointing'):
                self.teacher.enable_gradient_checkpointing(use_reentrant=False)
            
            # Print model sizes
            vae_params = sum(p.numel() for p in self.vae.parameters())
            teacher_params = sum(p.numel() for p in self.teacher.parameters())
            self.logger.info(f"VAE parameters: {vae_params:,}")
            self.logger.info(f"Teacher parameters: {teacher_params:,}")
            
            # Ensure models are in training mode and gradients are enabled
            self.vae.train()
            self.teacher.train()
            
            # Explicitly enable gradients for all parameters
            for param in self.vae.parameters():
                param.requires_grad = True
            for param in self.teacher.parameters():
                param.requires_grad = True
            
            # Verify gradient settings
            vae_grad_params = sum(1 for p in self.vae.parameters() if p.requires_grad)
            teacher_grad_params = sum(1 for p in self.teacher.parameters() if p.requires_grad)
            self.logger.info(f"VAE trainable parameters: {vae_grad_params:,}")
            self.logger.info(f"Teacher trainable parameters: {teacher_grad_params:,}")
            
            if self.args.compile:
                self.logger.info("Compiling models with torch.compile()...")
                try:
                    # Use more conservative compilation settings
                    self.vae = torch.compile(
                        self.vae,
                        mode="reduce-overhead",
                        fullgraph=False,
                        dynamic=True
                    )
                    self.teacher = torch.compile(
                        self.teacher,
                        mode="reduce-overhead",
                        fullgraph=False,
                        dynamic=True
                    )
                    self.logger.info("Models compiled successfully")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {str(e)}. Continuing without compilation.")
                    self.args.compile = False
            
            # Log memory usage after model initialization
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                self.logger.info(f"GPU memory after model init: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
            
            # Verify models are on correct device
            self.logger.info(f"VAE device: {next(self.vae.parameters()).device}")
            self.logger.info(f"Teacher device: {next(self.teacher.parameters()).device}")
            
        except Exception as e:
            self.logger.error(f"Error setting up models: {str(e)}", exc_info=True)
            raise
        
        self.logger.info("Models initialized successfully")
    
    def _setup_optimizers(self):
        """Configure optimizers and schedulers for both models."""
        self.vae_optimizer = torch.optim.AdamW(
            self.vae.parameters(),
            lr=self.args.vae_lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999)
        )
        self.teacher_optimizer = torch.optim.AdamW(
            self.teacher.parameters(),
            lr=self.args.teacher_lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999)
        )
        self.vae_scheduler = CosineAnnealingWarmRestarts(
            self.vae_optimizer,
            T_0=self.args.scheduler_t0,
            T_mult=2,
            eta_min=self.args.min_lr
        )
        self.teacher_scheduler = CosineAnnealingWarmRestarts(
            self.teacher_optimizer,
            T_0=self.args.scheduler_t0,
            T_mult=2,
            eta_min=self.args.min_lr
        )
    
    def _setup_data(self):
        """Load dataset and create data loaders for training and validation."""
        self.logger.info("Loading dataset...")
        try:
            # Check if data directory exists
            if not os.path.exists(self.args.data_dir):
                raise ValueError(f"Data directory {self.args.data_dir} does not exist")
            
            # List available files
            sprite_files = list(Path(self.args.data_dir).glob("sprites*.npy"))
            label_files = list(Path(self.args.data_dir).glob("labels*.csv"))
            
            self.logger.info(f"Found {len(sprite_files)} sprite files and {len(label_files)} label files")
            
            if not sprite_files or not label_files:
                raise ValueError(f"No sprite or label files found in {self.args.data_dir}")
            
            # Initialize dataset with progress monitoring
            self.logger.info("Initializing dataset...")
            dataset = PixelArtDataset(self.args.data_dir, teacher_model=None)
            self.logger.info(f"Dataset initialized with {len(dataset)} samples")
            
            # Split into train and validation sets
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            self.logger.info(f"Splitting dataset: {train_size} training samples, {val_size} validation samples")
            
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            # Create data loaders with error checking
            self.logger.info("Creating data loaders...")
            
            # More conservative DataLoader settings
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=min(4, self.args.num_workers),  # Reduce number of workers
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
                multiprocessing_context='spawn',
                timeout=120  # Add timeout
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=min(4, self.args.num_workers),  # Reduce number of workers
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
                multiprocessing_context='spawn',
                timeout=120  # Add timeout
            )
            
            # Test data loading
            self.logger.info("Testing data loader...")
            test_batch = next(iter(self.train_loader))
            self.logger.info("Successfully loaded first batch")
            self.logger.info(f"Batch keys: {test_batch.keys()}")
            self.logger.info(f"Image shape: {test_batch['image'].shape}")
            
            self.logger.info("Data loaders created successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up data: {str(e)}", exc_info=True)
            raise
    
    def _handle_interrupt(self, signum, frame):
        """Handle keyboard interrupt gracefully"""
        self.logger.info("\nTraining interrupted. Saving checkpoint and cleaning up...")
        self._save_checkpoint(is_interrupted=True)
        self.writer.close()
        sys.exit(0)
    
    def _save_checkpoint(self, is_best=False, is_interrupted=False):
        """Save model checkpoints."""
        checkpoint = {
            'global_step': self.global_step,
            'vae_state_dict': self.vae.state_dict(),
            'teacher_state_dict': self.teacher.state_dict(),
            'vae_optimizer': self.vae_optimizer.state_dict(),
            'teacher_optimizer': self.teacher_optimizer.state_dict(),
            'vae_scheduler': self.vae_scheduler.state_dict(),
            'teacher_scheduler': self.teacher_scheduler.state_dict(),
            'best_loss': self.best_loss,
            'args': vars(self.args)
        }
        ckpt_path = self.checkpoints_dir / 'latest.pt'
        torch.save(checkpoint, ckpt_path)
        self.logger.info(f"Checkpoint saved at step {self.global_step}")
        if is_best:
            best_path = self.checkpoints_dir / 'best.pt'
            shutil.copy(ckpt_path, best_path)
            self.logger.info("Best checkpoint updated")
        if is_interrupted:
            self.logger.info("Interrupted checkpoint saved")
    
    def _cleanup_cuda(self):
        """Clean up CUDA memory and old files."""
        torch.cuda.empty_cache()
    
    def _log_metrics(self, metrics: dict):
        """Log metrics to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.global_step)
    
    def _generate_samples(self):
        """Generate and save sample images from the VAE."""
        save_start = time.time()
        self.vae.eval()
        with torch.no_grad():
            z = torch.randn(self.args.sample_count, self.args.latent_dim, device=self.device)
            # Create a dummy prompt embedding (zeros) for generation
            prompt_embedding = torch.zeros(self.args.sample_count, self.args.latent_dim, device=self.device)
            recon, _ = self.vae(z, prompt_embedding)
            # Denormalize images from [-1, 1] to [0, 255]
            images = ((recon + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
            
            # Save each sample image
            sample_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, img in enumerate(images):
                # Convert from CHW to HWC
                img = np.transpose(img, (1, 2, 0))
                sample_path = self.output_dir / f"sample_{self.global_step}_{i}_{sample_time}.png"
                Image.fromarray(img).save(sample_path)
            
            save_time = time.time() - save_start
            self.logger.info(f"Generated and saved {self.args.sample_count} samples in {save_time:.2f}s")
        
        self.vae.train()
    
    def _compute_losses(self, batch):
        """Compute losses including RL rewards"""
        with autocast(device_type='cuda', dtype=self.amp_dtype) if self.use_amp else nullcontext():
            # VAE forward pass
            recon, mu, logvar = self.vae(batch['image'])
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(recon, batch['image'], reduction='mean')
            
            # KL divergence loss with numerical stability
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Get prompt embeddings from batch
            prompt_embeddings = batch.get('prompt_embedding')
            
            # Teacher evaluation and rewards
            teacher_out = self.teacher(recon.detach(), prompt_embeddings)
            quality_scores = teacher_out['quality_scores']
            semantic_score = teacher_out['semantic_score']
            
            # Compute rewards
            quality_reward = quality_scores.mean(dim=1, keepdim=True)
            semantic_reward = semantic_score if semantic_score is not None else torch.zeros_like(quality_reward)
            total_reward = quality_reward + self.semantic_weight * semantic_reward
            
            # Update baseline with moving average
            if self.baseline is None:
                self.baseline = total_reward.mean().item()
            else:
                self.baseline = (self.baseline_momentum * self.baseline + 
                               (1 - self.baseline_momentum) * total_reward.mean().item())
            
            # Compute advantage (reward - baseline)
            advantage = (total_reward - self.baseline).detach()
            
            # Scale rewards for stability
            advantage = advantage * self.reward_scale
            
            # Policy gradient loss (negative because we want to maximize reward)
            pg_loss = -(advantage * recon_loss).mean()
            
            # VAE total loss (combine reconstruction, KL, and policy gradient)
            vae_loss = (self.args.recon_weight * recon_loss +
                       self.args.kl_weight * kl_loss +
                       pg_loss)
            
            # Teacher loss (quality assessment)
            quality_loss = -torch.mean(quality_scores)
            teacher_loss = self.args.quality_weight * quality_loss
            
            metrics = {
                'recon_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'quality_loss': quality_loss.item(),
                'pg_loss': pg_loss.item(),
                'semantic_reward': semantic_reward.mean().item(),
                'quality_reward': quality_reward.mean().item(),
                'baseline': self.baseline,
                'advantage': advantage.mean().item(),
                'vae_loss': vae_loss.item(),
                'teacher_loss': teacher_loss.item(),
                'total_loss': (vae_loss + teacher_loss).item(),
                'quality_scores': quality_scores.mean().item()
            }
            
            return vae_loss, teacher_loss, metrics, recon, teacher_out
    
    def _save_eval_samples(self, recon_images, teacher_output, batch):
        """Save comparison samples with input and generated images side by side, along with evaluation scores"""
        if not hasattr(self, '_last_eval_save') or \
           self.global_step - self._last_eval_save >= self.args.eval_save_freq:
            save_start = time.time()
            self._last_eval_save = self.global_step
            
            with torch.no_grad():
                # Get quality scores and ensure they're on CPU
                quality_scores = teacher_output['quality_scores'][:4].cpu()
                semantic_scores = teacher_output.get('semantic_score', None)
                if semantic_scores is not None:
                    semantic_scores = semantic_scores[:4].cpu()
                
                # Create a comparison grid
                num_samples = min(4, len(recon_images))
                grid_height = num_samples
                grid_width = 2  # Original and Generated side by side
                
                # Create a white background image
                comparison_img = Image.new('RGB', (grid_width * 128 + (grid_width-1) * 10, 
                                                 grid_height * 128 + (grid_height-1) * 10 + 30), 
                                         color='white')
                
                for i in range(num_samples):
                    # Get original image from batch
                    orig_img = batch['image'][i]
                    orig_img = ((orig_img.cpu().numpy() + 1) * 127.5).astype(np.uint8)
                    orig_img = np.transpose(orig_img, (1, 2, 0))
                    orig_img = Image.fromarray(orig_img)
                    
                    # Get generated image
                    gen_img = recon_images[i]
                    gen_img = ((gen_img.cpu().numpy() + 1) * 127.5).astype(np.uint8)
                    gen_img = np.transpose(gen_img, (1, 2, 0))
                    gen_img = Image.fromarray(gen_img)
                    
                    # Calculate positions
                    y_pos = i * (128 + 10)  # 10 pixels padding between rows
                    
                    # Paste images
                    comparison_img.paste(orig_img, (0, y_pos))
                    comparison_img.paste(gen_img, (128 + 10, y_pos))  # 10 pixels padding between columns
                    
                    # Add text with scores
                    draw = ImageDraw.Draw(comparison_img)
                    try:
                        font = ImageFont.truetype("arial.ttf", 12)
                    except:
                        font = ImageFont.load_default()
                    
                    # Format scores
                    quality_text = f"Quality: {quality_scores[i].mean():.3f}"
                    if semantic_scores is not None:
                        semantic_text = f"Semantic: {semantic_scores[i].item():.3f}"
                    else:
                        semantic_text = ""
                    
                    # Draw text
                    text_y = y_pos + 128 + 2
                    draw.text((0, text_y), "Original", fill='black', font=font)
                    draw.text((128 + 10, text_y), 
                            f"Generated | {quality_text} | {semantic_text}", 
                            fill='black', font=font)
                
                # Save the comparison image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comparison_path = self.eval_samples_dir / f'comparison_{self.global_step}_{timestamp}.png'
                comparison_img.save(comparison_path)
            
            save_time = time.time() - save_start
            self.logger.info(f"Saved comparison image with {num_samples} samples in {save_time:.2f}s")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model and training state from checkpoint."""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model states
            self.vae.load_state_dict(checkpoint['vae_state_dict'])
            self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
            
            # Load optimizer states
            self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer'])
            self.teacher_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
            
            # Load scheduler states
            self.vae_scheduler.load_state_dict(checkpoint['vae_scheduler'])
            self.teacher_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
            
            # Load training state
            self.global_step = checkpoint['global_step']
            self.best_loss = checkpoint['best_loss']
            
            self.logger.info(f"Successfully loaded checkpoint from step {self.global_step}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}", exc_info=True)
            return False
    
    def train(self):
        """Main training loop with improvements"""
        self.logger.info("Starting hybrid training...")
        try:
            # Initial memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated() / 1e9
                self.logger.info(f"Initial GPU memory usage: {initial_memory:.2f} GB")
            
            # Set models to train mode and enable gradients
            self.vae.train()
            self.teacher.train()
            torch.set_grad_enabled(True)
            
            # Move models to GPU explicitly
            self.vae = self.vae.cuda()
            self.teacher = self.teacher.cuda()
            
            for epoch in range(self.args.num_epochs):
                epoch_start_time = time.time()
                epoch_losses = []
                self.logger.info(f"\nEpoch {epoch+1}/{self.args.num_epochs}")
                
                # Calculate total steps for this epoch
                total_steps = len(self.train_loader)
                batch_times = []
                
                self.logger.info("Starting epoch iteration...")
                
                for batch_idx, batch in enumerate(self.train_loader):
                    self.logger.info(f"Processing batch {batch_idx}")
                    batch_start_time = time.time()
                    
                    try:
                        # Print detailed progress
                        if batch_idx % 5 == 0:
                            self.logger.info(
                                f"Batch {batch_idx}/{total_steps} "
                                f"({(batch_idx/total_steps)*100:.1f}%) - "
                                f"Step {self.global_step}"
                            )
                            if batch_times:
                                avg_time = sum(batch_times) / len(batch_times)
                                eta = avg_time * (total_steps - batch_idx)
                                self.logger.info(f"Average batch time: {avg_time:.2f}s, ETA: {eta/60:.2f} minutes")
                        
                        # Process batch
                        try:
                            # Move images to GPU and generate prompt embeddings
                            images = batch['image'].cuda(non_blocking=True)
                            
                            # Zero gradients before forward pass
                            self.vae_optimizer.zero_grad(set_to_none=True)
                            self.teacher_optimizer.zero_grad(set_to_none=True)
                            
                            # Forward pass through VAE first
                            with autocast(device_type='cuda', dtype=self.amp_dtype) if self.use_amp else nullcontext():
                                recon, mu, logvar = self.vae(images)
                                
                                # Generate prompt embeddings with teacher
                                with torch.no_grad():
                                    teacher_out = self.teacher(images)
                                    prompt_embeddings = teacher_out['prompt_embedding']
                                
                                # Compute losses
                                # Reconstruction loss (MSE)
                                recon_loss = F.mse_loss(recon, images, reduction='mean')
                                
                                # KL divergence loss
                                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                                
                                # Teacher evaluation
                                teacher_eval = self.teacher(recon.detach(), prompt_embeddings)
                                quality_scores = teacher_eval['quality_scores']
                                semantic_score = teacher_eval.get('semantic_score', None)
                                
                                # Compute rewards and losses
                                quality_reward = quality_scores.mean(dim=1, keepdim=True)
                                semantic_reward = semantic_score if semantic_score is not None else torch.zeros_like(quality_reward)
                                total_reward = quality_reward + self.semantic_weight * semantic_reward
                                
                                # Update baseline
                                if self.baseline is None:
                                    self.baseline = total_reward.mean().item()
                                else:
                                    self.baseline = (self.baseline_momentum * self.baseline + 
                                                   (1 - self.baseline_momentum) * total_reward.mean().item())
                                
                                # Compute advantage
                                advantage = (total_reward - self.baseline).detach()
                                advantage = advantage * self.reward_scale
                                
                                # Compute final losses
                                pg_loss = -(advantage * recon_loss).mean()
                                vae_loss = (self.args.recon_weight * recon_loss +
                                          self.args.kl_weight * kl_loss +
                                          pg_loss)
                                
                                quality_loss = -torch.mean(quality_scores)
                                teacher_loss = self.args.quality_weight * quality_loss
                                
                                # Scale losses for gradient accumulation
                                vae_loss = vae_loss / self.args.gradient_accumulation_steps
                                teacher_loss = teacher_loss / self.args.gradient_accumulation_steps
                            
                            # Backward pass
                            if self.use_amp:
                                self.scaler.scale(vae_loss).backward()
                                self.scaler.scale(teacher_loss).backward()
                            else:
                                vae_loss.backward()
                                teacher_loss.backward()
                            
                            # Step if we've accumulated enough gradients
                            if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                                if self.use_amp:
                                    self.scaler.unscale_(self.vae_optimizer)
                                    self.scaler.unscale_(self.teacher_optimizer)
                                    
                                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), self.args.max_grad_norm)
                                torch.nn.utils.clip_grad_norm_(self.teacher.parameters(), self.args.max_grad_norm)
                                
                                if self.use_amp:
                                    self.scaler.step(self.vae_optimizer)
                                    self.scaler.step(self.teacher_optimizer)
                                    self.scaler.update()
                                else:
                                    self.vae_optimizer.step()
                                    self.teacher_optimizer.step()
                                
                                # Step schedulers
                                self.vae_scheduler.step()
                                self.teacher_scheduler.step()
                            
                            # Log metrics
                            metrics = {
                                'recon_loss': recon_loss.item(),
                                'kl_loss': kl_loss.item(),
                                'quality_loss': quality_loss.item(),
                                'pg_loss': pg_loss.item(),
                                'semantic_reward': semantic_reward.mean().item(),
                                'quality_reward': quality_reward.mean().item(),
                                'baseline': self.baseline,
                                'advantage': advantage.mean().item(),
                                'vae_loss': vae_loss.item(),
                                'teacher_loss': teacher_loss.item(),
                                'total_loss': (vae_loss + teacher_loss).item(),
                                'quality_scores': quality_scores.mean().item()
                            }
                            
                            if self.global_step % self.args.log_every == 0:
                                self._log_metrics(metrics)
                                current_memory = torch.cuda.memory_allocated() / 1e9
                                self.logger.info(
                                    f"Step {self.global_step} - "
                                    f"VAE Loss: {metrics['vae_loss']:.4f}, "
                                    f"Teacher Loss: {metrics['teacher_loss']:.4f}, "
                                    f"Quality: {metrics['quality_scores']:.4f}, "
                                    f"GPU Memory: {current_memory:.2f} GB"
                                )
                            
                            # Update counters and save samples
                            self.global_step += 1
                            epoch_losses.append(metrics['total_loss'])
                            
                            if self.global_step % self.args.eval_save_freq == 0:
                                self._save_eval_samples(recon, teacher_eval, {'image': images})
                            
                            # Update batch timing
                            batch_time = time.time() - batch_start_time
                            batch_times.append(batch_time)
                            
                            # Cleanup
                            torch.cuda.empty_cache()
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                self.logger.error(f"GPU OOM: {str(e)}")
                                torch.cuda.empty_cache()
                                continue
                            raise e
                            
                    except Exception as e:
                        self.logger.error(f"Error in batch {batch_idx}: {str(e)}", exc_info=True)
                        continue
                
                # Epoch summary with timing
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = np.mean(epoch_losses)
                self.logger.info(
                    f"\nEpoch {epoch+1} Summary:\n"
                    f"Time: {epoch_time/60:.2f} minutes\n"
                    f"Average Loss: {avg_epoch_loss:.4f}\n"
                    f"Best Loss: {self.best_loss:.4f}\n"
                    f"Learning Rates: VAE = {self.vae_optimizer.param_groups[0]['lr']:.6f}, "
                    f"Teacher = {self.teacher_optimizer.param_groups[0]['lr']:.6f}\n"
                    f"Memory Usage: {torch.cuda.memory_allocated()/1e9:.2f} GB"
                )
                
                # Early stopping check
                self.early_stopping(avg_epoch_loss)
                if self.early_stopping.early_stop:
                    self.logger.info("Early stopping triggered")
                    break
                
                if avg_epoch_loss < self.best_loss:
                    self.best_loss = avg_epoch_loss
                    self._save_checkpoint(is_best=True)
                
                self._cleanup_cuda()
                
        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted by user. Saving checkpoint and cleaning up...")
            self._save_checkpoint(is_interrupted=True)
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}", exc_info=True)
            raise
        finally:
            self.writer.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("Training completed.")

# --------------------------
# Main entry point
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Hybrid Training for Lunaris: Generator and Evaluator")
    # Data paths
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing sprites*.npy and labels*.csv files')
    parser.add_argument('--output_dir', type=str, default='output', help='Base output directory')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint file to resume training from')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Number of steps to accumulate gradients')
    parser.add_argument('--chunk_size', type=int, default=32, help='Chunk size for attention computation')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile()')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent space dimension')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension for MoE teacher')
    parser.add_argument('--num_experts', type=int, default=4, help='Number of experts in the teacher model')
    
    # Optimizer parameters
    parser.add_argument('--vae_lr', type=float, default=1e-4, help='Learning rate for VAE')
    parser.add_argument('--teacher_lr', type=float, default=1e-4, help='Learning rate for teacher model')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizers')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--scheduler_t0', type=int, default=10, help='T0 for cosine annealing scheduler')
    
    # Loss weights
    parser.add_argument('--recon_weight', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--kl_weight', type=float, default=0.1, help='KL divergence loss weight')
    parser.add_argument('--quality_weight', type=float, default=0.5, help='Quality loss weight from teacher evaluation')
    
    # Logging and checkpointing
    parser.add_argument('--log_every', type=int, default=100, help='Log every N steps')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--sample_every', type=int, default=500, help='Generate sample images every N steps')
    parser.add_argument('--keep_n_checkpoints', type=int, default=5, help='Keep only the last N periodic checkpoints')
    parser.add_argument('--early_stopping_patience', type=int, default=7,
                        help='Number of epochs to wait before early stopping')
    parser.add_argument('--eval_save_freq', type=int, default=500,
                        help='Save evaluated samples every N steps')
    
    # Add RL-specific arguments
    parser.add_argument('--reward_scale', type=float, default=0.1,
                       help='Scale factor for RL rewards')
    parser.add_argument('--semantic_weight', type=float, default=0.5,
                       help='Weight for semantic matching reward')
    parser.add_argument('--baseline_momentum', type=float, default=0.9,
                       help='Momentum for reward baseline updates')
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    trainer = TrainingManager(args)
    trainer.train()

if __name__ == "__main__":
    main()
