import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import argparse
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import sys
import signal
import glob
import shutil
import psutil
import logging
from pathlib import Path
import gc
import torch.compiler
import datetime
import json

# Import the model from lunar_core_model.py
from lunar_core_model import LunarCoreVAE

# Configure logging
def setup_logging(log_dir):
    """Configure logging with improved format."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
    
    # Configure root logger with a more professional format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("LunarCore")

def validate_training_parameters(args):
    """Validate all training parameters before starting."""
    logger = logging.getLogger("LunarCore.validation")
    
    # Validate directories and files
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist!")
    
    sprites_path = os.path.join(args.data_dir, "sprites.npy")
    labels_path = os.path.join(args.data_dir, "labels.csv")
    
    if not os.path.exists(sprites_path):
        raise ValueError(f"Sprites file not found: {sprites_path}")
    if not os.path.exists(labels_path):
        raise ValueError(f"Labels file not found: {labels_path}")
    
    # Validate numeric parameters
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {args.batch_size}")
    if args.epochs <= 0:
        raise ValueError(f"Number of epochs must be positive, got {args.epochs}")
    if args.learning_rate <= 0:
        raise ValueError(f"Learning rate must be positive, got {args.learning_rate}")
    if args.latent_dim <= 0:
        raise ValueError(f"Latent dimension must be positive, got {args.latent_dim}")
    
    # Validate checkpoint if provided
    if args.checkpoint and not os.path.exists(args.checkpoint):
        raise ValueError(f"Checkpoint file {args.checkpoint} does not exist!")
    
    logger.info("All training parameters validated successfully")
    return True

def check_system_memory(batch_size, image_size=(128, 128), num_channels=3):
    """Check if system has enough memory for training with 128×128 images."""
    logger = logging.getLogger("LunarCore.memory")
    
    # Calculate approximate memory requirements for 128×128 images
    sample_size = batch_size * image_size[0] * image_size[1] * num_channels * 4  # 4 bytes per float32
    model_size = 500 * 1024 * 1024  # Increased model size estimate for 128×128 architecture (500MB)
    
    # Check RAM with increased safety margin for larger images
    available_ram = psutil.virtual_memory().available
    required_ram = sample_size * 4  # Increased safety factor for larger images
    
    if required_ram > available_ram:
        raise MemoryError(f"Not enough RAM for 128×128 training. Required: {required_ram/1e9:.2f}GB, Available: {available_ram/1e9:.2f}GB")
    
    # Check CUDA memory if available with adjusted requirements
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_required = sample_size * 6  # Increased factor for GPU memory requirements with larger images
        
        if gpu_memory_required > gpu_memory:
            logger.warning(f"GPU memory might be insufficient for 128×128 images. Required: {gpu_memory_required/1e9:.2f}GB, Available: {gpu_memory/1e9:.2f}GB")
    
    logger.info(f"Memory check passed for 128×128 training. RAM Available: {available_ram/1e9:.2f}GB")
    return True

def optimize_num_workers():
    """Calculate optimal number of workers based on system CPU cores."""
    cpu_count = psutil.cpu_count(logical=False)
    if cpu_count is None:
        return 0
    return min(4, max(1, cpu_count - 1))  # Leave one core free for system

def cleanup_cuda_memory():
    """Limpa completamente o cache CUDA e libera toda memória."""
    if torch.cuda.is_available():
        try:
            # Limpa o cache CUDA
            torch.cuda.empty_cache()
            
            # Força sincronização de todas as streams CUDA
            torch.cuda.synchronize()
            
            # Reseta o estado do allocator CUDA
            torch.cuda.memory.empty_cache()
            
            # Limpa qualquer memória em cache
            torch.cuda.empty_cache()  # Chamada adicional para garantir
            
            # Força o coletor de lixo do Python
            gc.collect()
            
        except Exception as e:
            logger = logging.getLogger("LunarCore.cuda")
            logger.warning(f"Error cleaning CUDA memory: {str(e)}")

def cleanup_memory():
    """Função melhorada para limpeza de memória."""
    logger = logging.getLogger("LunarCore.memory")
    
    try:
        # Limpa referências cíclicas
        gc.collect()
        
        # Limpa cache CUDA se disponível
        cleanup_cuda_memory()
        
        # Limpa variáveis não utilizadas
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except:
                pass
                
    except Exception as e:
        logger.warning(f"Error during memory cleanup: {str(e)}")

def reset_cuda_device():
    """Reseta completamente o dispositivo CUDA."""
    logger = logging.getLogger("LunarCore.cuda")
    
    if torch.cuda.is_available():
        try:
            # Limpa toda a memória CUDA
            cleanup_cuda_memory()
            
            # Reseta todos os dispositivos CUDA
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            # Força reinicialização do ambiente CUDA
            current_device = torch.cuda.current_device()
            torch.cuda.empty_cache()  # Corrigido: chamada direta ao invés de através do device
            
            logger.info(f"CUDA device reset successfully")
            logger.info(f"→ Current device: {torch.cuda.get_device_name(current_device)}")
            logger.info(f"→ Memory allocated: {torch.cuda.memory_allocated(current_device)/1e9:.2f}GB")
            logger.info(f"→ Memory cached: {torch.cuda.memory_reserved(current_device)/1e9:.2f}GB")
            
        except Exception as e:
            logger.warning(f"Error resetting CUDA device: {str(e)}")

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.logger = logging.getLogger("LunarCore.early_stopping")

    def __call__(self, val_loss, model, epoch, optimizer, scheduler, output_dirs, metrics=None):
        try:
            if self.best_loss is None:
                self.best_loss = val_loss
                self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, output_dirs, metrics)
            elif val_loss > self.best_loss + self.min_delta:
                self.counter += 1
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.logger.info("Early stopping triggered")
            else:
                self.best_loss = val_loss
                self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, output_dirs, metrics)
                self.counter = 0
        except Exception as e:
            self.logger.error("Erro durante early stopping check", exc_info=True)
            raise

    def save_checkpoint(self, val_loss, model, epoch, optimizer, scheduler, output_dirs, metrics=None):
        try:
            if self.verbose:
                self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            
            # Save best model
            best_model_path = os.path.join(output_dirs["checkpoints.best"], "best_model.pth")
            safe_save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=val_loss,
                output_path=best_model_path,
                metrics=metrics,
                is_best=True
            )
            
            # Save periodic checkpoint
            metrics_str = "_".join([f"{k}_{v:.4f}" for k, v in (metrics or {}).items()])
            checkpoint_name = f"checkpoint_epoch_{epoch}_valloss_{val_loss:.4f}"
            if metrics_str:
                checkpoint_name += f"_{metrics_str}"
            checkpoint_name += ".pth"
            
            checkpoint_path = os.path.join(output_dirs["checkpoints.periodic"], checkpoint_name)
            safe_save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=val_loss,
                output_path=checkpoint_path,
                metrics=metrics,
                is_best=False
            )
            
            self.val_loss_min = val_loss
            
        except Exception as e:
            self.logger.error("Erro ao salvar checkpoint", exc_info=True)
            raise

# Dataset
class PixelArtDataset(Dataset):
    """
    Dataset class for loading pixel art images from sprites.npy and labels from labels.csv.
    Handles 128×128 pixel art images and their corresponding labels/prompts.
    """
    def __init__(self, sprites_path, labels_path, transform=None):
        """
        Initialize the dataset.
        
        Args:
            sprites_path (str): Path to sprites.npy file containing image data
            labels_path (str): Path to labels.csv file containing labels/prompts
            transform (callable, optional): Optional transform to be applied on images
        """
        self.transform = transform
        
        # Load sprite data from .npy file
        try:
            self.sprites = np.load(sprites_path)
            if len(self.sprites.shape) != 4:
                raise ValueError(f"Expected 4D array (N,H,W,C), got shape {self.sprites.shape}")
            
            # Normalização específica para pixel art
            if self.sprites.dtype == np.uint8:
                # Normaliza para [-1, 1] mantendo valores discretos
                self.sprites = (self.sprites.astype(np.float32) / 127.5) - 1.0
                # Arredonda para manter valores discretos
                self.sprites = np.round(self.sprites * 32) / 32
            
            self.sprites = np.transpose(self.sprites, (0, 3, 1, 2))
            
        except Exception as e:
            raise ValueError(f"Error loading sprites from {sprites_path}: {str(e)}")
        
        # Load labels from CSV file
        try:
            self.labels_df = pd.read_csv(labels_path)
            if len(self.labels_df) != len(self.sprites):
                raise ValueError(
                    f"Mismatch between number of sprites ({len(self.sprites)}) "
                    f"and labels ({len(self.labels_df)})"
                )
        except Exception as e:
            raise ValueError(f"Error loading labels from {labels_path}: {str(e)}")
        
        # Convert to tensors for efficiency
        self.sprites = torch.from_numpy(self.sprites)
        
        # Verify image dimensions
        N, C, H, W = self.sprites.shape
        if H != 128 or W != 128:
            raise ValueError(
                f"Expected 128×128 images, got {H}×{W}. "
                "Please ensure sprites are in the correct resolution."
            )
        
        print(f"Dataset loaded successfully:")
        print(f"→ Number of samples: {N}")
        print(f"→ Image dimensions: {H}×{W}")
        print(f"→ Channels: {C}")
        print(f"→ Label columns: {', '.join(self.labels_df.columns)}")

    def __len__(self):
        return len(self.sprites)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            tuple: (image, label) where image is a tensor and label is a string
        """
        # Get image and convert to float32 tensor
        image = self.sprites[idx].float()
        
        # Get corresponding label
        label = self.labels_df.iloc[idx].to_dict()
        
        # Apply transforms if specified
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

def calculate_metrics(recon_images, images):
    """Calculate additional metrics for model evaluation with pixel art focus."""
    with torch.no_grad():
        # Mean Absolute Error (mais adequado para pixel art)
        mae = torch.mean(torch.abs(recon_images - images)).item()
        
        # Peak Signal-to-Noise Ratio (PSNR)
        mse = torch.mean((recon_images - images) ** 2).item()
        psnr = 20 * np.log10(2.0) - 10 * np.log10(mse)
        
        # Pixel Accuracy (específico para pixel art)
        pixel_accuracy = torch.mean((torch.abs(recon_images - images) < 0.1).float()).item()
        
        return {
            "mae": mae,
            "psnr": psnr,
            "pixel_accuracy": pixel_accuracy
        }

def save_comparison_grid(epoch, original_images, generated_images, output_dirs, num_images=8):
    """
    Save a grid of original vs generated images side by side with high quality for pixel art.
    """
    # Create the comparisons directory if it doesn't exist
    progress_dir = os.path.join(output_dirs["outputs.progress"], f'epoch_{epoch:04d}')
    os.makedirs(progress_dir, exist_ok=True)

    # Convert tensors to numpy arrays and move to CPU if necessary
    if torch.is_tensor(original_images):
        original_images = original_images.detach().cpu().float().numpy()
    if torch.is_tensor(generated_images):
        generated_images = generated_images.detach().cpu().float().numpy()

    # Take only the specified number of images
    original_images = original_images[:num_images]
    generated_images = generated_images[:num_images]

    # Create a figure with a grid of image pairs
    fig, axes = plt.subplots(num_images, 2, figsize=(20, 4*num_images), dpi=300)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    # Desabilitar interpolação para matplotlib
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.resample'] = False
    plt.rcParams['image.composite_image'] = False

    for i in range(num_images):
        # Original image
        orig_img = np.transpose(original_images[i], (1, 2, 0))
        orig_img = (orig_img + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
        orig_img = np.clip(orig_img, 0, 1)  # Ensure values are in [0, 1]
        
        # Forçar pixels nítidos
        axes[i, 0].imshow(orig_img, interpolation='nearest')
        axes[i, 0].set_aspect('equal')
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=16)

        # Generated image
        gen_img = np.transpose(generated_images[i], (1, 2, 0))
        gen_img = (gen_img + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
        gen_img = np.clip(gen_img, 0, 1)  # Ensure values are in [0, 1]
        
        # Forçar pixels nítidos
        axes[i, 1].imshow(gen_img, interpolation='nearest')
        axes[i, 1].set_aspect('equal')
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Generated', fontsize=16)

    # Add epoch information
    plt.suptitle(f'Training Progress - Epoch {epoch}', y=0.95, fontsize=18)
    
    # Save the figure with high quality and no interpolation
    comparison_path = os.path.join(progress_dir, f'comparison.png')
    plt.savefig(comparison_path, bbox_inches='tight', dpi=300, 
                pad_inches=0.2, facecolor='white', edgecolor='none',
                transparent=False)
    plt.close()

    # Save individual images in high quality
    details_dir = os.path.join(progress_dir, 'individual')
    os.makedirs(details_dir, exist_ok=True)

    # Convert back to tensors for save_image
    original_images_tensor = torch.from_numpy(original_images)
    generated_images_tensor = torch.from_numpy(generated_images)

    # Normalize tensors from [-1, 1] to [0, 1]
    original_images_tensor = (original_images_tensor + 1) / 2
    generated_images_tensor = (generated_images_tensor + 1) / 2

    for i in range(num_images):
        # Save original with nearest neighbor interpolation
        save_image(
            original_images_tensor[i],
            os.path.join(details_dir, f'original_{i+1}.png'),
            normalize=False,
            nrow=1,
            padding=0,
            scale_each=False
        )
        # Save generated with nearest neighbor interpolation
        save_image(
            generated_images_tensor[i],
            os.path.join(details_dir, f'generated_{i+1}.png'),
            normalize=False,
            nrow=1,
            padding=0,
            scale_each=False
        )

def find_best_checkpoint(output_dirs):
    """Find the best checkpoint based on validation loss and metrics."""
    logger = logging.getLogger("LunarCore.checkpoint")
    
    # Search for checkpoints in best and periodic directories
    best_checkpoints = glob.glob(os.path.join(output_dirs["checkpoints.best"], "*.pth"))
    periodic_checkpoints = glob.glob(os.path.join(output_dirs["checkpoints.periodic"], "*.pth"))
    
    all_checkpoints = best_checkpoints + periodic_checkpoints
    if not all_checkpoints:
        return None
    
    # Find best checkpoint
    best_checkpoint = None
    best_loss = float('inf')
    best_epoch = -1
    
    for ckpt in all_checkpoints:
        try:
            checkpoint = torch.load(ckpt, map_location='cpu')
            if "val_loss" not in checkpoint:
                logger.warning(f"Skipping corrupted checkpoint: {os.path.basename(ckpt)}")
                continue
            
            val_loss = checkpoint["val_loss"]
            epoch = checkpoint["epoch"]
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = ckpt
                best_epoch = epoch
                
        except Exception as e:
            logger.warning(f"Skipping invalid checkpoint {os.path.basename(ckpt)}: {str(e)}")
            continue
    
    if best_checkpoint:
        logger.info(f"Found best checkpoint from epoch {best_epoch} with loss {best_loss:.4f}")
        logger.info(f"Path: {os.path.relpath(best_checkpoint, output_dirs['outputs.main'])}")
    
    return best_checkpoint

def cleanup_old_files(output_dirs, keep_best=True, keep_last_n=5):
    """Clean up old training files with improved management."""
    logger = logging.getLogger("LunarCore.cleanup")
    logger.info("Starting cleanup process")
    
    # Define cleanup targets
    cleanup_targets = [
        ("Periodic checkpoints", os.path.join(output_dirs["checkpoints.periodic"], "*.pth")),
        ("Interrupt checkpoints", os.path.join(output_dirs["checkpoints.interrupt"], "*.pth")),
        ("Sample outputs", os.path.join(output_dirs["outputs.samples"], "*")),
        ("Comparison outputs", os.path.join(output_dirs["outputs.progress"], "*")),
        ("Metric files", os.path.join(output_dirs["outputs.metrics"], "*.json"))
    ]
    
    total_space_saved = 0
    kept_files = []
    
    # Keep best checkpoints if requested
    if keep_best:
        best_checkpoints = glob.glob(os.path.join(output_dirs["checkpoints.best"], "*.pth"))
        kept_files.extend(best_checkpoints)
        logger.info(f"Keeping {len(best_checkpoints)} best checkpoints")
    
    # Process each cleanup target
    for desc, pattern in cleanup_targets:
        files = glob.glob(pattern)
        if files:
            # Sort files by modification time (newest first)
            files.sort(key=os.path.getmtime, reverse=True)
            
            # Keep the most recent files
            if keep_last_n > 0:
                kept_files.extend(files[:keep_last_n])
                files_to_remove = files[keep_last_n:]
            else:
                files_to_remove = files
            
            # Remove old files
            for f in files_to_remove:
                try:
                    size = os.path.getsize(f)
                    os.remove(f)
                    total_space_saved += size
                    logger.info(f"Removed: {os.path.relpath(f, output_dirs['outputs.main'])}")
                except Exception as e:
                    logger.error(f"Error removing {f}: {str(e)}")
    
    # Convert bytes to human-readable format
    if total_space_saved > 1024*1024*1024:
        space_str = f"{total_space_saved/(1024*1024*1024):.2f} GB"
    else:
        space_str = f"{total_space_saved/(1024*1024):.2f} MB"
    
    logger.info(f"Cleanup completed. Total space freed: {space_str}")
    if kept_files:
        logger.info("Kept files:")
        for f in kept_files:
            logger.info(f"    {os.path.relpath(f, output_dirs['outputs.main'])}")

def safe_save_checkpoint(model, optimizer, scheduler, epoch, val_loss, output_path, metrics=None, is_best=False):
    """
    Função segura para salvar checkpoints com tratamento de exceções detalhado.
    
    Args:
        model: Modelo PyTorch
        optimizer: Otimizador
        scheduler: Scheduler do learning rate
        epoch: Época atual
        val_loss: Loss de validação
        output_path: Caminho para salvar o checkpoint
        metrics: Dicionário com métricas adicionais (opcional)
        is_best: Flag indicando se é o melhor modelo até agora
    """
    logger = logging.getLogger("LunarCore.checkpoint")
    
    try:
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepara o checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
            "metrics": metrics or {}
        }
        
        # Salva o checkpoint com tratamento de erro de I/O
        try:
            torch.save(checkpoint, output_path)
            logger.info(f"Checkpoint salvo com sucesso em: {output_path}")
            
            if is_best:
                best_path = os.path.join(os.path.dirname(output_path), "best_model.pth")
                torch.save(checkpoint, best_path)
                logger.info(f"Melhor modelo atualizado em: {best_path}")
                
        except IOError as io_err:
            logger.error(f"Erro de I/O ao salvar checkpoint: {str(io_err)}", exc_info=True)
            raise
            
    except Exception as e:
        logger.error(f"Erro ao preparar/salvar checkpoint: {str(e)}", exc_info=True)
        raise

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Carrega checkpoint com tratamento de exceções detalhado.
    """
    logger = logging.getLogger("LunarCore.checkpoint")
    
    try:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        
        # Carrega estado do modelo
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Estado do modelo carregado de: {checkpoint_path}")
        except Exception as model_err:
            logger.error("Erro ao carregar estado do modelo", exc_info=True)
            raise
            
        # Carrega estado do optimizer se fornecido
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Estado do optimizer carregado")
            except Exception as opt_err:
                logger.error("Erro ao carregar estado do optimizer", exc_info=True)
                raise
                
        # Carrega estado do scheduler se fornecido
        if scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Estado do scheduler carregado")
            except Exception as sched_err:
                logger.error("Erro ao carregar estado do scheduler", exc_info=True)
                raise
                
        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('val_loss', float('inf'))
        metrics = checkpoint.get('metrics', {})
        
        logger.info(f"Checkpoint carregado com sucesso - Época: {epoch}, Val Loss: {val_loss:.4f}")
        return epoch, val_loss, metrics
        
    except Exception as e:
        logger.error(f"Erro ao carregar checkpoint: {str(e)}", exc_info=True)
        raise

def setup_pytorch_optimizations():
    """Configure PyTorch 2.6 optimizations and features."""
    logger = logging.getLogger("LunarCore.setup")
    
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    
    # Keep default dtype as float32 for stability
    torch.set_default_dtype(torch.float32)
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Default dtype: {torch.get_default_dtype()}")
    logger.info("PyTorch optimizations configured")

def setup_model_compilation(model):
    """Setup model compilation using torch.compile."""
    logger = logging.getLogger("LunarCore.compilation")
    
    try:
        # Compile model with PyTorch optimizations
        compiled_model = torch.compile(
            model,
            fullgraph=True,
            dynamic=True
        )
        logger.info("Model successfully compiled with torch.compile")
        return compiled_model
    except Exception as e:
        logger.warning(f"Could not compile model: {str(e)}")
        logger.warning("Falling back to eager mode")
        return model

def setup_directory_structure(base_dir):
    """Setup and validate the project directory structure."""
    logger = logging.getLogger("LunarCore.setup")
    
    # Define directory structure
    dirs = {
        'logs': {
            'main': 'logs',                    # Log files directory
            'training': 'logs/training',        # Training specific logs
            'errors': 'logs/errors'            # Error logs
        },
        'checkpoints': {
            'main': 'checkpoints',             # Main checkpoint directory
            'best': 'checkpoints/best',        # Best model checkpoints
            'periodic': 'checkpoints/periodic', # Regular interval checkpoints
            'interrupt': 'checkpoints/interrupt' # Interrupt checkpoints
        },
        'outputs': {
            'main': 'outputs',                 # Main outputs directory
            'samples': 'outputs/samples',       # Generated samples
            'metrics': 'outputs/metrics',       # Training metrics
            'progress': 'outputs/progress',     # Training progress images
            'reports': 'outputs/reports'        # Training reports
        },
        'tensorboard': 'tensorboard'           # Tensorboard logs
    }
    
    # Create directory structure
    created_dirs = {}
    for category, value in dirs.items():
        if isinstance(value, dict):
            for subcategory, path in value.items():
                full_path = os.path.join(base_dir, path)
                os.makedirs(full_path, exist_ok=True)
                created_dirs[f"{category}.{subcategory}"] = full_path
        else:
            full_path = os.path.join(base_dir, value)
            os.makedirs(full_path, exist_ok=True)
            created_dirs[category] = full_path
    
    logger.info("Directory structure initialized:")
    for name, path in created_dirs.items():
        logger.info(f"    {name:20} → {os.path.relpath(path, base_dir)}")
    
    return created_dirs

def save_training_artifacts(epoch, images, recon_images, metrics, output_dirs):
    """Save training artifacts with improved organization."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comparison grid in the progress directory
    save_comparison_grid(
        epoch,
        images.cpu(),
        recon_images.cpu(),
        output_dirs,
        num_images=8
    )
    
    # Save individual samples in the samples directory
    samples_dir = os.path.join(
        output_dirs["outputs.samples"],
        f"epoch_{epoch:04d}"
    )
    os.makedirs(samples_dir, exist_ok=True)
    
    # Normalize tensors from [-1, 1] to [0, 1] before saving
    images = (images + 1) / 2
    recon_images = (recon_images + 1) / 2
    
    for i in range(min(8, len(images))):
        # Save original and reconstruction in a pair directory
        pair_dir = os.path.join(samples_dir, f'pair_{i+1}')
        os.makedirs(pair_dir, exist_ok=True)
        
        save_image(
            images[i],
            os.path.join(pair_dir, 'original.png'),
            normalize=False
        )
        save_image(
            recon_images[i],
            os.path.join(pair_dir, 'reconstruction.png'),
            normalize=False
        )
    
    # Save metrics in the metrics directory
    metrics_file = os.path.join(
        output_dirs["outputs.metrics"],
        f"metrics_epoch_{epoch:04d}.json"
    )
    with open(metrics_file, 'w') as f:
        json.dump({
            "epoch": epoch,
            "timestamp": timestamp,
            "metrics": metrics
        }, f, indent=4)

def generate_training_report(epoch, train_metrics, val_metrics, output_dirs, args):
    """Generate a detailed training report."""
    logger = logging.getLogger("LunarCore.report")
    
    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "epoch": epoch,
        "training_config": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "latent_dim": args.latent_dim,
            "kl_weight": args.kl_weight,
            "mixed_precision": args.mixed_precision,
            "compile": args.compile
        },
        "training_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "hardware_info": {
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "pytorch_version": torch.__version__
        },
        "memory_usage": {
            "ram": {
                "total": psutil.virtual_memory().total / (1024**3),
                "available": psutil.virtual_memory().available / (1024**3),
                "percent": psutil.virtual_memory().percent
            }
        }
    }
    
    if torch.cuda.is_available():
        report["memory_usage"]["gpu"] = {
            "total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "allocated": torch.cuda.memory_allocated(0) / (1024**3),
            "cached": torch.cuda.memory_reserved(0) / (1024**3)
        }
    
    # Save report in a more organized structure
    report_dir = os.path.join(output_dirs["outputs.reports"], f"epoch_{epoch:04d}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Save detailed report
    report_file = os.path.join(report_dir, "training_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save summary in a separate file for quick reference
    summary_file = os.path.join(report_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("[Training Report Summary]\n")
        f.write(f"Epoch {epoch}/{args.epochs}\n")
        f.write(f"Training Loss: {train_metrics.get('loss', 'N/A'):.4f}\n")
        f.write(f"Validation Loss: {val_metrics.get('loss', 'N/A'):.4f}\n")
        f.write(f"Learning Rate: {args.learning_rate:.6f}\n")
        f.write(f"Device: {report['hardware_info']['device']}\n")
        f.write(f"Memory Usage:\n")
        f.write(f"    RAM: {report['memory_usage']['ram']['percent']}% used\n")
        if torch.cuda.is_available():
            f.write(f"    GPU: {report['memory_usage']['gpu']['allocated']:.2f}GB allocated\n")
    
    # Log summary
    logger.info("\n[Training Report Summary]")
    logger.info(f"Epoch {epoch}/{args.epochs}")
    logger.info(f"Training Loss: {train_metrics.get('loss', 'N/A'):.4f}")
    logger.info(f"Validation Loss: {val_metrics.get('loss', 'N/A'):.4f}")
    logger.info(f"Learning Rate: {args.learning_rate:.6f}")
    logger.info(f"Device: {report['hardware_info']['device']}")
    logger.info(f"Memory Usage:")
    logger.info(f"    RAM: {report['memory_usage']['ram']['percent']}% used")
    if torch.cuda.is_available():
        logger.info(f"    GPU: {report['memory_usage']['gpu']['allocated']:.2f}GB allocated")
    
    return report

def get_transforms():
    """
    Get the transforms for the dataset.
    Simplified version without rotations for pixel art testing.
    """
    return transforms.Compose([
        transforms.Lambda(lambda x: x)  # Identity transform - no augmentations for testing
    ])

def quantize_colors(x, n_colors=32):
    """
    Quantiza as cores da imagem para um número específico de níveis.
    Args:
        x (torch.Tensor): Tensor de imagem normalizado entre -1 e 1
        n_colors (int): Número de níveis de quantização por canal
    Returns:
        torch.Tensor: Tensor quantizado com valores discretos
    """
    # Primeiro normaliza para [0, 1]
    x = (x + 1.0) / 2.0
    # Quantiza para níveis discretos
    x = torch.round(x * (n_colors - 1)) / (n_colors - 1)
    # Volta para [-1, 1]
    x = x * 2.0 - 1.0
    return x

def setup_signal_handler(model, optimizer, scheduler, epoch, output_dirs):
    """
    Configura o handler de sinal para interrupção segura do treinamento.
    """
    logger = logging.getLogger("LunarCore.signal_handler")
    
    def signal_handler(sig, frame):
        logger.info("Interrupção detectada. Salvando checkpoint final...")
        try:
            # Salva checkpoint de interrupção
            interrupt_path = os.path.join(output_dirs["checkpoints.interrupt"], f"interrupt_checkpoint_epoch_{epoch}.pth")
            safe_save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=float('inf'),  # Não temos val_loss no momento da interrupção
                output_path=interrupt_path,
                metrics=None
            )
            logger.info("Checkpoint de interrupção salvo com sucesso")
            
        except Exception as e:
            logger.error("Erro ao salvar checkpoint de interrupção", exc_info=True)
            
        finally:
            sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    logger.info("Handler de sinal configurado para salvamento seguro de checkpoints")

def main():
    # Add signal handler for graceful interruption
    def signal_handler(signum, frame):
        logger = logging.getLogger("LunarCore.signal")
        logger.info("\nInterrupting training... Saving checkpoint and cleaning up...")
        
        try:
            if 'model' in locals() and 'optimizer' in locals() and 'epoch' in locals():
                # Move model to CPU before saving to avoid CUDA errors
                model_cpu = model.to('cpu')
                
                checkpoint_path = os.path.join(
                    output_dirs["checkpoints.interrupt"],
                    f"interrupt_checkpoint_{datetime.datetime.now():%Y%m%d_%H%M%S}.pth"
                )
                
                # Create checkpoint with CPU tensors
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model_cpu.state_dict(),
                    "optimizer_state_dict": {
                        k: v.cpu() if torch.is_tensor(v) else v 
                        for k, v in optimizer.state_dict().items()
                    },
                    "scheduler_state_dict": scheduler.state_dict() if 'scheduler' in locals() else None,
                    "loss": train_loss if 'train_loss' in locals() else None,
                    "val_loss": val_loss if 'val_loss' in locals() else None,
                }
                
                # Save checkpoint
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Checkpoint saved to {os.path.relpath(checkpoint_path, output_dirs['outputs.main'])}")
                
                # Move model back to original device if needed
                if torch.cuda.is_available():
                    model.to('cuda')
                
        except Exception as e:
            logger.error(f"Failed to save interrupt checkpoint: {str(e)}")
        
        finally:
            # Cleanup
            if 'writer' in locals():
                writer.close()
            cleanup_memory()
            
            # Force exit to avoid hanging
            os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Command line arguments
    parser = argparse.ArgumentParser(description="LunarCoreVAE Training")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load (optional)",
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0005, help="Learning rate"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=128, help="Latent space dimension"
    )
    parser.add_argument(
        "--kl_weight", type=float, default=0.00001, help="KL divergence weight"
    )
    parser.add_argument(
        "--data_dir", type=str, default=".", help="Training data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--save_every", type=int, default=10, help="Save checkpoint every X epochs"
    )
    parser.add_argument(
        "--lr_step_size", type=int, default=10, help="Reduce lr every X epochs"
    )
    parser.add_argument(
        "--lr_gamma", type=float, default=0.8, help="LR reduction factor"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,  # Aumentado de 7 para 20
        help="Number of epochs to wait for improvement before early stopping"
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.001,  # Aumentado para ser menos sensível
        help="Minimum change in validation loss to qualify as an improvement"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for model optimization"
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision training"
    )

    args = parser.parse_args()

    # Setup directory structure
    output_dirs = setup_directory_structure(args.output_dir)
    
    # Setup logging with new directory structure
    logger = setup_logging(output_dirs["logs.main"])
    setup_pytorch_optimizations()
    logger.info("Starting LunarCore training session")

    try:
        # Reset CUDA device before starting
        logger.info("\n[CUDA Memory Cleanup]")
        reset_cuda_device()
        
        # Validate parameters
        validate_training_parameters(args)
        
        # Check system memory
        check_system_memory(args.batch_size)

        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Check for old files to clean up
        has_old_files = (
            len(glob.glob(os.path.join(output_dirs["checkpoints.periodic"], "*.pth"))) > 0 or
            len(glob.glob(os.path.join(output_dirs["checkpoints.best"], "*.pth"))) > 0 or
            len(glob.glob(os.path.join(output_dirs["checkpoints.interrupt"], "*.pth"))) > 0
        )

        if has_old_files:
            logger.warning("Old training files detected")
            user_input = input("Do you want to clean up files from previous training sessions? (y/n): ")
            if user_input.lower() == 'y':
                cleanup_old_files(output_dirs, keep_best=True)
                logger.info("Cleanup completed")
            else:
                logger.info("Keeping old files")

        # Check and load best checkpoint automatically if not specified
        if args.checkpoint is None:
            best_checkpoint = find_best_checkpoint(output_dirs)
            if best_checkpoint:
                logger.info(f"\n[Automatic Checkpoint]")
                logger.info(f"Found best checkpoint: {os.path.basename(best_checkpoint)}")
                user_input = input("Do you want to continue training from this checkpoint? (y/n): ")
                if user_input.lower() == 'y':
                    args.checkpoint = best_checkpoint
                    logger.info("Using found checkpoint.")
                else:
                    logger.info("Starting new training.")

        # Hyperparameters
        epochs = args.epochs
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        latent_dim = args.latent_dim
        kl_weight = args.kl_weight
        data_dir = args.data_dir
        output_dir = args.output_dir
        save_every = args.save_every
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Loss function específica para pixel art
        def pixel_art_loss(recon_x, x):
            # Perda MSE por canal de cor separadamente
            mse_r = torch.mean((recon_x[:, 0, :, :] - x[:, 0, :, :]) ** 2)
            mse_g = torch.mean((recon_x[:, 1, :, :] - x[:, 1, :, :]) ** 2)
            mse_b = torch.mean((recon_x[:, 2, :, :] - x[:, 2, :, :]) ** 2)
            mse = 0.7 * (mse_r + mse_g + mse_b) / 3.0
            
            # Perda de cores (penaliza diferenças na paleta de cores)
            color_loss = 0.3 * (
                torch.mean(torch.abs(torch.mean(recon_x[:, 0, :, :], dim=(1,2)) - torch.mean(x[:, 0, :, :], dim=(1,2)))) +
                torch.mean(torch.abs(torch.mean(recon_x[:, 1, :, :], dim=(1,2)) - torch.mean(x[:, 1, :, :], dim=(1,2)))) +
                torch.mean(torch.abs(torch.mean(recon_x[:, 2, :, :], dim=(1,2)) - torch.mean(x[:, 2, :, :], dim=(1,2))))
            )
            
            # Perda de bordas (mantém pixels nítidos)
            edge_loss = 0.1 * (
                torch.mean(torch.abs(
                    recon_x[:, :, 1:, :] - recon_x[:, :, :-1, :] -
                    (x[:, :, 1:, :] - x[:, :, :-1, :])
                )) + 
                torch.mean(torch.abs(
                    recon_x[:, :, :, 1:] - recon_x[:, :, :, :-1] -
                    (x[:, :, :, 1:] - x[:, :, :, :-1])
                ))
            )
            
            # Perda de quantização de cores (força valores discretos)
            n_colors = 32  # Número de níveis de cor
            # Discretização mais forte para cores
            discreteness_loss = 0.2 * torch.mean(
                torch.abs(torch.round(recon_x * 8) / 8 - recon_x)
            )
            
            return mse + color_loss + edge_loss + discreteness_loss

        # Substitui a loss function padrão
        loss_fn = pixel_art_loss

        # Remove all transformations - pixel art should not be transformed
        transform = None  # Removendo todas as transformações para preservar a qualidade do pixel art

        # Load dataset from .npy and .csv files
        try:
            sprites_path = os.path.join(data_dir, "sprites.npy")
            labels_path = os.path.join(data_dir, "labels.csv")
            
            logger.info(f"Loading dataset from:")
            logger.info(f"→ Sprites: {os.path.relpath(sprites_path)}")
            logger.info(f"→ Labels: {os.path.relpath(labels_path)}")
            
            dataset = PixelArtDataset(
                sprites_path=sprites_path,
                labels_path=labels_path,
                transform=transform
            )
            
            logger.info(f"Dataset loaded successfully for 128×128 training")
            logger.info(f"Number of training images: {len(dataset)}")
            
        except Exception as e:
            logger.error(f"ERROR loading dataset: {str(e)}")
            sys.exit(1)

        # Split dataset with same ratios
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        logger.info(f"Dataset split:")
        logger.info(f"→ Training samples: {train_size}")
        logger.info(f"→ Validation samples: {val_size}")

        # Adjust DataLoader settings for 128×128 images
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=optimize_num_workers(),
            pin_memory=True,
            persistent_workers=True if optimize_num_workers() > 0 else False,
            prefetch_factor=2 if optimize_num_workers() > 0 else None,
            drop_last=True,  # Ensure consistent batch sizes
            generator=torch.Generator().manual_seed(42)  # Added for reproducibility
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=optimize_num_workers(),
            pin_memory=True,
            persistent_workers=True if optimize_num_workers() > 0 else False,
            prefetch_factor=2 if optimize_num_workers() > 0 else None,
            drop_last=True,  # Ensure consistent batch sizes
            generator=torch.Generator().manual_seed(42)  # Added for reproducibility
        )

        # Function to save images (updated for 128×128)
        def save_images(epoch, input_img, generated_img, img_name):
            output_dir_epoch = os.path.join(output_dir, f"epoch_{epoch}")
            os.makedirs(output_dir_epoch, exist_ok=True)

            # Denormalize images from [-1, 1] to [0, 1]
            input_img = (input_img + 1) / 2
            generated_img = (generated_img + 1) / 2

            # Save high-resolution images
            input_pil = transforms.ToPILImage()(input_img.squeeze(0))
            generated_pil = transforms.ToPILImage()(generated_img.squeeze(0))

            # Save with high quality
            input_pil.save(os.path.join(output_dir_epoch, f"{img_name}_input.png"), quality=95, optimize=True)
            generated_pil.save(os.path.join(output_dir_epoch, f"{img_name}_generated.png"), quality=95, optimize=True)

        # Model and optimizer
        model = LunarCoreVAE(latent_dim=latent_dim).to(device)
        
        # Apply torch.compile if requested
        if args.compile:
            model = setup_model_compilation(model)

        # Initialize weights
        def weights_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        model.apply(weights_init)

        # Optimizer com weight decay reduzido
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0003,  # Learning rate menor
            weight_decay=1e-7,  # Weight decay menor
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        # Scheduler com warmup mais longo
        def warmup_lambda(epoch):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            return 0.95 ** ((epoch - warmup_epochs) // 5)  # Decay mais suave
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_lambda,
            verbose=False
        )

        # Gradient scaler for mixed precision
        scaler = torch.amp.GradScaler() if args.mixed_precision else None

        # Load checkpoint if specified
        start_epoch = 0
        best_val_loss = float('inf')
        if args.checkpoint:
            logger.info(f"\n[Loading Checkpoint]")
            start_epoch, best_val_loss = load_checkpoint(args.checkpoint, model, optimizer, scheduler)
            
            logger.info(f"→ Starting epoch: {start_epoch}")
            logger.info(f"→ Validation loss: {best_val_loss:.4f}")
            logger.info(f"→ Best loss so far: {best_val_loss:.4f}")
            
            # Adjust learning rate based on loaded epoch
            for _ in range(start_epoch):
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"→ Current learning rate: {current_lr:.6f}")
        else:
            logger.info("\n[New Training]")
            logger.info("Starting training from scratch.")

        # TensorBoard
        writer = SummaryWriter(os.path.join(output_dirs["tensorboard"], f"experiment_{datetime.datetime.now():%Y%m%d_%H%M%S}"))

        # Enable cuDNN benchmark
        torch.backends.cudnn.benchmark = True

        # Initialize early stopping with adjusted parameters
        early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            verbose=True
        )

        # Training loop with adjusted logging for 128×128
        for epoch in range(start_epoch, epochs):
            logger.info("-" * 100)
            logger.info(f"EPOCH {epoch+1}/{epochs} (128×128 Resolution)")
            logger.info("-" * 100)
            
            # Training Phase
            logger.info("[TRAINING PHASE]")
            model.train()
            train_loss = 0.0
            max_grad_norm = 0.0
            
            # Log learning rate at the start of each epoch
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.6f}")

            for i, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Training Progress (128×128)", leave=False)):
                images = images.to(device)
                optimizer.zero_grad(set_to_none=True)

                try:
                    if args.mixed_precision:
                        with torch.amp.autocast('cuda'):
                            recon_images, mean, logvar = model(images)
                            # Aplica quantização de cores
                            recon_images = quantize_colors(recon_images, levels=8)
                            recon_loss = loss_fn(recon_images, images)
                            kl_divergence = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                            # Annealing mais lento do KL weight
                            kl_weight_current = min(1.0, epoch / 100) * kl_weight  # Annealing mais lento
                            loss = recon_loss + kl_weight_current * kl_divergence

                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"Batch {i+1}/{len(train_dataloader)} | NaN/Inf loss detected")
                            logger.warning(f"         | Recon Loss: {recon_loss.item():.4f}")
                            logger.warning(f"         | KL Div: {kl_divergence.item():.4f}")
                            continue

                        scaler.scale(loss).backward()
                        # Unscale before gradient clipping
                        scaler.unscale_(optimizer)
                        # Stronger gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        recon_images, mean, logvar = model(images)
                        # Aplica quantização de cores
                        recon_images = quantize_colors(recon_images, levels=8)
                        recon_loss = loss_fn(recon_images, images)
                        kl_divergence = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                        # Annealing mais lento do KL weight
                        kl_weight_current = min(1.0, epoch / 100) * kl_weight  # Annealing mais lento
                        loss = recon_loss + kl_weight_current * kl_divergence

                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"Batch {i+1}/{len(train_dataloader)} | NaN/Inf loss detected")
                            logger.warning(f"         | Recon Loss: {recon_loss.item():.4f}")
                            logger.warning(f"         | KL Div: {kl_divergence.item():.4f}")
                            continue

                        loss.backward()
                        # Stronger gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()

                    train_loss += loss.item()

                    if (i + 1) % 10 == 0:
                        current_lr = optimizer.param_groups[0]['lr']
                        logger.info(f"Batch {i+1}/{len(train_dataloader)} | Metrics:")
                        logger.info(f"         | Loss: {loss.item():.4f}")
                        logger.info(f"         | Recon Loss: {recon_loss.item():.4f}")
                        logger.info(f"         | KL Div: {kl_divergence.item():.4f}")
                        logger.info(f"         | KL Weight: {kl_weight_current:.6f}")
                        logger.info(f"         | Learning Rate: {current_lr:.6f}")

                except RuntimeError as e:
                    logger.error(f"Batch {i+1}/{len(train_dataloader)} | Runtime Error: {str(e)}")
                    continue

            avg_train_loss = train_loss / len(train_dataloader)
            
            # Validation Phase with adjusted logging
            logger.info("\n[VALIDATION PHASE (128×128)]")
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            # Inicializa todas as métricas necessárias
            val_metrics = {
                "mae": 0.0,
                "psnr": 0.0,
                "pixel_accuracy": 0.0  # Adicionando a métrica que faltava
            }
            
            with torch.no_grad():
                for i, (images, labels) in enumerate(tqdm(val_dataloader, desc="Validation Progress", leave=False)):
                    images = images.to(device)

                    with torch.amp.autocast('cuda'):
                        recon_images, mean, logvar = model(images)
                        # Aplica quantização de cores na validação também
                        recon_images = quantize_colors(recon_images, levels=8)
                        recon_loss = loss_fn(recon_images, images)
                        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                        loss = recon_loss + kl_weight * kl_divergence

                    val_loss += loss.item()
                    val_recon_loss += recon_loss.item()
                    val_kl_loss += kl_divergence.item()
                    
                    batch_metrics = calculate_metrics(recon_images, images)
                    for k, v in batch_metrics.items():
                        val_metrics[k] += v

                    if i == 0 and (epoch + 1) % 5 == 0:
                        save_comparison_grid(
                            epoch + 1,
                            images.cpu(),
                            recon_images.cpu(),
                            output_dirs,
                            num_images=6  # Reduced number of images due to larger size
                        )
            
            # Calculate averages
            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_recon_loss = val_recon_loss / len(val_dataloader)
            avg_val_kl_loss = val_kl_loss / len(val_dataloader)
            for k in val_metrics:
                val_metrics[k] /= len(val_dataloader)
            
            # Epoch Summary
            logger.info("\n[EPOCH SUMMARY]")
            logger.info("Training Metrics:")
            logger.info("    Average Loss: %.4f", avg_train_loss)
            logger.info("\nValidation Metrics:")
            logger.info("    Total Loss: %.4f", avg_val_loss)
            logger.info("    Recon Loss: %.4f", avg_val_recon_loss)
            logger.info("    KL Loss: %.4f", avg_val_kl_loss)
            logger.info("\nAdditional Metrics:")
            for k, v in val_metrics.items():
                logger.info("    %s: %.4f", k.upper(), v)
            
            # Early stopping check
            early_stopping(
                avg_val_loss,
                model,
                epoch + 1,
                optimizer,
                scheduler,
                output_dirs,
                val_metrics
            )
            
            if early_stopping.early_stop:
                logger.info("\n[EARLY STOPPING]")
                logger.info("Training interrupted: No improvement in validation metrics")
                break
            
            # Move scheduler step to after validation but before checkpoint saving
            scheduler.step()  # Atualiza o learning rate para o próximo epoch
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate for next epoch: {current_lr:.6f}")

            # Save artifacts with new directory structure
            if (epoch + 1) % args.save_every == 0:
                save_training_artifacts(
                    epoch + 1,
                    images,
                    recon_images,
                    val_metrics,
                    output_dirs
                )
            
            # Save checkpoint with new directory structure
            is_best = avg_val_loss < best_val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                avg_val_loss,
                output_dirs,
                val_metrics,
                is_best
            )

            # Generate and save training report
            train_metrics = {
                "loss": avg_train_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            val_metrics_full = {
                "loss": avg_val_loss,
                "recon_loss": avg_val_recon_loss,
                "kl_loss": avg_val_kl_loss,
                **val_metrics
            }
            report = generate_training_report(
                epoch + 1,
                train_metrics,
                val_metrics_full,
                output_dirs,
                args
            )

        writer.close()
        logger.info("\n" + "=" * 100)
        logger.info("TRAINING COMPLETED")
        logger.info("Best validation loss: %.4f", early_stopping.best_loss)
        logger.info("=" * 100)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        cleanup_memory()
        raise

    finally:
        cleanup_memory()
        logger.info("Training session ended")

if __name__ == '__main__':
    freeze_support()
    main()
