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
    
    # Validate directories
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist!")
    
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

def check_system_memory(batch_size, image_size=(16, 16), num_channels=3):
    """Check if system has enough memory for training."""
    logger = logging.getLogger("LunarCore.memory")
    
    # Calculate approximate memory requirements
    sample_size = batch_size * image_size[0] * image_size[1] * num_channels * 4  # 4 bytes per float32
    model_size = 100 * 1024 * 1024  # Approximate model size in bytes (100MB)
    
    # Check RAM
    available_ram = psutil.virtual_memory().available
    required_ram = sample_size * 3  # Factor for safety margin
    
    if required_ram > available_ram:
        raise MemoryError(f"Not enough RAM. Required: {required_ram/1e9:.2f}GB, Available: {available_ram/1e9:.2f}GB")
    
    # Check CUDA memory if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_required = sample_size * 4  # Factor for GPU memory requirements
        
        if gpu_memory_required > gpu_memory:
            logger.warning(f"GPU memory might be insufficient. Required: {gpu_memory_required/1e9:.2f}GB, Available: {gpu_memory/1e9:.2f}GB")
    
    logger.info(f"Memory check passed. RAM Available: {available_ram/1e9:.2f}GB")
    return True

def optimize_num_workers():
    """Calculate optimal number of workers based on system CPU cores."""
    cpu_count = psutil.cpu_count(logical=False)
    if cpu_count is None:
        return 0
    return min(4, max(1, cpu_count - 1))  # Leave one core free for system

def cleanup_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    def __call__(self, val_loss, model, epoch, optimizer, scheduler, output_dirs, metrics=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, output_dirs, metrics)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, output_dirs, metrics)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer, scheduler, output_dirs, metrics=None):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics or {}
        }
        
        # Save best model
        best_model_path = os.path.join(output_dirs["checkpoints.best"], "best_model.pth")
        torch.save(checkpoint, best_model_path)
        
        # Save checkpoint with metrics in filename
        metrics_str = "_".join([f"{k}_{v:.4f}" for k, v in (metrics or {}).items()])
        checkpoint_name = f"checkpoint_epoch_{epoch}_valloss_{val_loss:.4f}"
        if metrics_str:
            checkpoint_name += f"_{metrics_str}"
        checkpoint_name += ".pth"
        
        checkpoint_path = os.path.join(output_dirs["checkpoints.periodic"], checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        
        self.val_loss_min = val_loss

# Dataset
class PixelArtDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Directory and image validations
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory {root_dir} does not exist!")
        
        # Comprehensive image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        
        self.image_files = [
            f for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f)) and 
               os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        # Detailed log about found images
        print(f"Images found in directory {root_dir}: {len(self.image_files)}")
        
        if len(self.image_files) == 0:
            raise ValueError(
                f"No images found in directory {root_dir}. "
                f"Please verify images are in correct format: {', '.join(image_extensions)}"
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, -1

def calculate_metrics(recon_images, images):
    """Calculate additional metrics for model evaluation."""
    with torch.no_grad():
        # Mean Absolute Error
        mae = torch.mean(torch.abs(recon_images - images)).item()
        
        # Peak Signal-to-Noise Ratio (PSNR)
        mse = torch.mean((recon_images - images) ** 2).item()
        psnr = 20 * np.log10(2.0) - 10 * np.log10(mse)  # 2.0 because images are in range [-1, 1]
        
        return {
            "mae": mae,
            "psnr": psnr
        }

def save_comparison_grid(epoch, original_images, generated_images, output_dir, num_images=8):
    """
    Save a grid of original vs generated images side by side.
    """
    # Create the comparisons directory if it doesn't exist
    comparisons_dir = os.path.join(output_dir, 'training_progress')
    os.makedirs(comparisons_dir, exist_ok=True)

    # Convert tensors to numpy arrays and move to CPU if necessary
    if torch.is_tensor(original_images):
        original_images = original_images.detach().cpu().float().numpy()
    if torch.is_tensor(generated_images):
        generated_images = generated_images.detach().cpu().float().numpy()

    # Take only the specified number of images
    original_images = original_images[:num_images]
    generated_images = generated_images[:num_images]

    # Create a figure with a grid of image pairs
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 2*num_images))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    for i in range(num_images):
        # Original image
        orig_img = np.transpose(original_images[i], (1, 2, 0))
        orig_img = (orig_img + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
        orig_img = np.clip(orig_img, 0, 1)  # Ensure values are in [0, 1]
        orig_img = orig_img.astype(np.float32)  # Convert to float32
        axes[i, 0].imshow(orig_img)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original')

        # Generated image
        gen_img = np.transpose(generated_images[i], (1, 2, 0))
        gen_img = (gen_img + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
        gen_img = np.clip(gen_img, 0, 1)  # Ensure values are in [0, 1]
        gen_img = gen_img.astype(np.float32)  # Convert to float32
        
        # Check if image is all gray
        if np.std(gen_img) < 0.01:
            logger = logging.getLogger("LunarCore")
            logger.warning(f"Generated image {i} appears to be mostly gray. Stats: mean={np.mean(gen_img):.4f}, std={np.std(gen_img):.4f}")
        
        axes[i, 1].imshow(gen_img)
        axes[i, 1].axis('off')
        if i == 0:
            axes[i, 1].set_title('Generated')

    # Add epoch information
    plt.suptitle(f'Training Progress - Epoch {epoch}', y=0.95)
    
    # Save the figure
    comparison_path = os.path.join(comparisons_dir, f'comparison_epoch_{epoch:04d}.png')
    plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
    plt.close()

    # Save individual images in high quality for detailed inspection
    details_dir = os.path.join(comparisons_dir, f'epoch_{epoch:04d}_details')
    os.makedirs(details_dir, exist_ok=True)

    # Convert back to tensors for save_image and normalize to [0, 1] range
    original_images_tensor = torch.from_numpy(original_images)
    generated_images_tensor = torch.from_numpy(generated_images)

    # Normalize tensors from [-1, 1] to [0, 1] before saving
    original_images_tensor = (original_images_tensor + 1) / 2
    generated_images_tensor = (generated_images_tensor + 1) / 2

    for i in range(num_images):
        # Save original
        save_image(
            original_images_tensor[i],
            os.path.join(details_dir, f'original_{i+1}.png'),
            normalize=False  # Already normalized to [0, 1]
        )
        # Save generated
        save_image(
            generated_images_tensor[i],
            os.path.join(details_dir, f'generated_{i+1}.png'),
            normalize=False  # Already normalized to [0, 1]
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
        ("Comparison outputs", os.path.join(output_dirs["outputs.comparisons"], "*")),
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

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load checkpoint with validation and fallback."""
    logger = logging.getLogger("LunarCore.checkpoint")
    
    try:
        logger.info(f"Loading checkpoint: {os.path.basename(checkpoint_path)}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Validate checkpoint contents
        required_keys = ["model_state_dict", "optimizer_state_dict", "epoch", "val_loss"]
        if not all(key in checkpoint for key in required_keys):
            raise ValueError("Checkpoint missing required keys")
            
        # Load model state with error checking
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as e:
            logger.error(f"Error loading model state: {str(e)}")
            raise
            
        # Load optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler if available
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", checkpoint["val_loss"])
        
        logger.info(f"Successfully loaded checkpoint from epoch {start_epoch}")
        logger.info(f"Validation loss: {checkpoint['val_loss']:.4f}")
        logger.info(f"Best loss so far: {best_val_loss:.4f}")
        
        return start_epoch, best_val_loss
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
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
        'logs': 'logs',                    # Log files
        'checkpoints': {
            'main': 'checkpoints',         # Main checkpoint directory
            'best': 'checkpoints/best',    # Best model checkpoints
            'periodic': 'checkpoints/periodic', # Regular interval checkpoints
            'interrupt': 'checkpoints/interrupt' # Interrupt checkpoints
        },
        'outputs': {
            'main': 'outputs',             # Main outputs directory
            'samples': 'outputs/samples',   # Generated samples
            'metrics': 'outputs/metrics',   # Training metrics
            'comparisons': 'outputs/comparisons' # Training progress comparisons
        },
        'tensorboard': 'runs'              # Tensorboard logs
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

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, output_dirs, metrics=None, is_best=False):
    """Save model checkpoint with improved organization."""
    logger = logging.getLogger("LunarCore.checkpoint")
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "metrics": metrics or {},
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Save periodic checkpoint
    metrics_str = "_".join([f"{k}_{v:.4f}" for k, v in (metrics or {}).items()])
    checkpoint_name = f"model_epoch_{epoch:04d}_loss_{val_loss:.4f}"
    if metrics_str:
        checkpoint_name += f"_{metrics_str}"
    checkpoint_name += ".pth"
    
    periodic_path = os.path.join(output_dirs["checkpoints.periodic"], checkpoint_name)
    torch.save(checkpoint, periodic_path)
    logger.info(f"Saved periodic checkpoint: {os.path.basename(periodic_path)}")
    
    # Save best model if applicable
    if is_best:
        best_path = os.path.join(output_dirs["checkpoints.best"], "best_model.pth")
        torch.save(checkpoint, best_path)
        # Also save a dated copy of the best model
        dated_best_path = os.path.join(
            output_dirs["checkpoints.best"],
            f"best_model_epoch_{epoch:04d}_loss_{val_loss:.4f}.pth"
        )
        shutil.copy2(best_path, dated_best_path)
        logger.info(f"Saved new best model: loss = {val_loss:.4f}")

def save_training_artifacts(epoch, images, recon_images, metrics, output_dirs):
    """Save training artifacts with improved organization."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comparison grid
    save_comparison_grid(
        epoch,
        images.cpu(),
        recon_images.cpu(),
        output_dirs["outputs.comparisons"],
        num_images=8
    )
    
    # Save individual samples
    samples_dir = os.path.join(
        output_dirs["outputs.samples"],
        f"epoch_{epoch:04d}"
    )
    os.makedirs(samples_dir, exist_ok=True)
    
    # Normalize tensors from [-1, 1] to [0, 1] before saving
    images = (images + 1) / 2
    recon_images = (recon_images + 1) / 2
    
    for i in range(min(8, len(images))):
        # Save original
        save_image(
            images[i],
            os.path.join(samples_dir, f'original_{i+1}.png'),
            normalize=False  # Already normalized
        )
        # Save reconstruction
        save_image(
            recon_images[i],
            os.path.join(samples_dir, f'reconstruction_{i+1}.png'),
            normalize=False  # Already normalized
        )
    
    # Save metrics
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
    
    # Save report
    report_file = os.path.join(
        output_dirs["outputs.metrics"],
        f"training_report_epoch_{epoch:04d}.json"
    )
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    
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
        "--batch_size", type=int, default=512, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=128, help="Latent space dimension"
    )
    parser.add_argument(
        "--kl_weight", type=float, default=0.0005, help="KL divergence weight"
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
        default=7,
        help="Number of epochs to wait for improvement before early stopping"
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.0001,
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
    logger = setup_logging(output_dirs["logs"])
    setup_pytorch_optimizations()
    logger.info("Starting LunarCore training session")

    try:
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

        # Loss function (MSE + KL Divergence)
        loss_fn = nn.MSELoss(reduction="mean")

        # Transformations
        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomCrop((16, 16)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Dataset and DataLoader
        try:
            dataset = PixelArtDataset(root_dir=data_dir, transform=transform)
        except ValueError as e:
            logger.error(f"ERROR loading dataset: {e}")
            sys.exit(1)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # DataLoaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=optimize_num_workers(),
            pin_memory=True,
            persistent_workers=True if optimize_num_workers() > 0 else False,
            prefetch_factor=2 if optimize_num_workers() > 0 else None,
            drop_last=True  # Ensure consistent batch sizes
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=optimize_num_workers(),
            pin_memory=True,
            persistent_workers=True if optimize_num_workers() > 0 else False,
            prefetch_factor=2 if optimize_num_workers() > 0 else None,
            drop_last=True  # Ensure consistent batch sizes
        )

        # Function to save images
        def save_images(epoch, input_img, generated_img, img_name):
            output_dir_epoch = os.path.join(output_dir, f"epoch_{epoch}")
            os.makedirs(output_dir_epoch, exist_ok=True)

            input_img = (input_img + 1) / 2
            generated_img = (generated_img + 1) / 2

            input_pil = transforms.ToPILImage()(input_img.squeeze(0))
            generated_pil = transforms.ToPILImage()(generated_img.squeeze(0))

            input_pil.save(os.path.join(output_dir_epoch, f"{img_name}_input.png"))
            generated_pil.save(os.path.join(output_dir_epoch, f"{img_name}_generated.png"))

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

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            eps=1e-8
        )

        # Scheduler with warmup
        def warmup_lambda(epoch):
            warmup_epochs = 5
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs  # Começa em 1/warmup_epochs e vai até 1
            return args.lr_gamma ** ((epoch - warmup_epochs) // args.lr_step_size)  # Decai após o warmup
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_lambda,
            verbose=False  # Não precisamos do log do scheduler pois já logamos manualmente
        )

        # Gradient scaler for mixed precision
        scaler = torch.amp.GradScaler() if args.mixed_precision else None

        # Load checkpoint if specified
        start_epoch = 0
        best_val_loss = float('inf')
        if args.checkpoint:
            logger.info(f"\n[Loading Checkpoint]")
            start_epoch, best_val_loss = load_checkpoint(args.checkpoint, model, optimizer, scheduler, device)
            
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

        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            verbose=True
        )

        # Training loop with mixed precision updates
        for epoch in range(start_epoch, epochs):
            logger.info("-" * 100)
            logger.info(f"EPOCH {epoch+1}/{epochs}")
            logger.info("-" * 100)
            
            # Training Phase
            logger.info("[TRAINING PHASE]")
            model.train()
            train_loss = 0.0
            max_grad_norm = 0.0
            
            # Log learning rate at the start of each epoch
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.6f}")

            for i, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Training Progress", leave=False)):
                images = images.to(device)
                optimizer.zero_grad(set_to_none=True)

                try:
                    if args.mixed_precision:
                        with torch.amp.autocast('cuda'):
                            recon_images, mean, logvar = model(images)
                            recon_loss = loss_fn(recon_images, images)
                            kl_divergence = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                            # Add KL annealing
                            kl_weight_current = min(1.0, epoch / 50) * kl_weight  # Slower annealing
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
                        recon_loss = loss_fn(recon_images, images)
                        kl_divergence = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                        # Add KL annealing
                        kl_weight_current = min(1.0, epoch / 50) * kl_weight  # Slower annealing
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
            
            # Validation Phase
            logger.info("\n[VALIDATION PHASE]")
            model.eval()
            val_loss = 0.0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_metrics = {"mae": 0.0, "psnr": 0.0}
            
            with torch.no_grad():
                for i, (images, labels) in enumerate(tqdm(val_dataloader, desc="Validation Progress", leave=False)):
                    images = images.to(device)

                    with torch.amp.autocast('cuda'):
                        recon_images, mean, logvar = model(images)
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
                        save_comparison_grid(epoch + 1, images.cpu(), recon_images.cpu(), output_dir, num_images=8)
            
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
