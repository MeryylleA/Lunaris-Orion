import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
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

# Import the model from lunar_core_model.py
from lunar_core_model import LunarCoreVAE

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

    def __call__(self, val_loss, model, epoch, optimizer, scheduler, output_dir, metrics=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, output_dir, metrics)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch, optimizer, scheduler, output_dir, metrics)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer, scheduler, output_dir, metrics=None):
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
        best_model_path = os.path.join(output_dir, "best_model.pth")
        torch.save(checkpoint, best_model_path)
        
        # Save checkpoint with metrics in filename
        metrics_str = "_".join([f"{k}_{v:.4f}" for k, v in (metrics or {}).items()])
        checkpoint_name = f"checkpoint_epoch_{epoch}_valloss_{val_loss:.4f}"
        if metrics_str:
            checkpoint_name += f"_{metrics_str}"
        checkpoint_name += ".pth"
        
        checkpoint_path = os.path.join(output_dir, checkpoint_name)
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
        original_images = original_images.cpu().numpy()
    if torch.is_tensor(generated_images):
        generated_images = generated_images.cpu().numpy()

    # Take only the specified number of images
    original_images = original_images[:num_images]
    generated_images = generated_images[:num_images]

    # Ensure arrays are float32 and handle NaN/Inf values
    original_images = np.nan_to_num(original_images.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    generated_images = np.nan_to_num(generated_images.astype(np.float32), nan=0.0, posinf=1.0, neginf=-1.0)

    # Create a figure with a grid of image pairs
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 2*num_images))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    for i in range(num_images):
        # Original image
        orig_img = np.transpose(original_images[i], (1, 2, 0))
        orig_img = (orig_img + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
        orig_img = np.clip(orig_img, 0, 1)  # Ensure values are in [0, 1]
        axes[i, 0].imshow(orig_img)
        axes[i, 0].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original')

        # Generated image
        gen_img = np.transpose(generated_images[i], (1, 2, 0))
        gen_img = (gen_img + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]
        gen_img = np.clip(gen_img, 0, 1)  # Ensure values are in [0, 1]
        
        # Check if image is all black
        if np.mean(gen_img) < 0.01:
            print(f"Warning: Generated image {i} appears to be mostly black. Raw stats: mean={np.mean(gen_img):.4f}, std={np.std(gen_img):.4f}")
        
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

    for i in range(num_images):
        # Save original
        orig_img = np.transpose(original_images[i], (1, 2, 0))
        orig_img = ((orig_img + 1) / 2.0 * 255).astype(np.uint8)
        orig_img = np.clip(orig_img, 0, 255)
        Image.fromarray(orig_img).save(
            os.path.join(details_dir, f'original_{i+1}.png')
        )

        # Save generated
        gen_img = np.transpose(generated_images[i], (1, 2, 0))
        gen_img = ((gen_img + 1) / 2.0 * 255).astype(np.uint8)
        gen_img = np.clip(gen_img, 0, 255)
        Image.fromarray(gen_img).save(
            os.path.join(details_dir, f'generated_{i+1}.png')
        )

def find_best_checkpoint(output_dir):
    """
    Find the best checkpoint based on validation loss.
    """
    # Search for all checkpoints
    checkpoint_pattern = os.path.join(output_dir, "checkpoint_epoch_*.pth")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
        
    # Extract validation loss from filename
    best_checkpoint = None
    best_loss = float('inf')
    
    for ckpt in checkpoints:
        try:
            # Extract loss from filename (format: checkpoint_epoch_X_valloss_Y.pth)
            val_loss = float(ckpt.split('valloss_')[1].split('_')[0])
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint = ckpt
        except:
            continue
    
    return best_checkpoint

def cleanup_old_files(output_dir, keep_best=True):
    """
    Clean up old training files, keeping only the best checkpoint if specified.
    """
    print("\n[File Cleanup]")
    
    # List of directories and files to check
    cleanup_targets = [
        ("Training progress directory", os.path.join(output_dir, "training_progress")),
        ("Old checkpoints", os.path.join(output_dir, "checkpoint_epoch_*.pth")),
        ("Interrupt checkpoints", os.path.join(output_dir, "interrupt_checkpoint.pth"))
    ]
    
    total_space_saved = 0
    
    for desc, path in cleanup_targets:
        if "*" in path:  # For glob patterns
            files = glob.glob(path)
            if keep_best and "checkpoint" in path:
                # Keep the best checkpoint
                best_checkpoint = find_best_checkpoint(output_dir)
                if best_checkpoint in files:
                    files.remove(best_checkpoint)
        else:  # For specific directories
            files = [path] if os.path.exists(path) else []
        
        for f in files:
            try:
                size = 0
                if os.path.isfile(f):
                    size = os.path.getsize(f)
                    os.remove(f)
                elif os.path.isdir(f):
                    size = sum(os.path.getsize(os.path.join(dirpath,filename))
                             for dirpath, dirnames, filenames in os.walk(f)
                             for filename in filenames)
                    shutil.rmtree(f)
                total_space_saved += size
                print(f"  → Removed: {f}")
            except Exception as e:
                print(f"  → Error removing {f}: {str(e)}")
    
    # Convert bytes to MB or GB for better readability
    if total_space_saved > 1024*1024*1024:
        space_str = f"{total_space_saved/(1024*1024*1024):.2f} GB"
    else:
        space_str = f"{total_space_saved/(1024*1024):.2f} MB"
    
    print(f"\nTotal space freed: {space_str}")

def main():
    # Add signal handler for graceful interruption
    def signal_handler(signum, frame):
        print("\nInterrupting training... Saving checkpoint and cleaning up...")
        if 'model' in locals() and 'optimizer' in locals() and 'epoch' in locals():
            checkpoint_path = os.path.join(args.output_dir, "interrupt_checkpoint.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if 'scheduler' in locals() else None,
                "loss": train_loss if 'train_loss' in locals() else None,
                "val_loss": val_loss if 'val_loss' in locals() else None,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        if 'writer' in locals():
            writer.close()
        sys.exit(0)

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
        "--num_workers", type=int, default=0, help="Number of workers for DataLoader"
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

    args = parser.parse_args()

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory {args.data_dir} does not exist!")
        print("Please verify the image directory path.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check for old files to clean up
    has_old_files = (
        os.path.exists(os.path.join(args.output_dir, "training_progress")) or
        len(glob.glob(os.path.join(args.output_dir, "checkpoint_epoch_*.pth"))) > 0 or
        os.path.exists(os.path.join(args.output_dir, "interrupt_checkpoint.pth"))
    )

    if has_old_files:
        print("\n[Old Files Detected]")
        user_input = input("Do you want to clean up files from previous training sessions? (y/n): ")
        if user_input.lower() == 'y':
            cleanup_old_files(args.output_dir, keep_best=True)
            print("Cleanup completed.")
        else:
            print("Keeping old files.")

    # Check and load best checkpoint automatically if not specified
    if args.checkpoint is None:
        best_checkpoint = find_best_checkpoint(args.output_dir)
        if best_checkpoint:
            print(f"\n[Automatic Checkpoint]")
            print(f"Found best checkpoint: {os.path.basename(best_checkpoint)}")
            user_input = input("Do you want to continue training from this checkpoint? (y/n): ")
            if user_input.lower() == 'y':
                args.checkpoint = best_checkpoint
                print("Using found checkpoint.")
            else:
                print("Starting new training.")

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
        print(f"ERROR loading dataset: {e}")
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
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
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
    
    # Initialize weights properly
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(weights_init)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,  # L2 regularization
        eps=1e-8  # For numerical stability
    )

    # Scheduler with warmup
    def warmup_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return args.lr_gamma ** ((epoch - warmup_epochs) // args.lr_step_size)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)

    # Gradient scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda')

    # Load checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.checkpoint:
        print(f"\n[Loading Checkpoint]")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        if "best_val_loss" in checkpoint:
            best_val_loss = checkpoint["best_val_loss"]
        
        print(f"→ Starting epoch: {start_epoch}")
        if "val_loss" in checkpoint:
            print(f"→ Validation loss: {checkpoint['val_loss']:.4f}")
        print(f"→ Best loss so far: {best_val_loss:.4f}")
        
        # Adjust learning rate based on loaded epoch
        for _ in range(start_epoch):
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"→ Current learning rate: {current_lr:.6f}")
    else:
        print("\n[New Training]")
        print("Starting training from scratch.")

    # TensorBoard
    writer = SummaryWriter(f"runs/lunar_core_experiment_{start_epoch}")

    # Enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True
    )

    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")
        
        # Training Phase
        print("\n[Training Phase]")
        model.train()
        train_loss = 0.0
        max_grad_norm = 0.0
        
        for i, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Train)")):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)

            try:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    recon_images, mean, logvar = model(images)
                    recon_loss = loss_fn(recon_images, images)
                    kl_divergence = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
                    kl_weight_current = min(1.0, epoch / 10) * kl_weight
                    loss = recon_loss + kl_weight_current * kl_divergence

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Warning: NaN/Inf loss detected in batch {i+1}")
                    print(f"  → Recon Loss: {recon_loss.item():.4f}")
                    print(f"  → KL Div: {kl_divergence.item():.4f}")
                    continue

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                if (i + 1) % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"\n  Batch {i+1}/{len(train_dataloader)}:")
                    print(f"  → Loss: {loss.item():.4f}")
                    print(f"  → Recon Loss: {recon_loss.item():.4f}")
                    print(f"  → KL Div: {kl_divergence.item():.4f}")
                    print(f"  → Learning Rate: {current_lr:.6f}")

            except RuntimeError as e:
                print(f"\n  Error in batch {i+1}: {str(e)}")
                continue

        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation Phase
        print("\n[Validation Phase]")
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        val_metrics = {"mae": 0.0, "psnr": 0.0}
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Val)")):
                images = images.to(device)

                with torch.cuda.amp.autocast():
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
        print("\n[Epoch Summary]")
        print(f"  Training:")
        print(f"  → Average Loss: {avg_train_loss:.4f}")
        print(f"\n  Validation:")
        print(f"  → Total Loss: {avg_val_loss:.4f}")
        print(f"  → Recon Loss: {avg_val_recon_loss:.4f}")
        print(f"  → KL Loss: {avg_val_kl_loss:.4f}")
        print(f"\n  Validation Metrics:")
        for k, v in val_metrics.items():
            print(f"  → {k}: {v:.4f}")
        
        # Early stopping check
        early_stopping(
            avg_val_loss,
            model,
            epoch + 1,
            optimizer,
            scheduler,
            output_dir,
            val_metrics
        )
        
        if early_stopping.early_stop:
            print("\n[Early Stopping]")
            print("Training interrupted: No improvement in validation")
            break
        
        scheduler.step()

    writer.close()
    print("\nTraining completed!")
    print(f"Best validation loss: {early_stopping.best_loss:.4f}")

if __name__ == '__main__':
    freeze_support()
    main()
