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

# Import the model from the lunar_core_model.py file
from lunar_core_model import LunarCoreVAE

# Dataset
class PixelArtDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            f
            for f in os.listdir(root_dir)
            if f.endswith((".jpg", ".JPEG", ".png"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, -1

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Training of LunarCoreVAE")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint to be loaded (optional)",
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
        "--latent_dim", type=int, default=128, help="Dimension of the latent space"
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

    args = parser.parse_args()

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
    dataset = PixelArtDataset(root_dir=data_dir, transform=transform)
    
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

    # Instantiate the model
    model = LunarCoreVAE(latent_dim=latent_dim).to(device)

    # Weight initialization
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    # Optimizer (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning Rate Scheduler
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    # Load checkpoint, if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        if "best_val_loss" in checkpoint:
            best_val_loss = checkpoint["best_val_loss"]
        loaded_loss = checkpoint["loss"]
        print(f"Checkpoint loaded from epoch {start_epoch}")
        print(f"Loss: {loaded_loss:.4f}")
        if "val_loss" in checkpoint:
            print(f"Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"Best validation loss so far: {best_val_loss:.4f}")

    # TensorBoard
    writer = SummaryWriter(f"runs/lunar_core_experiment_{start_epoch}")

    # Enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    # Training loop
    for epoch in range(start_epoch, epochs):
        # Training mode
        model.train()
        train_loss = 0.0
        
        # Training loop
        for i, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Train)")):
            images = images.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                recon_images, mean, logvar = model(images)
                recon_loss = loss_fn(recon_images, images)
                kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                loss = recon_loss + kl_weight * kl_divergence

            # Backprop with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # Print statistics
            if (i + 1) % 10 == 0:  # Reduce print frequency
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Div: {kl_divergence.item():.4f}"
                )

            # Add metrics to TensorBoard
            step = epoch * len(train_dataloader) + i
            writer.add_scalar('Loss/Train/Total', loss.item(), step)
            writer.add_scalar('Loss/Train/Reconstruction', recon_loss.item(), step)
            writer.add_scalar('Loss/Train/KL_Divergence', kl_divergence.item(), step)

        # Calculate average training loss
        train_loss /= len(train_dataloader)
        
        # Validation mode
        model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        
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

                # Save validation images (first batch only)
                if i == 0 and (epoch + 1) % 5 == 0:
                    img_name = f"val_batch_{epoch+1}"
                    save_images(epoch + 1, images[0].cpu(), recon_images[0].cpu(), img_name)
                    writer.add_images('Images/Val/Original', images, epoch + 1)
                    writer.add_images('Images/Val/Reconstructed', recon_images, epoch + 1)

        # Calculate average validation losses
        val_loss /= len(val_dataloader)
        val_recon_loss /= len(val_dataloader)
        val_kl_loss /= len(val_dataloader)

        # Add validation metrics to TensorBoard
        writer.add_scalar('Loss/Val/Total', val_loss, epoch)
        writer.add_scalar('Loss/Val/Reconstruction', val_recon_loss, epoch)
        writer.add_scalar('Loss/Val/KL_Divergence', val_kl_loss, epoch)

        # Print epoch statistics
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, "best_model.pth")
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": val_loss,
                "best_val_loss": best_val_loss,
            }, best_model_path)
            print(f"Best model saved at {best_model_path} with validation loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step()

        # Save regular checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            os.makedirs(output_dir, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # Close TensorBoard writer
    writer.close()

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    freeze_support()
    main()
