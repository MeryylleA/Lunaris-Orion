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

# Importar o modelo do arquivo lunar_core_model.py
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
    # Argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Treinamento do LunarCoreVAE")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Caminho para o checkpoint a ser carregado (opcional)",
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Número de épocas para treinar"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Tamanho do batch"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Taxa de aprendizado"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=128, help="Dimensão do espaço latente"
    )
    parser.add_argument(
        "--kl_weight", type=float, default=0.0005, help="Peso da divergência KL"
    )
    parser.add_argument(
        "--data_dir", type=str, default=".", help="Diretório dos dados de treinamento"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Diretório de saída"
    )
    parser.add_argument(
        "--save_every", type=int, default=10, help="Salvar checkpoint a cada X épocas"
    )
    parser.add_argument(
        "--lr_step_size", type=int, default=10, help="Reduzir lr a cada X épocas"
    )
    parser.add_argument(
        "--lr_gamma", type=float, default=0.8, help="Fator de redução da lr"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Número de workers para o DataLoader"
    )

    args = parser.parse_args()

    # Hiperparâmetros
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    latent_dim = args.latent_dim
    kl_weight = args.kl_weight
    data_dir = args.data_dir
    output_dir = args.output_dir
    save_every = args.save_every
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Função de perda (MSE + KL Divergence)
    loss_fn = nn.MSELoss(reduction="mean")

    # Transformações
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.Pad(2, padding_mode='reflect'),
        transforms.RandomCrop((16, 16)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset e DataLoader
    dataset = PixelArtDataset(root_dir=data_dir, transform=transform)
    
    # Dividir dataset
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

    # Função para salvar as imagens
    def save_images(epoch, input_img, generated_img, img_name):
        output_dir_epoch = os.path.join(output_dir, f"epoch_{epoch}")
        os.makedirs(output_dir_epoch, exist_ok=True)

        input_img = (input_img + 1) / 2
        generated_img = (generated_img + 1) / 2

        input_pil = transforms.ToPILImage()(input_img.squeeze(0))
        generated_pil = transforms.ToPILImage()(generated_img.squeeze(0))

        input_pil.save(os.path.join(output_dir_epoch, f"{img_name}_input.png"))
        generated_pil.save(os.path.join(output_dir_epoch, f"{img_name}_generated.png"))

    # Instanciar o modelo
    model = LunarCoreVAE(latent_dim=latent_dim).to(device)

    # Inicialização dos pesos
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.2)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    # Otimizador (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning Rate Scheduler
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    # Carregar checkpoint, se especificado
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
        print(f"Checkpoint carregado da época {start_epoch}")
        print(f"Loss: {loaded_loss:.4f}")
        if "val_loss" in checkpoint:
            print(f"Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"Melhor perda de validação até agora: {best_val_loss:.4f}")

    # TensorBoard
    writer = SummaryWriter(f"runs/lunar_core_experiment_{start_epoch}")

    # Ativar o benchmark do cuDNN
    torch.backends.cudnn.benchmark = True

    # Loop de treinamento
    for epoch in range(start_epoch, epochs):
        # Modo de treinamento
        model.train()
        train_loss = 0.0
        
        # Loop de treinamento
        for i, (images, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Train)")):
            images = images.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                recon_images, mean, logvar = model(images)
                recon_loss = loss_fn(recon_images, images)
                kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                loss = recon_loss + kl_weight * kl_divergence

            # Backprop com scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # Imprimir estatísticas
            if (i + 1) % 10 == 0:  # Reduzir a frequência de impressão
                print(
                    f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Div: {kl_divergence.item():.4f}"
                )

            # Adicionar métricas ao TensorBoard
            step = epoch * len(train_dataloader) + i
            writer.add_scalar('Loss/Train/Total', loss.item(), step)
            writer.add_scalar('Loss/Train/Reconstruction', recon_loss.item(), step)
            writer.add_scalar('Loss/Train/KL_Divergence', kl_divergence.item(), step)

        # Calcular perda média de treinamento
        train_loss /= len(train_dataloader)
        
        # Modo de validação
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

                # Salvar imagens de validação (primeira batch apenas)
                if i == 0 and (epoch + 1) % 5 == 0:
                    img_name = f"val_batch_{epoch+1}"
                    save_images(epoch + 1, images[0].cpu(), recon_images[0].cpu(), img_name)
                    writer.add_images('Images/Val/Original', images, epoch + 1)
                    writer.add_images('Images/Val/Reconstructed', recon_images, epoch + 1)

        # Calcular perdas médias de validação
        val_loss /= len(val_dataloader)
        val_recon_loss /= len(val_dataloader)
        val_kl_loss /= len(val_dataloader)

        # Adicionar métricas de validação ao TensorBoard
        writer.add_scalar('Loss/Val/Total', val_loss, epoch)
        writer.add_scalar('Loss/Val/Reconstruction', val_recon_loss, epoch)
        writer.add_scalar('Loss/Val/KL_Divergence', val_kl_loss, epoch)

        # Imprimir estatísticas da época
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Salvar o melhor modelo
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
            print(f"Melhor modelo salvo em {best_model_path} com perda de validação: {val_loss:.4f}")

        # Atualizar a taxa de aprendizado
        scheduler.step()

        # Salvar checkpoint regular
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
            print(f"Checkpoint salvo em {checkpoint_path}")

    # Fechar o writer do TensorBoard
    writer.close()

    print("Treinamento concluído!")
    print(f"Melhor perda de validação: {best_val_loss:.4f}")

if __name__ == '__main__':
    freeze_support()
    main()
