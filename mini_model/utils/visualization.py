"""
Utilitários para visualização de imagens durante treinamento.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from torchvision.utils import make_grid
import wandb

class VisualizationManager:
    """Gerencia visualizações durante treinamento."""
    
    def __init__(self, output_dir: str, num_samples: int = 8):
        """
        Args:
            output_dir: Diretório para salvar visualizações
            num_samples: Número de amostras para visualizar
        """
        self.output_dir = Path(output_dir) / "visualizations"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        
        # Criar subdiretórios
        (self.output_dir / "train").mkdir(exist_ok=True)
        (self.output_dir / "val").mkdir(exist_ok=True)
        (self.output_dir / "test").mkdir(exist_ok=True)
        (self.output_dir / "progress").mkdir(exist_ok=True)
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Converte tensor para imagem numpy."""
        # Converter de [-1, 1] para [0, 1]
        img = (tensor.detach() + 1) / 2
        img = img.clamp(0, 1)
        
        # Converter para numpy
        if len(img.shape) == 4:
            img = img[0]  # Pegar primeira imagem do batch
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        
        return img
    
    def save_image_grid(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        style_labels: torch.Tensor,
        phase: str,
        epoch: int,
        batch_idx: int
    ):
        """Salva grid de imagens reais e geradas."""
        # Selecionar subset de imagens
        n = min(self.num_samples, real_images.size(0))
        real_images = real_images[:n].detach()
        generated_images = generated_images[:n].detach()
        style_labels = style_labels[:n].detach()
        
        # Criar grid
        real_grid = make_grid(real_images, nrow=n, normalize=True, padding=2)
        gen_grid = make_grid(generated_images, nrow=n, normalize=True, padding=2)
        
        # Criar figura
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Plotar imagens reais
        ax1.imshow(self._tensor_to_image(real_grid))
        ax1.set_title("Imagens Reais")
        ax1.axis("off")
        
        # Plotar imagens geradas
        ax2.imshow(self._tensor_to_image(gen_grid))
        ax2.set_title("Imagens Geradas")
        ax2.axis("off")
        
        # Adicionar informações de estilo
        style_info = [f"Style {s.argmax().item()}" for s in style_labels]
        fig.suptitle(f"Epoch {epoch}, Batch {batch_idx}\nEstilos: {', '.join(style_info)}")
        
        # Salvar
        save_path = self.output_dir / phase / f"epoch_{epoch:03d}_batch_{batch_idx:04d}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        # Logging no wandb
        if wandb.run is not None:
            wandb.log({
                f"{phase}/examples": wandb.Image(str(save_path)),
                "epoch": epoch
            })
    
    def save_progress_grid(
        self,
        images: List[Tuple[torch.Tensor, torch.Tensor]],
        epoch: int
    ):
        """Salva grid mostrando progresso do treinamento."""
        n_pairs = len(images)
        fig, axes = plt.subplots(2, n_pairs, figsize=(4*n_pairs, 8))
        
        for i, (real, gen) in enumerate(images):
            # Plotar imagem real
            axes[0, i].imshow(self._tensor_to_image(real.detach()))
            axes[0, i].set_title(f"Real {i+1}")
            axes[0, i].axis("off")
            
            # Plotar imagem gerada
            axes[1, i].imshow(self._tensor_to_image(gen.detach()))
            axes[1, i].set_title(f"Gerada {i+1}")
            axes[1, i].axis("off")
        
        plt.suptitle(f"Progresso do Treinamento - Epoch {epoch}")
        
        # Salvar
        save_path = self.output_dir / "progress" / f"epoch_{epoch:03d}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        # Logging no wandb
        if wandb.run is not None:
            wandb.log({
                "progress/examples": wandb.Image(str(save_path)),
                "epoch": epoch
            })
    
    def visualize_attention(
        self,
        attention_maps: torch.Tensor,
        images: torch.Tensor,
        epoch: int,
        phase: str = "train"
    ):
        """Visualiza mapas de atenção."""
        n = min(self.num_samples, images.size(0))
        
        fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
        
        for i in range(n):
            # Plotar imagem original
            axes[0, i].imshow(self._tensor_to_image(images[i].detach()))
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis("off")
            
            # Plotar mapa de atenção
            # Calcular média sobre as heads e reduzir para 2D
            attention = attention_maps[i]  # [H, W, head, head]
            attention = attention.mean(dim=(2, 3))  # Média sobre as heads [H, W]
            axes[1, i].imshow(attention.cpu().numpy(), cmap="hot")
            axes[1, i].set_title(f"Atenção {i+1}")
            axes[1, i].axis("off")
        
        plt.suptitle(f"Mapas de Atenção - Epoch {epoch}")
        
        # Salvar
        save_path = self.output_dir / phase / f"attention_epoch_{epoch:03d}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        # Logging no wandb
        if wandb.run is not None:
            wandb.log({
                f"{phase}/attention": wandb.Image(str(save_path)),
                "epoch": epoch
            }) 