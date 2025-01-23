"""
Utilitários para gerenciamento de checkpoints.
"""

import os
from pathlib import Path
import torch
from typing import Optional, Dict, Any

class CheckpointManager:
    """Gerencia salvamento e carregamento de checkpoints."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        save_best_only: bool = True,
        max_checkpoints: int = 5
    ):
        """
        Args:
            checkpoint_dir: Diretório para salvar checkpoints
            model: Modelo PyTorch
            optimizer: Otimizador
            scheduler: Learning rate scheduler (opcional)
            save_best_only: Se deve salvar apenas o melhor modelo
            max_checkpoints: Número máximo de checkpoints a manter
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_best_only = save_best_only
        self.max_checkpoints = max_checkpoints
        
        self.best_val_loss = float('inf')
        self.checkpoints = []
    
    def save(self, epoch: int, val_loss: float) -> None:
        """Salva checkpoint."""
        # Verificar se deve salvar
        if self.save_best_only and val_loss >= self.best_val_loss:
            return
        
        # Atualizar melhor loss
        self.best_val_loss = min(val_loss, self.best_val_loss)
        
        # Criar checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Salvar checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, str(checkpoint_path))
        
        # Adicionar à lista de checkpoints
        self.checkpoints.append(checkpoint_path)
        
        # Remover checkpoints antigos se necessário
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
    
    def load(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Carrega checkpoint."""
        if checkpoint_path is None:
            # Carregar último checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
            if not checkpoints:
                return {}
            checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = Path(checkpoint_path)
        
        # Carregar checkpoint
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        
        # Restaurar estados
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint 