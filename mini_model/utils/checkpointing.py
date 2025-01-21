"""
Gerenciamento de checkpoints do modelo com backup e rotação.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from datetime import datetime

class CheckpointManager:
    """Gerenciador de checkpoints com backup e rotação."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        backup_dir: str,
        max_checkpoints: int = 5,
        keep_best: bool = True
    ):
        """
        Args:
            checkpoint_dir: Diretório para checkpoints ativos
            backup_dir: Diretório para backups
            max_checkpoints: Número máximo de checkpoints mantidos
            keep_best: Se deve manter sempre o melhor checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.backup_dir = Path(backup_dir)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        
        # Criar diretórios
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Estado
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        self.best_checkpoint: Optional[str] = None
        self.best_val_loss: float = float("inf")
        
        # Carregar metadados existentes
        self._load_metadata()
    
    def _load_metadata(self):
        """Carrega metadados de checkpoints existentes."""
        metadata_file = self.checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                data = json.load(f)
                self.checkpoints = data["checkpoints"]
                self.best_checkpoint = data.get("best_checkpoint")
                self.best_val_loss = data.get("best_val_loss", float("inf"))
    
    def _save_metadata(self):
        """Salva metadados dos checkpoints."""
        metadata_file = self.checkpoint_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump({
                "checkpoints": self.checkpoints,
                "best_checkpoint": self.best_checkpoint,
                "best_val_loss": self.best_val_loss
            }, f, indent=2)
    
    def _get_checkpoint_name(self, epoch: int) -> str:
        """Gera nome para novo checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"checkpoint_epoch_{epoch:03d}_{timestamp}.pt"
    
    def _rotate_checkpoints(self):
        """Remove checkpoints antigos mantendo limite máximo."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
            
        # Ordenar por data
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1]["timestamp"]
        )
        
        # Manter N mais recentes e o melhor
        keep_count = self.max_checkpoints
        if self.keep_best and self.best_checkpoint:
            keep_count -= 1
        
        # Remover mais antigos
        for name, _ in sorted_checkpoints[:-keep_count]:
            if name == self.best_checkpoint:
                continue
                
            # Mover para backup
            src = self.checkpoint_dir / name
            dst = self.backup_dir / name
            if src.exists():
                shutil.move(str(src), str(dst))
            
            del self.checkpoints[name]
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Salva checkpoint do modelo.
        
        Args:
            model: Modelo PyTorch
            optimizer: Otimizador
            scheduler: Learning rate scheduler
            epoch: Época atual
            metadata: Metadados adicionais
        """
        # Gerar nome
        name = self._get_checkpoint_name(epoch)
        path = self.checkpoint_dir / name
        
        # Preparar estado
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch
        }
        
        if scheduler is not None:
            state["scheduler_state"] = scheduler.state_dict()
            
        if metadata:
            state["metadata"] = metadata
        
        # Salvar checkpoint
        torch.save(state, path)
        
        # Atualizar metadados
        self.checkpoints[name] = {
            "epoch": epoch,
            "timestamp": datetime.now().timestamp(),
            "metadata": metadata or {}
        }
        
        # Atualizar melhor modelo
        val_loss = metadata.get("val_loss") if metadata else None
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_checkpoint = name
        
        # Rotação
        self._rotate_checkpoints()
        
        # Salvar metadados
        self._save_metadata()
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Carrega checkpoint mais recente."""
        if not self.checkpoints:
            return None
            
        # Encontrar mais recente
        latest = max(
            self.checkpoints.items(),
            key=lambda x: x[1]["timestamp"]
        )
        
        return self.load(latest[0])
    
    def load_best(self) -> Optional[Dict[str, Any]]:
        """Carrega melhor checkpoint."""
        if not self.best_checkpoint:
            return None
            
        return self.load(self.best_checkpoint)
    
    def load(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Carrega checkpoint específico.
        
        Args:
            name: Nome do arquivo de checkpoint
            
        Returns:
            Dict com estados do modelo, otimizador e metadados
        """
        path = self.checkpoint_dir / name
        if not path.exists():
            path = self.backup_dir / name
            if not path.exists():
                return None
        
        return torch.load(path)
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Retorna informações sobre checkpoints salvos."""
        return {
            "total_checkpoints": len(self.checkpoints),
            "best_checkpoint": self.best_checkpoint,
            "best_val_loss": self.best_val_loss,
            "checkpoints": self.checkpoints
        } 