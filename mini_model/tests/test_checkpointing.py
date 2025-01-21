"""
Testes para o sistema de gerenciamento de checkpoints.
"""

import unittest
import tempfile
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timedelta
import shutil

from ..utils.checkpointing import CheckpointManager

class SimpleModel(nn.Module):
    """Modelo simples para testes."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

class TestCheckpointManager(unittest.TestCase):
    """Testes para o gerenciador de checkpoints."""
    
    def setUp(self):
        # Criar diretórios temporários
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.backup_dir = Path(self.temp_dir) / "backups"
        
        # Criar gerenciador
        self.manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            backup_dir=str(self.backup_dir),
            max_checkpoints=3,
            keep_best=True
        )
        
        # Criar modelo e otimizador para testes
        self.model = SimpleModel()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1
        )
    
    def test_initialization(self):
        """Testa inicialização do gerenciador."""
        self.assertEqual(self.manager.max_checkpoints, 3)
        self.assertTrue(self.manager.keep_best)
        self.assertTrue(self.checkpoint_dir.exists())
        self.assertTrue(self.backup_dir.exists())
    
    def test_checkpoint_saving(self):
        """Testa salvamento de checkpoints."""
        # Salvar checkpoint
        metadata = {"val_loss": 1.0}
        self.manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=0,
            metadata=metadata
        )
        
        # Verificar arquivo
        self.assertEqual(len(list(self.checkpoint_dir.glob("*.pt"))), 1)
        
        # Verificar metadados
        metadata_file = self.checkpoint_dir / "metadata.json"
        self.assertTrue(metadata_file.exists())
        
        with open(metadata_file) as f:
            saved_metadata = json.load(f)
            self.assertEqual(len(saved_metadata["checkpoints"]), 1)
            self.assertEqual(
                saved_metadata["best_val_loss"],
                metadata["val_loss"]
            )
    
    def test_checkpoint_loading(self):
        """Testa carregamento de checkpoints."""
        # Salvar estado inicial
        initial_state = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Modificar modelo significativamente
        with torch.no_grad():
            for param in self.model.parameters():
                param.add_(torch.ones_like(param) * 10.0)  # Adicionando um valor grande
        
        # Salvar checkpoint
        self.manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=0
        )
        
        # Carregar checkpoint mais recente
        checkpoint = self.manager.load_latest()
        self.assertIsNotNone(checkpoint)
        
        # Restaurar modelo ao estado inicial
        self.model.load_state_dict(initial_state)
        
        # Verificar que estado é diferente do checkpoint
        checkpoint_state = checkpoint["model_state"]
        for name, param in self.model.named_parameters():
            self.assertFalse(
                torch.allclose(param, checkpoint_state[name], rtol=1e-3),
                f"Parâmetro {name} não foi modificado como esperado"
            )
    
    def test_checkpoint_rotation(self):
        """Testa rotação de checkpoints."""
        # Salvar múltiplos checkpoints
        for i in range(5):  # Mais que max_checkpoints
            self.manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=i,
                metadata={"val_loss": float(5 - i)}  # Melhor loss no último
            )
        
        # Verificar número de checkpoints
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        self.assertEqual(len(checkpoints), self.manager.max_checkpoints)
        
        # Verificar backups
        backups = list(self.backup_dir.glob("*.pt"))
        self.assertEqual(len(backups), 2)  # 5 total - 3 ativos
    
    def test_best_checkpoint_handling(self):
        """Testa gerenciamento do melhor checkpoint."""
        # Salvar checkpoints com diferentes losses
        losses = [3.0, 1.0, 2.0, 4.0]
        for i, loss in enumerate(losses):
            self.manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=i,
                metadata={"val_loss": loss}
            )
        
        # Verificar melhor checkpoint
        self.assertEqual(self.manager.best_val_loss, min(losses))
        
        # Carregar melhor checkpoint
        best_checkpoint = self.manager.load_best()
        self.assertIsNotNone(best_checkpoint)
        self.assertEqual(
            best_checkpoint["metadata"]["val_loss"],
            min(losses)
        )
    
    def test_metadata_persistence(self):
        """Testa persistência de metadados."""
        # Salvar checkpoint com metadados
        metadata = {
            "val_loss": 1.0,
            "learning_rate": 0.001,
            "epoch": 10
        }
        self.manager.save(
            model=self.model,
            optimizer=self.optimizer,
            epoch=10,
            metadata=metadata
        )
        
        # Criar novo gerenciador
        new_manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            backup_dir=str(self.backup_dir)
        )
        
        # Verificar metadados carregados
        self.assertEqual(new_manager.best_val_loss, metadata["val_loss"])
        self.assertEqual(len(new_manager.checkpoints), 1)
        
        checkpoint_info = new_manager.get_checkpoint_info()
        saved_metadata = list(checkpoint_info["checkpoints"].values())[0]["metadata"]
        self.assertEqual(saved_metadata["val_loss"], metadata["val_loss"])
        self.assertEqual(saved_metadata["learning_rate"], metadata["learning_rate"])
    
    def test_checkpoint_cleanup(self):
        """Testa limpeza de checkpoints antigos."""
        # Salvar checkpoints com timestamps diferentes
        base_time = datetime.now()
        
        for i in range(5):
            # Simular checkpoint antigo
            name = self.manager._get_checkpoint_name(i)
            path = self.checkpoint_dir / name
            
            # Salvar checkpoint
            self.manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=i
            )
            
            # Modificar timestamp nos metadados
            timestamp = (base_time - timedelta(days=i)).timestamp()
            self.manager.checkpoints[name]["timestamp"] = timestamp
        
        # Forçar rotação
        self.manager._rotate_checkpoints()
        
        # Verificar que checkpoints mais antigos foram movidos
        active_checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        self.assertEqual(len(active_checkpoints), self.manager.max_checkpoints)
        
        backup_checkpoints = list(self.backup_dir.glob("*.pt"))
        self.assertEqual(len(backup_checkpoints), 2)
    
    def test_error_handling(self):
        """Testa tratamento de erros."""
        # Testar carregamento de checkpoint inexistente
        result = self.manager.load("invalid.pt")
        self.assertIsNone(result)
        
        # Testar carregamento sem checkpoints
        result = self.manager.load_latest()
        self.assertIsNone(result)
        
        result = self.manager.load_best()
        self.assertIsNone(result)
    
    def tearDown(self):
        """Limpa arquivos temporários."""
        shutil.rmtree(self.temp_dir) 