"""
Testes para o sistema de treinamento multi-GPU.
"""

import unittest
import torch
import tempfile
from pathlib import Path
import yaml
import numpy as np
from unittest.mock import MagicMock, patch
import os
import shutil

from ..training.trainer import GPUManager, MultiGPUTrainer, Trainer, train_distributed
from ..core.mini_arch import Pixel16Generator
from ..data.pixel16_dataset import Pixel16Dataset

def train_distributed(config_path: str, data_dir: str, output_dir: str, world_size: int):
    """Função auxiliar para treinamento distribuído."""
    pass  # Mock para testes

class TestGPUManager(unittest.TestCase):
    """Testes para o gerenciador de GPUs."""
    
    def setUp(self):
        self.has_cuda = torch.cuda.is_available()
    
    def test_get_gpu_info(self):
        """Testa coleta de informações das GPUs."""
        gpu_info = GPUManager.get_gpu_info()
        
        if self.has_cuda:
            self.assertGreater(len(gpu_info), 0)
            for gpu in gpu_info:
                self.assertIn("index", gpu)
                self.assertIn("name", gpu)
                self.assertIn("total_memory", gpu)
                self.assertIn("free_memory", gpu)
                self.assertIn("compute_capability", gpu)
        else:
            self.assertEqual(len(gpu_info), 0)
    
    def test_select_best_gpus(self):
        """Testa seleção das melhores GPUs."""
        if not self.has_cuda:
            gpus = GPUManager.select_best_gpus()
            self.assertEqual(len(gpus), 0)
            return
            
        # Testar seleção automática
        gpus = GPUManager.select_best_gpus()
        self.assertEqual(len(gpus), torch.cuda.device_count())
        
        # Testar seleção com limite
        num_gpus = min(2, torch.cuda.device_count())
        gpus = GPUManager.select_best_gpus(num_gpus)
        self.assertEqual(len(gpus), num_gpus)
    
    def test_optimize_gpu_settings(self):
        """Testa otimização de configurações de GPU."""
        if not self.has_cuda:
            GPUManager.optimize_gpu_settings()  # Não deve lançar erro
            return
            
        GPUManager.optimize_gpu_settings()
        
        # Verificar configurações
        self.assertTrue(torch.backends.cudnn.enabled)
        self.assertTrue(torch.backends.cudnn.benchmark)
        
        # Verificar TF32 em GPUs compatíveis
        if torch.cuda.get_device_capability()[0] >= 8:
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)

class TestMultiGPUTrainer(unittest.TestCase):
    """Testes para o treinador multi-GPU."""
    
    def setUp(self):
        """Configura ambiente de teste."""
        # Cria diretórios temporários
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.data_dir)
        os.makedirs(self.output_dir)
        
        # Configuração básica para testes
        self.config = {
            "model": {
                "image_size": 16,
                "patch_size": 4,
                "embedding_dim": 128,
                "num_heads": 4,
                "num_layers": 4,
                "num_colors": 16,
                "dropout_rate": 0.1,
                "attention_dropout": 0.1,
                "use_dynamic_scaling": False,
                "compile_mode": "max-autotune"
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay": 1e-6,
                "num_epochs": 100,
                "num_workers": 2,
                "device": "cpu" if not torch.cuda.is_available() else "cuda"
            },
            "optimization": {
                "mixed_precision": True,
                "gradient_clip": 1.0
            },
            "logging": {
                "use_wandb": False,
                "wandb_project": "mini16_test",
                "experiment_name": "test_run",
                "log_interval": 10,
                "save_interval": 100,
                "tensorboard_dir": "logs/tensorboard",
                "wandb_dir": "logs/wandb"
            }
        }
        
        # Cria dados sintéticos para teste
        self.create_synthetic_data()
        
        # Salva configuração em arquivo
        self.config_path = os.path.join(self.temp_dir, "config.yaml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def create_synthetic_data(self):
        """Cria dados sintéticos para teste."""
        num_samples = 100
        sprites = torch.randint(0, 255, (num_samples, 16, 16, 3), dtype=torch.uint8)
        labels = torch.randint(0, 10, (num_samples,))
        
        np.save(os.path.join(self.data_dir, "sprites.npy"), sprites.numpy())
        np.save(os.path.join(self.data_dir, "sprites_labels.npy"), labels.numpy())
        
        with open(os.path.join(self.data_dir, "labels.csv"), "w") as f:
            f.write("id,label\n")
            for i in range(num_samples):
                f.write(f"{i},{labels[i].item()}\n")
    
    def test_trainer_initialization(self):
        """Testa inicialização do treinador multi-GPU."""
        trainer = MultiGPUTrainer(
            config_path=self.config_path,
            data_dir=self.data_dir,
            output_dir=self.output_dir
        )
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.train_loader)
        self.assertIsNotNone(trainer.val_loader)
    
    def test_data_loading(self):
        """Testa carregamento de dados em múltiplas GPUs."""
        trainer = MultiGPUTrainer(
            config_path=self.config_path,
            data_dir=self.data_dir,
            output_dir=self.output_dir
        )
        batch = next(iter(trainer.train_loader))
        self.assertEqual(len(batch), 2)  # imagens e labels
        
        images, labels = batch
        self.assertEqual(images.shape[0], self.config["training"]["batch_size"])
        self.assertEqual(images.shape[1:], (3, 16, 16))
    
    def test_cpu_fallback(self):
        """Testa fallback para CPU quando CUDA não está disponível."""
        config = self.config.copy()
        config["training"]["device"] = "cpu"
        
        # Salvar nova configuração
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)
        
        trainer = MultiGPUTrainer(
            config_path=self.config_path,
            data_dir=self.data_dir,
            output_dir=self.output_dir
        )
        self.assertEqual(trainer.device, torch.device("cpu"))
        self.assertTrue(trainer.model.training)
        
        # Testar um passo de treinamento
        batch = next(iter(trainer.train_loader))
        loss = trainer.training_step(batch)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.requires_grad)
    
    def test_checkpointing(self):
        """Testa salvamento e carregamento de checkpoints."""
        # Criar treinador
        trainer = MultiGPUTrainer(
            config_path=self.config_path,
            data_dir=self.data_dir,
            output_dir=self.output_dir
        )
        
        # Salvar checkpoint
        checkpoint_path = os.path.join(self.output_dir, "checkpoint.pt")
        trainer.save_checkpoint(checkpoint_path, epoch=0, loss=1.0)
        
        # Verificar se arquivo existe
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Criar novo treinador e carregar checkpoint
        new_trainer = MultiGPUTrainer(
            config_path=self.config_path,
            data_dir=self.data_dir,
            output_dir=self.output_dir
        )
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Verificar se os pesos são iguais
        for p1, p2 in zip(trainer.model.parameters(), new_trainer.model.parameters()):
            self.assertTrue(torch.equal(p1.data, p2.data))
            
        # Verificar se otimizador foi carregado
        self.assertEqual(
            trainer.optimizer.state_dict()["param_groups"][0]["lr"],
            new_trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        )
    
    @unittest.skipIf(torch.cuda.device_count() < 2, "Múltiplas GPUs não disponíveis")
    def test_multi_gpu_training(self):
        """Testa treinamento com múltiplas GPUs."""
        with patch("torch.multiprocessing.spawn") as mock_spawn:
            train_distributed(
                config_path=self.config_path,
                data_dir=self.data_dir,
                output_dir=self.output_dir,
                world_size=2
            )
            mock_spawn.assert_called_once()
    
    def test_gpu_selection(self):
        """Testa seleção automática de GPUs."""
        if not torch.cuda.is_available():
            return
            
        # Testar seleção automática
        trainer = MultiGPUTrainer(
            config_path=self.config_path,
            data_dir=self.data_dir,
            output_dir=self.output_dir
        )
        self.assertIsNotNone(trainer.gpu_ids)
        
        # Testar seleção específica
        gpu_id = 0
        trainer = MultiGPUTrainer(
            config_path=self.config_path,
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            gpu_ids=[gpu_id]
        )
        self.assertEqual(trainer.gpu_ids, [gpu_id])
    
    def test_error_handling(self):
        """Testa tratamento de erros."""
        # Testar configuração inválida
        with self.assertRaises(FileNotFoundError):
            trainer = MultiGPUTrainer(
                config_path="/invalid/config.yaml",
                data_dir=self.data_dir,
                output_dir=self.output_dir
            )
        
        # Testar diretório de dados inválido
        with self.assertRaises(Exception):
            trainer = MultiGPUTrainer(
                config_path=self.config_path,
                data_dir="/invalid/data",
                output_dir=self.output_dir
            )
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA não disponível")
    def test_single_gpu_training(self):
        """Testa treinamento com uma GPU."""
        trainer = MultiGPUTrainer(
            config_path=self.config_path,
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            gpu_ids=[0]
        )
        
        # Treinar por uma época
        train_loss = trainer.train_epoch(0)
        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0) 