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

from ..training.trainer import GPUManager, MultiGPUTrainer, Trainer
from ..core.mini_arch import Pixel16Generator
from ..data.pixel16_dataset import Pixel16Dataset

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
            return
            
        # Testar seleção automática
        gpus = GPUManager.select_best_gpus()
        self.assertEqual(len(gpus), torch.cuda.device_count())
        
        # Testar seleção com limite
        num_gpus = min(2, torch.cuda.device_count())
        gpus = GPUManager.select_best_gpus(num_gpus)
        self.assertEqual(len(gpus), num_gpus)
        
        # Verificar ordenação
        if len(gpus) > 1:
            gpu_info = GPUManager.get_gpu_info()
            first_gpu = next(g for g in gpu_info if g["index"] == gpus[0])
            second_gpu = next(g for g in gpu_info if g["index"] == gpus[1])
            
            # Primeira GPU deve ter maior capacidade ou memória
            self.assertGreaterEqual(
                first_gpu["compute_capability"],
                second_gpu["compute_capability"]
            )
    
    def test_optimize_gpu_settings(self):
        """Testa otimização de configurações de GPU."""
        if not self.has_cuda:
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
        self.has_cuda = torch.cuda.is_available()
        
        # Criar diretórios temporários
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.output_dir = Path(self.temp_dir) / "output"
        self.config_path = Path(self.temp_dir) / "config.yaml"
        
        # Criar diretórios
        self.data_dir.mkdir()
        self.output_dir.mkdir()
        
        # Criar dados de teste
        self._create_test_data()
        
        # Criar configuração
        self._create_test_config()
    
    def _create_test_data(self):
        """Cria dados sintéticos para teste."""
        # Criar sprites sintéticos
        num_samples = 100
        sprites = np.random.randint(
            0, 255,
            size=(num_samples, 16, 16, 3),
            dtype=np.uint8
        )
        np.save(self.data_dir / "sprites.npy", sprites)
        
        # Criar labels
        labels = np.random.randn(num_samples, 64)
        np.save(self.data_dir / "sprites_labels.npy", labels)
        
        # Criar CSV
        with open(self.data_dir / "labels.csv", "w") as f:
            f.write("id,name,style\n")
            for i in range(num_samples):
                f.write(f"{i},sprite_{i},style_{i}\n")
    
    def _create_test_config(self):
        """Cria configuração de teste."""
        config = {
            "model": {
                "image_size": 16,
                "patch_size": 4,
                "embedding_dim": 128,
                "ff_dim": 256,
                "num_layers": 4,
                "num_heads": 4,
                "dropout_rate": 0.1,
                "attention_dropout": 0.1,
                "num_colors": 16,
                "style_dim": 64
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay": 1e-6,
                "num_epochs": 2,
                "warmup_steps": 100,
                "gradient_clip_val": 1.0,
                "num_workers": 2
            },
            "optimization": {
                "mixed_precision": True,
                "compile_mode": "reduce-overhead",
                "target_throughput": 100
            }
        }
        
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)
    
    def test_trainer_initialization(self):
        """Testa inicialização do treinador."""
        if not self.has_cuda:
            return
            
        trainer = MultiGPUTrainer(
            config_path=str(self.config_path),
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            gpu_ids=[0]
        )
        
        # Verificar componentes
        self.assertIsInstance(trainer.model, (Pixel16Generator, torch.nn.parallel.DistributedDataParallel))
        self.assertIsInstance(trainer.train_dataset, Pixel16Dataset)
        self.assertIsInstance(trainer.val_dataset, Pixel16Dataset)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA não disponível")
    def test_single_gpu_training(self):
        """Testa treinamento com uma GPU."""
        trainer = MultiGPUTrainer(
            config_path=str(self.config_path),
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            gpu_ids=[0]
        )
        
        # Treinar por uma época
        train_loss = trainer.train_epoch(0)
        self.assertIsInstance(train_loss, float)
        self.assertGreater(train_loss, 0)
    
    @unittest.skipIf(torch.cuda.device_count() < 2, "Múltiplas GPUs não disponíveis")
    def test_multi_gpu_training(self):
        """Testa treinamento com múltiplas GPUs."""
        with patch("torch.multiprocessing.spawn") as mock_spawn:
            train_distributed(
                config_path=str(self.config_path),
                data_dir=str(self.data_dir),
                output_dir=str(self.output_dir),
                world_size=2
            )
            mock_spawn.assert_called_once()
    
    def test_gpu_selection(self):
        """Testa seleção automática de GPUs."""
        if not self.has_cuda:
            return
            
        # Testar seleção automática
        trainer = MultiGPUTrainer(
            config_path=str(self.config_path),
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        self.assertIsNotNone(trainer.gpu_ids)
        
        # Testar seleção específica
        gpu_id = 0
        trainer = MultiGPUTrainer(
            config_path=str(self.config_path),
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            gpu_ids=[gpu_id]
        )
        self.assertEqual(trainer.gpu_ids, [gpu_id])
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA não disponível")
    def test_data_loading(self):
        """Testa carregamento e processamento de dados."""
        # Criar configuração sem CUDA
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Configurar para usar CPU
        if "training" not in config:
            config["training"] = {}
        config["training"]["device"] = "cpu"
        config["training"]["use_cuda"] = False
        
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)
        
        trainer = MultiGPUTrainer(
            config_path=str(self.config_path),
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        
        # Verificar dataloaders
        batch = next(iter(trainer.train_loader))
        self.assertEqual(len(batch), 2)  # (sprites, style_labels)
        
        sprites, style_labels = batch
        self.assertEqual(sprites.shape[1:], (3, 16, 16))
        self.assertEqual(style_labels.shape[1], 64)
    
    def test_checkpointing(self):
        """Testa salvamento e carregamento de checkpoints."""
        if not self.has_cuda:
            return
            
        trainer = MultiGPUTrainer(
            config_path=str(self.config_path),
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            gpu_ids=[0]
        )
        
        # Salvar checkpoint
        val_loss = 1.0
        trainer._save_checkpoint(epoch=0, val_loss=val_loss)
        
        # Verificar arquivo
        checkpoint_dir = self.output_dir / "checkpoints"
        self.assertTrue(checkpoint_dir.exists())
        self.assertTrue(any(checkpoint_dir.glob("*.pt")))
    
    def test_error_handling(self):
        """Testa tratamento de erros."""
        # Testar configuração inválida
        with self.assertRaises(AssertionError):
            trainer = MultiGPUTrainer(
                config_path="/invalid/path",
                data_dir=str(self.data_dir),
                output_dir=str(self.output_dir)
            )
        
        # Testar diretório de dados inválido
        with self.assertRaises(Exception):
            trainer = MultiGPUTrainer(
                config_path=str(self.config_path),
                data_dir="/invalid/path",
                output_dir=str(self.output_dir)
            )
    
    def tearDown(self):
        """Limpa arquivos temporários."""
        import shutil
        shutil.rmtree(self.temp_dir) 