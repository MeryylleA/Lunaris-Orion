"""
Testes para o sistema de monitoramento de recursos.
"""

import unittest
import time
from unittest.mock import MagicMock, patch
import torch
import numpy as np

from ..utils.monitoring import ResourceMonitor

class TestResourceMonitor(unittest.TestCase):
    """Testes para o monitor de recursos."""
    
    def setUp(self):
        self.monitor = ResourceMonitor(
            check_interval=0.1,
            memory_threshold=90.0,
            window_size=10
        )
    
    def test_initialization(self):
        """Testa inicialização do monitor."""
        self.assertEqual(self.monitor.check_interval, 0.1)
        self.assertEqual(self.monitor.memory_threshold, 90.0)
        self.assertEqual(self.monitor.window_size, 10)
        self.assertFalse(self.monitor.running)
        self.assertIsNone(self.monitor.thread)
    
    def test_start_stop(self):
        """Testa início e parada do monitoramento."""
        try:
            # Iniciar
            self.monitor.start()
            self.assertTrue(self.monitor.running)
            self.assertIsNotNone(self.monitor.thread)
            self.assertTrue(self.monitor.thread.is_alive())
            
            # Parar
            self.monitor.stop()
            self.assertFalse(self.monitor.running)
            
            # Esperar thread terminar
            if self.monitor.thread:
                self.monitor.thread.join(timeout=1.0)
                self.assertFalse(self.monitor.thread.is_alive())
        except Exception as e:
            self.monitor.stop()  # Garantir que o monitor é parado mesmo em caso de erro
            raise e
    
    def test_resource_checking(self):
        """Testa checagem de recursos."""
        stats = self.monitor._check_resources()
        
        # Verificar métricas básicas
        self.assertIn("cpu", stats)
        self.assertIn("ram", stats)
        self.assertGreaterEqual(stats["cpu"], 0)
        self.assertGreaterEqual(stats["ram"], 0)
        
        # Verificar métricas de GPU se disponível
        if torch.cuda.is_available():
            self.assertIn("gpu_memory", stats)
            self.assertIn("gpu_utilization", stats)
            self.assertGreater(len(stats["gpu_memory"]), 0)
            self.assertGreater(len(stats["gpu_utilization"]), 0)
    
    def test_metrics_update(self):
        """Testa atualização de métricas."""
        # Criar dados de teste
        test_stats = {
            "cpu": 50.0,
            "ram": 75.0,
            "gpu_memory": [60.0, 70.0],
            "gpu_utilization": [80.0, 85.0]
        }
        
        # Atualizar métricas
        self.monitor._update_metrics(test_stats)
        
        # Verificar histórico
        summary = self.monitor.get_stats_summary()
        self.assertIn("cpu", summary)
        self.assertIn("ram", summary)
        
        # Verificar estatísticas
        for key in ["cpu", "ram"]:
            stats = summary[key]
            self.assertIn("mean", stats)
            self.assertIn("min", stats)
            self.assertIn("max", stats)
            self.assertIn("std", stats)
    
    def test_alert_system(self):
        """Testa sistema de alertas."""
        # Mock para callback
        mock_callback = MagicMock()
        self.monitor.add_alert_callback(mock_callback)
        
        # Simular uso crítico de recursos
        test_stats = {
            "ram": 95.0,  # Acima do threshold
            "gpu_memory": [80.0, 95.0]  # Segunda GPU crítica
        }
        
        # Verificar alertas
        self.monitor._check_alerts(test_stats)
        
        # Verificar se callback foi chamado
        mock_callback.assert_called()
        
        # Verificar mensagem de alerta
        call_args = mock_callback.call_args[0]
        self.assertIn("RAM crítico", call_args[0])
        self.assertIn("GPU 1 crítica", call_args[0])
    
    def test_window_management(self):
        """Testa gerenciamento da janela de métricas."""
        # Simular várias atualizações
        test_stats = {"cpu": 50.0}
        
        for i in range(20):  # Mais que window_size
            self.monitor._update_metrics(test_stats)
            time.sleep(0.01)
        
        # Verificar tamanho do histórico
        timestamp = time.time()
        cutoff = timestamp - (self.monitor.window_size * self.monitor.check_interval)
        
        for key, values in self.monitor.metrics.items():
            for t, _ in values:
                self.assertGreater(t, cutoff)
    
    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_resource_monitoring_accuracy(self, mock_memory, mock_cpu):
        """Testa precisão do monitoramento."""
        # Configurar mocks
        mock_cpu.return_value = 75.5
        mock_memory_obj = MagicMock()
        mock_memory_obj.percent = 82.3
        mock_memory.return_value = mock_memory_obj
        
        # Coletar métricas
        stats = self.monitor._check_resources()
        
        # Verificar valores
        self.assertEqual(stats["cpu"], 75.5)
        self.assertEqual(stats["ram"], 82.3)
    
    def test_stats_summary_calculation(self):
        """Testa cálculo de resumo estatístico."""
        # Adicionar dados de teste
        test_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        timestamp = time.time()
        
        for value in test_values:
            self.monitor.metrics["test"].append((timestamp, value))
            timestamp += 1
        
        # Calcular resumo
        summary = self.monitor.get_stats_summary()
        test_stats = summary["test"]
        
        # Verificar estatísticas
        self.assertEqual(test_stats["mean"], np.mean(test_values))
        self.assertEqual(test_stats["min"], np.min(test_values))
        self.assertEqual(test_stats["max"], np.max(test_values))
        self.assertEqual(test_stats["std"], np.std(test_values))
    
    def test_multiple_callbacks(self):
        """Testa múltiplos callbacks de alerta."""
        # Criar callbacks
        callbacks = [MagicMock() for _ in range(3)]
        for callback in callbacks:
            self.monitor.add_alert_callback(callback)
        
        # Simular alerta com dados básicos
        test_stats = {
            "ram": 95.0,
            "cpu": 80.0,
            "gpu_memory": [] if not torch.cuda.is_available() else [80.0],
            "gpu_utilization": [] if not torch.cuda.is_available() else [85.0]
        }
        
        self.monitor._check_alerts(test_stats)
        
        # Verificar que todos foram chamados
        for callback in callbacks:
            callback.assert_called_once()
            
            # Verificar que os argumentos estão corretos
            args = callback.call_args[0]
            self.assertIsInstance(args[0], str)  # Mensagem de alerta
            self.assertEqual(args[1], test_stats)  # Estatísticas
    
    def tearDown(self):
        """Limpa recursos."""
        if self.monitor.running:
            self.monitor.stop() 