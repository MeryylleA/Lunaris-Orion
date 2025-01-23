"""
Monitoramento de recursos do sistema durante treinamento.
"""

import time
import threading
from typing import Dict, List, Callable, Any, Optional
import psutil
import torch
import numpy as np
from collections import defaultdict

class ResourceMonitor:
    """Monitor de recursos do sistema."""
    
    def __init__(self, window_size: int = 60, check_interval: float = 1.0,
                 memory_threshold: float = 90.0, alert_callbacks: List[Callable] = None):
        """Inicializa o monitor de recursos."""
        self.window_size = window_size
        self.check_interval = check_interval
        self.memory_threshold = memory_threshold
        self.alert_callbacks = alert_callbacks or []
        
        self.metrics = defaultdict(list)
        self.running = False
        self.monitor_thread = None
        
        # Verificar disponibilidade de GPU
        self.has_gpu = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.has_gpu else 0
        
        if not self.has_gpu:
            print("\nGPU não disponível - monitoramento apenas de CPU/RAM")
    
    def start(self):
        """Inicia monitoramento em thread separada."""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        """Para monitoramento."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
            self.monitor_thread = None
    
    def add_alert_callback(self, callback: Callable[[str, Dict], None]):
        """Adiciona callback para alertas."""
        self.alert_callbacks.append(callback)
    
    def _check_resources(self) -> Dict[str, Any]:
        """Coleta métricas do sistema."""
        stats = {}
        
        try:
            # CPU e RAM (sempre disponíveis)
            stats["cpu"] = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            stats["ram"] = memory.percent
            
            # GPU (apenas se disponível)
            if self.has_gpu:
                gpu_stats = {"memory": [], "utilization": []}
                for i in range(self.num_gpus):
                    try:
                        gpu_memory = torch.cuda.memory_reserved(i) / torch.cuda.get_device_properties(i).total_memory * 100
                        gpu_stats["memory"].append(gpu_memory)
                        
                        # Utilização via nvidia-smi se disponível
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_stats["utilization"].append(util.gpu)
                        except:
                            gpu_stats["utilization"].append(0.0)
                    except:
                        gpu_stats["memory"].append(0.0)
                        gpu_stats["utilization"].append(0.0)
                
                if gpu_stats["memory"]:
                    stats["gpu_memory"] = gpu_stats["memory"]
                if gpu_stats["utilization"]:
                    stats["gpu_utilization"] = gpu_stats["utilization"]
        except:
            pass
        
        return stats
    
    def _check_alerts(self, stats: Dict[str, Any]):
        """Verifica condições para alertas."""
        alerts = []
        
        # Memória RAM
        if stats["ram"] > self.memory_threshold:
            alerts.append(
                f"Uso de RAM crítico: {stats['ram']:.1f}%"
            )
        
        # Memória GPU
        for i, mem in enumerate(stats["gpu_memory"]):
            if mem > self.memory_threshold:
                alerts.append(
                    f"Memória GPU {i} crítica: {mem:.1f}%"
                )
        
        # Disparar callbacks
        if alerts and self.alert_callbacks:
            message = "\n".join(alerts)
            for callback in self.alert_callbacks:
                callback(message, stats)
    
    def _update_metrics(self, stats: Dict[str, Any]):
        """Atualiza métricas com novas medições."""
        timestamp = time.time()
        
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.metrics[key].append((timestamp, value))
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    self.metrics[f"{key}_{i}"].append((timestamp, v))
        
        # Limitar tamanho do histórico
        cutoff = timestamp - (self.window_size * self.check_interval)
        for key in self.metrics:
            self.metrics[key] = [
                (t, v) for t, v in self.metrics[key]
                if t > cutoff
            ]
    
    def _monitor_loop(self):
        """Loop principal de monitoramento."""
        while self.running:
            try:
                # Coletar métricas silenciosamente
                stats = self._check_resources()
                if stats:
                    self._update_metrics(stats)
                    self._check_alerts(stats)
            except Exception:
                pass  # Ignora erros silenciosamente
            
            time.sleep(self.check_interval)
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Retorna resumo estatístico das métricas."""
        summary = {}
        
        for key, values in self.metrics.items():
            if not values:
                continue
                
            # Extrair apenas valores
            vals = [v for _, v in values]
            
            # Calcular estatísticas
            summary[key] = {
                "mean": np.mean(vals),
                "min": np.min(vals),
                "max": np.max(vals),
                "std": np.std(vals)
            }
        
        return summary 