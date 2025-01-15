import os
import time
import psutil
import torch
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

class TrainingMonitor:
    def __init__(self, output_dir: str, log_interval: int = 60):
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval
        self.metrics: List[Dict] = []
        self.start_time = time.time()
        
        # Configura logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TrainingMonitor')
        
        # Verifica CUDA
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device_name = torch.cuda.get_device_name(0)
            self.logger.info(f"GPU detectada: {self.device_name}")
        else:
            self.logger.warning("CUDA não disponível!")
    
    def get_gpu_stats(self) -> Optional[Dict]:
        """Coleta estatísticas da GPU"""
        if not self.cuda_available:
            return None
            
        return {
            'gpu_utilization': torch.cuda.utilization(),
            'gpu_memory_used': torch.cuda.memory_allocated() / 1024**3,  # GB
            'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
            'gpu_temp': torch.cuda.temperature()
        }
    
    def get_system_stats(self) -> Dict:
        """Coleta estatísticas do sistema"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
    
    def get_training_stats(self, global_step: int, loss: float, learning_rate: float) -> Dict:
        """Coleta estatísticas do treinamento"""
        elapsed_time = time.time() - self.start_time
        return {
            'step': global_step,
            'loss': loss,
            'learning_rate': learning_rate,
            'elapsed_time': elapsed_time,
            'steps_per_second': global_step / elapsed_time if elapsed_time > 0 else 0
        }
    
    def log_metrics(self, global_step: int, loss: float, learning_rate: float):
        """Registra todas as métricas"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            **self.get_training_stats(global_step, loss, learning_rate),
            **self.get_system_stats()
        }
        
        if self.cuda_available:
            metrics.update(self.get_gpu_stats())
            
        self.metrics.append(metrics)
        
        # Log a cada intervalo
        if len(self.metrics) % self.log_interval == 0:
            self._save_metrics()
            self._log_current_stats(metrics)
    
    def _save_metrics(self):
        """Salva métricas em CSV"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.output_dir / 'metrics.csv', index=False)
        
        # Gera gráficos
        self._plot_metrics(df)
    
    def _plot_metrics(self, df: pd.DataFrame):
        """Gera gráficos das métricas"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Configura estilo
            plt.style.use('dark_background')
            sns.set_theme(style="darkgrid")
            
            # Plot de loss
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df, x='step', y='loss')
            plt.title('Training Loss')
            plt.savefig(self.output_dir / 'loss.png')
            plt.close()
            
            if self.cuda_available:
                # Plot de GPU
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                sns.lineplot(data=df, x='step', y='gpu_utilization', ax=ax1)
                ax1.set_title('GPU Utilization (%)')
                
                sns.lineplot(data=df, x='step', y='gpu_memory_used', ax=ax2)
                ax2.set_title('GPU Memory Usage (GB)')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'gpu_metrics.png')
                plt.close()
                
        except ImportError:
            self.logger.warning("matplotlib e seaborn são necessários para gerar gráficos")
    
    def _log_current_stats(self, metrics: Dict):
        """Loga estatísticas atuais"""
        self.logger.info(
            f"Step {metrics['step']}: "
            f"Loss = {metrics['loss']:.4f}, "
            f"LR = {metrics['learning_rate']:.6f}, "
            f"Steps/s = {metrics['steps_per_second']:.2f}"
        )
        
        if self.cuda_available:
            self.logger.info(
                f"GPU: "
                f"Utilization = {metrics['gpu_utilization']}%, "
                f"Memory = {metrics['gpu_memory_used']:.1f}GB/{metrics['gpu_memory_total']:.1f}GB, "
                f"Temp = {metrics['gpu_temp']}°C"
            )
    
    def finish(self):
        """Finaliza o monitoramento"""
        self._save_metrics()
        
        # Calcula e loga estatísticas finais
        df = pd.DataFrame(self.metrics)
        total_time = time.time() - self.start_time
        
        final_stats = {
            'total_steps': len(df),
            'total_time': f"{total_time/3600:.1f}h",
            'avg_loss': df['loss'].mean(),
            'min_loss': df['loss'].min(),
            'avg_steps_per_second': df['steps_per_second'].mean()
        }
        
        self.logger.info("Treinamento finalizado!")
        self.logger.info(f"Estatísticas finais: {final_stats}")
        
        # Salva estatísticas finais
        with open(self.output_dir / 'final_stats.txt', 'w') as f:
            for k, v in final_stats.items():
                f.write(f"{k}: {v}\n") 