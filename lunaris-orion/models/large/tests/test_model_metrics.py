"""
Large Model Metrics Generator
---------------------------
Generates comprehensive metrics and visualizations for the Large model, including:
- Training metrics (loss, learning rate, etc.)
- GPU utilization and memory usage
- Attention patterns and cache statistics
- Model architecture analysis
- Performance benchmarks
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from torch.cuda import nvml
import psutil
import GPUtil

from ..model import LargeModel, ModelConfig
from ..lunar_cache import CacheConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMetricsAnalyzer:
    """Analyzer for Large model metrics and performance."""
    
    def __init__(self, model: Optional[LargeModel] = None):
        self.model = model if model is not None else self._create_test_model()
        self.metrics = {
            'performance': [],
            'memory': [],
            'attention': [],
            'cache': [],
            'architecture': self._analyze_architecture()
        }
    
    def _create_test_model(self) -> LargeModel:
        """Create a test model instance."""
        config = ModelConfig(
            embedding_dim=1024,
            num_heads=16,
            num_layers=24,
            ffn_dim=4096,
            max_sequence_length=2048
        )
        return LargeModel(config)
    
    def _analyze_architecture(self) -> Dict:
        """Analyze model architecture and parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        layer_params = {}
        for name, module in self.model.named_modules():
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                layer_params[name] = params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'layer_parameters': layer_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture_config': self.model.config.__dict__
        }
    
    def benchmark_inference(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        sequence_length: int = 512,
        num_runs: int = 100
    ) -> Dict:
        """Benchmark model inference performance."""
        logger.info("Running inference benchmark...")
        
        results = {
            'batch_size': [],
            'latency_ms': [],
            'throughput': [],
            'memory_usage_mb': []
        }
        
        device = next(self.model.parameters()).device
        
        for batch_size in batch_sizes:
            latencies = []
            memory_usage = []
            
            # Generate dummy input
            x = torch.randint(
                0, self.model.config.vocab_size,
                (batch_size, sequence_length),
                device=device
            )
            
            # Warmup
            for _ in range(10):
                self.model(x)
            
            # Benchmark
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                self.model(x)
                end_event.record()
                
                torch.cuda.synchronize()
                latency = start_event.elapsed_time(end_event)
                latencies.append(latency)
                
                memory_usage.append(
                    torch.cuda.memory_allocated() / 1024**2
                )
            
            avg_latency = np.mean(latencies)
            throughput = batch_size * (1000 / avg_latency)  # samples/second
            
            results['batch_size'].append(batch_size)
            results['latency_ms'].append(avg_latency)
            results['throughput'].append(throughput)
            results['memory_usage_mb'].append(np.mean(memory_usage))
            
            logger.info(f"Batch size {batch_size}: {avg_latency:.2f}ms, {throughput:.2f} samples/s")
        
        self.metrics['performance'].append(results)
        return results
    
    def analyze_attention_patterns(
        self,
        sequence_length: int = 512,
        batch_size: int = 4
    ) -> Dict:
        """Analyze attention patterns across layers."""
        logger.info("Analyzing attention patterns...")
        
        device = next(self.model.parameters()).device
        x = torch.randint(
            0, self.model.config.vocab_size,
            (batch_size, sequence_length),
            device=device
        )
        
        # Get attention maps
        self.model(x)
        attention_maps = self.model.get_attention_maps()
        
        # Analyze patterns
        analysis = {
            'layer_attention': {},
            'average_attention': {},
            'attention_stats': {}
        }
        
        for layer_name, attn_map in attention_maps.items():
            # Convert to numpy for analysis
            attn_np = attn_map.detach().cpu().numpy()
            
            analysis['layer_attention'][layer_name] = {
                'mean': float(np.mean(attn_np)),
                'std': float(np.std(attn_np)),
                'sparsity': float((attn_np < 0.01).mean())
            }
        
        self.metrics['attention'].append(analysis)
        return analysis
    
    def monitor_gpu_metrics(self, duration_sec: int = 60) -> Dict:
        """Monitor GPU metrics during model operation."""
        logger.info("Monitoring GPU metrics...")
        
        metrics = {
            'timestamp': [],
            'gpu_utilization': [],
            'memory_used': [],
            'temperature': [],
            'power_usage': []
        }
        
        try:
            nvml.nvmlInit()
            device = nvml.nvmlDeviceGetHandleByIndex(0)
            
            start_time = datetime.now()
            while (datetime.now() - start_time).seconds < duration_sec:
                info = nvml.nvmlDeviceGetUtilizationRates(device)
                memory = nvml.nvmlDeviceGetMemoryInfo(device)
                temp = nvml.nvmlDeviceGetTemperature(device, nvml.NVML_TEMPERATURE_GPU)
                power = nvml.nvmlDeviceGetPowerUsage(device) / 1000.0
                
                metrics['timestamp'].append(datetime.now().isoformat())
                metrics['gpu_utilization'].append(info.gpu)
                metrics['memory_used'].append(memory.used / 1024**2)
                metrics['temperature'].append(temp)
                metrics['power_usage'].append(power)
                
                time.sleep(1)
        
        except Exception as e:
            logger.warning(f"Failed to monitor GPU metrics: {str(e)}")
        
        self.metrics['memory'].append(metrics)
        return metrics
    
    def generate_visualizations(self, save_dir: str = 'model_analysis'):
        """Generate visualizations of model metrics."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot performance metrics
        if self.metrics['performance']:
            perf = self.metrics['performance'][-1]
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(perf['batch_size'], perf['latency_ms'], marker='o')
            plt.title('Inference Latency vs Batch Size')
            plt.xlabel('Batch Size')
            plt.ylabel('Latency (ms)')
            
            plt.subplot(1, 2, 2)
            plt.plot(perf['batch_size'], perf['throughput'], marker='o')
            plt.title('Throughput vs Batch Size')
            plt.xlabel('Batch Size')
            plt.ylabel('Throughput (samples/s)')
            
            plt.tight_layout()
            plt.savefig(save_path / 'performance_metrics.png')
            plt.close()
        
        # Plot attention patterns
        if self.metrics['attention']:
            attn = self.metrics['attention'][-1]
            layer_means = [v['mean'] for v in attn['layer_attention'].values()]
            layer_names = list(attn['layer_attention'].keys())
            
            plt.figure(figsize=(12, 6))
            plt.bar(layer_names, layer_means)
            plt.title('Average Attention by Layer')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_path / 'attention_patterns.png')
            plt.close()
        
        # Plot GPU metrics
        if self.metrics['memory']:
            mem = self.metrics['memory'][-1]
            
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.plot(mem['gpu_utilization'])
            plt.title('GPU Utilization')
            plt.ylabel('Utilization %')
            
            plt.subplot(2, 2, 2)
            plt.plot(mem['memory_used'])
            plt.title('Memory Usage')
            plt.ylabel('Memory (MB)')
            
            plt.subplot(2, 2, 3)
            plt.plot(mem['temperature'])
            plt.title('GPU Temperature')
            plt.ylabel('Temperature (°C)')
            
            plt.subplot(2, 2, 4)
            plt.plot(mem['power_usage'])
            plt.title('Power Usage')
            plt.ylabel('Power (W)')
            
            plt.tight_layout()
            plt.savefig(save_path / 'gpu_metrics.png')
            plt.close()
    
    def save_metrics(self, save_dir: str = 'model_analysis'):
        """Save all metrics to JSON file."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = save_path / f'model_metrics_{timestamp}.json'
        
        # Convert numpy values to Python native types
        metrics_dict = {
            k: self._convert_to_serializable(v)
            for k, v in self.metrics.items()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        logger.info(f"Saved metrics to {metrics_file}")
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/torch values to Python native types."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        return obj

def run_model_analysis(model: Optional[LargeModel] = None):
    """Run comprehensive model analysis and generate report."""
    logger.info("Starting model analysis...")
    
    analyzer = ModelMetricsAnalyzer(model)
    
    # Run analysis
    analyzer.benchmark_inference()
    analyzer.analyze_attention_patterns()
    analyzer.monitor_gpu_metrics(duration_sec=60)
    
    # Generate outputs
    analyzer.generate_visualizations()
    analyzer.save_metrics()
    
    logger.info("Analysis completed. Check model_analysis directory for outputs.")

if __name__ == '__main__':
    run_model_analysis() 