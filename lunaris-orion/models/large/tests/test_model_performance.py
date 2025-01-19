"""
Model Performance Tests
---------------------
Tests the Large model's performance and generates metrics for:
- Training metrics (loss, gradients)
- GPU utilization and memory usage
- Attention pattern analysis
- Model architecture analysis
- Performance benchmarks
"""

import torch
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import GPUtil

from ..model import LargeModel, ModelConfig
from ..lunar_cache import LunarCache, CacheConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPerformanceTester:
    """Tests model performance and generates metrics."""
    
    def __init__(self, model: Optional[LargeModel] = None):
        self.model = model if model is not None else self._create_test_model()
        self.results = {
            'training_metrics': [],
            'gpu_metrics': [],
            'attention_patterns': [],
            'architecture_analysis': [],
            'benchmarks': []
        }
        
        # Initialize device
        self.device = next(self.model.parameters()).device
        
        # Set up hooks for gradient analysis
        self._setup_gradient_hooks()
    
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
    
    def _setup_gradient_hooks(self):
        """Set up hooks to track gradients during training."""
        self.gradient_stats = []
        
        def hook_fn(grad):
            if grad is not None:
                self.gradient_stats.append({
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'norm': grad.norm().item()
                })
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: hook_fn(grad))
    
    def test_training_metrics(
        self,
        num_steps: int = 100,
        batch_size: int = 32,
        sequence_length: int = 512
    ) -> Dict:
        """Test training metrics including loss and gradients."""
        logger.info("Testing training metrics...")
        
        results = {
            'loss_values': [],
            'gradient_norms': [],
            'learning_rates': []
        }
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        
        for step in range(num_steps):
            # Generate random batch
            input_ids = torch.randint(
                0, self.model.config.vocab_size,
                (batch_size, sequence_length),
                device=self.device
            )
            target_ids = torch.randint(
                0, self.model.config.vocab_size,
                (batch_size, sequence_length),
                device=self.device
            )
            
            # Forward pass
            outputs = self.model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, self.model.config.vocab_size),
                target_ids.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Record metrics
            results['loss_values'].append(loss.item())
            results['gradient_norms'].append(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).item()
            )
            results['learning_rates'].append(scheduler.get_last_lr()[0])
            
            if step % 10 == 0:
                logger.info(f"Step {step}: Loss = {loss.item():.4f}")
        
        # Compute statistics
        stats = {
            'final_loss': results['loss_values'][-1],
            'avg_loss': np.mean(results['loss_values']),
            'loss_std': np.std(results['loss_values']),
            'avg_gradient_norm': np.mean(results['gradient_norms']),
            'gradient_norm_std': np.std(results['gradient_norms'])
        }
        
        self.results['training_metrics'].append({
            'raw_results': results,
            'stats': stats,
            'gradient_stats': self.gradient_stats
        })
        
        logger.info(f"Training metrics test results: {stats}")
        return stats
    
    def test_gpu_metrics(self, duration: int = 60) -> Dict:
        """Monitor GPU metrics over specified duration."""
        logger.info(f"Monitoring GPU metrics for {duration} seconds...")
        
        results = {
            'utilization': [],
            'memory_used': [],
            'temperature': [],
            'power_usage': []
        }
        
        start_time = time.time()
        while time.time() - start_time < duration:
            gpu = GPUtil.getGPUs()[0]  # Assuming first GPU
            
            results['utilization'].append(gpu.load * 100)
            results['memory_used'].append(gpu.memoryUsed)
            results['temperature'].append(gpu.temperature)
            results['power_usage'].append(gpu.powerUsage if hasattr(gpu, 'powerUsage') else 0)
            
            time.sleep(1)  # Sample every second
        
        # Compute statistics
        stats = {
            'avg_utilization': np.mean(results['utilization']),
            'max_utilization': np.max(results['utilization']),
            'avg_memory_used': np.mean(results['memory_used']),
            'max_memory_used': np.max(results['memory_used']),
            'avg_temperature': np.mean(results['temperature']),
            'max_temperature': np.max(results['temperature']),
            'avg_power_usage': np.mean(results['power_usage']),
            'max_power_usage': np.max(results['power_usage'])
        }
        
        self.results['gpu_metrics'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"GPU metrics test results: {stats}")
        return stats
    
    def analyze_attention_patterns(
        self,
        num_samples: int = 10,
        sequence_length: int = 512
    ) -> Dict:
        """Analyze attention patterns across layers."""
        logger.info("Analyzing attention patterns...")
        
        results = {
            'attention_maps': [],
            'attention_stats': []
        }
        
        # Generate random input
        input_ids = torch.randint(
            0, self.model.config.vocab_size,
            (num_samples, sequence_length),
            device=self.device
        )
        
        # Get attention maps from each layer
        attention_maps = []
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, output_attentions=True)
            attention_maps = outputs.attentions
        
        # Analyze attention patterns
        for layer_idx, attn_map in enumerate(attention_maps):
            # Average over heads and batch
            avg_attention = attn_map.mean(dim=(0, 1))
            
            # Compute statistics
            stats = {
                'sparsity': (avg_attention < 0.1).float().mean().item(),
                'entropy': -(avg_attention * torch.log(avg_attention + 1e-10)).sum().item(),
                'max_attention': avg_attention.max().item(),
                'mean_attention': avg_attention.mean().item()
            }
            
            results['attention_maps'].append(avg_attention.cpu().numpy())
            results['attention_stats'].append({
                'layer': layer_idx,
                'stats': stats
            })
        
        self.results['attention_patterns'].append(results)
        
        logger.info("Attention pattern analysis completed")
        return results
    
    def analyze_architecture(self) -> Dict:
        """Analyze model architecture and parameters."""
        logger.info("Analyzing model architecture...")
        
        results = {
            'layer_sizes': {},
            'parameter_stats': {},
            'memory_analysis': {}
        }
        
        # Analyze layer sizes
        total_params = 0
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                num_params = sum(p.numel() for p in module.parameters())
                results['layer_sizes'][name] = num_params
                total_params += num_params
        
        # Analyze parameter statistics
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stats = {
                    'mean': param.mean().item(),
                    'std': param.std().item(),
                    'norm': param.norm().item(),
                    'shape': list(param.shape),
                    'size': param.numel()
                }
                results['parameter_stats'][name] = stats
        
        # Memory analysis
        memory_stats = {
            'total_parameters': total_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'gpu_memory_allocated': torch.cuda.memory_allocated() / (1024 * 1024),
            'gpu_memory_cached': torch.cuda.memory_reserved() / (1024 * 1024)
        }
        results['memory_analysis'] = memory_stats
        
        self.results['architecture_analysis'].append(results)
        
        logger.info(f"Architecture analysis completed: {memory_stats}")
        return results
    
    def run_benchmarks(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        sequence_lengths: List[int] = [128, 256, 512, 1024]
    ) -> Dict:
        """Run performance benchmarks with various configurations."""
        logger.info("Running performance benchmarks...")
        
        results = {
            'inference_time': {},
            'memory_usage': {},
            'throughput': {}
        }
        
        self.model.eval()
        with torch.no_grad():
            for batch_size in batch_sizes:
                for seq_len in sequence_lengths:
                    key = f"batch_{batch_size}_seq_{seq_len}"
                    results['inference_time'][key] = []
                    results['memory_usage'][key] = []
                    
                    # Generate input
                    input_ids = torch.randint(
                        0, self.model.config.vocab_size,
                        (batch_size, seq_len),
                        device=self.device
                    )
                    
                    # Warmup
                    for _ in range(3):
                        self.model(input_ids)
                    
                    # Benchmark
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    for _ in range(10):
                        torch.cuda.empty_cache()
                        start_event.record()
                        self.model(input_ids)
                        end_event.record()
                        torch.cuda.synchronize()
                        
                        results['inference_time'][key].append(
                            start_event.elapsed_time(end_event)
                        )
                        results['memory_usage'][key].append(
                            torch.cuda.memory_allocated() / (1024 * 1024)
                        )
                    
                    # Calculate throughput (tokens/second)
                    avg_time = np.mean(results['inference_time'][key])
                    throughput = (batch_size * seq_len) / (avg_time / 1000)  # Convert ms to s
                    results['throughput'][key] = throughput
                    
                    logger.info(
                        f"Benchmark {key}: "
                        f"Avg time = {avg_time:.2f}ms, "
                        f"Throughput = {throughput:.2f} tokens/s"
                    )
        
        self.results['benchmarks'].append(results)
        return results
    
    def generate_visualizations(self, save_dir: str = 'performance_tests'):
        """Generate visualizations of test results."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot training metrics
        if self.results['training_metrics']:
            metrics = self.results['training_metrics'][-1]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(metrics['raw_results']['loss_values'])
            plt.title('Training Loss')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            
            plt.subplot(1, 3, 2)
            plt.plot(metrics['raw_results']['gradient_norms'])
            plt.title('Gradient Norms')
            plt.xlabel('Step')
            plt.ylabel('Norm')
            
            plt.subplot(1, 3, 3)
            plt.plot(metrics['raw_results']['learning_rates'])
            plt.title('Learning Rate')
            plt.xlabel('Step')
            plt.ylabel('LR')
            
            plt.tight_layout()
            plt.savefig(save_path / 'training_metrics.png')
            plt.close()
        
        # Plot GPU metrics
        if self.results['gpu_metrics']:
            metrics = self.results['gpu_metrics'][-1]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(metrics['raw_results']['utilization'])
            plt.title('GPU Utilization')
            plt.xlabel('Time (s)')
            plt.ylabel('Utilization (%)')
            
            plt.subplot(1, 3, 2)
            plt.plot(metrics['raw_results']['memory_used'])
            plt.title('GPU Memory Usage')
            plt.xlabel('Time (s)')
            plt.ylabel('Memory (MB)')
            
            plt.subplot(1, 3, 3)
            plt.plot(metrics['raw_results']['temperature'])
            plt.title('GPU Temperature')
            plt.xlabel('Time (s)')
            plt.ylabel('Temperature (°C)')
            
            plt.tight_layout()
            plt.savefig(save_path / 'gpu_metrics.png')
            plt.close()
        
        # Plot attention patterns
        if self.results['attention_patterns']:
            patterns = self.results['attention_patterns'][-1]
            
            for layer_idx, attn_map in enumerate(patterns['attention_maps']):
                plt.figure(figsize=(10, 8))
                sns.heatmap(attn_map, cmap='viridis')
                plt.title(f'Layer {layer_idx} Attention Pattern')
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                plt.savefig(save_path / f'attention_layer_{layer_idx}.png')
                plt.close()
        
        # Plot architecture analysis
        if self.results['architecture_analysis']:
            arch = self.results['architecture_analysis'][-1]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            sizes = list(arch['layer_sizes'].values())
            labels = list(arch['layer_sizes'].keys())
            plt.pie(sizes, labels=labels, autopct='%1.1f%%')
            plt.title('Parameter Distribution Across Layers')
            
            plt.subplot(1, 2, 2)
            norms = [stats['norm'] for stats in arch['parameter_stats'].values()]
            names = list(arch['parameter_stats'].keys())
            plt.bar(range(len(norms)), norms)
            plt.xticks(range(len(norms)), names, rotation=45)
            plt.title('Parameter Norms')
            
            plt.tight_layout()
            plt.savefig(save_path / 'architecture_analysis.png')
            plt.close()
        
        # Plot benchmarks
        if self.results['benchmarks']:
            bench = self.results['benchmarks'][-1]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            for key, times in bench['inference_time'].items():
                plt.plot(times, label=key)
            plt.title('Inference Time')
            plt.xlabel('Run')
            plt.ylabel('Time (ms)')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            throughputs = list(bench['throughput'].values())
            configs = list(bench['throughput'].keys())
            plt.bar(range(len(throughputs)), throughputs)
            plt.xticks(range(len(throughputs)), configs, rotation=45)
            plt.title('Throughput')
            plt.ylabel('Tokens/second')
            
            plt.tight_layout()
            plt.savefig(save_path / 'benchmarks.png')
            plt.close()
    
    def save_results(self, save_dir: str = 'performance_tests'):
        """Save test results to JSON file."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = save_path / f'performance_results_{timestamp}.json'
        
        # Convert numpy values to Python native types
        results_dict = {
            k: self._convert_to_serializable(v)
            for k, v in self.results.items()
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        logger.info(f"Saved results to {results_file}")
    
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

def run_performance_tests(model: Optional[LargeModel] = None):
    """Run all performance tests and generate report."""
    logger.info("Starting model performance tests...")
    
    tester = ModelPerformanceTester(model)
    
    # Run tests
    tester.test_training_metrics()
    tester.test_gpu_metrics()
    tester.analyze_attention_patterns()
    tester.analyze_architecture()
    tester.run_benchmarks()
    
    # Generate outputs
    tester.generate_visualizations()
    tester.save_results()
    
    logger.info("Performance tests completed. Check performance_tests directory for outputs.")

if __name__ == '__main__':
    run_performance_tests() 