"""
Model Quantization Tests
----------------------
Tests the Large model's quantization functionality, including:
- Different quantization methods (dynamic, static, per-channel)
- Performance impact analysis
- Output quality validation
- Memory reduction measurement
- Hardware compatibility checks
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
from torch.ao.quantization import (
    get_default_qconfig,
    quantize_dynamic,
    prepare,
    convert,
    QConfig
)

from ..model import LargeModel, ModelConfig
from ..lunar_cache import LunarCache, CacheConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelQuantizationTester:
    """Tests model quantization functionality."""
    
    def __init__(self, model: Optional[LargeModel] = None):
        self.model = model if model is not None else self._create_test_model()
        self.results = {
            'quantization_methods': [],
            'performance_impact': [],
            'output_quality': [],
            'memory_reduction': [],
            'hardware_compatibility': []
        }
        
        # Initialize device
        self.device = next(self.model.parameters()).device
        
        # Store original model state
        self.original_state = {
            'model_size': self._get_model_size(self.model),
            'state_dict': self.model.state_dict()
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
    
    def _get_model_size(self, model: torch.nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / (1024 * 1024)
    
    def test_quantization_methods(
        self,
        methods: List[str] = ['dynamic', 'static', 'per_channel']
    ) -> Dict:
        """Test different quantization methods."""
        logger.info("Testing quantization methods...")
        
        results = {
            'methods': {},
            'model_sizes': {},
            'quantization_time': {},
            'success_rate': {}
        }
        
        for method in methods:
            logger.info(f"Testing {method} quantization...")
            
            try:
                # Create a copy of the model for quantization
                model_copy = self._create_test_model()
                model_copy.load_state_dict(self.original_state['state_dict'])
                
                start_time = time.time()
                
                if method == 'dynamic':
                    # Dynamic quantization
                    quantized_model = quantize_dynamic(
                        model_copy,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                elif method == 'static':
                    # Static quantization
                    model_copy.eval()
                    qconfig = get_default_qconfig('fbgemm')
                    model_copy.qconfig = qconfig
                    prepared_model = prepare(model_copy)
                    
                    # Calibrate with sample data
                    self._calibrate_model(prepared_model)
                    
                    quantized_model = convert(prepared_model)
                else:  # per_channel
                    # Per-channel quantization
                    model_copy.eval()
                    qconfig = QConfig(
                        activation=torch.ao.quantization.default_observer,
                        weight=torch.ao.quantization.per_channel_weight_observer
                    )
                    model_copy.qconfig = qconfig
                    prepared_model = prepare(model_copy)
                    
                    # Calibrate with sample data
                    self._calibrate_model(prepared_model)
                    
                    quantized_model = convert(prepared_model)
                
                quantization_time = time.time() - start_time
                
                # Record results
                results['methods'][method] = {
                    'success': True,
                    'error': None
                }
                results['model_sizes'][method] = self._get_model_size(quantized_model)
                results['quantization_time'][method] = quantization_time
                results['success_rate'][method] = 1.0
                
                logger.info(
                    f"{method} quantization successful: "
                    f"Size reduction = {self.original_state['model_size'] / results['model_sizes'][method]:.2f}x"
                )
                
            except Exception as e:
                logger.error(f"Error during {method} quantization: {str(e)}")
                results['methods'][method] = {
                    'success': False,
                    'error': str(e)
                }
                results['success_rate'][method] = 0.0
        
        # Compute statistics
        stats = {
            'successful_methods': sum(
                1 for m in results['methods'].values() if m['success']
            ),
            'avg_size_reduction': np.mean([
                self.original_state['model_size'] / size
                for size in results['model_sizes'].values()
            ]) if results['model_sizes'] else 0,
            'avg_quantization_time': np.mean(list(results['quantization_time'].values()))
        }
        
        self.results['quantization_methods'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Quantization methods test results: {stats}")
        return stats
    
    def _calibrate_model(
        self,
        model: torch.nn.Module,
        num_batches: int = 10,
        batch_size: int = 32,
        sequence_length: int = 512
    ):
        """Calibrate model for static quantization."""
        model.eval()
        with torch.no_grad():
            for _ in range(num_batches):
                input_ids = torch.randint(
                    0, model.config.vocab_size,
                    (batch_size, sequence_length),
                    device=self.device
                )
                model(input_ids)
    
    def test_performance_impact(
        self,
        quantized_models: Dict[str, torch.nn.Module],
        num_runs: int = 100,
        batch_size: int = 32,
        sequence_length: int = 512
    ) -> Dict:
        """Test performance impact of quantization."""
        logger.info("Testing performance impact...")
        
        results = {
            'inference_time': {},
            'throughput': {},
            'memory_usage': {},
            'cpu_usage': {}
        }
        
        # Test original model first
        results['inference_time']['original'] = []
        results['memory_usage']['original'] = []
        results['cpu_usage']['original'] = []
        
        input_ids = torch.randint(
            0, self.model.config.vocab_size,
            (batch_size, sequence_length),
            device=self.device
        )
        
        # Warmup
        for _ in range(3):
            self.model(input_ids)
        
        # Benchmark original model
        for _ in range(num_runs):
            torch.cuda.empty_cache()
            start_time = time.time()
            self.model(input_ids)
            inference_time = time.time() - start_time
            
            results['inference_time']['original'].append(inference_time)
            results['memory_usage']['original'].append(
                torch.cuda.memory_allocated() / (1024 * 1024)
            )
            results['cpu_usage']['original'].append(
                psutil.cpu_percent()
            )
        
        # Test each quantized model
        for method, model in quantized_models.items():
            results['inference_time'][method] = []
            results['memory_usage'][method] = []
            results['cpu_usage'][method] = []
            
            # Warmup
            for _ in range(3):
                model(input_ids)
            
            # Benchmark
            for _ in range(num_runs):
                torch.cuda.empty_cache()
                start_time = time.time()
                model(input_ids)
                inference_time = time.time() - start_time
                
                results['inference_time'][method].append(inference_time)
                results['memory_usage'][method].append(
                    torch.cuda.memory_allocated() / (1024 * 1024)
                )
                results['cpu_usage'][method].append(
                    psutil.cpu_percent()
                )
        
        # Calculate throughput
        for method in ['original'] + list(quantized_models.keys()):
            avg_time = np.mean(results['inference_time'][method])
            results['throughput'][method] = (batch_size * sequence_length) / avg_time
        
        # Compute statistics
        stats = {
            'speedup': {
                method: (
                    np.mean(results['inference_time']['original']) /
                    np.mean(results['inference_time'][method])
                )
                for method in quantized_models.keys()
            },
            'memory_reduction': {
                method: (
                    np.mean(results['memory_usage']['original']) /
                    np.mean(results['memory_usage'][method])
                )
                for method in quantized_models.keys()
            },
            'throughput_improvement': {
                method: (
                    results['throughput'][method] /
                    results['throughput']['original']
                )
                for method in quantized_models.keys()
            }
        }
        
        self.results['performance_impact'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Performance impact test results: {stats}")
        return stats
    
    def test_output_quality(
        self,
        quantized_models: Dict[str, torch.nn.Module],
        num_samples: int = 100,
        batch_size: int = 32,
        sequence_length: int = 512
    ) -> Dict:
        """Test output quality of quantized models."""
        logger.info("Testing output quality...")
        
        results = {
            'output_diff': {},
            'correlation': {},
            'relative_error': {}
        }
        
        for method, model in quantized_models.items():
            results['output_diff'][method] = []
            results['correlation'][method] = []
            results['relative_error'][method] = []
            
            for _ in range(num_samples):
                # Generate input
                input_ids = torch.randint(
                    0, self.model.config.vocab_size,
                    (batch_size, sequence_length),
                    device=self.device
                )
                
                # Get outputs
                with torch.no_grad():
                    original_output = self.model(input_ids)
                    quantized_output = model(input_ids)
                
                # Calculate metrics
                diff = torch.abs(original_output - quantized_output)
                correlation = torch.corrcoef(
                    torch.stack([
                        original_output.view(-1),
                        quantized_output.view(-1)
                    ])
                )[0, 1]
                relative_error = torch.norm(diff) / torch.norm(original_output)
                
                results['output_diff'][method].append(diff.mean().item())
                results['correlation'][method].append(correlation.item())
                results['relative_error'][method].append(relative_error.item())
        
        # Compute statistics
        stats = {
            'avg_output_diff': {
                method: np.mean(diffs)
                for method, diffs in results['output_diff'].items()
            },
            'avg_correlation': {
                method: np.mean(corrs)
                for method, corrs in results['correlation'].items()
            },
            'avg_relative_error': {
                method: np.mean(errors)
                for method, errors in results['relative_error'].items()
            }
        }
        
        self.results['output_quality'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Output quality test results: {stats}")
        return stats
    
    def test_memory_reduction(
        self,
        quantized_models: Dict[str, torch.nn.Module]
    ) -> Dict:
        """Test memory reduction achieved by quantization."""
        logger.info("Testing memory reduction...")
        
        results = {
            'model_size': {'original': self.original_state['model_size']},
            'memory_usage': {'original': []},
            'peak_memory': {'original': []},
            'state_dict_size': {'original': self._get_state_dict_size(self.model)}
        }
        
        # Measure memory usage for original model
        input_ids = torch.randint(
            0, self.model.config.vocab_size,
            (32, 512),
            device=self.device
        )
        
        torch.cuda.empty_cache()
        self.model(input_ids)
        results['memory_usage']['original'].append(
            torch.cuda.memory_allocated() / (1024 * 1024)
        )
        results['peak_memory']['original'].append(
            torch.cuda.max_memory_allocated() / (1024 * 1024)
        )
        
        # Test each quantized model
        for method, model in quantized_models.items():
            results['model_size'][method] = self._get_model_size(model)
            results['memory_usage'][method] = []
            results['peak_memory'][method] = []
            results['state_dict_size'][method] = self._get_state_dict_size(model)
            
            torch.cuda.empty_cache()
            model(input_ids)
            results['memory_usage'][method].append(
                torch.cuda.memory_allocated() / (1024 * 1024)
            )
            results['peak_memory'][method].append(
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            )
        
        # Compute statistics
        stats = {
            'size_reduction': {
                method: results['model_size']['original'] / size
                for method, size in results['model_size'].items()
                if method != 'original'
            },
            'memory_reduction': {
                method: (
                    np.mean(results['memory_usage']['original']) /
                    np.mean(results['memory_usage'][method])
                )
                for method in quantized_models.keys()
            },
            'state_dict_reduction': {
                method: (
                    results['state_dict_size']['original'] /
                    results['state_dict_size'][method]
                )
                for method in quantized_models.keys()
            }
        }
        
        self.results['memory_reduction'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Memory reduction test results: {stats}")
        return stats
    
    def _get_state_dict_size(self, model: torch.nn.Module) -> float:
        """Calculate size of model's state dict in MB."""
        with io.BytesIO() as buffer:
            torch.save(model.state_dict(), buffer)
            return buffer.tell() / (1024 * 1024)
    
    def test_hardware_compatibility(
        self,
        quantized_models: Dict[str, torch.nn.Module]
    ) -> Dict:
        """Test hardware compatibility of quantized models."""
        logger.info("Testing hardware compatibility...")
        
        results = {
            'cpu_compatibility': {},
            'gpu_compatibility': {},
            'inference_success': {}
        }
        
        input_ids = torch.randint(
            0, self.model.config.vocab_size,
            (32, 512)
        )
        
        for method, model in quantized_models.items():
            # Test CPU compatibility
            try:
                cpu_model = model.cpu()
                cpu_model(input_ids.cpu())
                results['cpu_compatibility'][method] = True
            except Exception as e:
                logger.error(f"CPU compatibility error for {method}: {str(e)}")
                results['cpu_compatibility'][method] = False
            
            # Test GPU compatibility
            try:
                gpu_model = model.cuda()
                gpu_model(input_ids.cuda())
                results['gpu_compatibility'][method] = True
            except Exception as e:
                logger.error(f"GPU compatibility error for {method}: {str(e)}")
                results['gpu_compatibility'][method] = False
            
            # Test inference success
            results['inference_success'][method] = (
                results['cpu_compatibility'][method] or
                results['gpu_compatibility'][method]
            )
        
        # Compute statistics
        stats = {
            'cpu_compatible_methods': sum(results['cpu_compatibility'].values()),
            'gpu_compatible_methods': sum(results['gpu_compatibility'].values()),
            'fully_compatible_methods': sum(
                1 for method in quantized_models.keys()
                if results['cpu_compatibility'][method] and
                results['gpu_compatibility'][method]
            )
        }
        
        self.results['hardware_compatibility'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Hardware compatibility test results: {stats}")
        return stats
    
    def generate_visualizations(self, save_dir: str = 'quantization_tests'):
        """Generate visualizations of test results."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot performance impact
        if self.results['performance_impact']:
            perf = self.results['performance_impact'][-1]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            methods = ['original'] + list(perf['raw_results']['throughput'].keys())
            throughputs = [perf['raw_results']['throughput'][m] for m in methods]
            plt.bar(range(len(methods)), throughputs)
            plt.xticks(range(len(methods)), methods, rotation=45)
            plt.title('Throughput Comparison')
            plt.ylabel('Tokens/second')
            
            plt.subplot(1, 3, 2)
            memory_usage = [
                np.mean(perf['raw_results']['memory_usage'][m])
                for m in methods
            ]
            plt.bar(range(len(methods)), memory_usage)
            plt.xticks(range(len(methods)), methods, rotation=45)
            plt.title('Memory Usage')
            plt.ylabel('Memory (MB)')
            
            plt.subplot(1, 3, 3)
            cpu_usage = [
                np.mean(perf['raw_results']['cpu_usage'][m])
                for m in methods
            ]
            plt.bar(range(len(methods)), cpu_usage)
            plt.xticks(range(len(methods)), methods, rotation=45)
            plt.title('CPU Usage')
            plt.ylabel('CPU %')
            
            plt.tight_layout()
            plt.savefig(save_path / 'performance_impact.png')
            plt.close()
        
        # Plot output quality
        if self.results['output_quality']:
            quality = self.results['output_quality'][-1]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            methods = list(quality['raw_results']['output_diff'].keys())
            diffs = [np.mean(quality['raw_results']['output_diff'][m]) for m in methods]
            plt.bar(range(len(methods)), diffs)
            plt.xticks(range(len(methods)), methods, rotation=45)
            plt.title('Average Output Difference')
            plt.ylabel('Difference')
            
            plt.subplot(1, 3, 2)
            corrs = [np.mean(quality['raw_results']['correlation'][m]) for m in methods]
            plt.bar(range(len(methods)), corrs)
            plt.xticks(range(len(methods)), methods, rotation=45)
            plt.title('Output Correlation')
            plt.ylabel('Correlation')
            
            plt.subplot(1, 3, 3)
            errors = [np.mean(quality['raw_results']['relative_error'][m]) for m in methods]
            plt.bar(range(len(methods)), errors)
            plt.xticks(range(len(methods)), methods, rotation=45)
            plt.title('Relative Error')
            plt.ylabel('Error')
            
            plt.tight_layout()
            plt.savefig(save_path / 'output_quality.png')
            plt.close()
        
        # Plot memory reduction
        if self.results['memory_reduction']:
            memory = self.results['memory_reduction'][-1]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            methods = list(memory['raw_results']['model_size'].keys())
            sizes = [memory['raw_results']['model_size'][m] for m in methods]
            plt.bar(range(len(methods)), sizes)
            plt.xticks(range(len(methods)), methods, rotation=45)
            plt.title('Model Size')
            plt.ylabel('Size (MB)')
            
            plt.subplot(1, 3, 2)
            memory_usage = [
                np.mean(memory['raw_results']['memory_usage'][m])
                for m in methods
            ]
            plt.bar(range(len(methods)), memory_usage)
            plt.xticks(range(len(methods)), methods, rotation=45)
            plt.title('Memory Usage')
            plt.ylabel('Memory (MB)')
            
            plt.subplot(1, 3, 3)
            state_dict_sizes = [memory['raw_results']['state_dict_size'][m] for m in methods]
            plt.bar(range(len(methods)), state_dict_sizes)
            plt.xticks(range(len(methods)), methods, rotation=45)
            plt.title('State Dict Size')
            plt.ylabel('Size (MB)')
            
            plt.tight_layout()
            plt.savefig(save_path / 'memory_reduction.png')
            plt.close()
    
    def save_results(self, save_dir: str = 'quantization_tests'):
        """Save test results to JSON file."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = save_path / f'quantization_results_{timestamp}.json'
        
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

def run_quantization_tests(model: Optional[LargeModel] = None):
    """Run all quantization tests and generate report."""
    logger.info("Starting model quantization tests...")
    
    tester = ModelQuantizationTester(model)
    
    # Run quantization methods test first to get quantized models
    quantization_stats = tester.test_quantization_methods()
    
    # Get successful quantized models
    quantized_models = {
        method: model
        for method, info in quantization_stats['methods'].items()
        if info['success']
    }
    
    # Run remaining tests with quantized models
    if quantized_models:
        tester.test_performance_impact(quantized_models)
        tester.test_output_quality(quantized_models)
        tester.test_memory_reduction(quantized_models)
        tester.test_hardware_compatibility(quantized_models)
    
    # Generate outputs
    tester.generate_visualizations()
    tester.save_results()
    
    logger.info("Quantization tests completed. Check quantization_tests directory for outputs.")

if __name__ == '__main__':
    run_quantization_tests() 