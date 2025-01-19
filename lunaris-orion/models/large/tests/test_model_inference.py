"""
Model Inference Tests
-------------------
Tests the Large model's inference functionality, including:
- Generation quality
- Inference speed
- Memory efficiency
- Batch processing
- Error handling
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
from PIL import Image
import io
import pytest

from ..model import LargeModel, ModelConfig
from ..lunar_cache import LunarCache, CacheConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def model_config():
    return ModelConfig(
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        ffn_dim=4096,
        max_sequence_length=2048,
        image_size=64,
        dropout=0.1
    )

@pytest.fixture
def model(model_config):
    return LargeModel(model_config)

def test_model_initialization(model):
    """Test if model is properly initialized."""
    assert model is not None
    assert isinstance(model, LargeModel)
    assert model.config.embedding_dim == 1024
    assert model.config.num_heads == 16

def test_model_forward_pass(model):
    """Test model forward pass."""
    batch_size = 2
    seq_length = 128
    
    # Create dummy input
    input_ids = torch.randint(
        0, model.config.vocab_size,
        (batch_size, seq_length)
    )
    
    # Run forward pass
    with torch.no_grad():
        output = model(input_ids)
    
    # Check output shape
    expected_shape = (batch_size, 3, 64, 64)  # Assuming 64x64 pixel art output
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()

def test_model_generation(model):
    """Test model generation with a prompt."""
    prompt = "A pixel art landscape"
    
    with torch.no_grad():
        output = model.generate(prompt, temperature=0.7)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape[1] == 3  # RGB channels
    assert 0 <= output.min() <= output.max() <= 1  # Check value range

def test_model_batch_processing(model):
    """Test model batch processing capabilities."""
    batch_size = 4
    seq_length = 64
    
    # Create batch of inputs
    input_ids = torch.randint(
        0, model.config.vocab_size,
        (batch_size, seq_length)
    )
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    assert len(outputs) == batch_size
    assert all(not torch.isnan(output).any() for output in outputs)

class ModelInferenceTester:
    """Tests model inference functionality."""
    
    def __init__(self, model: Optional[LargeModel] = None):
        self.model = model if model is not None else self._create_test_model()
        self.results = {
            'generation_quality': [],
            'inference_speed': [],
            'memory_efficiency': [],
            'batch_processing': [],
            'error_handling': []
        }
        
        # Initialize device
        self.device = next(self.model.parameters()).device
    
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
    
    def test_generation_quality(
        self,
        num_samples: int = 10,
        prompts: Optional[List[str]] = None,
        save_images: bool = True
    ) -> Dict:
        """Test generation quality with different prompts."""
        logger.info("Testing generation quality...")
        
        if prompts is None:
            prompts = [
                "A pixel art landscape with mountains and trees",
                "A pixel art character in a fighting pose",
                "A pixel art spaceship in orbit",
                "A pixel art dungeon with treasure",
                "A pixel art city at night"
            ]
        
        results = {
            'generations': [],
            'generation_times': [],
            'prompt_lengths': [],
            'image_sizes': []
        }
        
        save_path = Path('inference_tests/generations')
        if save_images:
            save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                for j in range(num_samples // len(prompts)):
                    start_time = time.time()
                    
                    # Generate image
                    try:
                        output = self.model.generate(
                            prompt,
                            temperature=0.8,
                            top_p=0.9
                        )
                        
                        generation_time = time.time() - start_time
                        
                        # Convert output tensor to image
                        if isinstance(output, torch.Tensor):
                            image = self._tensor_to_image(output)
                        else:
                            image = output
                        
                        # Save image if requested
                        if save_images:
                            image_path = save_path / f"sample_{i}_{j}.png"
                            image.save(image_path)
                        
                        # Record results
                        results['generations'].append({
                            'prompt': prompt,
                            'success': True,
                            'image_path': str(image_path) if save_images else None
                        })
                        results['generation_times'].append(generation_time)
                        results['prompt_lengths'].append(len(prompt))
                        results['image_sizes'].append(image.size)
                        
                    except Exception as e:
                        logger.error(f"Generation failed for prompt: {prompt}")
                        logger.error(str(e))
                        results['generations'].append({
                            'prompt': prompt,
                            'success': False,
                            'error': str(e)
                        })
        
        # Compute statistics
        stats = {
            'success_rate': sum(
                1 for g in results['generations'] if g['success']
            ) / len(results['generations']),
            'avg_generation_time': np.mean(results['generation_times']),
            'std_generation_time': np.std(results['generation_times']),
            'avg_prompt_length': np.mean(results['prompt_lengths']),
            'unique_image_sizes': list(set(results['image_sizes']))
        }
        
        self.results['generation_quality'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Generation quality test results: {stats}")
        return stats
    
    def test_inference_speed(
        self,
        num_runs: int = 100,
        batch_sizes: List[int] = [1, 4, 8, 16],
        sequence_lengths: List[int] = [128, 256, 512, 1024]
    ) -> Dict:
        """Test inference speed with different configurations."""
        logger.info("Testing inference speed...")
        
        results = {
            'latencies': {},
            'throughput': {},
            'memory_usage': {}
        }
        
        self.model.eval()
        with torch.no_grad():
            for batch_size in batch_sizes:
                for seq_len in sequence_lengths:
                    key = f"batch_{batch_size}_seq_{seq_len}"
                    results['latencies'][key] = []
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
                    
                    for _ in range(num_runs):
                        torch.cuda.empty_cache()
                        start_event.record()
                        self.model(input_ids)
                        end_event.record()
                        torch.cuda.synchronize()
                        
                        results['latencies'][key].append(
                            start_event.elapsed_time(end_event)
                        )
                        results['memory_usage'][key].append(
                            torch.cuda.memory_allocated() / (1024 * 1024)
                        )
                    
                    # Calculate throughput (tokens/second)
                    avg_latency = np.mean(results['latencies'][key])
                    throughput = (batch_size * seq_len) / (avg_latency / 1000)
                    results['throughput'][key] = throughput
                    
                    logger.info(
                        f"Config {key}: "
                        f"Avg latency = {avg_latency:.2f}ms, "
                        f"Throughput = {throughput:.2f} tokens/s"
                    )
        
        # Compute statistics
        stats = {
            'min_latency': min(
                np.mean(lats) for lats in results['latencies'].values()
            ),
            'max_throughput': max(results['throughput'].values()),
            'avg_memory_usage': {
                key: np.mean(usage)
                for key, usage in results['memory_usage'].items()
            }
        }
        
        self.results['inference_speed'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Inference speed test results: {stats}")
        return stats
    
    def test_memory_efficiency(
        self,
        max_batch_size: int = 64,
        sequence_length: int = 512,
        step_size: int = 4
    ) -> Dict:
        """Test memory efficiency with increasing batch sizes."""
        logger.info("Testing memory efficiency...")
        
        results = {
            'batch_sizes': [],
            'memory_usage': [],
            'peak_memory': [],
            'oom_occurred': False,
            'max_stable_batch': None
        }
        
        self.model.eval()
        with torch.no_grad():
            batch_size = step_size
            while batch_size <= max_batch_size:
                try:
                    torch.cuda.empty_cache()
                    
                    # Generate input
                    input_ids = torch.randint(
                        0, self.model.config.vocab_size,
                        (batch_size, sequence_length),
                        device=self.device
                    )
                    
                    # Forward pass
                    self.model(input_ids)
                    
                    # Record memory usage
                    current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    
                    results['batch_sizes'].append(batch_size)
                    results['memory_usage'].append(current_memory)
                    results['peak_memory'].append(peak_memory)
                    
                    logger.info(
                        f"Batch size {batch_size}: "
                        f"Memory = {current_memory:.2f}MB, "
                        f"Peak = {peak_memory:.2f}MB"
                    )
                    
                    batch_size += step_size
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        results['oom_occurred'] = True
                        results['max_stable_batch'] = batch_size - step_size
                        logger.warning(
                            f"OOM at batch size {batch_size}, "
                            f"max stable batch size: {results['max_stable_batch']}"
                        )
                        break
                    else:
                        raise e
        
        # Compute statistics
        stats = {
            'max_stable_batch': results['max_stable_batch'],
            'memory_per_sample': np.mean([
                mem / bs
                for mem, bs in zip(results['memory_usage'], results['batch_sizes'])
            ]),
            'peak_memory': max(results['peak_memory']) if results['peak_memory'] else 0,
            'memory_scaling': np.polyfit(
                results['batch_sizes'],
                results['memory_usage'],
                1
            )[0] if len(results['batch_sizes']) > 1 else 0
        }
        
        self.results['memory_efficiency'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Memory efficiency test results: {stats}")
        return stats
    
    def test_batch_processing(
        self,
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
        num_batches: int = 10
    ) -> Dict:
        """Test batch processing capabilities."""
        logger.info("Testing batch processing...")
        
        results = {
            'processing_times': {},
            'success_rates': {},
            'memory_usage': {},
            'output_consistency': {}
        }
        
        self.model.eval()
        with torch.no_grad():
            for batch_size in batch_sizes:
                results['processing_times'][batch_size] = []
                results['success_rates'][batch_size] = []
                results['memory_usage'][batch_size] = []
                results['output_consistency'][batch_size] = []
                
                for _ in range(num_batches):
                    try:
                        # Generate prompts
                        prompts = [
                            "A pixel art scene"
                            for _ in range(batch_size)
                        ]
                        
                        start_time = time.time()
                        
                        # Generate images
                        outputs = self.model.generate_batch(
                            prompts,
                            temperature=0.8,
                            top_p=0.9
                        )
                        
                        processing_time = time.time() - start_time
                        
                        # Record results
                        results['processing_times'][batch_size].append(processing_time)
                        results['success_rates'][batch_size].append(1.0)
                        results['memory_usage'][batch_size].append(
                            torch.cuda.memory_allocated() / (1024 * 1024)
                        )
                        
                        # Check output consistency
                        if isinstance(outputs, torch.Tensor):
                            consistency = torch.std(
                                outputs.float().view(batch_size, -1),
                                dim=0
                            ).mean().item()
                        else:
                            consistency = 0.0
                        results['output_consistency'][batch_size].append(consistency)
                        
                    except Exception as e:
                        logger.error(f"Batch processing failed for size {batch_size}")
                        logger.error(str(e))
                        results['success_rates'][batch_size].append(0.0)
        
        # Compute statistics
        stats = {
            'avg_processing_time': {
                bs: np.mean(times)
                for bs, times in results['processing_times'].items()
            },
            'success_rate': {
                bs: np.mean(rates)
                for bs, rates in results['success_rates'].items()
            },
            'avg_memory_usage': {
                bs: np.mean(usage)
                for bs, usage in results['memory_usage'].items()
            },
            'avg_consistency': {
                bs: np.mean(cons)
                for bs, cons in results['output_consistency'].items()
            }
        }
        
        self.results['batch_processing'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Batch processing test results: {stats}")
        return stats
    
    def test_error_handling(self) -> Dict:
        """Test error handling during inference."""
        logger.info("Testing error handling...")
        
        results = {
            'error_cases': [],
            'recovery_success': []
        }
        
        # Test cases
        test_cases = [
            {
                'name': 'empty_prompt',
                'prompt': "",
                'expected_error': True
            },
            {
                'name': 'very_long_prompt',
                'prompt': "a " * 1000,
                'expected_error': True
            },
            {
                'name': 'invalid_temperature',
                'prompt': "A pixel art scene",
                'temperature': -1.0,
                'expected_error': True
            },
            {
                'name': 'normal_case',
                'prompt': "A pixel art scene",
                'expected_error': False
            }
        ]
        
        for case in test_cases:
            try:
                if 'temperature' in case:
                    output = self.model.generate(
                        case['prompt'],
                        temperature=case['temperature']
                    )
                else:
                    output = self.model.generate(case['prompt'])
                
                results['error_cases'].append({
                    'case': case['name'],
                    'error_occurred': False,
                    'expected_error': case['expected_error']
                })
                
                if not case['expected_error']:
                    results['recovery_success'].append(True)
                
            except Exception as e:
                results['error_cases'].append({
                    'case': case['name'],
                    'error_occurred': True,
                    'error_message': str(e),
                    'expected_error': case['expected_error']
                })
                
                if case['expected_error']:
                    results['recovery_success'].append(True)
                else:
                    results['recovery_success'].append(False)
        
        # Test memory recovery
        try:
            initial_memory = torch.cuda.memory_allocated()
            
            # Try to cause OOM
            large_batch = torch.randint(
                0, self.model.config.vocab_size,
                (1000, 1000),
                device=self.device
            )
            self.model(large_batch)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Check if memory is properly recovered
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()
                memory_recovered = final_memory <= initial_memory * 1.1  # Allow 10% overhead
                results['recovery_success'].append(memory_recovered)
        
        # Compute statistics
        stats = {
            'expected_errors_caught': sum(
                1 for case in results['error_cases']
                if case['error_occurred'] == case['expected_error']
            ) / len(results['error_cases']),
            'recovery_success_rate': np.mean(results['recovery_success'])
        }
        
        self.results['error_handling'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Error handling test results: {stats}")
        return stats
    
    def generate_visualizations(self, save_dir: str = 'inference_tests'):
        """Generate visualizations of test results."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot inference speed
        if self.results['inference_speed']:
            speed = self.results['inference_speed'][-1]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            for key, latencies in speed['raw_results']['latencies'].items():
                plt.plot(latencies, label=key)
            plt.title('Inference Latencies')
            plt.xlabel('Run')
            plt.ylabel('Latency (ms)')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            throughputs = speed['raw_results']['throughput']
            plt.bar(range(len(throughputs)), list(throughputs.values()))
            plt.xticks(
                range(len(throughputs)),
                list(throughputs.keys()),
                rotation=45
            )
            plt.title('Throughput')
            plt.ylabel('Tokens/second')
            
            plt.subplot(1, 3, 3)
            memory_usage = speed['stats']['avg_memory_usage']
            plt.bar(range(len(memory_usage)), list(memory_usage.values()))
            plt.xticks(
                range(len(memory_usage)),
                list(memory_usage.keys()),
                rotation=45
            )
            plt.title('Average Memory Usage')
            plt.ylabel('Memory (MB)')
            
            plt.tight_layout()
            plt.savefig(save_path / 'inference_speed.png')
            plt.close()
        
        # Plot memory efficiency
        if self.results['memory_efficiency']:
            memory = self.results['memory_efficiency'][-1]
            
            plt.figure(figsize=(10, 5))
            
            plt.plot(
                memory['raw_results']['batch_sizes'],
                memory['raw_results']['memory_usage'],
                'b-',
                label='Memory Usage'
            )
            plt.plot(
                memory['raw_results']['batch_sizes'],
                memory['raw_results']['peak_memory'],
                'r--',
                label='Peak Memory'
            )
            
            if memory['raw_results']['max_stable_batch']:
                plt.axvline(
                    x=memory['raw_results']['max_stable_batch'],
                    color='g',
                    linestyle=':',
                    label='Max Stable Batch'
                )
            
            plt.title('Memory Scaling with Batch Size')
            plt.xlabel('Batch Size')
            plt.ylabel('Memory (MB)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(save_path / 'memory_efficiency.png')
            plt.close()
        
        # Plot batch processing
        if self.results['batch_processing']:
            batch = self.results['batch_processing'][-1]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            processing_times = batch['stats']['avg_processing_time']
            plt.bar(
                range(len(processing_times)),
                list(processing_times.values())
            )
            plt.xticks(
                range(len(processing_times)),
                list(processing_times.keys())
            )
            plt.title('Average Processing Time')
            plt.xlabel('Batch Size')
            plt.ylabel('Time (s)')
            
            plt.subplot(1, 3, 2)
            success_rates = batch['stats']['success_rate']
            plt.bar(
                range(len(success_rates)),
                list(success_rates.values())
            )
            plt.xticks(
                range(len(success_rates)),
                list(success_rates.keys())
            )
            plt.title('Success Rate')
            plt.xlabel('Batch Size')
            plt.ylabel('Rate')
            
            plt.subplot(1, 3, 3)
            consistency = batch['stats']['avg_consistency']
            plt.bar(
                range(len(consistency)),
                list(consistency.values())
            )
            plt.xticks(
                range(len(consistency)),
                list(consistency.keys())
            )
            plt.title('Output Consistency')
            plt.xlabel('Batch Size')
            plt.ylabel('Consistency Score')
            
            plt.tight_layout()
            plt.savefig(save_path / 'batch_processing.png')
            plt.close()
    
    def save_results(self, save_dir: str = 'inference_tests'):
        """Save test results to JSON file."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = save_path / f'inference_results_{timestamp}.json'
        
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
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert a tensor to a PIL Image."""
        # Assuming tensor is in the format [C, H, W] with values in [0, 1]
        tensor = tensor.cpu().clamp(0, 1)
        if tensor.shape[0] == 1:
            # Grayscale
            tensor = tensor.repeat(3, 1, 1)
        tensor = (tensor * 255).byte()
        return Image.fromarray(
            tensor.permute(1, 2, 0).numpy(),
            mode='RGB'
        )

def run_inference_tests(model: Optional[LargeModel] = None):
    """Run all inference tests and generate report."""
    logger.info("Starting model inference tests...")
    
    tester = ModelInferenceTester(model)
    
    # Run tests
    tester.test_generation_quality()
    tester.test_inference_speed()
    tester.test_memory_efficiency()
    tester.test_batch_processing()
    tester.test_error_handling()
    
    # Generate outputs
    tester.generate_visualizations()
    tester.save_results()
    
    logger.info("Inference tests completed. Check inference_tests directory for outputs.")

if __name__ == '__main__':
    run_inference_tests() 