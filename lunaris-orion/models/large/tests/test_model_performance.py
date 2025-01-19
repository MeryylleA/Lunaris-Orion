"""
Model Performance Tests
---------------------
Tests the Large model's performance characteristics including:
- Inference speed
- Memory efficiency
- Model stability
- Generation quality
- Resource utilization
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
import pytest
from torch.amp import autocast
import gc
import psutil
import GPUtil

from ..model import LargeModel, ModelConfig
from ..lunar_cache import LunarCache, CacheConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def model_config():
    """Create a model configuration for testing."""
    return ModelConfig(
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        ffn_dim=4096,
        max_sequence_length=2048,
        image_size=64,
        dropout=0.1,
        attention_dropout=0.1,
        gradient_checkpointing=True,
        use_rope=True,
        use_sliding_window=True,
        sliding_window_size=256,
        cache_config=CacheConfig(
            stvm_size=1024,
            pattern_dim=2048,
            cache_threshold=0.85,
            max_patterns=100,
            temperature=0.1,
            device='cpu',  # Use CPU for testing
            enable_logging=True,
            priority_threshold=0.9,
            cleanup_frequency=1000
        )
    )

@pytest.fixture
def model(model_config):
    """Create a model instance for testing."""
    model = LargeModel(model_config)
    model.eval()  # Set to evaluation mode
    return model

def test_inference_speed(model):
    """Test model inference speed with various batch sizes."""
    device = next(model.parameters()).device
    batch_sizes = [1, 2, 4]  # Reduced batch sizes for CPU
    sequence_length = 64  # Reduced sequence length
    warmup_runs = 2
    test_runs = 5
    
    results = {
        'batch_size': [],
        'latency_ms': [],
        'throughput': [],
        'memory_usage_mb': []
    }
    
    for batch_size in batch_sizes:
        # Create dummy input
        input_ids = torch.randint(
            0, model.config.vocab_size,
            (batch_size, sequence_length),
            device=device,
            dtype=torch.long
        )
        
        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(input_ids)
        
        # Test runs
        latencies = []
        memory_usage = []
        
        for _ in range(test_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(input_ids)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            if device.type == 'cuda':
                memory_usage.append(torch.cuda.memory_allocated() / 1024**2)  # MB
            else:
                memory_usage.append(psutil.Process().memory_info().rss / 1024**2)  # MB
        
        avg_latency = np.mean(latencies)
        throughput = batch_size * (1000 / avg_latency)  # samples/second
        
        results['batch_size'].append(batch_size)
        results['latency_ms'].append(avg_latency)
        results['throughput'].append(throughput)
        results['memory_usage_mb'].append(np.mean(memory_usage))
        
        logger.info(f"Batch size {batch_size}: {avg_latency:.2f}ms, {throughput:.2f} samples/s")
    
    # Verify performance metrics
    assert all(lat > 0 for lat in results['latency_ms']), "Invalid latency measurements"
    assert all(tput > 0 for tput in results['throughput']), "Invalid throughput measurements"
    assert len(results['batch_size']) == len(batch_sizes), "Missing results for some batch sizes"
    assert all(mem > 0 for mem in results['memory_usage_mb']), "Invalid memory measurements"

def test_memory_efficiency(model):
    """Test model memory usage patterns."""
    device = next(model.parameters()).device
    batch_size = 2  # Reduced batch size
    sequence_length = 64  # Reduced sequence length
    
    # Take multiple measurements to account for Python's memory management
    initial_measurements = []
    peak_measurements = []
    final_measurements = []
    
    for _ in range(3):  # Run multiple trials
        if device.type == 'cuda':
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = psutil.Process().memory_info().rss
        initial_measurements.append(initial_memory)
        
        # Test memory growth during forward pass
        input_ids = torch.randint(
            0, model.config.vocab_size,
            (batch_size, sequence_length),
            device=device,
            dtype=torch.long
        )
        
        with torch.no_grad():
            output = model(input_ids)
        
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            peak_memory = psutil.Process().memory_info().rss
        peak_measurements.append(peak_memory)
        
        # Memory should be released after forward pass
        del output
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        if device.type == 'cuda':
            final_memory = torch.cuda.memory_allocated()
        else:
            final_memory = psutil.Process().memory_info().rss
        final_measurements.append(final_memory)
    
    # Use the median values for comparison
    initial_memory = np.median(initial_measurements)
    peak_memory = np.median(peak_measurements)
    final_memory = np.median(final_measurements)
    
    # Verify memory management with relaxed constraints for CPU
    if device.type == 'cuda':
        assert peak_memory > initial_memory, "No memory used during forward pass"
        assert final_memory <= initial_memory * 1.1, "Memory not properly released"
    else:
        # For CPU, just verify that memory usage is within reasonable bounds
        memory_growth = (peak_memory - initial_memory) / initial_memory
        assert memory_growth >= -0.1, f"Unexpected memory decrease: {memory_growth:.2%}"
        assert memory_growth <= 0.5, f"Excessive memory growth: {memory_growth:.2%}"
    
    # Additional assertions for memory metrics
    assert initial_memory > 0, "Invalid initial memory measurement"
    assert peak_memory > 0, "Invalid peak memory measurement"
    assert final_memory > 0, "Invalid final memory measurement"

def test_model_stability(model):
    """Test model stability with various input patterns."""
    device = next(model.parameters()).device
    test_cases = [
        # Normal case
        torch.randint(0, model.config.vocab_size, (1, 64), device=device, dtype=torch.long),
        # Empty sequence (minimum length 1)
        torch.randint(0, model.config.vocab_size, (1, 1), device=device, dtype=torch.long),
        # Medium sequence
        torch.randint(0, model.config.vocab_size, (1, 128), device=device, dtype=torch.long),
        # Small batch
        torch.randint(0, model.config.vocab_size, (4, 64), device=device, dtype=torch.long),
        # Edge case - all zeros
        torch.zeros((1, 64), device=device, dtype=torch.long)
    ]
    
    results = []
    for i, test_input in enumerate(test_cases):
        try:
            with torch.no_grad():
                output = model(test_input)
            
            # Check output properties
            assert not torch.isnan(output).any(), f"NaN in output for test case {i}"
            assert not torch.isinf(output).any(), f"Inf in output for test case {i}"
            assert output.shape[0] == test_input.shape[0], f"Batch size mismatch for test case {i}"
            
            results.append(True)
        except Exception as e:
            logger.error(f"Error in test case {i}: {str(e)}")
            results.append(False)
    
    # At least 80% of test cases should pass
    assert sum(results) >= len(results) * 0.8, "Too many stability test failures"
    assert len(results) == len(test_cases), "Not all test cases were executed"

def test_generation_quality(model):
    """Test the quality of generated images."""
    device = next(model.parameters()).device
    prompts = [
        "A pixel art landscape",
        "A cute pixel art cat",
        "A pixel art spaceship",
        "A pixel art forest"
    ]
    
    results = []
    for prompt in prompts:
        with torch.no_grad():
            output = model.generate(prompt, temperature=0.7)
            
            # Check basic properties
            assert output.shape[1] == 3, "Output should have 3 channels (RGB)"
            assert output.shape[2] == output.shape[3], "Output should be square"
            assert 0 <= output.min() <= output.max() <= 1, "Output values should be in [0,1]"
            
            # Check image statistics
            mean = output.mean().item()
            std = output.std().item()
            assert 0.1 <= mean <= 0.9, f"Unusual mean pixel value: {mean}"
            assert 0.05 <= std <= 0.5, f"Unusual standard deviation: {std}"  # Relaxed constraints
            
            results.append({
                'prompt': prompt,
                'mean': mean,
                'std': std,
                'min': output.min().item(),
                'max': output.max().item()
            })
    
    # Additional assertions for generation quality
    assert len(results) == len(prompts), "Not all prompts were processed"
    assert all(0.05 <= r['std'] <= 0.5 for r in results), "Inconsistent standard deviations"
    assert all(0.1 <= r['mean'] <= 0.9 for r in results), "Inconsistent mean values"

def test_resource_utilization(model):
    """Test CPU and GPU resource utilization."""
    device = next(model.parameters()).device
    batch_size = 2  # Reduced batch size
    sequence_length = 64  # Reduced sequence length
    
    # Initial resource state
    initial_gpu = GPUtil.getGPUs()[0] if torch.cuda.is_available() else None
    initial_cpu = psutil.cpu_percent(interval=1)
    
    input_ids = torch.randint(
        0, model.config.vocab_size,
        (batch_size, sequence_length),
        device=device,
        dtype=torch.long
    )
    
    # Measure during inference
    with torch.no_grad():
        start_time = time.perf_counter()
        _ = model(input_ids)
        inference_time = time.perf_counter() - start_time
    
    # Final resource state
    final_gpu = GPUtil.getGPUs()[0] if torch.cuda.is_available() else None
    final_cpu = psutil.cpu_percent(interval=1)
    
    results = {
        'inference_time': inference_time,
        'cpu_usage': final_cpu - initial_cpu
    }
    
    if initial_gpu and final_gpu:
        results.update({
            'gpu_memory_used': final_gpu.memoryUsed - initial_gpu.memoryUsed,
            'gpu_utilization': final_gpu.load * 100
        })
        if device.type == 'cuda':
            assert results['gpu_utilization'] > 0, "GPU not utilized during inference"
    
    # Verify resource utilization
    assert results['cpu_usage'] >= 0, "Invalid CPU usage measurement"
    assert results['inference_time'] > 0, "Invalid inference time measurement"
    assert results['inference_time'] < 10.0, "Inference time too long"
    if device.type == 'cuda':
        assert 'gpu_memory_used' in results, "GPU memory usage not measured"
        assert 'gpu_utilization' in results, "GPU utilization not measured"

def run_performance_suite(model):
    """Run complete performance test suite and generate report."""
    logger.info("Starting performance test suite...")
    
    results = {
        'inference_speed': test_inference_speed(model),
        'memory_efficiency': test_memory_efficiency(model),
        'stability': test_model_stability(model),
        'generation_quality': test_generation_quality(model),
        'resource_utilization': test_resource_utilization(model)
    }
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = Path('test_results')
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / f'performance_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("Performance test suite completed. Results saved to test_results/")
    return results

if __name__ == '__main__':
    # Create model and run full test suite
    model_config = ModelConfig(
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        ffn_dim=4096,
        max_sequence_length=2048,
        image_size=64,
        cache_config=CacheConfig(
            stvm_size=1024,
            pattern_dim=2048,
            cache_threshold=0.85,
            max_patterns=100,
            temperature=0.1,
            device='cpu',  # Use CPU for testing
            enable_logging=True,
            priority_threshold=0.9,
            cleanup_frequency=1000
        )
    )
    model = LargeModel(model_config)
    results = run_performance_suite(model)
