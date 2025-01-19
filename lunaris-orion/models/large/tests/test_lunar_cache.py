"""
LunarCache Test Suite
--------------------
Comprehensive tests for the LunarCache memory system, including:
- Pattern storage and retrieval
- Cache hit rates and efficiency
- Memory management
- Priority system
- Performance metrics
"""

import torch
import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import json
from datetime import datetime

from ..lunar_cache import LunarCache, CacheConfig
from ..model import LargeModel, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LunarCacheTestSuite:
    """Test suite for LunarCache system with metrics collection."""
    
    def __init__(self, cache_size: int = 1024, pattern_dim: int = 1024):
        self.config = CacheConfig(
            stvm_size=cache_size,
            pattern_dim=pattern_dim,
            cache_threshold=0.85,
            max_patterns=100,
            temperature=0.1,
            enable_logging=True,
            priority_threshold=0.9,
            cleanup_frequency=1000
        )
        self.cache = LunarCache(self.config)
        self.metrics = {
            'hit_rates': [],
            'memory_usage': [],
            'pattern_scores': [],
            'priorities': [],
            'query_times': [],
            'cache_sizes': []
        }
        self.test_results = {}
    
    def generate_test_patterns(
        self,
        num_patterns: int,
        batch_size: int = 32
    ) -> torch.Tensor:
        """Generate test patterns with controlled similarity."""
        patterns = []
        base_pattern = torch.randn(self.config.pattern_dim)
        
        for i in range(num_patterns):
            # Create variations of base pattern
            noise = torch.randn(self.config.pattern_dim) * 0.1
            pattern = base_pattern + noise
            pattern = pattern / pattern.norm()
            patterns.append(pattern)
        
        return torch.stack(patterns)
    
    def run_cache_benchmark(
        self,
        num_patterns: int = 1000,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """Run comprehensive cache benchmark."""
        logger.info("Starting cache benchmark...")
        
        # Generate test patterns
        patterns = self.generate_test_patterns(num_patterns, batch_size)
        
        # Metrics to track
        total_hits = 0
        total_queries = 0
        memory_usage = []
        query_times = []
        
        # Run benchmark
        for i in range(0, num_patterns, batch_size):
            batch = patterns[i:i+batch_size]
            
            # Time the query
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            enhanced_patterns, cache_info = self.cache(batch)
            end_time.record()
            
            torch.cuda.synchronize()
            query_time = start_time.elapsed_time(end_time)
            
            # Update metrics
            total_hits += sum(info['hit'] for info in cache_info['cache_info'])
            total_queries += len(batch)
            query_times.append(query_time)
            
            # Track memory
            memory_usage.append(
                torch.cuda.memory_allocated() / 1024**2  # MB
            )
        
        # Calculate final metrics
        metrics = {
            'hit_rate': total_hits / total_queries,
            'avg_query_time_ms': np.mean(query_times),
            'max_memory_mb': max(memory_usage),
            'avg_memory_mb': np.mean(memory_usage),
            'total_patterns_processed': num_patterns
        }
        
        # Update test metrics
        self.metrics['hit_rates'].append(metrics['hit_rate'])
        self.metrics['memory_usage'].append(metrics['max_memory_mb'])
        self.metrics['query_times'].append(metrics['avg_query_time_ms'])
        
        logger.info(f"Benchmark results: {metrics}")
        return metrics
    
    def test_priority_system(self, num_patterns: int = 500) -> Dict[str, float]:
        """Test the priority system of the cache."""
        logger.info("Testing priority system...")
        
        # Generate patterns with different priorities
        patterns = self.generate_test_patterns(num_patterns)
        priorities = torch.linspace(0, 1, num_patterns)
        
        high_priority_hits = 0
        low_priority_hits = 0
        
        # Process patterns
        for i, (pattern, priority) in enumerate(zip(patterns, priorities)):
            score = self.cache.score_pattern(pattern)[0]
            if score > self.config.cache_threshold:
                self.cache.stvm.update(pattern, score.item(), priority.item())
        
        # Query patterns and check priorities
        for pattern in patterns:
            cached_patterns, similarities = self.cache.stvm.query(pattern)
            if len(cached_patterns) > 0:
                max_similarity = similarities.max().item()
                if max_similarity > self.config.priority_threshold:
                    high_priority_hits += 1
                else:
                    low_priority_hits += 1
        
        results = {
            'high_priority_hits': high_priority_hits,
            'low_priority_hits': low_priority_hits,
            'total_queries': num_patterns,
            'high_priority_ratio': high_priority_hits / num_patterns
        }
        
        logger.info(f"Priority system test results: {results}")
        return results
    
    def generate_visualizations(self, save_dir: str = 'test_results'):
        """Generate visualizations of cache performance."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot hit rates over time
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['hit_rates'])
        plt.title('Cache Hit Rate Over Time')
        plt.xlabel('Test Iteration')
        plt.ylabel('Hit Rate')
        plt.savefig(save_path / 'hit_rates.png')
        plt.close()
        
        # Plot memory usage
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['memory_usage'])
        plt.title('Memory Usage Over Time')
        plt.xlabel('Test Iteration')
        plt.ylabel('Memory Usage (MB)')
        plt.savefig(save_path / 'memory_usage.png')
        plt.close()
        
        # Plot query times
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['query_times'])
        plt.title('Query Times Over Time')
        plt.xlabel('Test Iteration')
        plt.ylabel('Query Time (ms)')
        plt.savefig(save_path / 'query_times.png')
        plt.close()
    
    def save_metrics(self, save_dir: str = 'test_results'):
        """Save test metrics to JSON file."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = save_path / f'cache_metrics_{timestamp}.json'
        
        with open(metrics_file, 'w') as f:
            json.dump({
                'config': self.config.__dict__,
                'metrics': self.metrics,
                'test_results': self.test_results
            }, f, indent=4)
        
        logger.info(f"Saved metrics to {metrics_file}")

@pytest.fixture
def cache_config():
    return CacheConfig(
        stvm_size=1024,
        pattern_dim=1024,
        cache_threshold=0.85,
        max_patterns=100,
        temperature=0.1,
        enable_logging=True,
        priority_threshold=0.9,
        cleanup_frequency=1000
    )

@pytest.fixture
def lunar_cache(cache_config):
    return LunarCache(cache_config)

def test_cache_initialization(lunar_cache):
    """Test if cache is properly initialized."""
    assert lunar_cache is not None
    assert lunar_cache.stvm is not None
    assert lunar_cache.pattern_scorer is not None

def test_pattern_generation(lunar_cache):
    """Test pattern generation and scoring."""
    pattern = torch.randn(1, lunar_cache.config.pattern_dim)
    score, priority = lunar_cache.score_pattern(pattern)
    
    assert score is not None
    assert priority is not None
    assert 0 <= score <= 1
    assert 0 <= priority <= 1

def test_cache_update_and_query(lunar_cache):
    """Test cache update and query operations."""
    # Generate test pattern
    pattern = torch.randn(1, lunar_cache.config.pattern_dim)
    
    # Update cache
    score, priority = lunar_cache.score_pattern(pattern)
    lunar_cache.stvm.update(pattern[0], score.item(), priority.item())
    
    # Query cache
    cached_patterns, similarities = lunar_cache.stvm.query(pattern[0])
    
    assert len(cached_patterns) > 0
    assert len(similarities) > 0
    assert similarities[0] > lunar_cache.config.cache_threshold

def test_cache_memory_management(lunar_cache):
    """Test cache memory management and cleanup."""
    initial_patterns = (~lunar_cache.stvm.empty_slots).sum().item()
    
    # Add many patterns
    for _ in range(100):
        pattern = torch.randn(1, lunar_cache.config.pattern_dim)
        score, priority = lunar_cache.score_pattern(pattern)
        lunar_cache.stvm.update(pattern[0], score.item(), priority.item())
    
    # Force cleanup
    lunar_cache.stvm._cleanup_memory()
    
    final_patterns = (~lunar_cache.stvm.empty_slots).sum().item()
    assert final_patterns <= lunar_cache.config.stvm_size

def run_all_tests():
    """Run all cache tests and generate report."""
    logger.info("Starting comprehensive cache testing...")
    
    # Initialize test suite
    test_suite = LunarCacheTestSuite()
    
    # Run tests
    test_suite.test_results['benchmark'] = test_suite.run_cache_benchmark()
    test_suite.test_results['priority'] = test_suite.test_priority_system()
    
    # Generate visualizations
    test_suite.generate_visualizations()
    
    # Save metrics
    test_suite.save_metrics()
    
    logger.info("Testing completed. Check test_results directory for outputs.")

if __name__ == '__main__':
    run_all_tests() 