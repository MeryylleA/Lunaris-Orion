"""
Model-Cache Integration Tests
---------------------------
Tests the integration between the Large model and LunarCache system, including:
- Cache impact on model performance
- Memory efficiency
- Pattern reuse effectiveness
- Error handling and recovery
"""

import torch
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from ..model import LargeModel, ModelConfig
from ..lunar_cache import LunarCache, CacheConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCacheIntegrationTester:
    """Tests integration between model and cache system."""
    
    def __init__(self, model: Optional[LargeModel] = None):
        self.model = model if model is not None else self._create_test_model()
        self.results = {
            'cache_impact': [],
            'memory_efficiency': [],
            'pattern_reuse': [],
            'error_handling': []
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
    
    def test_cache_impact(
        self,
        num_samples: int = 1000,
        batch_size: int = 32,
        sequence_length: int = 512
    ) -> Dict:
        """Test impact of cache on model performance."""
        logger.info("Testing cache impact...")
        
        results = {
            'with_cache': {'latency': [], 'memory': []},
            'without_cache': {'latency': [], 'memory': []}
        }
        
        device = next(self.model.parameters()).device
        
        # Test with cache enabled
        self.model.toggle_cache(True)
        logger.info("Testing with cache enabled...")
        
        for i in range(0, num_samples, batch_size):
            batch = torch.randint(
                0, self.model.config.vocab_size,
                (batch_size, sequence_length),
                device=device
            )
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            self.model(batch)
            end_event.record()
            
            torch.cuda.synchronize()
            results['with_cache']['latency'].append(
                start_event.elapsed_time(end_event)
            )
            results['with_cache']['memory'].append(
                torch.cuda.memory_allocated() / 1024**2
            )
        
        # Test without cache
        self.model.toggle_cache(False)
        logger.info("Testing with cache disabled...")
        
        for i in range(0, num_samples, batch_size):
            batch = torch.randint(
                0, self.model.config.vocab_size,
                (batch_size, sequence_length),
                device=device
            )
            
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            self.model(batch)
            end_event.record()
            
            torch.cuda.synchronize()
            results['without_cache']['latency'].append(
                start_event.elapsed_time(end_event)
            )
            results['without_cache']['memory'].append(
                torch.cuda.memory_allocated() / 1024**2
            )
        
        # Compute statistics
        stats = {
            'with_cache': {
                'avg_latency': np.mean(results['with_cache']['latency']),
                'avg_memory': np.mean(results['with_cache']['memory']),
                'latency_std': np.std(results['with_cache']['latency']),
                'memory_std': np.std(results['with_cache']['memory'])
            },
            'without_cache': {
                'avg_latency': np.mean(results['without_cache']['latency']),
                'avg_memory': np.mean(results['without_cache']['memory']),
                'latency_std': np.std(results['without_cache']['latency']),
                'memory_std': np.std(results['without_cache']['memory'])
            }
        }
        
        self.results['cache_impact'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Cache impact test results: {stats}")
        return stats
    
    def test_pattern_reuse(
        self,
        num_patterns: int = 1000,
        similarity_threshold: float = 0.8
    ) -> Dict:
        """Test effectiveness of pattern reuse."""
        logger.info("Testing pattern reuse effectiveness...")
        
        if not self.model.cache_enabled:
            self.model.toggle_cache(True)
        
        results = {
            'reuse_rate': [],
            'pattern_similarities': [],
            'cache_hits': []
        }
        
        # Generate base patterns
        device = next(self.model.parameters()).device
        base_patterns = torch.randn(
            10, self.model.config.embedding_dim,
            device=device
        )
        base_patterns = base_patterns / base_patterns.norm(dim=-1, keepdim=True)
        
        # Generate test patterns with controlled similarity
        for i in range(num_patterns):
            # Select random base pattern
            base_idx = torch.randint(0, len(base_patterns), (1,))
            base = base_patterns[base_idx]
            
            # Add noise to create similar pattern
            noise = torch.randn_like(base) * (1 - similarity_threshold)
            pattern = base + noise
            pattern = pattern / pattern.norm()
            
            # Query cache
            enhanced_pattern, cache_info = self.model.lunar_cache(pattern.unsqueeze(0))
            
            # Record results
            results['reuse_rate'].append(
                float(cache_info['stats']['cache_hits'] / cache_info['stats']['queries'])
            )
            if cache_info['cache_info'][0]['hit']:
                results['pattern_similarities'].append(
                    cache_info['cache_info'][0]['max_similarity']
                )
            results['cache_hits'].append(cache_info['cache_info'][0]['hit'])
        
        # Compute statistics
        stats = {
            'avg_reuse_rate': np.mean(results['reuse_rate']),
            'avg_similarity': np.mean(results['pattern_similarities']) if results['pattern_similarities'] else 0,
            'hit_rate': np.mean(results['cache_hits']),
            'total_patterns': num_patterns
        }
        
        self.results['pattern_reuse'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Pattern reuse test results: {stats}")
        return stats
    
    def test_error_handling(self) -> Dict:
        """Test error handling and recovery."""
        logger.info("Testing error handling...")
        
        results = {
            'recovery_success': [],
            'error_types': []
        }
        
        # Test cache initialization errors
        try:
            bad_config = CacheConfig(stvm_size=-1)  # Invalid size
            self.model.lunar_cache = LunarCache(bad_config)
            results['error_types'].append('Failed to catch invalid config')
        except ValueError:
            results['recovery_success'].append('Caught invalid config')
        
        # Test cache overflow
        try:
            device = next(self.model.parameters()).device
            large_batch = torch.randn(
                1000, self.model.config.embedding_dim,
                device=device
            )
            self.model.lunar_cache(large_batch)
            results['recovery_success'].append('Handled large batch')
        except Exception as e:
            results['error_types'].append(f'Failed to handle large batch: {str(e)}')
        
        # Test cache recovery after error
        try:
            self.model.toggle_cache(False)
            self.model.toggle_cache(True)
            results['recovery_success'].append('Cache recovery successful')
        except Exception as e:
            results['error_types'].append(f'Cache recovery failed: {str(e)}')
        
        stats = {
            'total_tests': len(results['recovery_success']) + len(results['error_types']),
            'successful_recoveries': len(results['recovery_success']),
            'errors': len(results['error_types'])
        }
        
        self.results['error_handling'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Error handling test results: {stats}")
        return stats
    
    def generate_visualizations(self, save_dir: str = 'integration_tests'):
        """Generate visualizations of test results."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot cache impact
        if self.results['cache_impact']:
            impact = self.results['cache_impact'][-1]
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            data = [
                impact['raw_results']['with_cache']['latency'],
                impact['raw_results']['without_cache']['latency']
            ]
            plt.boxplot(data, labels=['With Cache', 'Without Cache'])
            plt.title('Latency Distribution')
            plt.ylabel('Latency (ms)')
            
            plt.subplot(1, 2, 2)
            data = [
                impact['raw_results']['with_cache']['memory'],
                impact['raw_results']['without_cache']['memory']
            ]
            plt.boxplot(data, labels=['With Cache', 'Without Cache'])
            plt.title('Memory Usage Distribution')
            plt.ylabel('Memory (MB)')
            
            plt.tight_layout()
            plt.savefig(save_path / 'cache_impact.png')
            plt.close()
        
        # Plot pattern reuse
        if self.results['pattern_reuse']:
            reuse = self.results['pattern_reuse'][-1]
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(reuse['raw_results']['reuse_rate'])
            plt.title('Pattern Reuse Rate Over Time')
            plt.xlabel('Pattern Index')
            plt.ylabel('Reuse Rate')
            
            plt.subplot(1, 2, 2)
            if reuse['raw_results']['pattern_similarities']:
                plt.hist(reuse['raw_results']['pattern_similarities'], bins=50)
                plt.title('Pattern Similarity Distribution')
                plt.xlabel('Similarity')
                plt.ylabel('Count')
            
            plt.tight_layout()
            plt.savefig(save_path / 'pattern_reuse.png')
            plt.close()
    
    def save_results(self, save_dir: str = 'integration_tests'):
        """Save test results to JSON file."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = save_path / f'integration_results_{timestamp}.json'
        
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

def run_integration_tests(model: Optional[LargeModel] = None):
    """Run all integration tests and generate report."""
    logger.info("Starting model-cache integration tests...")
    
    tester = ModelCacheIntegrationTester(model)
    
    # Run tests
    tester.test_cache_impact()
    tester.test_pattern_reuse()
    tester.test_error_handling()
    
    # Generate outputs
    tester.generate_visualizations()
    tester.save_results()
    
    logger.info("Integration tests completed. Check integration_tests directory for outputs.")

if __name__ == '__main__':
    run_integration_tests() 