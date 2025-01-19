"""
Model Training Tests
------------------
Tests the Large model's training functionality, including:
- Training loop stability
- Loss convergence
- Memory management
- Gradient flow
- Checkpoint saving/loading
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
import shutil

from ..model import LargeModel, ModelConfig
from ..lunar_cache import LunarCache, CacheConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainingTester:
    """Tests model training functionality."""
    
    def __init__(self, model: Optional[LargeModel] = None):
        self.model = model if model is not None else self._create_test_model()
        self.results = {
            'training_stability': [],
            'loss_convergence': [],
            'memory_tracking': [],
            'gradient_flow': [],
            'checkpoint_tests': []
        }
        
        # Initialize device
        self.device = next(self.model.parameters()).device
        
        # Set up gradient tracking
        self._setup_gradient_tracking()
    
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
    
    def _setup_gradient_tracking(self):
        """Set up hooks for tracking gradients."""
        self.gradient_stats = []
        
        def hook_fn(grad):
            if grad is not None:
                self.gradient_stats.append({
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'norm': grad.norm().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item()
                })
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: hook_fn(grad))
    
    def test_training_stability(
        self,
        num_epochs: int = 5,
        steps_per_epoch: int = 100,
        batch_size: int = 32,
        sequence_length: int = 512,
        learning_rate: float = 1e-4
    ) -> Dict:
        """Test training stability over multiple epochs."""
        logger.info("Testing training stability...")
        
        results = {
            'epoch_losses': [],
            'step_losses': [],
            'gradient_stats': [],
            'memory_usage': [],
            'training_time': []
        }
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * steps_per_epoch
        )
        
        epoch_start_time = time.time()
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for step in range(steps_per_epoch):
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
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Record metrics
                epoch_losses.append(loss.item())
                results['step_losses'].append(loss.item())
                results['memory_usage'].append(
                    torch.cuda.memory_allocated() / (1024 * 1024)
                )
                
                if step % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}, Step {step}: "
                        f"Loss = {loss.item():.4f}, "
                        f"Memory = {results['memory_usage'][-1]:.2f}MB"
                    )
            
            # Record epoch metrics
            results['epoch_losses'].append(np.mean(epoch_losses))
            results['training_time'].append(time.time() - epoch_start_time)
            
            logger.info(
                f"Epoch {epoch} completed: "
                f"Avg Loss = {results['epoch_losses'][-1]:.4f}"
            )
        
        # Compute stability metrics
        stats = {
            'final_loss': results['epoch_losses'][-1],
            'loss_std': np.std(results['epoch_losses']),
            'loss_trend': np.polyfit(
                range(len(results['epoch_losses'])),
                results['epoch_losses'],
                1
            )[0],
            'avg_memory_usage': np.mean(results['memory_usage']),
            'total_training_time': results['training_time'][-1]
        }
        
        self.results['training_stability'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Training stability test results: {stats}")
        return stats
    
    def test_loss_convergence(
        self,
        target_loss: float = 0.1,
        max_steps: int = 1000,
        batch_size: int = 32,
        sequence_length: int = 512,
        learning_rate: float = 1e-4,
        patience: int = 50
    ) -> Dict:
        """Test loss convergence to target value."""
        logger.info("Testing loss convergence...")
        
        results = {
            'losses': [],
            'steps_to_converge': None,
            'converged': False,
            'stopped_early': False
        }
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        best_loss = float('inf')
        steps_without_improvement = 0
        
        for step in range(max_steps):
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
            
            current_loss = loss.item()
            results['losses'].append(current_loss)
            
            # Check convergence
            if current_loss <= target_loss:
                results['converged'] = True
                results['steps_to_converge'] = step + 1
                logger.info(f"Converged to target loss in {step + 1} steps")
                break
            
            # Early stopping check
            if current_loss < best_loss:
                best_loss = current_loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
            
            if steps_without_improvement >= patience:
                results['stopped_early'] = True
                logger.info(f"Early stopping triggered after {step + 1} steps")
                break
            
            if step % 10 == 0:
                logger.info(f"Step {step}: Loss = {current_loss:.4f}")
        
        # Compute convergence metrics
        stats = {
            'final_loss': results['losses'][-1],
            'converged': results['converged'],
            'steps_to_converge': results['steps_to_converge'],
            'stopped_early': results['stopped_early'],
            'loss_variance': np.var(results['losses'])
        }
        
        self.results['loss_convergence'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Loss convergence test results: {stats}")
        return stats
    
    def test_memory_management(
        self,
        num_steps: int = 100,
        batch_sizes: List[int] = [8, 16, 32, 64],
        sequence_length: int = 512
    ) -> Dict:
        """Test memory management under different batch sizes."""
        logger.info("Testing memory management...")
        
        results = {
            'memory_usage': {},
            'peak_memory': {},
            'oom_events': {}
        }
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size {batch_size}")
            
            results['memory_usage'][batch_size] = []
            results['peak_memory'][batch_size] = 0
            results['oom_events'][batch_size] = 0
            
            try:
                for step in range(num_steps):
                    torch.cuda.empty_cache()
                    
                    # Generate random batch
                    input_ids = torch.randint(
                        0, self.model.config.vocab_size,
                        (batch_size, sequence_length),
                        device=self.device
                    )
                    
                    # Forward pass
                    outputs = self.model(input_ids)
                    
                    # Record memory usage
                    current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    
                    results['memory_usage'][batch_size].append(current_memory)
                    results['peak_memory'][batch_size] = max(
                        results['peak_memory'][batch_size],
                        peak_memory
                    )
                    
                    if step % 10 == 0:
                        logger.info(
                            f"Batch size {batch_size}, Step {step}: "
                            f"Memory = {current_memory:.2f}MB, "
                            f"Peak = {peak_memory:.2f}MB"
                        )
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results['oom_events'][batch_size] += 1
                    logger.warning(f"OOM error with batch size {batch_size}")
                    torch.cuda.empty_cache()
                else:
                    raise e
        
        # Compute memory statistics
        stats = {
            'max_stable_batch': max(
                [bs for bs in batch_sizes if results['oom_events'][bs] == 0],
                default=None
            ),
            'memory_scaling': {
                bs: np.mean(results['memory_usage'][bs])
                for bs in batch_sizes
                if results['memory_usage'][bs]
            },
            'peak_memory': {
                bs: results['peak_memory'][bs]
                for bs in batch_sizes
            },
            'oom_events': results['oom_events']
        }
        
        self.results['memory_tracking'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Memory management test results: {stats}")
        return stats
    
    def test_gradient_flow(
        self,
        num_steps: int = 100,
        batch_size: int = 32,
        sequence_length: int = 512
    ) -> Dict:
        """Test gradient flow through the model."""
        logger.info("Testing gradient flow...")
        
        results = {
            'gradient_norms': [],
            'gradient_stats': [],
            'parameter_updates': []
        }
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # Store initial parameter values
        initial_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
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
            
            # Record gradient statistics
            grad_norms = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.norm().item()
            
            results['gradient_norms'].append(grad_norms)
            results['gradient_stats'].append(self.gradient_stats[-1])
            
            optimizer.step()
            
            if step % 10 == 0:
                logger.info(f"Step {step}: Processed gradients")
        
        # Compute parameter updates
        for name, final_param in self.model.named_parameters():
            if name in initial_params:
                update = (final_param - initial_params[name]).norm().item()
                results['parameter_updates'].append({
                    'name': name,
                    'update_norm': update
                })
        
        # Compute gradient flow statistics
        stats = {
            'avg_gradient_norm': np.mean([
                np.mean(list(norms.values()))
                for norms in results['gradient_norms']
            ]),
            'gradient_norm_std': np.std([
                np.mean(list(norms.values()))
                for norms in results['gradient_norms']
            ]),
            'nan_gradients': any(
                stat['has_nan'] for stat in results['gradient_stats']
            ),
            'inf_gradients': any(
                stat['has_inf'] for stat in results['gradient_stats']
            ),
            'largest_update': max(
                update['update_norm']
                for update in results['parameter_updates']
            )
        }
        
        self.results['gradient_flow'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Gradient flow test results: {stats}")
        return stats
    
    def test_checkpointing(
        self,
        checkpoint_dir: str = 'test_checkpoints'
    ) -> Dict:
        """Test checkpoint saving and loading."""
        logger.info("Testing checkpointing functionality...")
        
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'save_success': False,
            'load_success': False,
            'state_matches': False,
            'memory_usage': []
        }
        
        try:
            # Generate some random data
            input_ids = torch.randint(
                0, self.model.config.vocab_size,
                (32, 512),
                device=self.device
            )
            
            # Get initial outputs
            self.model.eval()
            with torch.no_grad():
                initial_outputs = self.model(input_ids)
            
            # Save checkpoint
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.model.config
            }
            
            checkpoint_file = checkpoint_path / 'test_checkpoint.pt'
            torch.save(checkpoint, checkpoint_file)
            results['save_success'] = True
            
            # Create new model instance
            new_model = self._create_test_model()
            new_model = new_model.to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_file)
            new_model.load_state_dict(checkpoint['model_state_dict'])
            results['load_success'] = True
            
            # Compare outputs
            new_model.eval()
            with torch.no_grad():
                new_outputs = new_model(input_ids)
            
            # Check if outputs match
            outputs_match = torch.allclose(
                initial_outputs,
                new_outputs,
                rtol=1e-4,
                atol=1e-4
            )
            results['state_matches'] = outputs_match
            
            # Record memory usage
            results['memory_usage'].append(
                torch.cuda.memory_allocated() / (1024 * 1024)
            )
            
        except Exception as e:
            logger.error(f"Error during checkpointing: {str(e)}")
        finally:
            # Cleanup
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
        
        # Compute checkpoint statistics
        stats = {
            'save_success': results['save_success'],
            'load_success': results['load_success'],
            'state_matches': results['state_matches'],
            'memory_overhead': np.mean(results['memory_usage'])
        }
        
        self.results['checkpoint_tests'].append({
            'raw_results': results,
            'stats': stats
        })
        
        logger.info(f"Checkpointing test results: {stats}")
        return stats
    
    def generate_visualizations(self, save_dir: str = 'training_tests'):
        """Generate visualizations of test results."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Plot training stability
        if self.results['training_stability']:
            stability = self.results['training_stability'][-1]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(stability['raw_results']['epoch_losses'])
            plt.title('Epoch Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            
            plt.subplot(1, 3, 2)
            plt.plot(stability['raw_results']['memory_usage'])
            plt.title('Memory Usage')
            plt.xlabel('Step')
            plt.ylabel('Memory (MB)')
            
            plt.subplot(1, 3, 3)
            plt.plot(stability['raw_results']['training_time'])
            plt.title('Training Time')
            plt.xlabel('Epoch')
            plt.ylabel('Time (s)')
            
            plt.tight_layout()
            plt.savefig(save_path / 'training_stability.png')
            plt.close()
        
        # Plot loss convergence
        if self.results['loss_convergence']:
            convergence = self.results['loss_convergence'][-1]
            
            plt.figure(figsize=(10, 5))
            plt.plot(convergence['raw_results']['losses'])
            plt.axhline(
                y=convergence['stats']['final_loss'],
                color='r',
                linestyle='--',
                label='Final Loss'
            )
            plt.title('Loss Convergence')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(save_path / 'loss_convergence.png')
            plt.close()
        
        # Plot memory management
        if self.results['memory_tracking']:
            memory = self.results['memory_tracking'][-1]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            for batch_size, usage in memory['raw_results']['memory_usage'].items():
                if usage:  # Only plot if we have data
                    plt.plot(usage, label=f'Batch {batch_size}')
            plt.title('Memory Usage by Batch Size')
            plt.xlabel('Step')
            plt.ylabel('Memory (MB)')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            batch_sizes = list(memory['stats']['memory_scaling'].keys())
            memory_usage = list(memory['stats']['memory_scaling'].values())
            plt.bar(range(len(batch_sizes)), memory_usage)
            plt.xticks(range(len(batch_sizes)), batch_sizes)
            plt.title('Average Memory Usage')
            plt.xlabel('Batch Size')
            plt.ylabel('Memory (MB)')
            
            plt.tight_layout()
            plt.savefig(save_path / 'memory_management.png')
            plt.close()
        
        # Plot gradient flow
        if self.results['gradient_flow']:
            gradients = self.results['gradient_flow'][-1]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 2, 1)
            grad_norms = np.array([
                list(norms.values())
                for norms in gradients['raw_results']['gradient_norms']
            ])
            plt.imshow(grad_norms.T, aspect='auto', cmap='viridis')
            plt.colorbar(label='Gradient Norm')
            plt.title('Gradient Norms Over Time')
            plt.xlabel('Step')
            plt.ylabel('Layer')
            
            plt.subplot(1, 2, 2)
            updates = gradients['raw_results']['parameter_updates']
            update_norms = [update['update_norm'] for update in updates]
            plt.bar(range(len(updates)), update_norms)
            plt.xticks(
                range(len(updates)),
                [update['name'] for update in updates],
                rotation=45
            )
            plt.title('Parameter Updates')
            plt.ylabel('Update Norm')
            
            plt.tight_layout()
            plt.savefig(save_path / 'gradient_flow.png')
            plt.close()
    
    def save_results(self, save_dir: str = 'training_tests'):
        """Save test results to JSON file."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = save_path / f'training_results_{timestamp}.json'
        
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

def run_training_tests(model: Optional[LargeModel] = None):
    """Run all training tests and generate report."""
    logger.info("Starting model training tests...")
    
    tester = ModelTrainingTester(model)
    
    # Run tests
    tester.test_training_stability()
    tester.test_loss_convergence()
    tester.test_memory_management()
    tester.test_gradient_flow()
    tester.test_checkpointing()
    
    # Generate outputs
    tester.generate_visualizations()
    tester.save_results()
    
    logger.info("Training tests completed. Check training_tests directory for outputs.")

if __name__ == '__main__':
    run_training_tests() 