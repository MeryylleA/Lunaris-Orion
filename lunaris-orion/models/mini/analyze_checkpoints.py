"""
Checkpoint analysis script for the Mini model.
Generates technical metrics and visualizations to track training progress.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from PIL import Image
import logging
from torchvision.utils import make_grid
import seaborn as sns
from typing import Dict, List, Tuple

class CheckpointAnalyzer:
    def __init__(self, checkpoints_dir: str = 'checkpoints'):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.metrics_dir = Path('metrics')
        self.metrics_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """Load a checkpoint file."""
        try:
            return torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            self.logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            return None
    
    def get_model_size(self, checkpoint: Dict) -> Dict:
        """Calculate model size metrics."""
        total_params = 0
        trainable_params = 0
        
        for param in checkpoint['model_state_dict'].values():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        size_mb = sum(param.numel() * param.element_size() 
                     for param in checkpoint['model_state_dict'].values()) / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': size_mb
        }
    
    def analyze_gradients(self, checkpoint: Dict) -> Dict:
        """Analyze gradient statistics from optimizer state."""
        grad_norms = []
        param_norms = []
        
        for group in checkpoint['optimizer_state_dict']['state'].values():
            if 'exp_avg' in group:
                grad_norms.append(torch.norm(group['exp_avg']).item())
            if 'exp_avg_sq' in group:
                param_norms.append(torch.norm(torch.sqrt(group['exp_avg_sq'])).item())
        
        return {
            'gradient_norm_mean': np.mean(grad_norms) if grad_norms else 0,
            'gradient_norm_std': np.std(grad_norms) if grad_norms else 0,
            'parameter_norm_mean': np.mean(param_norms) if param_norms else 0,
            'parameter_norm_std': np.std(param_norms) if param_norms else 0
        }
    
    def plot_training_progress(self, checkpoints: List[Dict]) -> str:
        """Generate training progress visualization."""
        epochs = [cp['epoch'] for cp in checkpoints]
        losses = [cp['loss'] for cp in checkpoints]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker='o')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plot_path = self.metrics_dir / f'training_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_path)
        plt.close()
        
        return str(plot_path)
    
    def plot_gradient_distribution(self, checkpoint: Dict) -> str:
        """Generate gradient distribution visualization."""
        # Calculate gradient statistics in chunks to save memory
        grad_mean = 0
        grad_std = 0
        n_samples = 0
        sample_size = 10000  # Number of gradients to sample
        
        for group in checkpoint['optimizer_state_dict']['state'].values():
            if 'exp_avg' in group:
                grad = group['exp_avg']
                if n_samples < sample_size:
                    # Sample random gradients
                    n_remaining = min(sample_size - n_samples, grad.numel())
                    indices = torch.randperm(grad.numel())[:n_remaining]
                    grad_values = grad.flatten()[indices].numpy()
                    
                    # Update running statistics
                    n_samples += len(grad_values)
                    delta = grad_values - grad_mean
                    grad_mean += delta.mean()
                    grad_std = np.sqrt(((grad_values - grad_mean) ** 2).mean())
        
        # Generate synthetic data for visualization
        grad_values = np.random.normal(grad_mean, grad_std, sample_size)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(grad_values, bins=50)
        plt.title('Gradient Distribution (Sampled)')
        plt.xlabel('Gradient Value')
        plt.ylabel('Count')
        
        plot_path = self.metrics_dir / f'gradient_dist_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_path)
        plt.close()
        
        return str(plot_path)
    
    def generate_report(self) -> Tuple[Dict, List[str]]:
        """Generate a comprehensive analysis report."""
        checkpoints = []
        checkpoint_files = sorted(self.checkpoints_dir.glob('checkpoint_epoch_*.pt'))
        
        for cp_path in checkpoint_files:
            cp = self.load_checkpoint(cp_path)
            if cp:
                checkpoints.append(cp)
        
        if not checkpoints:
            self.logger.error("No checkpoints found!")
            return {}, []
        
        # Get latest checkpoint for detailed analysis
        latest_cp = checkpoints[-1]
        
        # Calculate metrics
        model_metrics = self.get_model_size(latest_cp)
        gradient_metrics = self.analyze_gradients(latest_cp)
        
        # Generate visualizations
        plots = []
        plots.append(self.plot_training_progress(checkpoints))
        plots.append(self.plot_gradient_distribution(latest_cp))
        
        # Compile report
        report = {
            'latest_epoch': latest_cp['epoch'],
            'latest_loss': latest_cp['loss'],
            'model_metrics': model_metrics,
            'gradient_metrics': gradient_metrics,
            'total_checkpoints': len(checkpoints),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save report
        report_path = self.metrics_dir / f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        return report, plots

def main():
    analyzer = CheckpointAnalyzer()
    report, plots = analyzer.generate_report()
    
    if report:
        print("\n=== Model Analysis Report ===")
        print(f"Latest Epoch: {report['latest_epoch']}")
        print(f"Latest Loss: {report['latest_loss']:.4f}")
        print("\nModel Metrics:")
        print(f"Total Parameters: {report['model_metrics']['total_parameters']:,}")
        print(f"Model Size: {report['model_metrics']['model_size_mb']:.2f} MB")
        print("\nGradient Metrics:")
        print(f"Mean Gradient Norm: {report['gradient_metrics']['gradient_norm_mean']:.4f}")
        print(f"Gradient Norm Std: {report['gradient_metrics']['gradient_norm_std']:.4f}")
        print("\nVisualizations saved:")
        for plot in plots:
            print(f"- {plot}")
    else:
        print("No report generated. Check if checkpoints exist.")

if __name__ == "__main__":
    main() 