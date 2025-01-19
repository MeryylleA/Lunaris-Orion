"""
Analyze training checkpoints and plot training metrics.
Enhanced with GPU acceleration and parallel processing.
"""

import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple, Generator
import pandas as pd
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import gc
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
from itertools import islice

console = Console()

def setup_logging():
    """Setup enhanced logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, console=console)]
    )

def get_optimal_device():
    """Get the optimal device for processing with memory management."""
    if torch.cuda.is_available():
        # Get GPU with most free memory
        gpu_memory = []
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated()
                gpu_memory.append((i, free_memory))
        
        device_id = max(gpu_memory, key=lambda x: x[1])[0]
        device = torch.device(f'cuda:{device_id}')
        
        # Set up GPU for optimal performance
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        
        console.log(f"[green]Using GPU {device_id} for analysis[/green]")
        console.log(f"[blue]Available GPU memory: {gpu_memory[device_id][1] / 1024**3:.2f} GB[/blue]")
    else:
        device = torch.device('cpu')
        console.log("[yellow]Using CPU for analysis (no GPU available)[/yellow]")
    return device

def chunk_generator(items: List, chunk_size: int) -> Generator:
    """Generate chunks of items for batch processing."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

def process_checkpoint(cp_file: Path, device: torch.device) -> Optional[Dict]:
    """Process a single checkpoint with optimized memory usage."""
    try:
        if cp_file.stat().st_size == 0:
            return None
            
        # Load checkpoint with memory optimization
        with torch.amp.autocast('cuda', enabled=device.type=='cuda'):
            # Use weights_only=True for faster loading and security
            checkpoint = torch.load(cp_file, map_location=device, weights_only=True)
            
            result = {
                'file': cp_file,
                'size': cp_file.stat().st_size / (1024 * 1024),  # MB
                'epoch': checkpoint.get('epoch', 0),
                'val_loss': checkpoint.get('val_loss', float('inf')),
                'train_loss': checkpoint.get('train_loss', float('inf')),
                'training_duration': checkpoint.get('training_duration', 0),
                'learning_rate': checkpoint.get('optimizer_state_dict', {}).get('param_groups', [{}])[0].get('lr', 0)
            }
            
            # Extract GPU metrics if available
            if 'gpu_info' in checkpoint:
                result['memory_allocated'] = checkpoint['gpu_info'].get('memory_allocated', 0) / (1024**3)  # GB
                result['memory_cached'] = checkpoint['gpu_info'].get('memory_cached', 0) / (1024**3)  # GB
            
            # Clear references immediately
            del checkpoint
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            return result
        
    except Exception as e:
        console.log(f"[red]Error processing checkpoint {cp_file.name}: {str(e)}[/red]")
        return None

def analyze_checkpoints(run_dir: Path) -> Dict:
    """Analyze checkpoints with optimized GPU acceleration and parallel processing."""
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found in {run_dir}")
    
    # Get checkpoint files
    checkpoint_files = sorted(
        [f for f in checkpoints_dir.glob("checkpoint_epoch_*.pt")],
        key=lambda x: int(re.findall(r'\d+', x.stem)[-1])
    )
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")
    
    console.log(f"[blue]Found {len(checkpoint_files)} checkpoints to analyze[/blue]")
    
    # Get optimal device
    device = get_optimal_device()
    
    # Calculate optimal batch size based on available memory
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        # Use 10% of available memory for batch processing
        memory_per_checkpoint = 200 * 1024 * 1024  # Estimate 200MB per checkpoint
        batch_size = max(1, int((total_memory * 0.1) / memory_per_checkpoint))
    else:
        batch_size = 2
    
    results = []
    processed_count = 0
    total_checkpoints = len(checkpoint_files)
    
    with Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        "{task.completed}/{task.total}",
        "•",
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "[green]Analyzing checkpoints...", 
            total=total_checkpoints,
            completed=0
        )
        
        # Process checkpoints in smaller batches
        for batch_start in range(0, total_checkpoints, batch_size):
            batch_end = min(batch_start + batch_size, total_checkpoints)
            batch = checkpoint_files[batch_start:batch_end]
            
            # Process batch
            batch_results = []
            for cp_file in batch:
                result = process_checkpoint(cp_file, device)
                if result is not None:
                    batch_results.append(result)
                processed_count += 1
                progress.update(task, completed=processed_count)
            
            results.extend(batch_results)
            
            # Force GPU memory cleanup after each batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
    
    if not results:
        raise ValueError("No valid metrics could be extracted from checkpoints")
    
    # Sort results by epoch
    results.sort(key=lambda x: x['epoch'])
    
    # Convert to metrics dictionary
    metrics = {
        'checkpoint_sizes': [r['size'] for r in results],
        'epochs': [r['epoch'] for r in results],
        'val_losses': [r['val_loss'] for r in results],
        'train_losses': [r['train_loss'] for r in results],
        'training_times': [r['training_duration'] for r in results],
        'learning_rates': [r['learning_rate'] for r in results],
        'memory_usage': [r.get('memory_allocated', 0) for r in results]
    }
    
    # Generate summary statistics
    summary = {
        'total_epochs': len(results),
        'best_val_loss': min(metrics['val_losses']),
        'best_epoch': metrics['epochs'][np.argmin(metrics['val_losses'])],
        'total_training_time': sum(metrics['training_times']),
        'avg_memory_usage': np.mean(metrics['memory_usage'])
    }
    
    console.log("[green]Analysis completed successfully![/green]")
    console.log(f"[blue]Summary:[/blue]")
    console.log(f"Total epochs analyzed: {summary['total_epochs']}")
    console.log(f"Best validation loss: {summary['best_val_loss']:.4f} at epoch {summary['best_epoch']}")
    console.log(f"Total training time: {summary['total_training_time'] / 3600:.2f} hours")
    console.log(f"Average memory usage: {summary['avg_memory_usage']:.2f} GB")
    
    return metrics

def plot_training_metrics(metrics: Dict, save_dir: Optional[Path] = None):
    """Plot training metrics with enhanced visualizations."""
    plt.style.use('seaborn')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(metrics['epochs'], metrics['train_losses'], label='Train Loss', color='blue', alpha=0.7)
    ax1.plot(metrics['epochs'], metrics['val_losses'], label='Val Loss', color='red', alpha=0.7)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Learning rate
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(metrics['epochs'], metrics['learning_rates'], color='green', alpha=0.7)
    ax2.set_title('Learning Rate Schedule')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.grid(True, alpha=0.3)
    
    # 3. Memory usage
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(metrics['epochs'], metrics['memory_usage'], color='purple', alpha=0.7)
    ax3.set_title('GPU Memory Usage')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Memory (GB)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Checkpoint sizes
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(metrics['epochs'], metrics['checkpoint_sizes'], color='orange', alpha=0.7)
    ax4.set_title('Checkpoint Sizes')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Size (MB)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        console.log(f"[green]Saved metrics plot to {save_path}[/green]")
    else:
        plt.show()

def main():
    """Main execution function."""
    setup_logging()
    
    try:
        # Find run directory (looking for run_YYYYMMDD_HHMMSS format)
        base_dir = Path(__file__).parent.resolve()  # Get the directory containing this script
        run_dir = base_dir / "runs"
        
        if not run_dir.exists():
            raise FileNotFoundError(f"No 'runs' directory found in {base_dir}")
        
        # Filter for actual run directories (starting with 'run_')
        runs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
        if not runs:
            raise FileNotFoundError(f"No run directories found in {run_dir}")
        
        # Sort runs by creation time and get the latest
        latest_run = max(runs, key=lambda x: x.stat().st_mtime)
        console.log(f"[blue]Analyzing latest run: {latest_run.name}[/blue]")
        
        # Analyze checkpoints
        metrics = analyze_checkpoints(latest_run)
        
        # Plot and save metrics
        plot_training_metrics(metrics, save_dir=latest_run)
        
        # Save metrics to JSON
        metrics_file = latest_run / 'checkpoint_analysis.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        console.log(f"[green]Saved analysis results to {metrics_file}[/green]")
        
    except Exception as e:
        console.log(f"[red]Error during analysis: {str(e)}[/red]")
        raise

if __name__ == '__main__':
    main() 