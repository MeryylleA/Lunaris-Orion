"""
Analyze training checkpoints and plot training metrics.
Enhanced with GPU acceleration when available.
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
from typing import Optional, Dict, List, Tuple
import pandas as pd
from rich.logging import RichHandler
from rich.progress import track
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

def get_device():
    """Get the best available device for processing."""
    if torch.cuda.is_available():
        # Get the GPU with most free memory
        gpu_memory = []
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated()
                gpu_memory.append((i, free_memory))
        
        device_id = max(gpu_memory, key=lambda x: x[1])[0]
        device = torch.device(f'cuda:{device_id}')
        logging.info(f"Using GPU {device_id} for analysis")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU for analysis (no GPU available)")
    return device

def process_checkpoint_batch(files: List[Path], device: torch.device) -> List[Dict]:
    """Process a batch of checkpoint files in parallel using GPU."""
    results = []
    
    for cp_file in files:
        try:
            if cp_file.stat().st_size == 0:
                continue
                
            # Load checkpoint to specified device
            checkpoint = torch.load(cp_file, map_location=device)
            
            # Extract metrics
            result = {
                'file': cp_file,
                'size': cp_file.stat().st_size / (1024 * 1024),  # MB
                'epoch': checkpoint.get('epoch', 0),
                'val_loss': checkpoint.get('val_loss', float('inf')),
                'training_duration': checkpoint.get('training_duration', 0)
            }
            
            # Extract GPU info if available
            if 'gpu_info' in checkpoint:
                result['memory_allocated'] = checkpoint['gpu_info'].get('memory_allocated', 0) / (1024 * 1024 * 1024)  # GB
            
            results.append(result)
            
            # Clear GPU memory if using CUDA
            if device.type == 'cuda':
                del checkpoint
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.warning(f"Error processing checkpoint {cp_file.name}: {str(e)}")
            continue
            
    return results

def analyze_checkpoints(run_dir: Path) -> Dict:
    """Analyze checkpoints and extract key metrics using GPU acceleration."""
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found in {run_dir}")
    
    # Only analyze epoch checkpoints, skip best.pt and other special files
    checkpoint_files = sorted(
        [f for f in checkpoints_dir.glob("checkpoint_epoch_*.pt")],
        key=lambda x: int(re.findall(r'\d+', x.stem)[-1])
    )
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir}")
    
    logging.info(f"Found {len(checkpoint_files)} checkpoints to analyze")
    
    # Get the best available device
    device = get_device()
    
    # Process checkpoints in batches
    batch_size = 5  # Adjust based on available memory
    results = []
    
    with track(range(0, len(checkpoint_files), batch_size), description="Analyzing checkpoints") as progress:
        for i in progress:
            batch = checkpoint_files[i:i + batch_size]
            batch_results = process_checkpoint_batch(batch, device)
            results.extend(batch_results)
    
    if not results:
        raise ValueError("No valid metrics could be extracted from checkpoints")
    
    # Convert results to metrics dictionary
    metrics = {
        'checkpoint_sizes': [],
        'epochs': [],
        'val_losses': [],
        'training_times': [],
        'memory_usage': []
    }
    
    # Sort results by epoch
    results.sort(key=lambda x: x['epoch'])
    
    for result in results:
        metrics['checkpoint_sizes'].append(result['size'])
        metrics['epochs'].append(result['epoch'])
        metrics['val_losses'].append(result['val_loss'])
        metrics['training_times'].append(result['training_duration'])
        if 'memory_allocated' in result:
            metrics['memory_usage'].append(result['memory_allocated'])
    
    return metrics

def reconstruct_history_from_checkpoints(run_dir: Path) -> Optional[Dict]:
    """Reconstruct training history from checkpoint files using GPU acceleration."""
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
        
    checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_epoch_*.pt"), 
                            key=lambda x: int(re.findall(r'\d+', x.stem)[-1]))
    
    if not checkpoint_files:
        return None
    
    # Get the best available device
    device = get_device()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': [],
        'learning_rate': []
    }
    
    # Process checkpoints in batches
    batch_size = 5  # Adjust based on available memory
    
    for i in track(range(0, len(checkpoint_files), batch_size), description="Reconstructing history"):
        batch = checkpoint_files[i:i + batch_size]
        for cp_file in batch:
            try:
                checkpoint = torch.load(cp_file, map_location=device)
                history['train_loss'].append(checkpoint.get('train_loss', float('nan')))
                history['val_loss'].append(checkpoint.get('val_loss', float('nan')))
                history['epochs'].append(checkpoint.get('epoch', 0))
                if 'optimizer_state_dict' in checkpoint:
                    lr = checkpoint['optimizer_state_dict']['param_groups'][0].get('lr', float('nan'))
                    history['learning_rate'].append(lr)
                
                # Clear GPU memory if using CUDA
                if device.type == 'cuda':
                    del checkpoint
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logging.warning(f"Error loading checkpoint {cp_file.name}: {str(e)}")
                continue
    
    # Remove any NaN values
    for key in history:
        if key != 'epochs':
            history[key] = [x for x, e in zip(history[key], history['epochs']) 
                          if not (isinstance(x, float) and np.isnan(x))]
    history['epochs'] = [e for e in history['epochs'] 
                        if any(not (isinstance(x, float) and np.isnan(x)) 
                              for x in [history[k][i] for k in history if k != 'epochs'])]
    
    return history if history['epochs'] else None

def find_latest_run() -> Path:
    """Find the most recent training run directory."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        raise FileNotFoundError("No 'runs' directory found")
    
    runs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not runs:
        raise FileNotFoundError("No run directories found in 'runs'")
    
    def parse_run_timestamp(run_path: Path) -> datetime:
        """Parse timestamp from run directory name with multiple format support."""
        try:
            # Try different date formats
            formats = [
                "%Y%m%d_%H%M%S",  # Standard format: YYYYMMDD_HHMMSS
                "%Y%m%d",         # Date only format: YYYYMMDD
                "%Y_%m_%d_%H%M",  # Alternative format with underscores
                "%Y%m%d%H%M"      # Compact format without separator
            ]
            
            # Extract timestamp part (handle both 'run_' prefix and no prefix)
            name = run_path.name
            if name.startswith('run_'):
                timestamp_str = name[4:]  # Remove 'run_' prefix
            else:
                timestamp_str = name
            
            # Try each format
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            # If no format matches, try to extract numbers and make best guess
            numbers = re.findall(r'\d+', timestamp_str)
            if numbers:
                # Join all numbers and try to parse as basic timestamp
                timestamp_str = numbers[0]
                if len(timestamp_str) >= 8:  # At least YYYYMMDD
                    return datetime.strptime(timestamp_str[:8], "%Y%m%d")
            
            raise ValueError(f"Could not parse timestamp from directory name: {name}")
            
        except Exception as e:
            logging.warning(f"Error parsing timestamp for {run_path}: {str(e)}")
            # Return a very old date as fallback
            return datetime(1900, 1, 1)
    
    latest_run = max(runs, key=parse_run_timestamp)
    logging.info(f"Found latest run directory: {latest_run}")
    return latest_run

def load_training_history(run_dir: Path) -> Dict:
    """Load training history from available metrics files."""
    # Try different possible locations and filenames
    possible_paths = [
        run_dir / "metrics.json",
        run_dir / "logs" / "metrics.json",
        run_dir / "logs" / "training_history.json",
        run_dir / "training_history.json",
        run_dir / "history.json"
    ]
    
    # Try to find an existing metrics file
    found_files = [p for p in possible_paths if p.exists()]
    if not found_files:
        # If no metrics file found, try to reconstruct from checkpoints
        logging.warning("No metrics file found. Attempting to reconstruct from checkpoints...")
        history = reconstruct_history_from_checkpoints(run_dir)
        if history:
            return history
        raise FileNotFoundError(f"No metrics file found in {run_dir} and could not reconstruct from checkpoints")
    
    # Load the first found metrics file
    metrics_file = found_files[0]
    logging.info(f"Loading metrics from {metrics_file}")
    with open(metrics_file) as f:
        history = json.load(f)
    return history

def plot_training_curves(history: Dict, metrics: Dict, output_dir: Path):
    """Generate and save training visualization plots."""
    output_dir.mkdir(exist_ok=True)
    plt.style.use('seaborn')
    
    # Plot loss curves
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'loss_curves.png')
    plt.close()
    
    # Plot learning rate
    if 'learning_rate' in history:
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, history['learning_rate'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(output_dir / 'learning_rate.png')
        plt.close()
    
    # Plot GPU metrics
    if 'gpu_memory' in history and 'gpu_utilization' in history:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(epochs, history['gpu_memory'])
        ax1.set_title('GPU Memory Usage')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Memory Usage (%)')
        ax1.grid(True)
        
        ax2.plot(epochs, history['gpu_utilization'])
        ax2.set_title('GPU Utilization')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Utilization (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gpu_metrics.png')
        plt.close()
    
    # Plot checkpoint metrics
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['epochs'], metrics['checkpoint_sizes'])
    plt.title('Checkpoint Sizes Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Size (MB)')
    plt.grid(True)
    plt.savefig(output_dir / 'checkpoint_sizes.png')
    plt.close()

def generate_report(history: Dict, metrics: Dict, output_dir: Path):
    """Generate a detailed analysis report."""
    report = {
        'training_summary': {
            'total_epochs': len(history['train_loss']),
            'best_val_loss': min(history['val_loss']),
            'best_epoch': history['val_loss'].index(min(history['val_loss'])) + 1,
            'final_val_loss': history['val_loss'][-1],
            'total_training_time': metrics['training_times'][-1] if metrics['training_times'] else 0
        },
        'checkpoint_analysis': {
            'num_checkpoints': len(metrics['checkpoint_sizes']),
            'total_size_gb': sum(metrics['checkpoint_sizes']) / 1024,
            'avg_size_mb': np.mean(metrics['checkpoint_sizes']),
            'max_size_mb': max(metrics['checkpoint_sizes'])
        }
    }
    
    # Save report
    with open(output_dir / 'analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logging.info("\n=== Training Analysis Report ===")
    logging.info(f"Total Epochs: {report['training_summary']['total_epochs']}")
    logging.info(f"Best Validation Loss: {report['training_summary']['best_val_loss']:.4f} (Epoch {report['training_summary']['best_epoch']})")
    logging.info(f"Final Validation Loss: {report['training_summary']['final_val_loss']:.4f}")
    logging.info(f"Total Training Time: {report['training_summary']['total_training_time']/3600:.1f} hours")
    logging.info(f"\nCheckpoints:")
    logging.info(f"Total Size: {report['checkpoint_analysis']['total_size_gb']:.2f} GB")
    logging.info(f"Average Size: {report['checkpoint_analysis']['avg_size_mb']:.1f} MB")

def main():
    """Main function to analyze training run."""
    try:
        setup_logging()
        
        # Find latest run
        run_dir = find_latest_run()
        
        # Create output directory for analysis
        output_dir = run_dir / "analysis"
        output_dir.mkdir(exist_ok=True)
        
        # Load and analyze data
        history = load_training_history(run_dir)
        metrics = analyze_checkpoints(run_dir)
        
        # Generate visualizations and report
        plot_training_curves(history, metrics, output_dir)
        generate_report(history, metrics, output_dir)
        
        logging.info(f"\nAnalysis complete. Results saved to {output_dir}")
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 