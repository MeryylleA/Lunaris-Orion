#!/usr/bin/env python3
"""
Enhanced generation script for Lunaris-Orion.
Generates high-quality pixel art images from text prompts using trained models.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
import json

from models import LunarCoreVAE, LunarMoETeacher

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('generation.log')
    ]
)
logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """Initialize the image generator with trained models.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            logger.info("Successfully loaded checkpoint")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Initialize and load models
        self.vae = LunarCoreVAE().to(self.device)
        self.teacher = LunarMoETeacher().to(self.device)
        
        try:
            self.vae.load_state_dict(checkpoint['vae_state_dict'])
            self.teacher.load_state_dict(checkpoint['teacher_state_dict'])
            logger.info("Successfully loaded model weights")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise
        
        # Set models to evaluation mode
        self.vae.eval()
        self.teacher.eval()
        
        # Store configuration from checkpoint
        self.config = checkpoint.get('config', {})
        
    def generate(
        self,
        prompt: str,
        num_samples: int = 4,
        temperature: float = 1.0,
        quality_threshold: float = 0.7,
        max_attempts: int = 3,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate images from a prompt.
        
        Args:
            prompt: Text prompt for generation
            num_samples: Number of images to generate
            temperature: Sampling temperature (higher = more diverse)
            quality_threshold: Minimum quality score to accept
            max_attempts: Maximum generation attempts per image
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (generated images, quality scores)
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        best_images = []
        best_scores = []
        
        with torch.no_grad():
            attempts = 0
            while len(best_images) < num_samples and attempts < max_attempts * num_samples:
                # Generate latent vectors with temperature
                z = torch.randn(
                    num_samples - len(best_images),
                    self.vae.latent_dim,
                    device=self.device
                ) * temperature
                
                # Generate images
                images = self.vae.decode(z)
                
                # Get quality scores
                quality_scores = self.teacher.assess_quality(images)
                
                # Filter by quality threshold
                good_indices = quality_scores.squeeze() >= quality_threshold
                good_images = images[good_indices]
                good_scores = quality_scores[good_indices]
                
                best_images.extend(good_images[:num_samples - len(best_images)])
                best_scores.extend(good_scores[:num_samples - len(best_images)])
                
                attempts += 1
            
            if len(best_images) < num_samples:
                logger.warning(
                    f"Only generated {len(best_images)} images meeting quality threshold "
                    f"after {attempts} attempts"
                )
            
            return (
                torch.stack(best_images[:num_samples]),
                torch.stack(best_scores[:num_samples])
            )

    def save_images(
        self,
        images: torch.Tensor,
        scores: torch.Tensor,
        output_dir: str,
        prompt: str,
        save_metadata: bool = True
    ) -> List[str]:
        """Save generated images and optionally create a grid visualization.
        
        Args:
            images: Generated images tensor
            scores: Quality scores tensor
            output_dir: Directory to save images
            prompt: Original generation prompt
            save_metadata: Whether to save generation metadata
            
        Returns:
            List of saved image paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_paths = []
        metadata = {
            'timestamp': timestamp,
            'prompt': prompt,
            'config': self.config,
            'generations': []
        }
        
        # Save individual images
        for i, (image, score) in enumerate(zip(images, scores)):
            # Convert to PIL image
            image_np = ((image.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype('uint8')
            pil_image = Image.fromarray(image_np)
            
            # Save with metadata in filename
            filename = f"generated_{timestamp}_{i}_score_{score:.3f}.png"
            save_path = output_dir / filename
            pil_image.save(save_path)
            saved_paths.append(str(save_path))
            
            if save_metadata:
                metadata['generations'].append({
                    'filename': filename,
                    'quality_score': float(score),
                    'path': str(save_path)
                })
        
        # Create and save grid visualization
        if len(images) > 1:
            grid_size = int(np.ceil(np.sqrt(len(images))))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            fig.suptitle(f'Generated Images for prompt: "{prompt}"')
            
            for i, (image, score) in enumerate(zip(images, scores)):
                ax = axes[i // grid_size, i % grid_size]
                ax.imshow(((image.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype('uint8'))
                ax.set_title(f'Quality Score: {score:.3f}')
                ax.axis('off')
            
            # Hide empty subplots
            for i in range(len(images), grid_size * grid_size):
                axes[i // grid_size, i % grid_size].axis('off')
            
            plt.tight_layout()
            grid_path = output_dir / f'grid_{timestamp}.png'
            plt.savefig(grid_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_paths.append(str(grid_path))
            
            if save_metadata:
                metadata['grid_visualization'] = str(grid_path)
        
        # Save metadata
        if save_metadata:
            metadata_path = output_dir / f'metadata_{timestamp}.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        return saved_paths

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate pixel art images using Lunaris-Orion',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--prompt', type=str, default="A pixel art castle",
                      help='Text prompt for generation')
    parser.add_argument('--num_samples', type=int, default=4,
                      help='Number of images to generate')
    parser.add_argument('--output_dir', type=str, default='examples/output',
                      help='Directory to save generated images')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='Sampling temperature (higher = more diverse)')
    parser.add_argument('--quality_threshold', type=float, default=0.7,
                      help='Minimum quality score threshold')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (cuda/cpu, default: auto)')
    parser.add_argument('--no_metadata', action='store_true',
                      help='Disable metadata saving')
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # Initialize generator
        generator = ImageGenerator(args.checkpoint, device=args.device)
        
        # Generate images
        logger.info(f"Generating {args.num_samples} images for prompt: {args.prompt}")
        images, scores = generator.generate(
            prompt=args.prompt,
            num_samples=args.num_samples,
            temperature=args.temperature,
            quality_threshold=args.quality_threshold,
            seed=args.seed
        )
        
        # Save results
        logger.info("Saving generated images...")
        saved_paths = generator.save_images(
            images, scores, args.output_dir, args.prompt,
            save_metadata=not args.no_metadata
        )
        logger.info(f"Images saved to: {args.output_dir}")
        
        # Print quality scores
        logger.info("\nQuality Scores:")
        for i, score in enumerate(scores):
            logger.info(f"Image {i + 1}: {score:.3f}")
        
        logger.info(f"\nBest quality score: {scores.max():.3f}")
        logger.info(f"Average quality score: {scores.mean():.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main()) 