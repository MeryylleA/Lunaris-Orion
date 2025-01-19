"""
Optimized inference script for the Large model with H100-specific features.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import logging
from PIL import Image
import io
import argparse
from typing import Optional, Union, Tuple
import time
import numpy as np

from model import LargeModel
from config import config

class LargeInference:
    """
    Optimized inference class for the Large model with:
    - Flash Attention inference
    - Sliding window caching
    - Efficient memory management
    - Batch processing
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda',
        compile_model: bool = True
    ):
        self.device = torch.device(device)
        
        # Create model
        self.model = LargeModel(config['model'])
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Move model to device and optimize
        self.model.to(self.device)
        self.model.eval()
        
        # Enable Flash Attention if available
        self.model.enable_flash_attention()
        
        # Compile model for faster inference
        if compile_model and device == 'cuda':
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=True
            )
        
        logging.info(f"Model loaded and optimized for {device}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if it exists (from DDP training)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Load state dict
        self.model.load_state_dict(state_dict)
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    
    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        batch_size: int = 1,
        return_pil: bool = True
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Generate image from text prompt with optimized inference.
        
        Args:
            prompt: Text prompt to generate image from
            seed: Random seed for reproducibility
            temperature: Sampling temperature
            batch_size: Number of images to generate in parallel
            return_pil: Whether to return PIL Image or torch Tensor
        
        Returns:
            Generated image as PIL Image or torch Tensor
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Process prompt and generate
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # TODO: Implement tokenization
                tokens = torch.randint(0, config['model']['vocab_size'], (batch_size, 64))
                tokens = tokens.to(self.device)
                
                # Generate image
                output = self.model(tokens)
                
                # Apply temperature
                if temperature != 1.0:
                    output = output * temperature
                
                # Normalize output
                output = torch.clamp((output + 1) / 2, 0, 1)
            
            # Convert to PIL Image if requested
            if return_pil:
                if batch_size > 1:
                    images = []
                    for i in range(batch_size):
                        img_tensor = output[i].cpu()
                        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        images.append(Image.fromarray(img_np))
                    output = images
                else:
                    img_tensor = output[0].cpu()
                    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    output = Image.fromarray(img_np)
            
            # Log generation time
            generation_time = time.time() - start_time
            logging.info(f"Generated {batch_size} image(s) in {generation_time:.2f}s")
            
            return output
            
        except Exception as e:
            logging.error(f"Error during generation: {str(e)}")
            raise
    
    def get_memory_usage(self) -> Tuple[float, float]:
        """Get current GPU memory usage."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            return allocated, reserved
        return 0.0, 0.0

def main():
    parser = argparse.ArgumentParser(description='Generate images with Large model')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize model
    model = LargeInference(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Generate image
    output = model.generate_image(
        prompt=args.prompt,
        seed=args.seed,
        temperature=args.temperature,
        batch_size=args.batch_size
    )
    
    # Save image(s)
    if isinstance(output, list):
        for i, img in enumerate(output):
            path = args.output.replace('.png', f'_{i}.png')
            img.save(path)
            logging.info(f"Saved image to {path}")
    else:
        output.save(args.output)
        logging.info(f"Saved image to {args.output}")

if __name__ == '__main__':
    main() 