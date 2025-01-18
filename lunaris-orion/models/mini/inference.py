"""
Inference script for the Mini model.
Optimized for CPU usage and single image generation.
"""

import torch
from pathlib import Path
import logging
from PIL import Image
import argparse
from model import MiniModel
from transformers import GPT2Tokenizer
import json
import numpy as np

class PixelArtGenerator:
    def __init__(self, checkpoint_path: str = 'checkpoints/best_model.pt'):
        self.device = torch.device('cpu')
        logging.info(f'Using device: {self.device}')
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.config = checkpoint['config']
            logging.info(f'Loaded config: {json.dumps(self.config["model"], indent=2)}')
        except Exception as e:
            logging.error(f'Error loading checkpoint: {str(e)}')
            raise
        
        # Initialize model
        try:
            self.model = MiniModel(self.config['model'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            logging.info('Model initialized successfully')
        except Exception as e:
            logging.error(f'Error initializing model: {str(e)}')
            raise
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logging.info('Model loaded successfully')
    
    def generate(self, prompt: str, output_path: str = None, temperature: float = 0.7) -> Image.Image:
        """
        Generate a pixel art image from a text prompt.
        
        Args:
            prompt: Text description of the desired image
            output_path: Optional path to save the generated image
            temperature: Controls randomness in generation (0.0 to 1.0)
            
        Returns:
            PIL.Image: Generated pixel art image
        """
        try:
            logging.info(f'Generating image for prompt: "{prompt}" with temperature: {temperature}')
            
            # Tokenize prompt
            tokens = self.tokenizer(
                prompt,
                padding='max_length',
                truncation=True,
                max_length=self.config['model']['max_sequence_length'],
                return_tensors='pt'
            )
            logging.info(f'Tokenized prompt shape: {tokens["input_ids"].shape}')
            
            # Generate image
            with torch.no_grad():
                tokens = tokens['input_ids'].to(self.device)
                output = self.model.generate(tokens, temperature=temperature)
                logging.info(f'Model output shape: {output.shape}')
                
                # Validate output
                if output.shape[1] != 3:
                    raise ValueError(f'Expected 3 channels, got {output.shape[1]}')
                if torch.isnan(output).any():
                    raise ValueError('Output contains NaN values')
                if output.min() < 0 or output.max() > 1:
                    logging.warning(f'Output range: [{output.min():.2f}, {output.max():.2f}], clipping to [0, 1]')
                    output = torch.clamp(output, 0, 1)
                
                # Convert to PIL image
                image = output[0].cpu()  # Shape: [3, H, W]
                image = (image * 255).byte().permute(1, 2, 0).numpy()
                logging.info(f'Converted image shape: {image.shape}, dtype: {image.dtype}')
                logging.info(f'Image range: [{image.min()}, {image.max()}]')
                
                # Ensure valid image data
                if image.min() == image.max():
                    raise ValueError('Image has no variation (solid color)')
                
                image = Image.fromarray(image)
                
                # Save if output path provided
                if output_path:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    image.save(output_path)
                    logging.info(f'Saved image to: {output_path}')
                
                return image
                
        except Exception as e:
            logging.error(f'Error generating image: {str(e)}')
            raise

def main():
    parser = argparse.ArgumentParser(description='Generate pixel art from text prompt')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for image generation')
    parser.add_argument('--output', type=str, default='generated.png', help='Output image path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt', help='Model checkpoint path')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature (0.0 to 1.0)')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Generate image
    try:
        generator = PixelArtGenerator(args.checkpoint)
        image = generator.generate(args.prompt, args.output, args.temperature)
        logging.info('Generation completed successfully')
    except Exception as e:
        logging.error(f'Generation failed: {str(e)}')
        raise

if __name__ == '__main__':
    main() 