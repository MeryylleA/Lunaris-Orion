"""
Test script for the DiffusionDB dataset implementation.
"""

import torch
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from dataset import DiffusionDBDataset
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def test_dataset():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create dataset in dev mode
    dataset = DiffusionDBDataset(
        root_dir='data/diffusiondb',
        image_size=64,
        dev_mode=True,
        dev_samples=10  # Start with just 10 samples for quick testing
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=DiffusionDBDataset.collate_fn
    )
    
    # Test a single batch
    prompts, images = next(iter(dataloader))
    
    # Print shapes and data types
    logging.info(f"Prompts shape: {prompts.shape}")
    logging.info(f"Images shape: {images.shape}")
    logging.info(f"Prompts dtype: {prompts.dtype}")
    logging.info(f"Images dtype: {images.dtype}")
    
    # Decode a prompt to verify tokenization
    decoded_prompt = dataset.tokenizer.decode(prompts[0])
    logging.info(f"Sample prompt: {decoded_prompt}")
    
    # Save sample images
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)
    
    # Save individual images
    for i, image in enumerate(images):
        save_image(image, output_dir / f'sample_{i}.png')
    
    # Create a grid of images
    save_image(images, output_dir / 'grid.png', nrow=2)
    
    logging.info(f"Saved sample images to {output_dir}")

if __name__ == '__main__':
    test_dataset() 