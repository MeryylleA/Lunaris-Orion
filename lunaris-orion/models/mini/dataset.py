"""
Dataset implementation for DiffusionDB Pixel Art, with support for dev mode.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import logging
from typing import Tuple, Optional
from datasets import load_dataset
from transformers import GPT2Tokenizer

class DiffusionDBDataset(Dataset):
    """Dataset class for DiffusionDB pixel art images and prompts."""
    
    def __init__(self, 
                 root_dir: str,
                 image_size: int = 64,
                 transform: Optional[transforms.Compose] = None,
                 max_sequence_length: int = 128,
                 dev_mode: bool = False,
                 dev_samples: int = 10000):
        """
        Initialize the DiffusionDB Pixel Art dataset.
        
        Args:
            root_dir: Directory to store/load the dataset cache
            image_size: Size of the images (assumes square images)
            transform: Optional transform to be applied to images
            max_sequence_length: Maximum length of text prompts
            dev_mode: If True, only load a small subset of data
            dev_samples: Number of samples to use in dev mode
        """
        self.image_size = image_size
        self.max_sequence_length = max_sequence_length
        
        # Set up image transforms
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset from Hugging Face
        subset = '2k_random_1k' if dev_mode else '2k_all'
        self.dataset = load_dataset('jainr3/diffusiondb-pixelart', subset, split='train')
        
        if dev_mode and dev_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(dev_samples))
            logging.info(f'Dev mode: Using {dev_samples} samples')
        
        logging.info(f'Loaded dataset with {len(self.dataset)} samples')
    
    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """Process a PIL image."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image)
    
    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize the text prompt using GPT-2 tokenizer."""
        encoded = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(0)
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        sample = self.dataset[idx]
        
        # Get image and prompt
        image = sample['image']
        prompt = sample['text']
        
        # Process image and tokenize prompt
        image_tensor = self._process_image(image)
        prompt_tokens = self._tokenize_prompt(prompt)
        
        return prompt_tokens, image_tensor
    
    @staticmethod
    def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom collate function for DataLoader."""
        prompts, images = zip(*batch)
        return torch.stack(prompts), torch.stack(images) 