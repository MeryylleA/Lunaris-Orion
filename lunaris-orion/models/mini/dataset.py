"""
Dataset implementation for DiffusionDB Pixel Art, optimized for H100 GPU.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import logging
from typing import Tuple, Optional, Union
from pathlib import Path
from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np
import pandas as pd
import torchvision.transforms as T

class DiffusionDBDataset(Dataset):
    """Dataset class for loading pixel art images and prompts from DiffusionDB."""
    
    def __init__(self, 
                 root_dir: Union[str, Path],
                 image_size: int = 64,
                 max_sequence_length: int = 77,
                 dev_mode: bool = False,
                 dev_samples: int = 1000):
        """Initialize the dataset."""
        super().__init__()
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_size = image_size
        self.max_sequence_length = max_sequence_length
        self.dev_mode = dev_mode
        self.dev_samples = dev_samples
        
        # Set up image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ])
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load dataset with recommended configuration
        logging.info("Loading dataset from Hugging Face...")
        # No need for config as the dataset has a single configuration
        self.dataset = load_dataset(
            "jainr3/diffusiondb-pixelart",
            split="train",
            trust_remote_code=True
        )
        
        # If in dev mode, take a subset
        if dev_mode and dev_samples:
            self.dataset = self.dataset.select(range(min(dev_samples, len(self.dataset))))
        
        # Log dataset information
        if len(self.dataset) > 0:
            first_item = self.dataset[0]
            logging.info(f"Dataset fields available: {first_item.keys()}")
            if 'p' in first_item:  # According to the dataset structure, 'p' is the prompt field
                logging.info(f"Sample prompt: {first_item['p']}")
        
        logging.info(f"Dataset loaded with {len(self.dataset)} samples")
    
    def _load_image(self, image_data) -> torch.Tensor:
        """Load and transform an image."""
        try:
            # Handle PIL Image directly
            if isinstance(image_data, Image.Image):
                image = image_data.convert('RGB')
                image_tensor = self.transform(image)
                return image_tensor
            
            logging.error(f"Unexpected image data type: {type(image_data)}")
            return torch.zeros(3, self.image_size, self.image_size)
            
        except Exception as e:
            logging.error(f"Error loading image: {str(e)}")
            return torch.zeros(3, self.image_size, self.image_size)
    
    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize and pad a text prompt."""
        try:
            # Ensure we have a valid string
            if not isinstance(prompt, str) or not prompt:
                logging.warning(f"Invalid or empty prompt: {prompt}, using default text")
                prompt = "a pixel art image"  # Default prompt instead of empty string
            
            # Tokenize with padding and truncation
            tokens = self.tokenizer(
                prompt,
                padding='max_length',
                max_length=self.max_sequence_length,
                truncation=True,
                return_tensors='pt'
            )
            
            # Convert to long tensor and ensure correct shape
            token_ids = tokens['input_ids'].squeeze().long()
            
            # Handle case where we get a 0-dim tensor
            if token_ids.ndim == 0:
                token_ids = token_ids.unsqueeze(0)
            
            # Ensure we have the correct sequence length
            if token_ids.size(0) != self.max_sequence_length:
                if idx == 0:  # Only log for first item to avoid spam
                    logging.warning(f"Token sequence length mismatch: {token_ids.size(0)} vs {self.max_sequence_length}")
                # Pad or truncate to exact length
                if token_ids.size(0) < self.max_sequence_length:
                    padding = torch.full((self.max_sequence_length - token_ids.size(0),), self.tokenizer.pad_token_id, dtype=torch.long)
                    token_ids = torch.cat([token_ids, padding])
                else:
                    token_ids = token_ids[:self.max_sequence_length]
            
            return token_ids
            
        except Exception as e:
            logging.error(f"Error tokenizing prompt: {str(e)}")
            return torch.zeros(self.max_sequence_length, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        try:
            item = self.dataset[idx]
            
            # Load and process image
            image = self._load_image(item['image'])
            
            # Get prompt - try different possible field names
            prompt = None
            for field in ['text', 'prompt', 'p']:
                if field in item and item[field]:
                    prompt = item[field]
                    break
            
            if not prompt:
                logging.warning(f"No prompt found in item {idx}, using default")
                prompt = "a pixel art image"
            
            tokens = self._tokenize_prompt(prompt)
            
            # Ensure tokens are long tensors
            tokens = tokens.long()
            
            return tokens, image
            
        except Exception as e:
            logging.error(f"Error getting item {idx}: {str(e)}")
            return (
                torch.zeros(self.max_sequence_length, dtype=torch.long),
                torch.zeros(3, self.image_size, self.image_size)
            )

class LocalPixelArtDataset(Dataset):
    """Dataset loader for local 16x16 pixel art images."""
    
    def __init__(self, root_dir: str = "dataset/16x16", split="train", transform=None):
        """
        Args:
            root_dir: Path to dataset root directory
            split: 'train' or 'val' split
            transform: Optional transform to be applied on images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        
        # Load labels and sprites
        self.labels = pd.read_csv(self.root_dir / "labels.csv")
        self.sprites = np.load(self.root_dir / "sprites.npy")
        self.sprite_labels = np.load(self.root_dir / "sprites_labels.npy")
        
        # Split dataset (90% train, 10% val)
        n_samples = len(self.labels)
        indices = np.random.permutation(n_samples)
        split_idx = int(0.9 * n_samples)
        
        if split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
            
        # Setup transforms
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        logging.info(f"Loaded {len(self.indices)} samples for {split} split")
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        # Get true index from split indices
        true_idx = self.indices[idx]
        
        # Load image and label
        image = self.sprites[true_idx]
        label = self.sprite_labels[true_idx]
        
        # Convert image to tensor and normalize
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            # If image is [H, W, C], convert to [C, H, W]
            if image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            image = image / 255.0  # Normalize to [0, 1]
        
        if self.transform:
            image = self.transform(image)
            
        return label, image

# Data augmentation transforms
def get_train_transforms():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_val_transforms():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]) 