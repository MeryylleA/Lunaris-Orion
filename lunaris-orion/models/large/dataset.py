"""
Enhanced dataset implementation for the Large model with improved caching and preprocessing.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging
from PIL import Image
import numpy as np
from datasets import load_dataset
import torchvision.transforms as T
from typing import Optional, Tuple, Dict
import json
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
import io
import lmdb

class LargeDataset(Dataset):
    """
    Enhanced dataset for the Large model with:
    - LMDB caching for faster data loading
    - Parallel preprocessing
    - Smart batching
    - Advanced augmentation
    """
    def __init__(
        self,
        root_dir: str = 'data',
        image_size: int = 64,
        max_samples: Optional[int] = None,
        cache_dir: Optional[str] = None,
        use_augmentation: bool = True,
        num_workers: int = 4
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.max_samples = max_samples
        self.use_augmentation = use_augmentation
        self.num_workers = num_workers
        
        # Setup caching
        self.cache_dir = Path(cache_dir) if cache_dir else self.root_dir / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lmdb_path = self.cache_dir / 'data.lmdb'
        
        # Load dataset
        self._load_dataset()
        
        # Setup transforms
        self._setup_transforms()
        
        # Initialize LMDB environment
        self._setup_lmdb()
    
    def _load_dataset(self):
        """Load the dataset from Hugging Face and prepare indices."""
        logging.info("Loading dataset from Hugging Face...")
        
        # Load dataset
        self.dataset = load_dataset(
            "poloclub/diffusiondb",
            "2m_first_1k",
            cache_dir=str(self.cache_dir),
            num_proc=self.num_workers
        )['train']
        
        logging.info(f"Dataset fields available: {self.dataset.features.keys()}")
        
        # Apply max samples limit if specified
        if self.max_samples:
            self.dataset = self.dataset.select(range(min(self.max_samples, len(self.dataset))))
        
        logging.info(f"Dataset loaded with {len(self.dataset)} samples")
        
        # Create sample indices for smart batching
        self._create_sample_indices()
    
    def _create_sample_indices(self):
        """Create indices for smart batching based on sequence lengths."""
        self.indices = list(range(len(self.dataset)))
        
        # Sort indices by prompt length for more efficient batching
        prompt_lengths = [len(self.dataset[i]['text']) for i in self.indices]
        self.indices.sort(key=lambda i: prompt_lengths[i])
    
    def _setup_transforms(self):
        """Setup image transforms with optional augmentation."""
        transforms = [
            T.Resize((self.image_size, self.image_size), antialias=True),
            T.ToTensor(),
        ]
        
        if self.use_augmentation:
            transforms.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ])
        
        transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transform = T.Compose(transforms)
    
    def _setup_lmdb(self):
        """Initialize LMDB environment for caching."""
        self.env = lmdb.open(
            str(self.lmdb_path),
            map_size=1024 * 1024 * 1024 * 1024,  # 1TB
            create=True,
            readonly=False,
            meminit=False,
            map_async=True
        )
        
        # Prepare cache if needed
        if not self._is_cache_complete():
            self._prepare_cache()
    
    def _is_cache_complete(self) -> bool:
        """Check if LMDB cache is complete."""
        with self.env.begin() as txn:
            return txn.get(b'cache_complete') is not None
    
    def _prepare_cache(self):
        """Prepare LMDB cache with processed samples."""
        logging.info("Preparing dataset cache...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for idx in range(len(self.dataset)):
                futures.append(executor.submit(self._process_and_cache_sample, idx))
            
            # Wait for all processing to complete
            for future in futures:
                future.result()
        
        # Mark cache as complete
        with self.env.begin(write=True) as txn:
            txn.put(b'cache_complete', b'1')
        
        logging.info("Dataset cache preparation completed")
    
    def _process_and_cache_sample(self, idx: int):
        """Process and cache a single sample."""
        sample = self.dataset[idx]
        
        # Process image
        image = Image.open(io.BytesIO(sample['image']['bytes'])).convert('RGB')
        image_tensor = self.transform(image)
        
        # Process text
        text = sample['text']
        
        # Create cache key
        key = f"{idx}".encode()
        
        # Prepare data for caching
        cache_data = {
            'image': image_tensor.numpy(),
            'text': text
        }
        
        # Save to LMDB
        with self.env.begin(write=True) as txn:
            txn.put(key, json.dumps(cache_data).encode())
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset using LMDB cache."""
        true_idx = self.indices[idx]
        
        # Get data from LMDB
        with self.env.begin() as txn:
            data = json.loads(txn.get(f"{true_idx}".encode()))
        
        # Convert back to tensors
        image = torch.from_numpy(np.array(data['image']))
        text = data['text']
        
        # Apply augmentation if enabled
        if self.use_augmentation and self.training:
            image = self.transform(image)
        
        return text, image
    
    def get_sample_weight(self, idx: int) -> float:
        """Get sample weight for weighted sampling."""
        # Implement custom weighting logic here if needed
        return 1.0
    
    def set_epoch(self, epoch: int):
        """Set the epoch number for potential epoch-dependent behavior."""
        self.epoch = epoch
        
        # Optionally re-shuffle indices
        if epoch > 0:
            np.random.shuffle(self.indices) 