"""
Configuration file for the Mini model of Lunaris Orion.
Contains all hyperparameters and settings for model architecture, training, and inference.
Includes optimized settings for CPU training in dev mode.
"""

# Model Architecture
model_config = {
    'embedding_dim': 256,          # Reduced embedding dimension for CPU training
    'num_layers': 4,              # Reduced number of layers
    'num_heads': 4,               # Reduced number of attention heads
    'ff_dim': 1024,              # Reduced feed-forward network dimension
    'dropout_rate': 0.1,         # Dropout rate for regularization
    'max_sequence_length': 128,   # Maximum sequence length for text input
    'vocab_size': 50257,         # GPT-2 vocabulary size
    'image_size': 64,            # Target image size (64x64 for pixel art)
}

# Training Configuration
training_config = {
    'batch_size': 16,            # Smaller batch size for memory efficiency
    'learning_rate': 5e-5,       # Reduced learning rate for stability
    'epochs': 100,               # Number of training epochs (will be adjusted in dev mode)
    'warmup_steps': 100,         # Reduced warmup steps for smaller dataset
    'weight_decay': 0.01,        # Weight decay for regularization
    'gradient_clip_val': 1.0,    # Gradient clipping value
    'save_every_n_epochs': 1,    # Save checkpoint every epoch in dev mode
}

# Data Configuration
data_config = {
    'train_data_dir': 'data/train',
    'val_data_dir': 'data/val',
    'test_data_dir': 'data/test',
    'num_workers': 0,            # No multiprocessing for now to avoid issues
    'pin_memory': False,         # Disabled for CPU training
}

# Optimizer Configuration
optimizer_config = {
    'type': 'AdamW',            # Optimizer type
    'beta1': 0.9,               # Adam beta1
    'beta2': 0.999,             # Adam beta2
    'epsilon': 1e-8,            # Adam epsilon
}

# Inference Configuration
inference_config = {
    'temperature': 0.7,         # Sampling temperature
    'top_k': 50,               # Top-k sampling parameter
    'top_p': 0.9,              # Nucleus sampling parameter
    'num_inference_steps': 50,  # Number of denoising steps
}

# Logging Configuration
logging_config = {
    'log_dir': 'logs',
    'tensorboard': True,        # Enable TensorBoard logging
    'log_every_n_steps': 1,     # Log every step in dev mode
    'wandb': False,            # Disable Weights & Biases logging
}

# Hardware Configuration
hardware_config = {
    'device': 'cuda',          # Will be overridden by command line args
    'precision': '32',         # Use FP32 for CPU training
    'compile': False,          # Disabled for CPU training
}

# Save all configurations in a single dictionary
config = {
    'model': model_config,
    'training': training_config,
    'data': data_config,
    'optimizer': optimizer_config,
    'inference': inference_config,
    'logging': logging_config,
    'hardware': hardware_config,
} 