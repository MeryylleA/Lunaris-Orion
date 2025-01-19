"""
Configuration for the Large model optimized for multi-GPU H100 training.
Enhanced with anti-overfitting, quantization, and monitoring features.
"""

config = {
    'model': {
        # Model Architecture
        'embedding_dim': 2048,        # 8x larger than Mini
        'num_heads': 32,             # 8x more heads
        'num_layers': 32,            # 4x more layers
        'ffn_dim': 8192,            # 8x larger FFN
        'dropout': 0.1,
        'attention_dropout': 0.1,    # Separate dropout for attention
        'max_sequence_length': 2048,
        'vocab_size': 50257,         # GPT-2 vocabulary size
        'image_size': 64,           # Pixel art resolution
        
        # Advanced Features
        'use_rope': True,           # Rotary Position Embeddings
        'use_parallel_attention': True,  # Parallel attention/FFN blocks
        'use_swiglu': True,         # SwiGLU activations
        'use_sliding_window': True,  # Sliding Window Attention
        'sliding_window_size': 256,  # Local attention window size
        'gradient_checkpointing': True,  # Memory efficient checkpointing
        
        # Anti-overfitting Features
        'stochastic_depth_rate': 0.1,  # Stochastic depth drop rate
        'layer_scale_init_value': 1e-6,  # Layer scale initialization
        'gradient_clip_norm': 1.0,    # Gradient clipping norm
        'weight_decay': 0.01,        # Weight decay for regularization
        'label_smoothing': 0.1,      # Label smoothing factor
        'mixup_alpha': 0.2,          # Mixup augmentation alpha
        'cutmix_alpha': 1.0,         # CutMix augmentation alpha
        
        # Architecture Enhancements
        'use_cross_attention': True,  # Enable cross-attention
        'use_sdpa': True,            # Use PyTorch 2.0's SDPA
        'use_adaptive_layernorm': True,  # Use adaptive layer normalization
        'use_absolute_positions': True,  # Use absolute position embeddings
        'skip_connection_scaling': True,  # Learn skip connection scales
        
        # Quantization Settings
        'quantization': {
            'enabled': True,
            'backend': 'x86',        # or 'fbgemm', 'qnnpack'
            'dtype': 'qint8',        # Quantization data type
            'calibration_method': 'histogram',  # or 'minmax'
            'symmetric': True,       # Symmetric quantization
            'per_channel': True,     # Per-channel quantization
            'reduce_range': True,    # Reduce range for x86 compatibility
            'activation_dtype': 'quint8',  # Activation quantization type
            'observer_type': 'histogram',  # Type of observer to use
            'moving_average_constant': 0.01,  # Moving average constant
        },
        
        # Multi-GPU Training
        'sharding_strategy': 'zero3',  # ZeRO stage 3 for distributed training
        'mixed_precision': 'bf16',    # Use bfloat16 mixed precision
        'gradient_accumulation_steps': 4,
        'batch_size_per_gpu': 32,    # Per GPU batch size
        'gradient_clipping': 1.0,    # Gradient clipping threshold
        'loss_scaling': 'dynamic',   # Dynamic loss scaling for mixed precision
        
        # Optimization
        'optimizer': {
            'name': 'fused_adamw',   # Use fused AdamW implementation
            'lr': 2e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01,
            'fused': True,           # Enable kernel fusion
            'foreach': True,         # Enable vectorized updates
            'maximize': False,
            'capturable': True,      # Enable CUDA graphs
            'clip_grad_norm': 1.0,   # Gradient clipping norm
            'scale_parameter': True, # Scale learning rates by parameter size
            'relative_step': True,   # Use relative step sizes
            'warmup_init': True,     # Initialize with warmup
        },
        
        # Learning Rate Schedule
        'lr_schedule': {
            'warmup_epochs': 5,
            'min_lr': 1e-5,
            'cosine_decay': True,
            'decay_epochs': 995,     # Total epochs - warmup epochs
            'final_lr_factor': 0.1,  # Final LR = min_lr * final_lr_factor
            'cycle_momentum': True,  # Cycle momentum with LR
            'div_factor': 25.0,     # Initial LR division factor
            'final_div_factor': 1e4, # Final LR division factor
        },
        
        # Flash Attention Settings
        'flash_attention': {
            'enabled': True,
            'version': 2,            # Using Flash Attention v2
            'block_size': 128,
            'causal': False,
            'dropout': 0.1,
            'softmax_scale': None,   # Automatically determine scale
            'memory_efficient': True, # Use memory efficient attention
            'num_chunks': 4,         # Number of chunks for memory efficiency
        },
        
        # Memory Optimizations
        'memory': {
            'enable_checkpointing': True,
            'checkpoint_ratio': 0.5,   # Ratio of layers to checkpoint
            'offload_optimizer': False, # CPU optimizer state offloading
            'pin_memory': True,        # Pin memory for faster GPU transfer
            'find_unused_parameters': False,
            'detect_anomaly': False,   # Disable anomaly detection in DDP
            'benchmark_cudnn': True,   # Enable cuDNN benchmarking
            'deterministic': False,    # Non-deterministic for better performance
            'allow_tf32': True,        # Allow TF32 on Ampere
            'cudnn_benchmark': True,   # Enable cuDNN benchmarking
            'cuda_cache_size': 4096,   # CUDA cache size in MB
        },
    },
    
    'training': {
        'epochs': 1000,
        'eval_every': 1,
        'save_every': 5,
        'early_stopping_patience': 20,
        'gradient_clip_val': 1.0,
        'seed': 42,
        
        # Enhanced Checkpointing
        'save_top_k': 3,
        'checkpoint_dir': 'checkpoints',
        'backup_dir': 'backups',
        'backup_freq_hours': 4,
        'keep_last_n_checkpoints': 5,
        'save_optimizer_state': True,
        'save_scheduler_state': True,
        'save_grad_scaler': True,
        'save_rng_state': True,
        
        # Advanced Monitoring
        'log_every_n_steps': 10,
        'profiling': True,
        'track_memory': True,
        'track_gpu_stats': True,
        'log_gradient_flow': True,
        'log_parameter_norm': True,
        'log_learning_rate': True,
        'log_gpu_stats': True,
        'log_cpu_stats': True,
        'log_disk_usage': True,
        'log_network_usage': True,
        
        # Validation and Testing
        'validation': {
            'freq': 1,              # Validate every N epochs
            'metric': 'loss',       # Primary metric to track
            'mode': 'min',          # Minimize the metric
            'patience': 20,         # Early stopping patience
            'min_delta': 1e-4,      # Minimum change to count as improvement
            'save_images': True,    # Save validation images
            'num_samples': 64,      # Number of validation samples to generate
        },
        
        # Error Handling and Recovery
        'error_handling': {
            'max_retries': 3,       # Maximum number of retries on error
            'retry_delay': 60,      # Delay between retries in seconds
            'ignore_nan_loss': False,  # Stop on NaN loss
            'save_on_error': True,  # Save checkpoint on error
            'error_log_file': 'error_log.txt',  # Error log file
        },
        
        # Tensorboard Configuration
        'tensorboard': {
            'enabled': True,
            'log_graph': True,
            'log_parameters': True,
            'log_gradients': True,
            'log_images': True,
            'update_freq': 'epoch',
            'histogram_freq': 1,
            'write_grads': True,
            'write_images': True,
        },
        
        # Weights & Biases Integration
        'wandb': {
            'enabled': True,
            'project': 'large-pixel-art',
            'name': None,           # Auto-generated
            'tags': ['large-model', 'pixel-art'],
            'notes': 'H100 training run',
            'save_code': True,
            'log_freq': 10,
            'watch_model': True,
            'log_artifacts': True,
        },
    },
    
    'distributed': {
        'enabled': True,
        'backend': 'nccl',
        'num_gpus': 2,
        'find_unused_parameters': False,
        'sync_batch_norm': True,
        'gradient_as_bucket_view': True,  # Memory optimization for DDP
        'broadcast_buffers': False,      # Disable buffer broadcasting
        'bucket_cap_mb': 25,            # DDP bucket size in MB
        'static_graph': True,           # Enable static graph optimization
        'ddp_timeout': 1800,           # DDP timeout in seconds
        'nccl_debug': 'INFO',          # NCCL debug level
        'nccl_ib_timeout': 30,         # NCCL IB timeout
        'init_method': 'env://',       # DDP initialization method
    },
    
    'dataset': {
        'num_workers': 8,
        'prefetch_factor': 2,
        'pin_memory': True,
        'persistent_workers': True,
        'drop_last': True,
        'smart_batching': True,         # Group similar length sequences
        'max_tokens_per_batch': 8192,   # Dynamic batching
        'cache_mode': 'lmdb',          # Use LMDB for dataset caching
        
        # Enhanced Data Loading
        'loading': {
            'async_loading': True,     # Asynchronous data loading
            'cache_size_gb': 32,       # Cache size in GB
            'shuffle_buffer_size': 10000,  # Size of shuffle buffer
            'reshuffle_each_iteration': True,
            'deterministic': False,
            'drop_remainder': True,
            'experimental_optimization': {
                'map_and_batch_fusion': True,
                'parallel_batch': True,
                'parallel_interleave': True,
            },
        },
        
        # Advanced Augmentation
        'augmentation': {
            'enabled': True,
            'random_horizontal_flip': 0.5,
            'color_jitter': {
                'brightness': 0.1,
                'contrast': 0.1,
                'saturation': 0.1,
                'hue': 0.02,
            },
            'random_affine': {
                'degrees': 0,
                'translate': (0.05, 0.05),
                'scale': None,
                'shear': None,
            },
            'mixup': {
                'enabled': True,
                'alpha': 0.2,
                'prob': 0.5,
            },
            'cutmix': {
                'enabled': True,
                'alpha': 1.0,
                'prob': 0.5,
            },
            'random_erasing': {
                'enabled': True,
                'prob': 0.25,
                'mode': 'pixel',
                'max_count': 1,
            },
        },
        
        # Data Preprocessing
        'preprocessing': {
            'normalize': True,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'resize_mode': 'bilinear',
            'pad_mode': 'reflect',
            'interpolation': 'bicubic',
            'max_pixel_value': 255.0,
        },
    }
} 