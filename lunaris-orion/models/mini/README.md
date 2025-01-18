# Lunaris Orion - Mini Model

## Overview
The Mini model is a lightweight, efficient model designed for rapid pixel art generation. It prioritizes speed and computational efficiency while maintaining acceptable quality for basic pixel art generation tasks.

## Model Architecture
- Lightweight transformer-based architecture
- Optimized for 32x32 and 64x64 pixel art generation
- Fast inference with reduced parameter count
- Memory-efficient design

## Requirements
- Python 3.11
- PyTorch 2.4
- CUDA compatible GPU (optional, but recommended)

## Training
To train the model:
1. Prepare your dataset in the `data/` directory
2. Configure training parameters in `config.py`
3. Run training:
```bash
python train_mini.py --config config.py
```

### Training Configuration
Key parameters in `config.py`:
- `batch_size`: Number of images per batch
- `learning_rate`: Initial learning rate
- `epochs`: Number of training epochs
- `image_size`: Target image size (32 or 64)

## Testing
To test the model:
```bash
python test_mini.py --prompt "your prompt" --output output.png
```

## Model Details
- Input: Text prompts describing pixel art
- Output: Pixel art images (32x32 or 64x64)
- Training time: ~2-4 hours on modern GPU
- Inference time: <1 second per image

## Integration
The model integrates with the Discord bot through the `inference.py` module. See the main documentation for bot integration details. 