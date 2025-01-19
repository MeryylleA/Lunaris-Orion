# Lunaris Orion AI Models

This directory contains the implementation of two AI models for pixel art generation:
- **Mini**: A lightweight model optimized for speed and efficiency
- **Large**: A more complex model focused on high-quality output

## Project Structure

```
models/
├── requirements.txt        # Project dependencies
├── mini/                  # Mini model implementation
│   ├── README.md         # Mini model documentation
│   ├── config.py         # Configuration parameters
│   ├── model.py          # Model architecture
│   ├── dataset.py        # Data handling
│   └── train.py          # Training script
└── large/                # Large model implementation
    ├── README.md         # Large model documentation
    ├── config.py         # Configuration parameters
    ├── model.py          # Model architecture
    ├── dataset.py        # Data handling
    └── train.py          # Training script
```

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Mini Model

The Mini model is designed for rapid pixel art generation with reasonable quality. It features:
- Lightweight transformer architecture
- Fast inference time (<1s per image)
- Optimized for 32x32 and 64x64 pixel art
- Efficient training on consumer GPUs

See `mini/README.md` for detailed documentation.

## Large Model

The Large model is designed for high-quality pixel art generation. It features:
- Advanced transformer architecture
- Higher parameter count for better quality
- Support for larger image sizes
- Distributed training capability

See `large/README.md` for detailed documentation.

## Training

Each model has its own training script with configuration options. Basic training commands:

### Mini Model
```bash
cd mini
python train.py
```

### Large Model
```bash
cd large
python train.py
```

## Integration with Discord Bot

The models are designed to be easily integrated with the Discord bot. Key integration points:
1. Model loading and initialization
2. Prompt processing
3. Image generation
4. Error handling

See the bot's documentation for integration details.

## Development Guidelines

1. **Code Style**
   - Follow PEP 8
   - Use type hints
   - Write comprehensive docstrings

2. **Testing**
   - Write unit tests for new features
   - Test both CPU and GPU execution
   - Validate outputs thoroughly

3. **Documentation**
   - Keep READMEs updated
   - Document configuration options
   - Include usage examples

4. **Performance**
   - Profile code regularly
   - Optimize critical paths
   - Monitor memory usage

## Contributing

1. Create a new branch for your feature
2. Follow the development guidelines
3. Test thoroughly
4. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details. 