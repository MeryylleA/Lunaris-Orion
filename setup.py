from setuptools import setup, find_packages

setup(
    name="mini_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "psutil>=5.8.0",
        "pyyaml>=5.4.1",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
        "timm>=0.6.12",
        "torchvision>=0.15.1",
        "torchmetrics>=0.11.0",
        "torch-optimizer>=0.1.0",
        "optuna-4.2.0",
    ],
    python_requires=">=3.8",
) 