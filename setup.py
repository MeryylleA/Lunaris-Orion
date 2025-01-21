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
    ],
    python_requires=">=3.8",
) 