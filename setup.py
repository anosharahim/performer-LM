from setuptools import setup, find_packages

setup(
    name="performer",
    version="0.1.0",
    description="Transformer implementation in PyTorch",
    author="Your Name",
    packages=["performer"],  # Automatically find all packages
    install_requires=[
        "torch>=1.12.0",
        "numpy",
        "datasets",
        "tokenizers",
        "matplotlib",
        "seaborn",
    ],
    python_requires=">=3.8",
)
