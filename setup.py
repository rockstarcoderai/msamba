from setuptools import setup, find_packages

setup(
    name="enhanced-msamaba",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "PyYAML>=6.0",
        "wandb>=0.15.0",
        "pytest>=7.0.0",
        "tqdm>=4.65.0",
        "einops>=0.7.0",
        "transformers>=4.30.0",
        "librosa>=0.10.0",
        "opencv-python>=4.8.0",
    ],
    python_requires=">=3.8",
)
