"""
Colab Setup Utilities for Sketch-to-Image Project
Run this cell first in your Colab notebook
"""

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
from google.colab import drive

def setup_colab_environment():
    """
    Complete setup for Google Colab environment
    """
    print("Setting up Google Colab environment...")
    
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Set matplotlib to inline
    %matplotlib inline
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("CUDA not available, using CPU")
    
    # Set up working directory
    %cd /content
    print("Current working directory:", os.getcwd())
    
    return True

def install_requirements():
    """
    Install required packages for the project
    """
    print("Installing required packages...")
    
    # Install packages
    !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    !pip install pillow matplotlib numpy tqdm
    
    print("Package installation completed!")

def download_dataset():
    """
    Download and extract the FS2K dataset
    """
    print("Setting up dataset...")
    
    # Create directories
    !mkdir -p /content/dataset
    !mkdir -p /content/checkpoints
    !mkdir -p /content/results
    
    # If you have the dataset in Google Drive, copy it
    if os.path.exists('/content/drive/MyDrive/FS2K.zip'):
        print("Found FS2K.zip in Google Drive, copying...")
        !cp /content/drive/MyDrive/FS2K.zip /content/
        !unzip -q /content/FS2K.zip -d /content/
        print("Dataset extracted successfully!")
    else:
        print("Please upload FS2K.zip to your Google Drive or provide the dataset path")
    
    return True

def check_gpu_memory():
    """
    Check and display GPU memory usage
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Cached: {cached:.2f} GB")
        print(f"  Total: {total:.2f} GB")
        print(f"  Free: {total - allocated:.2f} GB")
    else:
        print("No GPU available")

def clear_gpu_memory():
    """
    Clear GPU memory
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory cleared")

# Run setup when imported
if __name__ == "__main__":
    setup_colab_environment()
    install_requirements()
    download_dataset()
    check_gpu_memory() 