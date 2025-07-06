import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import gc

class UNetGenerator(nn.Module):
    """
    U-Net Generator for Pix2Pix GAN
    """
    def __init__(self, input_channels: int = 3, output_channels: int = 3, 
                 num_filters: int = 64, num_layers: int = 8):
        super(UNetGenerator, self).__init__()
        
        self.num_layers = num_layers
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Conv2d(input_channels, num_filters, kernel_size=4, 
                                     stride=2, padding=1, bias=False))
        
        for i in range(num_layers - 1):
            in_channels = num_filters * (2 ** i)
            out_channels = num_filters * (2 ** (i + 1))
            self.encoder.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, 
                                        stride=2, padding=1, bias=False))
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        for i in range(num_layers - 1):
            in_channels = num_filters * (2 ** (num_layers - 1 - i))
            out_channels = num_filters * (2 ** (num_layers - 2 - i))
            self.decoder.append(nn.ConvTranspose2d(in_channels, out_channels, 
                                                 kernel_size=4, stride=2, padding=1, bias=False))
        
        # Final output layer
        self.final = nn.ConvTranspose2d(num_filters, output_channels, kernel_size=4, 
                                       stride=2, padding=1, bias=False)
        
        # Batch normalization and activation
        self.batch_norm = nn.ModuleList()
        for i in range(num_layers - 1):
            self.batch_norm.append(nn.BatchNorm2d(num_filters * (2 ** (i + 1))))
        
        self.decoder_batch_norm = nn.ModuleList()
        for i in range(num_layers - 1):
            self.decoder_batch_norm.append(nn.BatchNorm2d(num_filters * (2 ** (num_layers - 2 - i))))
    
    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < len(self.encoder) - 1:
                x = F.leaky_relu(self.batch_norm[i](x), 0.2)
            else:
                x = F.relu(x)
            encoder_outputs.append(x)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            x = F.relu(self.decoder_batch_norm[i](x))
            # Skip connection
            skip_connection = encoder_outputs[-(i + 2)]
            x = torch.cat([x, skip_connection], dim=1)
        
        # Final output
        x = self.final(x)
        return torch.tanh(x)

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix GAN
    """
    def __init__(self, input_channels: int = 6, num_filters: int = 64, num_layers: int = 3):
        super(PatchGANDiscriminator, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer (no batch norm)
        self.layers.append(nn.Conv2d(input_channels, num_filters, kernel_size=4, 
                                    stride=2, padding=1, bias=False))
        
        # Middle layers
        for i in range(num_layers - 1):
            in_channels = num_filters * (2 ** i)
            out_channels = num_filters * (2 ** (i + 1))
            self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, 
                                       stride=2, padding=1, bias=False))
            self.layers.append(nn.BatchNorm2d(out_channels))
        
        # Final layer
        self.layers.append(nn.Conv2d(num_filters * (2 ** (num_layers - 1)), 1, 
                                   kernel_size=4, stride=1, padding=1, bias=False))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.BatchNorm2d):
                x = F.leaky_relu(layer(x), 0.2)
            else:
                x = F.leaky_relu(layer(x), 0.2)
        return x

class Pix2PixDataset(Dataset):
    """
    Custom Dataset for Pix2Pix training
    """
    def __init__(self, data_dir: str, transform=None, split: str = 'train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Get list of image files
        self.image_files = []
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            for file in os.listdir(split_dir):
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    self.image_files.append(os.path.join(split_dir, file))
        
        print(f"Found {len(self.image_files)} images in {split} set")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load the side-by-side image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Split the image into sketch (left) and photo (right)
        _, _, w = image.shape
        sketch = image[:, :, :w//2]  # Left half
        photo = image[:, :, w//2:]   # Right half
        
        return sketch, photo

def get_transforms(image_size: int = 256):
    """
    Get transforms for training and validation
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size * 2)),  # Double width for side-by-side
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size * 2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform

def create_models(device):
    """
    Create and initialize the generator and discriminator models
    """
    generator = UNetGenerator(input_channels=3, output_channels=3, 
                            num_filters=64, num_layers=8).to(device)
    discriminator = PatchGANDiscriminator(input_channels=6, 
                                        num_filters=64, num_layers=3).to(device)
    
    # Initialize weights
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    return generator, discriminator

# Colab-specific optimizations
def setup_colab():
    """
    Setup function for Google Colab environment
    """
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("Running in Google Colab")
        
        # Mount Google Drive if needed
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
    except ImportError:
        IN_COLAB = False
        print("Running locally")
    
    return IN_COLAB

def get_device_info():
    """
    Get detailed device information for Colab
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Available GPU Memory: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
    
    return device

def plot_model_summary(generator, discriminator, device):
    """
    Plot model architecture summary for Colab
    """
    # Test with dummy data to get output shapes
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 256, 256).to(device)
    
    with torch.no_grad():
        generated = generator(input_tensor)
        disc_input = torch.cat([input_tensor, generated], dim=1)
        disc_output = discriminator(disc_input)
    
    # Print model information
    print("=" * 50)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 50)
    print(f"Generator Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"Total Parameters: {sum(p.numel() for p in generator.parameters()) + sum(p.numel() for p in discriminator.parameters()):,}")
    print(f"Input Shape: {input_tensor.shape}")
    print(f"Generator Output Shape: {generated.shape}")
    print(f"Discriminator Output Shape: {disc_output.shape}")
    print("=" * 50)

if __name__ == "__main__":
    # Test the models
    device = get_device_info()
    
    # Create models
    generator, discriminator = create_models(device)
    
    # Plot model summary
    plot_model_summary(generator, discriminator, device)
    
    # Test with dummy data
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # Test generator
    with torch.no_grad():
        generated = generator(input_tensor)
        print(f"Generator output shape: {generated.shape}")
    
    # Test discriminator
    with torch.no_grad():
        # Concatenate input and generated image
        disc_input = torch.cat([input_tensor, generated], dim=1)
        disc_output = discriminator(disc_input)
        print(f"Discriminator output shape: {disc_output.shape}")
    
    print("Model architecture test completed successfully!") 