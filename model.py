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


# Replace the entire UNetGenerator class with this one.
# In model.py, replace the entire UNetGenerator class with this one.

class UNetGenerator(nn.Module):
    """
    A standard U-Net Generator based on the Pix2Pix paper.
    It includes skip connections and caps the number of filters in deeper layers.
    """
    def __init__(self, input_channels=3, output_channels=3, num_filters=64):
        super(UNetGenerator, self).__init__()

        # Helper for a downsampling block
        def down_block(in_filters, out_filters, bn=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False)]
            if bn:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Helper for an upsampling block
        def up_block(in_filters, out_filters, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_filters, out_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_filters),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Encoder path
        self.down1 = down_block(input_channels, num_filters, bn=False) # 256 -> 128
        self.down2 = down_block(num_filters, num_filters * 2)         # 128 -> 64
        self.down3 = down_block(num_filters * 2, num_filters * 4)     # 64 -> 32
        self.down4 = down_block(num_filters * 4, num_filters * 8)     # 32 -> 16
        self.down5 = down_block(num_filters * 8, num_filters * 8)     # 16 -> 8
        self.down6 = down_block(num_filters * 8, num_filters * 8)     # 8 -> 4
        self.down7 = down_block(num_filters * 8, num_filters * 8)     # 4 -> 2
        
        # Bottleneck
        self.bottleneck = down_block(num_filters * 8, num_filters * 8) # 2 -> 1

        # Decoder path
        # The input channels are doubled due to skip connections (cat)
        self.up1 = up_block(num_filters * 8, num_filters * 8, dropout=True)
        self.up2 = up_block(num_filters * 8 * 2, num_filters * 8, dropout=True)
        self.up3 = up_block(num_filters * 8 * 2, num_filters * 8, dropout=True)
        self.up4 = up_block(num_filters * 8 * 2, num_filters * 8,dropout=True)
        self.up5 = up_block(num_filters * 8 * 2, num_filters * 4)
        self.up6 = up_block(num_filters * 4 * 2, num_filters * 2)
        self.up7 = up_block(num_filters * 2 * 2, num_filters)

        # Final layer to produce the output image
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, output_channels, 4, 2, 1),
            nn.Tanh() # Tanh activation scales output to [-1, 1]
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        b = self.bottleneck(d7)

        # Decoder with skip connections
        u1 = self.up1(b)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        
        return self.final_layer(torch.cat([u7, d1], 1))

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix GAN
    Takes concatenated sketch and generated photo (6 channels) as input
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

# In model.py, replace the ENTIRE Pix2PixDataset class with this corrected version.

class Pix2PixDataset(Dataset):
    """
    Corrected Custom Dataset for Pix2Pix that handles online data augmentation.
    It splits the image first, then applies the *same* random transformations
    to both the sketch and the photo to ensure they remain a matched pair.
    """
    def __init__(self, data_dir: str, split: str = 'train', image_size: int = 256):
        self.split = split
        self.image_size = image_size
        
        # Define the transformations that will be applied to both images.
        # We will handle RandomHorizontalFlip manually.
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # --- Load file paths ---
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
        # Load the combined side-by-side image
        image_path = self.image_files[idx]
        combined_image = Image.open(image_path).convert('RGB')
        
        # 1. Split the PIL image FIRST
        w, h = combined_image.size
        sketch_pil = combined_image.crop((0, 0, w//2, h))
        photo_pil = combined_image.crop((w//2, 0, w, h))
        
        # 2. Apply the same random flip to both images (for training set only)
        if self.split == 'train' and torch.rand(1) < 0.5:
            sketch_pil = transforms.functional.hflip(sketch_pil)
            photo_pil = transforms.functional.hflip(photo_pil)
        
        if self.split == 'train':
            # Apply color jitter to the photo only
            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            photo_pil = color_jitter(photo_pil)

        # 3. Apply the rest of the transformations (Resize, ToTensor, Normalize)
        sketch = self.transform(sketch_pil)
        photo = self.transform(photo_pil)
        
        return sketch, photo





def create_models(device):
    """
    Create and initialize the generator and discriminator models
    Generator: 3 channels (sketch) -> 3 channels (photo)
    Discriminator: 6 channels (sketch + photo) -> 1 channel (real/fake)
    """
    generator = UNetGenerator(input_channels=3, output_channels=3, 
                            num_filters=64).to(device)
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
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    return generator, discriminator

def create_small_models(device):
    """
    Create smaller models for memory-constrained environments
    """
    generator = UNetGenerator(input_channels=3, output_channels=3, 
                            num_filters=32).to(device)  # Reduced filters and layers
    discriminator = PatchGANDiscriminator(input_channels=6, 
                                        num_filters=32, num_layers=2).to(device)  # Reduced filters and layers
    
    # Initialize weights
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    print(f"Small Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Small Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    return generator, discriminator

def get_device_info():
    """
    Get detailed device information
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
    Plot model architecture summary
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