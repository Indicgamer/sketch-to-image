import os
import cv2
import numpy as np
import json
import random
from tqdm import tqdm
import argparse
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Pix2PixAugmenter:
    def __init__(self, input_path, output_path, augmentation_factor=3):
        """
        Initialize the Pix2Pix augmenter.
        
        Args:
            input_path (str): Path to the split dataset
            output_path (str): Path to save the augmented dataset
            augmentation_factor (int): How many augmented versions to create per image
        """
        self.input_path = input_path
        self.output_path = output_path
        self.augmentation_factor = augmentation_factor
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)
        
        # Define augmentation pipeline
        self.setup_augmentations()
    
    def setup_augmentations(self):
        """Setup augmentation pipelines for different scenarios."""
        
        # Geometric augmentations (applied to both sketch and photo)
        self.geometric_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                p=0.3
            ),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),
        ])
        
        # Color augmentations (applied to photo only)
        self.color_aug = A.Compose([
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.3
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.3
            ),
        ])
        
        # Noise augmentations
        self.noise_aug = A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.ISONoise(color_shift=(0.01, 0.05), p=0.1),
            A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.1),
        ])
        
        # Combined augmentation for training
        self.combined_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=8, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.08,
                rotate_limit=3,
                p=0.3
            ),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.08,
                hue=0.03,
                p=0.4
            ),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.15),
        ])
    
    def load_image(self, image_path):
        """Load image and convert to RGB."""
        image = cv2.imread(image_path)
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def save_image(self, image, output_path):
        """Save image in BGR format."""
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
    
    def split_merged_image(self, merged_image):
        """Split merged image into sketch and photo."""
        height, width = merged_image.shape[:2]
        half_width = width // 2
        
        sketch = merged_image[:, :half_width]
        photo = merged_image[:, half_width:]
        
        return sketch, photo
    
    def merge_images(self, sketch, photo):
        """Merge sketch and photo into single image."""
        height, width = sketch.shape[:2]
        merged = np.zeros((height, width * 2, 3), dtype=np.uint8)
        merged[:, :width] = sketch
        merged[:, width:] = photo
        return merged
    
    def augment_image_pair(self, sketch, photo, augmentation_type='combined'):
        """Apply augmentation to sketch-photo pair."""
        
        if augmentation_type == 'geometric':
            # Apply geometric augmentation to both
            augmented = self.geometric_aug(image=sketch, image1=photo)
            return augmented['image'], augmented['image1']
        
        elif augmentation_type == 'color':
            # Apply geometric to both, then color to photo only
            geo_aug = self.geometric_aug(image=sketch, image1=photo)
            sketch_aug = geo_aug['image']
            photo_aug = geo_aug['image1']
            
            # Apply color augmentation to photo only
            color_aug = self.color_aug(image=photo_aug)
            photo_aug = color_aug['image']
            
            return sketch_aug, photo_aug
        
        elif augmentation_type == 'noise':
            # Apply geometric + noise
            geo_aug = self.geometric_aug(image=sketch, image1=photo)
            sketch_aug = geo_aug['image']
            photo_aug = geo_aug['image1']
            
            # Apply noise to both
            noise_aug_sketch = self.noise_aug(image=sketch_aug)
            noise_aug_photo = self.noise_aug(image=photo_aug)
            
            return noise_aug_sketch['image'], noise_aug_photo['image']
        
        else:  # combined
            # Apply combined augmentation
            augmented = self.combined_aug(image=sketch, image1=photo)
            return augmented['image'], augmented['image1']
    
    def augment_dataset(self, split_name):
        """Augment dataset for a specific split."""
        print(f"\nAugmenting {split_name} split...")
        
        input_dir = os.path.join(self.input_path, split_name)
        output_dir = os.path.join(self.output_path, split_name)
        
        # Load metadata
        metadata_file = os.path.join(self.input_path, f'{split_name}_metadata.json')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        augmented_metadata = []
        augmentation_types = ['geometric', 'color', 'noise', 'combined']
        
        for i, item in enumerate(tqdm(metadata, desc=f"Augmenting {split_name}")):
            # Copy original image
            original_path = os.path.join(input_dir, item['filename'])
            original_output_path = os.path.join(output_dir, item['filename'])
            
            # Copy original image
            if os.path.exists(original_path):
                import shutil
                shutil.copy2(original_path, original_output_path)
                
                # Add original to metadata
                original_metadata_entry = {
                    'filename': item['filename'],
                    'original_source': item['original_source'],
                    'original_filename': item['original_filename'],
                    'split': split_name,
                    'index': i,
                    'augmentation_type': 'original',
                    'original_metadata': item.get('original_metadata', {})
                }
                augmented_metadata.append(original_metadata_entry)
            
            # Create augmented versions
            if split_name == 'train':  # Only augment training data
                image = self.load_image(original_path)
                if image is not None:
                    sketch, photo = self.split_merged_image(image)
                    
                    for aug_idx, aug_type in enumerate(augmentation_types):
                        try:
                            # Apply augmentation
                            aug_sketch, aug_photo = self.augment_image_pair(
                                sketch, photo, aug_type
                            )
                            
                            # Merge augmented images
                            aug_merged = self.merge_images(aug_sketch, aug_photo)
                            
                            # Save augmented image
                            aug_filename = f"{split_name}_{i:06d}_aug{aug_idx+1}.jpg"
                            aug_output_path = os.path.join(output_dir, aug_filename)
                            self.save_image(aug_merged, aug_output_path)
                            
                            # Add to metadata
                            aug_metadata_entry = {
                                'filename': aug_filename,
                                'original_source': item['original_source'],
                                'original_filename': item['original_filename'],
                                'split': split_name,
                                'index': i,
                                'augmentation_type': aug_type,
                                'augmentation_index': aug_idx + 1,
                                'original_metadata': item.get('original_metadata', {})
                            }
                            augmented_metadata.append(aug_metadata_entry)
                            
                        except Exception as e:
                            print(f"Error augmenting {item['filename']} with {aug_type}: {e}")
                            continue
        
        # Save augmented metadata
        output_metadata_file = os.path.join(self.output_path, f'{split_name}_metadata.json')
        with open(output_metadata_file, 'w') as f:
            json.dump(augmented_metadata, f, indent=2)
        
        return len(augmented_metadata)
    
    def augment_all_splits(self):
        """Augment all dataset splits."""
        print("Starting dataset augmentation...")
        
        splits = ['train', 'val', 'test']
        total_images = 0
        
        for split in splits:
            count = self.augment_dataset(split)
            total_images += count
            print(f"{split.capitalize()} split: {count} images")
        
        # Save overall metadata
        self.save_overall_metadata()
        
        print(f"\nAugmentation completed!")
        print(f"Total images: {total_images}")
        print(f"Output directory: {self.output_path}")
    
    def save_overall_metadata(self):
        """Save overall dataset metadata."""
        # Count images in each split
        train_count = len([f for f in os.listdir(os.path.join(self.output_path, 'train')) 
                          if f.endswith('.jpg')])
        val_count = len([f for f in os.listdir(os.path.join(self.output_path, 'val')) 
                        if f.endswith('.jpg')])
        test_count = len([f for f in os.listdir(os.path.join(self.output_path, 'test')) 
                         if f.endswith('.jpg')])
        
        overall_metadata = {
            'dataset_info': {
                'name': 'FS2K_Pix2Pix_Augmented',
                'description': 'FS2K dataset with augmentation for Pix2Pix GAN training',
                'total_samples': train_count + val_count + test_count,
                'train_samples': train_count,
                'val_samples': val_count,
                'test_samples': test_count,
                'augmentation_factor': self.augmentation_factor,
                'augmentation_types': ['geometric', 'color', 'noise', 'combined']
            },
            'augmentation_details': {
                'geometric': 'Horizontal flips, rotations, scaling, elastic transforms',
                'color': 'Brightness, contrast, saturation, hue adjustments',
                'noise': 'Gaussian noise, ISO noise, multiplicative noise',
                'combined': 'Mix of all augmentation types'
            },
            'usage': {
                'train': 'Use for training the Pix2Pix GAN model (includes augmented versions)',
                'val': 'Use for validation during training (original images only)',
                'test': 'Use for final evaluation (original images only)'
            }
        }
        
        metadata_file = os.path.join(self.output_path, 'dataset_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(overall_metadata, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Augment FS2K dataset for Pix2Pix GAN training')
    parser.add_argument('--input_path', type=str, default='split_data',
                        help='Path to the split dataset')
    parser.add_argument('--output_path', type=str, default='augmented_data',
                        help='Path to save the augmented dataset')
    parser.add_argument('--augmentation_factor', type=int, default=4,
                        help='Number of augmented versions per image (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize augmenter
    augmenter = Pix2PixAugmenter(
        input_path=args.input_path,
        output_path=args.output_path,
        augmentation_factor=args.augmentation_factor
    )
    
    # Augment dataset
    augmenter.augment_all_splits()
    
    print(f"\nDataset augmentation completed successfully!")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Augmentation factor: {args.augmentation_factor}")

if __name__ == "__main__":
    main() 