"""
Colab Data Preparation for FS2K Dataset
Complete pipeline: Download → Extract → Prepare → Split → Augment
"""

import os
import zipfile
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm
import json
from google.colab import files
import requests
from io import BytesIO

class FS2KDataPreparation:
    """
    Complete data preparation pipeline for FS2K dataset
    """
    def __init__(self, base_dir="/content"):
        self.base_dir = base_dir
        self.fs2k_dir = os.path.join(base_dir, "FS2K")
        self.prepared_dir = os.path.join(base_dir, "prepared_data")
        
        # Create directories
        os.makedirs(self.fs2k_dir, exist_ok=True)
        os.makedirs(self.prepared_dir, exist_ok=True)
        
    def find_uploaded_dataset(self):
        """
        Find the uploaded FS2K dataset
        """
        print("=" * 50)
        print("STEP 1: LOCATING UPLOADED DATASET")
        print("=" * 50)
        
        # Check common locations for uploaded files
        possible_locations = [
            os.path.join(self.base_dir, "FS2K.zip"),
            "/content/FS2K.zip",
            "FS2K.zip",
            os.path.join(self.base_dir, "FS2K", "FS2K.zip")
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                print(f"✅ Found FS2K.zip at: {location}")
                return location
        
        # If not found in common locations, check current directory
        current_files = os.listdir('.')
        zip_files = [f for f in current_files if f.lower().endswith('.zip')]
        
        if zip_files:
            print(f"Found zip files: {zip_files}")
            # Assume the first zip file is FS2K
            zip_path = zip_files[0]
            print(f"Using: {zip_path}")
            return zip_path
        
        print("❌ FS2K.zip not found!")
        print("Please make sure you've uploaded the FS2K.zip file")
        return None
    
    def extract_dataset(self, zip_path):
        """
        Extract FS2K dataset
        """
        print("\n" + "=" * 50)
        print("STEP 2: EXTRACTING DATASET")
        print("=" * 50)
        
        if not os.path.exists(zip_path):
            print(f"Zip file not found: {zip_path}")
            return False
        
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.base_dir)
        
        print("Extraction completed!")
        return True
    
    def prepare_side_by_side_images(self):
        """
        Create side-by-side images from sketch and photo pairs
        """
        print("\n" + "=" * 50)
        print("STEP 3: PREPARING SIDE-BY-SIDE IMAGES")
        print("=" * 50)
        
        # Find sketch and photo directories
        sketch_dirs = []
        photo_dirs = []
        
        for item in os.listdir(self.fs2k_dir):
            item_path = os.path.join(self.fs2k_dir, item)
            if os.path.isdir(item_path):
                if 'sketch' in item.lower():
                    sketch_dirs.append(item_path)
                elif 'photo' in item.lower():
                    photo_dirs.append(item_path)
        
        print(f"Found sketch directories: {len(sketch_dirs)}")
        print(f"Found photo directories: {len(photo_dirs)}")
        
        # Create output directory
        output_dir = os.path.join(self.prepared_dir, "side_by_side")
        os.makedirs(output_dir, exist_ok=True)
        
        total_pairs = 0
        processed_pairs = 0
        
        # Process each sketch/photo pair
        for sketch_dir in sketch_dirs:
            for photo_dir in photo_dirs:
                print(f"\nProcessing: {os.path.basename(sketch_dir)} ↔ {os.path.basename(photo_dir)}")
                
                # Get all sketch files
                sketch_files = []
                for ext in ['.jpg', '.jpeg', '.png']:
                    sketch_files.extend([f for f in os.listdir(sketch_dir) 
                                       if f.lower().endswith(ext)])
                
                # Get all photo files
                photo_files = []
                for ext in ['.jpg', '.jpeg', '.png']:
                    photo_files.extend([f for f in os.listdir(photo_dir) 
                                      if f.lower().endswith(ext)])
                
                print(f"Sketch files: {len(sketch_files)}")
                print(f"Photo files: {len(photo_files)}")
                
                # Match files by name (assuming they have similar names)
                for sketch_file in tqdm(sketch_files, desc="Processing pairs"):
                    # Try to find matching photo file
                    sketch_name = os.path.splitext(sketch_file)[0]
                    
                    # Look for matching photo file
                    matching_photo = None
                    for photo_file in photo_files:
                        photo_name = os.path.splitext(photo_file)[0]
                        if sketch_name in photo_name or photo_name in sketch_name:
                            matching_photo = photo_file
                            break
                    
                    if matching_photo:
                        try:
                            # Load images
                            sketch_path = os.path.join(sketch_dir, sketch_file)
                            photo_path = os.path.join(photo_dir, matching_photo)
                            
                            sketch_img = Image.open(sketch_path).convert('RGB')
                            photo_img = Image.open(photo_path).convert('RGB')
                            
                            # Resize both images to 256x256
                            sketch_img = sketch_img.resize((256, 256), Image.Resampling.LANCZOS)
                            photo_img = photo_img.resize((256, 256), Image.Resampling.LANCZOS)
                            
                            # Create side-by-side image
                            combined_img = Image.new('RGB', (512, 256))
                            combined_img.paste(sketch_img, (0, 0))
                            combined_img.paste(photo_img, (256, 0))
                            
                            # Save combined image
                            output_filename = f"pair_{processed_pairs:06d}.jpg"
                            output_path = os.path.join(output_dir, output_filename)
                            combined_img.save(output_path, 'JPEG', quality=95)
                            
                            processed_pairs += 1
                            
                        except Exception as e:
                            print(f"Error processing {sketch_file}: {e}")
                            continue
                
                total_pairs += len(sketch_files)
        
        print(f"\nTotal pairs processed: {processed_pairs}")
        print(f"Side-by-side images saved to: {output_dir}")
        return output_dir
    
    def split_dataset(self, side_by_side_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split dataset into train/validation/test sets
        """
        print("\n" + "=" * 50)
        print("STEP 4: SPLITTING DATASET")
        print("=" * 50)
        
        # Get all image files
        image_files = [f for f in os.listdir(side_by_side_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Total images: {len(image_files)}")
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split indices
        total = len(image_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split files
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        print(f"Train: {len(train_files)} images")
        print(f"Validation: {len(val_files)} images")
        print(f"Test: {len(test_files)} images")
        
        # Create split directories and copy files
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files_list in splits.items():
            split_dir = os.path.join(self.prepared_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            print(f"\nCopying {split_name} files...")
            for filename in tqdm(files_list, desc=f"Copying {split_name}"):
                src_path = os.path.join(side_by_side_dir, filename)
                dst_path = os.path.join(split_dir, filename)
                shutil.copy2(src_path, dst_path)
        
        # Save split information
        split_info = {
            'total_images': total,
            'train_count': len(train_files),
            'val_count': len(val_files),
            'test_count': len(test_files),
            'train_files': train_files,
            'val_files': val_files,
            'test_files': test_files
        }
        
        with open(os.path.join(self.prepared_dir, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Split information saved to: {os.path.join(self.prepared_dir, 'split_info.json')}")
        return True
    
    def augment_training_data(self, augmentation_factor=5):
        """
        Augment training data with various transformations
        """
        print("\n" + "=" * 50)
        print("STEP 5: AUGMENTING TRAINING DATA")
        print("=" * 50)
        
        train_dir = os.path.join(self.prepared_dir, 'train')
        if not os.path.exists(train_dir):
            print("Training directory not found!")
            return False
        
        # Get original training files
        original_files = [f for f in os.listdir(train_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Original training images: {len(original_files)}")
        print(f"Target augmented images: {len(original_files) * augmentation_factor}")
        
        # Create augmented directory
        augmented_dir = os.path.join(self.prepared_dir, 'train_augmented')
        os.makedirs(augmented_dir, exist_ok=True)
        
        # Copy original files first
        print("Copying original files...")
        for filename in tqdm(original_files, desc="Copying originals"):
            src_path = os.path.join(train_dir, filename)
            dst_path = os.path.join(augmented_dir, filename)
            shutil.copy2(src_path, dst_path)
        
        # Generate augmented versions
        augmentations_per_image = augmentation_factor - 1
        
        for filename in tqdm(original_files, desc="Generating augmentations"):
            image_path = os.path.join(train_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            try:
                # Load image
                img = Image.open(image_path).convert('RGB')
                
                # Split into sketch and photo
                sketch = img.crop((0, 0, 256, 256))
                photo = img.crop((256, 0, 512, 256))
                
                # Generate augmentations
                for i in range(augmentations_per_image):
                    # Apply random augmentation
                    augmented_sketch, augmented_photo = self.apply_augmentation(sketch, photo)
                    
                    # Combine back to side-by-side
                    combined = Image.new('RGB', (512, 256))
                    combined.paste(augmented_sketch, (0, 0))
                    combined.paste(augmented_photo, (256, 0))
                    
                    # Save augmented image
                    aug_filename = f"{base_name}_aug_{i+1:02d}.jpg"
                    aug_path = os.path.join(augmented_dir, aug_filename)
                    combined.save(aug_path, 'JPEG', quality=95)
                    
            except Exception as e:
                print(f"Error augmenting {filename}: {e}")
                continue
        
        # Replace original train directory with augmented one
        shutil.rmtree(train_dir)
        shutil.move(augmented_dir, train_dir)
        
        final_count = len([f for f in os.listdir(train_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"Final training images: {final_count}")
        print(f"Augmentation completed! Training data saved to: {train_dir}")
        return True
    
    def apply_augmentation(self, sketch, photo):
        """
        Apply random augmentation to sketch and photo pair
        """
        # Choose random augmentation type
        aug_type = random.choice(['geometric', 'color', 'noise', 'combined'])
        
        if aug_type == 'geometric':
            # Geometric transformations
            angle = random.uniform(-15, 15)
            sketch = sketch.rotate(angle, fillcolor='white')
            photo = photo.rotate(angle, fillcolor='white')
            
        elif aug_type == 'color':
            # Color transformations
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            
            sketch = ImageEnhance.Brightness(sketch).enhance(brightness_factor)
            sketch = ImageEnhance.Contrast(sketch).enhance(contrast_factor)
            
            photo = ImageEnhance.Brightness(photo).enhance(brightness_factor)
            photo = ImageEnhance.Contrast(photo).enhance(contrast_factor)
            
        elif aug_type == 'noise':
            # Add noise
            sketch_array = np.array(sketch)
            noise = np.random.normal(0, 10, sketch_array.shape).astype(np.uint8)
            sketch_array = np.clip(sketch_array + noise, 0, 255)
            sketch = Image.fromarray(sketch_array)
            
            photo_array = np.array(photo)
            noise = np.random.normal(0, 10, photo_array.shape).astype(np.uint8)
            photo_array = np.clip(photo_array + noise, 0, 255)
            photo = Image.fromarray(photo_array)
            
        else:  # combined
            # Combine multiple augmentations
            angle = random.uniform(-10, 10)
            brightness_factor = random.uniform(0.9, 1.1)
            
            sketch = sketch.rotate(angle, fillcolor='white')
            sketch = ImageEnhance.Brightness(sketch).enhance(brightness_factor)
            
            photo = photo.rotate(angle, fillcolor='white')
            photo = ImageEnhance.Brightness(photo).enhance(brightness_factor)
        
        return sketch, photo
    
    def run_complete_pipeline(self, augmentation_factor=5):
        """
        Run the complete data preparation pipeline
        """
        print("🚀 STARTING COMPLETE DATA PREPARATION PIPELINE")
        print("=" * 60)
        
        # Step 1: Find uploaded dataset
        zip_path = self.find_uploaded_dataset()
        if not zip_path:
            print("❌ Dataset not found. Please upload FS2K.zip manually.")
            return False
        
        # Step 2: Extract
        if not self.extract_dataset(zip_path):
            print("❌ Extraction failed.")
            return False
        
        # Step 3: Prepare side-by-side images
        side_by_side_dir = self.prepare_side_by_side_images()
        if not side_by_side_dir:
            print("❌ Side-by-side preparation failed.")
            return False
        
        # Step 4: Split dataset
        if not self.split_dataset(side_by_side_dir):
            print("❌ Dataset splitting failed.")
            return False
        
        # Step 5: Augment training data
        if not self.augment_training_data(augmentation_factor):
            print("❌ Data augmentation failed.")
            return False
        
        print("\n" + "=" * 60)
        print("✅ DATA PREPARATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Prepared data location: {self.prepared_dir}")
        print(f"📊 Dataset structure:")
        print(f"   - Train: {len(os.listdir(os.path.join(self.prepared_dir, 'train')))} images")
        print(f"   - Validation: {len(os.listdir(os.path.join(self.prepared_dir, 'val')))} images")
        print(f"   - Test: {len(os.listdir(os.path.join(self.prepared_dir, 'test')))} images")
        print("\n🎯 Ready for training!")
        
        return True

# Usage function for Colab
def prepare_fs2k_dataset_in_colab(augmentation_factor=5):
    """
    Convenience function to run the complete pipeline in Colab
    """
    preparator = FS2KDataPreparation()
    return preparator.run_complete_pipeline(augmentation_factor)

if __name__ == "__main__":
    # Example usage
    preparator = FS2KDataPreparation()
    preparator.run_complete_pipeline(augmentation_factor=5) 