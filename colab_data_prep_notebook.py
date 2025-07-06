"""
Colab Data Preparation - Simple Usage
Copy and paste these cells into your Colab notebook
"""

# Cell 1: Install and Import Dependencies
"""
!pip install pillow tqdm requests
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
"""

# Cell 2: Upload FS2K Dataset
"""
print("Please upload your FS2K.zip file:")
uploaded = files.upload()

# Move uploaded file to content directory
if 'FS2K.zip' in uploaded:
    !mv FS2K.zip /content/
    print("✅ FS2K.zip uploaded successfully!")
else:
    print("❌ FS2K.zip not found in uploaded files")
"""

# Cell 3: Extract Dataset
"""
print("📦 Extracting FS2K dataset...")
!unzip -q /content/FS2K.zip -d /content/
print("✅ Extraction completed!")
!ls -la /content/FS2K/
"""

# Cell 4: Prepare Side-by-Side Images
"""
import os
from PIL import Image
from tqdm import tqdm

def prepare_side_by_side_images():
    base_dir = "/content"
    fs2k_dir = os.path.join(base_dir, "FS2K")
    prepared_dir = os.path.join(base_dir, "prepared_data")
    
    os.makedirs(prepared_dir, exist_ok=True)
    output_dir = os.path.join(prepared_dir, "side_by_side")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find sketch and photo directories
    sketch_dirs = []
    photo_dirs = []
    
    for item in os.listdir(fs2k_dir):
        item_path = os.path.join(fs2k_dir, item)
        if os.path.isdir(item_path):
            if 'sketch' in item.lower():
                sketch_dirs.append(item_path)
            elif 'photo' in item.lower():
                photo_dirs.append(item_path)
    
    print(f"Found {len(sketch_dirs)} sketch directories")
    print(f"Found {len(photo_dirs)} photo directories")
    
    processed_pairs = 0
    
    for sketch_dir in sketch_dirs:
        for photo_dir in photo_dirs:
            print(f"Processing: {os.path.basename(sketch_dir)} ↔ {os.path.basename(photo_dir)}")
            
            sketch_files = [f for f in os.listdir(sketch_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            photo_files = [f for f in os.listdir(photo_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for sketch_file in tqdm(sketch_files, desc="Processing pairs"):
                sketch_name = os.path.splitext(sketch_file)[0]
                
                # Find matching photo
                matching_photo = None
                for photo_file in photo_files:
                    photo_name = os.path.splitext(photo_file)[0]
                    if sketch_name in photo_name or photo_name in sketch_name:
                        matching_photo = photo_file
                        break
                
                if matching_photo:
                    try:
                        sketch_path = os.path.join(sketch_dir, sketch_file)
                        photo_path = os.path.join(photo_dir, matching_photo)
                        
                        sketch_img = Image.open(sketch_path).convert('RGB')
                        photo_img = Image.open(photo_path).convert('RGB')
                        
                        # Resize to 256x256
                        sketch_img = sketch_img.resize((256, 256), Image.Resampling.LANCZOS)
                        photo_img = photo_img.resize((256, 256), Image.Resampling.LANCZOS)
                        
                        # Create side-by-side
                        combined_img = Image.new('RGB', (512, 256))
                        combined_img.paste(sketch_img, (0, 0))
                        combined_img.paste(photo_img, (256, 0))
                        
                        output_filename = f"pair_{processed_pairs:06d}.jpg"
                        output_path = os.path.join(output_dir, output_filename)
                        combined_img.save(output_path, 'JPEG', quality=95)
                        
                        processed_pairs += 1
                        
                    except Exception as e:
                        continue
    
    print(f"✅ Created {processed_pairs} side-by-side images")
    return output_dir

side_by_side_dir = prepare_side_by_side_images()
"""

# Cell 5: Split Dataset
"""
def split_dataset(side_by_side_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    prepared_dir = "/content/prepared_data"
    
    image_files = [f for f in os.listdir(side_by_side_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Total images: {len(image_files)}")
    
    random.shuffle(image_files)
    
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print(f"Train: {len(train_files)} images")
    print(f"Validation: {len(val_files)} images")
    print(f"Test: {len(test_files)} images")
    
    splits = {'train': train_files, 'val': val_files, 'test': test_files}
    
    for split_name, files_list in splits.items():
        split_dir = os.path.join(prepared_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"Copying {split_name} files...")
        for filename in tqdm(files_list, desc=f"Copying {split_name}"):
            src_path = os.path.join(side_by_side_dir, filename)
            dst_path = os.path.join(split_dir, filename)
            shutil.copy2(src_path, dst_path)
    
    print("✅ Dataset split completed!")

split_dataset(side_by_side_dir)
"""

# Cell 6: Augment Training Data
"""
def augment_training_data(augmentation_factor=5):
    prepared_dir = "/content/prepared_data"
    train_dir = os.path.join(prepared_dir, 'train')
    
    original_files = [f for f in os.listdir(train_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Original training images: {len(original_files)}")
    print(f"Target augmented images: {len(original_files) * augmentation_factor}")
    
    augmented_dir = os.path.join(prepared_dir, 'train_augmented')
    os.makedirs(augmented_dir, exist_ok=True)
    
    # Copy original files
    print("Copying original files...")
    for filename in tqdm(original_files, desc="Copying originals"):
        src_path = os.path.join(train_dir, filename)
        dst_path = os.path.join(augmented_dir, filename)
        shutil.copy2(src_path, dst_path)
    
    # Generate augmentations
    augmentations_per_image = augmentation_factor - 1
    
    def apply_augmentation(sketch, photo):
        aug_type = random.choice(['geometric', 'color', 'noise', 'combined'])
        
        if aug_type == 'geometric':
            angle = random.uniform(-15, 15)
            sketch = sketch.rotate(angle, fillcolor='white')
            photo = photo.rotate(angle, fillcolor='white')
            
        elif aug_type == 'color':
            brightness_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            
            sketch = ImageEnhance.Brightness(sketch).enhance(brightness_factor)
            sketch = ImageEnhance.Contrast(sketch).enhance(contrast_factor)
            
            photo = ImageEnhance.Brightness(photo).enhance(brightness_factor)
            photo = ImageEnhance.Contrast(photo).enhance(contrast_factor)
            
        elif aug_type == 'noise':
            sketch_array = np.array(sketch)
            noise = np.random.normal(0, 10, sketch_array.shape).astype(np.uint8)
            sketch_array = np.clip(sketch_array + noise, 0, 255)
            sketch = Image.fromarray(sketch_array)
            
            photo_array = np.array(photo)
            noise = np.random.normal(0, 10, photo_array.shape).astype(np.uint8)
            photo_array = np.clip(photo_array + noise, 0, 255)
            photo = Image.fromarray(photo_array)
            
        else:  # combined
            angle = random.uniform(-10, 10)
            brightness_factor = random.uniform(0.9, 1.1)
            
            sketch = sketch.rotate(angle, fillcolor='white')
            sketch = ImageEnhance.Brightness(sketch).enhance(brightness_factor)
            
            photo = photo.rotate(angle, fillcolor='white')
            photo = ImageEnhance.Brightness(photo).enhance(brightness_factor)
        
        return sketch, photo
    
    for filename in tqdm(original_files, desc="Generating augmentations"):
        image_path = os.path.join(train_dir, filename)
        base_name = os.path.splitext(filename)[0]
        
        try:
            img = Image.open(image_path).convert('RGB')
            sketch = img.crop((0, 0, 256, 256))
            photo = img.crop((256, 0, 512, 256))
            
            for i in range(augmentations_per_image):
                augmented_sketch, augmented_photo = apply_augmentation(sketch, photo)
                
                combined = Image.new('RGB', (512, 256))
                combined.paste(augmented_sketch, (0, 0))
                combined.paste(augmented_photo, (256, 0))
                
                aug_filename = f"{base_name}_aug_{i+1:02d}.jpg"
                aug_path = os.path.join(augmented_dir, aug_filename)
                combined.save(aug_path, 'JPEG', quality=95)
                
        except Exception as e:
            continue
    
    # Replace original train directory
    shutil.rmtree(train_dir)
    shutil.move(augmented_dir, train_dir)
    
    final_count = len([f for f in os.listdir(train_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"✅ Final training images: {final_count}")

augment_training_data(augmentation_factor=5)
"""

# Cell 7: Final Summary
"""
print("🎉 DATA PREPARATION COMPLETED!")
print("=" * 50)
print("📁 Dataset structure:")
!ls -la /content/prepared_data/
print("\n📊 Dataset counts:")
!echo "Train: $(ls /content/prepared_data/train/ | wc -l) images"
!echo "Validation: $(ls /content/prepared_data/val/ | wc -l) images"
!echo "Test: $(ls /content/prepared_data/test/ | wc -l) images"
print("\n🎯 Ready for training!")
""" 