"""
Single File Data Preparation for FS2K Dataset in Colab
Complete pipeline: Upload → Extract → Prepare → Split → Augment
Run this entire file in one Colab cell
"""

import os
import zipfile
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from tqdm import tqdm
import json
import requests
from io import BytesIO

def main():
    """
    Main function to run the complete data preparation pipeline
    """
    print("🚀 STARTING COMPLETE FS2K DATA PREPARATION")
    print("=" * 60)
    
    # Configuration
    base_dir = "/content"
    fs2k_dir = os.path.join(base_dir, "FS2K")
    prepared_dir = os.path.join(base_dir, "prepared_data")
    
    # Create directories
    os.makedirs(fs2k_dir, exist_ok=True)
    os.makedirs(prepared_dir, exist_ok=True)
    
    # STEP 1: LOCATE UPLOADED DATASET
    print("\n" + "=" * 50)
    print("STEP 1: LOCATING UPLOADED DATASET")
    print("=" * 50)
    
    # Check common locations for the uploaded zip file
    possible_locations = [
        os.path.join(base_dir, "FS2K.zip"),
        "/content/FS2K.zip",
        "FS2K.zip",
        os.path.join(base_dir, "FS2K", "FS2K.zip")
    ]
    
    zip_path = None
    for location in possible_locations:
        if os.path.exists(location):
            zip_path = location
            print(f"✅ Found FS2K.zip at: {location}")
            break
    
    # If not found in common locations, check current directory
    if not zip_path:
        current_files = os.listdir('.')
        zip_files = [f for f in current_files if f.lower().endswith('.zip')]
        
        if zip_files:
            zip_path = zip_files[0]
            print(f"✅ Found zip file: {zip_path}")
        else:
            print("❌ FS2K.zip not found!")
            print("Please make sure FS2K.zip is uploaded to the content folder")
            return False
    
    # STEP 2: EXTRACT DATASET
    print("\n" + "=" * 50)
    print("STEP 2: EXTRACTING DATASET")
    print("=" * 50)
    
    print(f"Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        print("✅ Extraction completed!")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False
    
    # Check extracted structure
    if os.path.exists(fs2k_dir):
        print(f"📁 Extracted to: {fs2k_dir}")
        print("📂 Contents:")
        for item in os.listdir(fs2k_dir):
            item_path = os.path.join(fs2k_dir, item)
            if os.path.isdir(item_path):
                file_count = len([f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"   📁 {item}/ ({file_count} images)")
    else:
        print("❌ FS2K directory not found after extraction")
        return False
    
    # STEP 3: PREPARE SIDE-BY-SIDE IMAGES
    print("\n" + "=" * 50)
    print("STEP 3: PREPARING SIDE-BY-SIDE IMAGES")
    print("=" * 50)
    
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
    
    # Create output directory
    output_dir = os.path.join(prepared_dir, "side_by_side")
    os.makedirs(output_dir, exist_ok=True)
    
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
            
            # Match files by name
            for sketch_file in tqdm(sketch_files, desc="Processing pairs"):
                sketch_name = os.path.splitext(sketch_file)[0]
                
                # Find matching photo file
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
    
    print(f"\n✅ Created {processed_pairs} side-by-side images")
    
    # STEP 4: SPLIT DATASET
    print("\n" + "=" * 50)
    print("STEP 4: SPLITTING DATASET")
    print("=" * 50)
    
    # Get all image files
    image_files = [f for f in os.listdir(output_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Total images: {len(image_files)}")
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split indices (70% train, 15% val, 15% test)
    total = len(image_files)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)
    
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
        split_dir = os.path.join(prepared_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"\nCopying {split_name} files...")
        for filename in tqdm(files_list, desc=f"Copying {split_name}"):
            src_path = os.path.join(output_dir, filename)
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
    
    with open(os.path.join(prepared_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✅ Split information saved to: {os.path.join(prepared_dir, 'split_info.json')}")
    
    # STEP 5: AUGMENT TRAINING DATA
    print("\n" + "=" * 50)
    print("STEP 5: AUGMENTING TRAINING DATA")
    print("=" * 50)
    
    augmentation_factor = 5  # Increase training data 5x
    
    train_dir = os.path.join(prepared_dir, 'train')
    original_files = [f for f in os.listdir(train_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Original training images: {len(original_files)}")
    print(f"Target augmented images: {len(original_files) * augmentation_factor}")
    
    # Create augmented directory
    augmented_dir = os.path.join(prepared_dir, 'train_augmented')
    os.makedirs(augmented_dir, exist_ok=True)
    
    # Copy original files first
    print("Copying original files...")
    for filename in tqdm(original_files, desc="Copying originals"):
        src_path = os.path.join(train_dir, filename)
        dst_path = os.path.join(augmented_dir, filename)
        shutil.copy2(src_path, dst_path)
    
    # Generate augmented versions
    augmentations_per_image = augmentation_factor - 1
    
    def apply_augmentation(sketch, photo):
        """Apply random augmentation to sketch and photo pair"""
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
                augmented_sketch, augmented_photo = apply_augmentation(sketch, photo)
                
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
    
    print(f"✅ Final training images: {final_count}")
    
    # FINAL SUMMARY
    print("\n" + "=" * 60)
    print("✅ DATA PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📁 Prepared data location: {prepared_dir}")
    print(f"📊 Dataset structure:")
    
    train_count = len([f for f in os.listdir(os.path.join(prepared_dir, 'train')) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    val_count = len([f for f in os.listdir(os.path.join(prepared_dir, 'val')) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    test_count = len([f for f in os.listdir(os.path.join(prepared_dir, 'test')) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"   - Train: {train_count} images (augmented {augmentation_factor}x)")
    print(f"   - Validation: {val_count} images")
    print(f"   - Test: {test_count} images")
    print(f"   - Total: {train_count + val_count + test_count} images")
    
    print("\n🎯 Ready for training!")
    print("📂 You can now use '/content/prepared_data' as your dataset directory")
    
    return True

if __name__ == "__main__":
    # Run the complete pipeline
    success = main()
    
    if success:
        print("\n🎉 All done! Your dataset is ready for training.")
    else:
        print("\n❌ Data preparation failed. Please check the error messages above.") 