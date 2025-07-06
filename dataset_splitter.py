import os
import json
import shutil
import random
from tqdm import tqdm
import argparse

class DatasetSplitter:
    def __init__(self, input_path, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Initialize the dataset splitter.
        
        Args:
            input_path (str): Path to the prepared dataset
            output_path (str): Path to save the split dataset
            train_ratio (float): Ratio for training data
            val_ratio (float): Ratio for validation data
            test_ratio (float): Ratio for test data
        """
        self.input_path = input_path
        self.output_path = output_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)
    
    def load_metadata(self, metadata_file):
        """Load metadata from JSON file."""
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def save_metadata(self, data, output_file):
        """Save metadata to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_all_images(self):
        """Get all image files from train and val directories."""
        all_images = []
        
        # Get train images
        train_dir = os.path.join(self.input_path, 'train')
        if os.path.exists(train_dir):
            train_files = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
            all_images.extend([('train', f) for f in train_files])
        
        # Get val images
        val_dir = os.path.join(self.input_path, 'val')
        if os.path.exists(val_dir):
            val_files = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]
            all_images.extend([('val', f) for f in val_files])
        
        return all_images
    
    def split_dataset(self):
        """Split the dataset into train, validation, and test sets."""
        print("Loading existing dataset...")
        
        # Get all images
        all_images = self.get_all_images()
        print(f"Total images found: {len(all_images)}")
        
        # Shuffle images
        random.shuffle(all_images)
        
        # Calculate split indices
        total_images = len(all_images)
        train_end = int(total_images * self.train_ratio)
        val_end = train_end + int(total_images * self.val_ratio)
        
        # Split images
        train_images = all_images[:train_end]
        val_images = all_images[train_end:val_end]
        test_images = all_images[val_end:]
        
        print(f"Train images: {len(train_images)} ({len(train_images)/total_images*100:.1f}%)")
        print(f"Validation images: {len(val_images)} ({len(val_images)/total_images*100:.1f}%)")
        print(f"Test images: {len(test_images)} ({len(test_images)/total_images*100:.1f}%)")
        
        # Process splits
        train_metadata = self.process_split(train_images, 'train')
        val_metadata = self.process_split(val_images, 'val')
        test_metadata = self.process_split(test_images, 'test')
        
        # Save overall metadata
        self.save_overall_metadata(train_metadata, val_metadata, test_metadata)
        
        print(f"\nDataset split completed!")
        print(f"Output directory: {self.output_path}")
    
    def process_split(self, images, split_name):
        """Process a split of images and copy them to the output directory."""
        print(f"\nProcessing {split_name} split...")
        
        metadata = []
        
        for i, (source_dir, filename) in enumerate(tqdm(images, desc=f"Processing {split_name}")):
            # Source and destination paths
            source_path = os.path.join(self.input_path, source_dir, filename)
            dest_filename = f"{split_name}_{i:06d}.jpg"
            dest_path = os.path.join(self.output_path, split_name, dest_filename)
            
            # Copy image
            shutil.copy2(source_path, dest_path)
            
            # Load original metadata if available
            original_metadata = self.get_original_metadata(source_dir, filename)
            
            # Create new metadata entry
            metadata_entry = {
                'filename': dest_filename,
                'original_source': source_dir,
                'original_filename': filename,
                'split': split_name,
                'index': i
            }
            
            # Add original metadata if available
            if original_metadata:
                metadata_entry['original_metadata'] = original_metadata
            
            metadata.append(metadata_entry)
        
        # Save split metadata
        metadata_file = os.path.join(self.output_path, f'{split_name}_metadata.json')
        self.save_metadata(metadata, metadata_file)
        
        return metadata
    
    def get_original_metadata(self, source_dir, filename):
        """Get original metadata for an image."""
        try:
            # Load original metadata
            metadata_file = os.path.join(self.input_path, f'{source_dir}_metadata.json')
            if os.path.exists(metadata_file):
                metadata = self.load_metadata(metadata_file)
                
                # Find matching entry
                for entry in metadata:
                    if entry.get('filename') == filename:
                        return entry
        except Exception as e:
            print(f"Warning: Could not load metadata for {filename}: {e}")
        
        return None
    
    def save_overall_metadata(self, train_metadata, val_metadata, test_metadata):
        """Save overall dataset metadata."""
        overall_metadata = {
            'dataset_info': {
                'name': 'FS2K_Pix2Pix_Split',
                'description': 'FS2K dataset split into train/val/test for Pix2Pix GAN training',
                'total_samples': len(train_metadata) + len(val_metadata) + len(test_metadata),
                'train_samples': len(train_metadata),
                'val_samples': len(val_metadata),
                'test_samples': len(test_metadata),
                'train_ratio': self.train_ratio,
                'val_ratio': self.val_ratio,
                'test_ratio': self.test_ratio
            },
            'splits': {
                'train': {
                    'count': len(train_metadata),
                    'metadata_file': 'train_metadata.json'
                },
                'val': {
                    'count': len(val_metadata),
                    'metadata_file': 'val_metadata.json'
                },
                'test': {
                    'count': len(test_metadata),
                    'metadata_file': 'test_metadata.json'
                }
            },
            'usage': {
                'train': 'Use for training the Pix2Pix GAN model',
                'val': 'Use for validation during training (early stopping, hyperparameter tuning)',
                'test': 'Use for final evaluation of model performance'
            }
        }
        
        metadata_file = os.path.join(self.output_path, 'dataset_metadata.json')
        self.save_metadata(overall_metadata, metadata_file)
    
    def create_sample_visualization(self, num_samples=3):
        """Create sample visualization showing images from each split."""
        print("\nCreating sample visualization...")
        
        splits = ['train', 'val', 'test']
        
        for split in splits:
            split_dir = os.path.join(self.output_path, split)
            if not os.path.exists(split_dir):
                continue
            
            # Get sample images
            files = [f for f in os.listdir(split_dir) if f.endswith('.jpg')]
            if len(files) == 0:
                continue
            
            # Select random samples
            sample_files = random.sample(files, min(num_samples, len(files)))
            
            print(f"{split.upper()} split samples:")
            for i, filename in enumerate(sample_files):
                print(f"  {i+1}. {filename}")
            print()

def main():
    parser = argparse.ArgumentParser(description='Split prepared FS2K dataset into train/val/test sets')
    parser.add_argument('--input_path', type=str, default='prepared_data',
                        help='Path to the prepared dataset')
    parser.add_argument('--output_path', type=str, default='split_data',
                        help='Path to save the split dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio for training data (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio for validation data (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio for test data (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--create_samples', action='store_true',
                        help='Create sample visualization')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Initialize splitter
    splitter = DatasetSplitter(
        input_path=args.input_path,
        output_path=args.output_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Split dataset
    splitter.split_dataset()
    
    # Create sample visualization if requested
    if args.create_samples:
        splitter.create_sample_visualization()
    
    print(f"\nDataset splitting completed successfully!")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")

if __name__ == "__main__":
    main() 