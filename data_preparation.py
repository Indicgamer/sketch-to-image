import json
import os
import cv2
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import random

class FS2KDataPreparation:
    def __init__(self, dataset_path, output_path, image_size=(256, 256)):
        """
        Initialize the data preparation class.
        
        Args:
            dataset_path (str): Path to the FS2K dataset
            output_path (str): Path to save the prepared data
            image_size (tuple): Target size for the merged images (width, height)
        """
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.image_size = image_size
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
        
        # Mapping between photo and sketch folders
        self.folder_mapping = {
            'photo1': 'sketch1',
            'photo2': 'sketch2', 
            'photo3': 'sketch3'
        }
    
    def load_annotations(self, annotation_file):
        """Load annotation file and return list of image metadata."""
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        return annotations
    
    def get_image_paths(self, image_name):
        """
        Get the paths for photo and sketch images based on image_name.
        
        Args:
            image_name (str): Image name from annotation (e.g., "photo1/image0110")
            
        Returns:
            tuple: (photo_path, sketch_path) or (None, None) if files don't exist
        """
        # Extract folder and filename
        folder, filename = image_name.split('/')
        
        # Get corresponding sketch folder
        sketch_folder = self.folder_mapping.get(folder)
        if not sketch_folder:
            return None, None
        
        # Construct paths
        photo_path = os.path.join(self.dataset_path, 'photo', folder, f"{filename}.jpg")
        
        # Handle different sketch file extensions
        sketch_path_jpg = os.path.join(self.dataset_path, 'sketch', sketch_folder, f"sketch{filename[5:]}.jpg")
        sketch_path_png = os.path.join(self.dataset_path, 'sketch', sketch_folder, f"sketch{filename[5:]}.png")
        
        # Check which sketch file exists
        if os.path.exists(sketch_path_jpg):
            sketch_path = sketch_path_jpg
        elif os.path.exists(sketch_path_png):
            sketch_path = sketch_path_png
        else:
            return None, None
        
        # Verify both files exist
        if os.path.exists(photo_path) and os.path.exists(sketch_path):
            return photo_path, sketch_path
        else:
            return None, None
    
    def load_and_resize_image(self, image_path):
        """Load and resize image to target size."""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.image_size)
            
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def merge_images_side_by_side(self, sketch_image, photo_image):
        """
        Merge sketch and photo images side by side.
        
        Args:
            sketch_image (numpy.ndarray): Sketch image (RGB)
            photo_image (numpy.ndarray): Photo image (RGB)
            
        Returns:
            numpy.ndarray: Merged image with sketch on left, photo on right
        """
        # Resize both images to half width for side-by-side layout
        half_width = self.image_size[0] // 2
        sketch_resized = cv2.resize(sketch_image, (half_width, self.image_size[1]))
        photo_resized = cv2.resize(photo_image, (half_width, self.image_size[1]))
        
        # Create merged image
        merged_image = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # Place sketch on left half
        merged_image[:, :half_width] = sketch_resized
        # Place photo on right half
        merged_image[:, half_width:] = photo_resized
        
        return merged_image
    
    def prepare_dataset(self, train_ratio=0.8):
        """
        Prepare the dataset by processing annotations and creating merged images.
        
        Args:
            train_ratio (float): Ratio of training data (0.8 = 80% train, 20% validation)
        """
        print("Loading annotations...")
        
        # Load train and test annotations
        train_annotations = self.load_annotations(os.path.join(self.dataset_path, 'anno_train.json'))
        test_annotations = self.load_annotations(os.path.join(self.dataset_path, 'anno_test.json'))
        
        # Combine all annotations
        all_annotations = train_annotations + test_annotations
        
        print(f"Total annotations: {len(all_annotations)}")
        
        # Shuffle annotations
        random.shuffle(all_annotations)
        
        # Split into train and validation
        split_idx = int(len(all_annotations) * train_ratio)
        train_data = all_annotations[:split_idx]
        val_data = all_annotations[split_idx:]
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Process training data
        self.process_annotations(train_data, 'train')
        
        # Process validation data
        self.process_annotations(val_data, 'val')
        
        # Save metadata
        self.save_metadata(train_data, val_data)
    
    def process_annotations(self, annotations, split_name):
        """
        Process annotations and create merged images.
        
        Args:
            annotations (list): List of annotation dictionaries
            split_name (str): 'train' or 'val'
        """
        print(f"\nProcessing {split_name} data...")
        
        valid_pairs = []
        failed_pairs = 0
        
        for i, annotation in enumerate(tqdm(annotations, desc=f"Processing {split_name}")):
            image_name = annotation['image_name']
            
            # Get image paths
            photo_path, sketch_path = self.get_image_paths(image_name)
            
            if photo_path is None or sketch_path is None:
                failed_pairs += 1
                continue
            
            # Load and resize images
            sketch_image = self.load_and_resize_image(sketch_path)
            photo_image = self.load_and_resize_image(photo_path)
            
            if sketch_image is None or photo_image is None:
                failed_pairs += 1
                continue
            
            # Merge images
            merged_image = self.merge_images_side_by_side(sketch_image, photo_image)
            
            # Save merged image
            output_filename = f"{split_name}_{i:06d}.jpg"
            output_path = os.path.join(self.output_path, split_name, output_filename)
            
            # Convert RGB to BGR for OpenCV
            merged_image_bgr = cv2.cvtColor(merged_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, merged_image_bgr)
            
            # Store metadata
            valid_pairs.append({
                'filename': output_filename,
                'original_image_name': image_name,
                'photo_path': photo_path,
                'sketch_path': sketch_path,
                'attributes': annotation
            })
        
        print(f"Successfully processed {len(valid_pairs)} pairs for {split_name}")
        print(f"Failed pairs: {failed_pairs}")
        
        # Save metadata for this split
        metadata_path = os.path.join(self.output_path, f'{split_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(valid_pairs, f, indent=2)
    
    def save_metadata(self, train_data, val_data):
        """Save overall dataset metadata."""
        metadata = {
            'dataset_info': {
                'name': 'FS2K_Pix2Pix',
                'description': 'FS2K dataset prepared for Pix2Pix GAN training',
                'image_size': self.image_size,
                'total_samples': len(train_data) + len(val_data),
                'train_samples': len(train_data),
                'val_samples': len(val_data)
            },
            'attributes': {
                'skin_color': 'RGB values for skin color',
                'lip_color': 'RGB values for lip color', 
                'eye_color': 'RGB values for eye color',
                'hair': 'Hair type (0/1)',
                'hair_color': 'Hair color category (0-4)',
                'gender': 'Gender (0/1)',
                'earring': 'Earring presence (0/1)',
                'smile': 'Smile presence (0/1)',
                'frontal_face': 'Frontal face (0/1)',
                'style': 'Style category (0/1)'
            }
        }
        
        metadata_path = os.path.join(self.output_path, 'dataset_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_sample_visualization(self, num_samples=5):
        """Create a sample visualization of the merged images."""
        print("\nCreating sample visualization...")
        
        # Get sample images from train set
        train_dir = os.path.join(self.output_path, 'train')
        train_files = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
        
        if len(train_files) == 0:
            print("No training images found for visualization")
            return
        
        # Select random samples
        sample_files = random.sample(train_files, min(num_samples, len(train_files)))
        
        # Create visualization
        fig_width = 10
        fig_height = 2 * num_samples
        
        # Load and display samples
        for i, filename in enumerate(sample_files):
            image_path = os.path.join(train_dir, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Split the merged image back into sketch and photo
            sketch = image[:, :self.image_size[0]//2]
            photo = image[:, self.image_size[0]//2:]
            
            # Save individual components for visualization
            sketch_path = os.path.join(self.output_path, f'sample_{i}_sketch.jpg')
            photo_path = os.path.join(self.output_path, f'sample_{i}_photo.jpg')
            merged_path = os.path.join(self.output_path, f'sample_{i}_merged.jpg')
            
            cv2.imwrite(sketch_path, cv2.cvtColor(sketch, cv2.COLOR_RGB2BGR))
            cv2.imwrite(photo_path, cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
            cv2.imwrite(merged_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        print(f"Sample visualization saved in {self.output_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare FS2K dataset for Pix2Pix GAN training')
    parser.add_argument('--dataset_path', type=str, default='FS2K/FS2K',
                        help='Path to the FS2K dataset')
    parser.add_argument('--output_path', type=str, default='prepared_data',
                        help='Path to save the prepared data')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 256],
                        help='Size of merged images (width height)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of training data (0.8 = 80% train, 20% validation)')
    parser.add_argument('--create_samples', action='store_true',
                        help='Create sample visualization')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Initialize data preparation
    data_prep = FS2KDataPreparation(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        image_size=tuple(args.image_size)
    )
    
    # Prepare dataset
    data_prep.prepare_dataset(train_ratio=args.train_ratio)
    
    # Create sample visualization if requested
    if args.create_samples:
        data_prep.create_sample_visualization()
    
    print(f"\nDataset preparation completed!")
    print(f"Output directory: {args.output_path}")
    print(f"Image size: {args.image_size}")
    print(f"Training ratio: {args.train_ratio}")

if __name__ == "__main__":
    main() 