import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import argparse
from model import UNetGenerator, get_transforms

class SketchToImageInference:
    """
    Inference class for sketch-to-image generation
    """
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create only the generator (no discriminator needed for inference)
        self.generator = UNetGenerator(input_channels=3, output_channels=3, 
                                     num_filters=64).to(self.device)
        
        # Load trained model
        self.load_model(model_path)
        
        # Use the exact same transforms as training
        self.transform = get_transforms(256)
    
    def load_model(self, model_path):
        """Load trained generator model"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'generator_state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
            else:
                # If it's just the generator state dict
                self.generator.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def preprocess_sketch(self, sketch_path):
        """Preprocess sketch image"""
        # Load and convert to RGB
        sketch = Image.open(sketch_path).convert('RGB')
        
        # Apply transforms (same as training)
        sketch_tensor = self.transform(sketch).unsqueeze(0).to(self.device)
        
        return sketch_tensor
    
    def generate_image(self, sketch_tensor):
        """Generate image from sketch tensor"""
        self.generator.eval()
        with torch.no_grad():
            generated = self.generator(sketch_tensor)
            # Denormalize (same as training)
            generated = (generated + 1) / 2
            generated = torch.clamp(generated, 0, 1)
        
        return generated
    
    def tensor_to_image(self, tensor):
        """Convert tensor to PIL image"""
        # Convert to numpy and transpose
        image = tensor.squeeze(0).cpu().numpy()
        image = (image * 255).astype('uint8')
        image = image.transpose(1, 2, 0)
        
        return Image.fromarray(image)
    
    def create_comparison_image(self, sketch_path, generated_image, output_path):
        """Create a side-by-side comparison of sketch and generated image"""
        # Load original sketch
        sketch = Image.open(sketch_path).convert('RGB')
        sketch = sketch.resize((256, 256))
        
        # Create comparison image
        comparison = Image.new('RGB', (512, 256))
        comparison.paste(sketch, (0, 0))
        comparison.paste(generated_image, (256, 0))
        
        # Save comparison
        comparison.save(output_path)
        print(f"Comparison image saved to: {output_path}")
    
    def process_single_image(self, sketch_path, output_path=None, save_comparison=True):
        """Process a single sketch image"""
        # Default to results folder if not specified
        if output_path is None:
            # Create results directory
            os.makedirs("results", exist_ok=True)
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(sketch_path))[0]
            output_path = os.path.join("results", f"generated_{base_name}.jpg")
        
        # Preprocess sketch
        sketch_tensor = self.preprocess_sketch(sketch_path)
        
        # Generate image
        generated_tensor = self.generate_image(sketch_tensor)
        
        # Convert to image and save
        generated_image = self.tensor_to_image(generated_tensor)
        generated_image.save(output_path)
        
        print(f"Generated image saved to: {output_path}")
        
        # Create comparison image if requested
        if save_comparison:
            comparison_path = output_path.replace('.jpg', '_comparison.jpg')
            self.create_comparison_image(sketch_path, generated_image, comparison_path)
    
    def process_batch(self, input_dir, output_dir=None, save_comparison=True):
        """Process all images in a directory"""
        # Default to results folder if not specified
        if output_dir is None:
            output_dir = "results"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
        
        print(f"Found {len(image_files)} images to process")
        print(f"Output will be saved to: {os.path.abspath(output_dir)}")
        
        for i, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"generated_{filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                self.process_single_image(input_path, output_path, save_comparison)
                print(f"Processed {i+1}/{len(image_files)}: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        print(f"\nAll generated images saved to: {os.path.abspath(output_dir)}")
        if save_comparison:
            print("Comparison images (sketch + generated) also saved with '_comparison' suffix")

def main():
    parser = argparse.ArgumentParser(description='Sketch-to-Image Inference')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained generator model')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input sketch image or directory')
    parser.add_argument('--output', type=str, default='results',
                       help='Path to output directory (default: results)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no_comparison', action='store_true',
                       help='Skip creating comparison images')
    
    args = parser.parse_args()
    
    # Create inference object
    inference = SketchToImageInference(args.model_path, args.device)
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        if args.output == 'results':
            # Use default results folder
            inference.process_single_image(args.input, save_comparison=not args.no_comparison)
        else:
            # Use specified output path
            inference.process_single_image(args.input, args.output, save_comparison=not args.no_comparison)
    elif os.path.isdir(args.input):
        # Directory of images
        inference.process_batch(args.input, args.output, save_comparison=not args.no_comparison)
    else:
        print(f"Input path does not exist: {args.input}")
    
    print(f"\nInference completed! Check the '{args.output}' folder for results.")

if __name__ == "__main__":
    main() 