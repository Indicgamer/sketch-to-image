#!/usr/bin/env python3
"""
Example script showing how to use the SketchToImageInference class programmatically.
This demonstrates how to integrate the inference functionality into your own Python code.
"""

import os
from inference import SketchToImageInference

def example_single_image():
    """Example: Process a single sketch image"""
    print("=== Single Image Processing Example ===")
    
    # Initialize the inference class
    model_path = "output_data/checkpoints/best_model.pth"  # Update this path
    inference = SketchToImageInference(model_path, device='cuda')
    
    # Process a single image
    sketch_path = "sketches/my_sketch.jpg"  # Update this path
    output_path = "results/generated_my_sketch.jpg"  # Optional: specify custom output
    
    if os.path.exists(sketch_path):
        inference.process_single_image(sketch_path, output_path, save_comparison=True)
        print(f"✅ Generated image saved to: {output_path}")
    else:
        print(f"❌ Sketch file not found: {sketch_path}")

def example_batch_processing():
    """Example: Process multiple sketch images"""
    print("\n=== Batch Processing Example ===")
    
    # Initialize the inference class
    model_path = "output_data/checkpoints/best_model.pth"  # Update this path
    inference = SketchToImageInference(model_path, device='cuda')
    
    # Process all images in a directory
    input_dir = "sketches/"  # Update this path
    output_dir = "results/batch_output/"  # Optional: specify custom output directory
    
    if os.path.exists(input_dir):
        inference.process_batch(input_dir, output_dir, save_comparison=True)
        print(f"✅ Batch processing completed. Check: {output_dir}")
    else:
        print(f"❌ Input directory not found: {input_dir}")

def example_custom_processing():
    """Example: Custom processing with more control"""
    print("\n=== Custom Processing Example ===")
    
    # Initialize the inference class
    model_path = "output_data/checkpoints/best_model.pth"  # Update this path
    inference = SketchToImageInference(model_path, device='cpu')  # Using CPU
    
    # Custom processing steps
    sketch_path = "sketches/custom_sketch.png"  # Update this path
    
    if os.path.exists(sketch_path):
        # Step 1: Preprocess the sketch
        sketch_tensor = inference.preprocess_sketch(sketch_path)
        print(f"✅ Preprocessed sketch tensor shape: {sketch_tensor.shape}")
        
        # Step 2: Generate the image
        generated_tensor = inference.generate_image(sketch_tensor)
        print(f"✅ Generated image tensor shape: {generated_tensor.shape}")
        
        # Step 3: Convert to PIL image
        generated_image = inference.tensor_to_image(generated_tensor)
        print(f"✅ Converted to PIL image size: {generated_image.size}")
        
        # Step 4: Save the result
        output_path = "results/custom_generated.jpg"
        generated_image.save(output_path)
        print(f"✅ Custom generated image saved to: {output_path}")
        
        # Step 5: Create comparison (optional)
        comparison_path = "results/custom_comparison.jpg"
        inference.create_comparison_image(sketch_path, generated_image, comparison_path)
        print(f"✅ Comparison image saved to: {comparison_path}")
    else:
        print(f"❌ Sketch file not found: {sketch_path}")

def example_error_handling():
    """Example: Error handling and validation"""
    print("\n=== Error Handling Example ===")
    
    try:
        # Try to load a non-existent model
        model_path = "non_existent_model.pth"
        inference = SketchToImageInference(model_path, device='cuda')
    except FileNotFoundError as e:
        print(f"❌ Expected error caught: {e}")
        print("✅ Error handling works correctly!")
    
    try:
        # Try to process a non-existent image
        model_path = "output_data/checkpoints/best_model.pth"  # Update this path
        if os.path.exists(model_path):
            inference = SketchToImageInference(model_path, device='cuda')
            inference.process_single_image("non_existent_sketch.jpg")
        else:
            print("⚠️  Model file not found, skipping error handling test")
    except Exception as e:
        print(f"❌ Error processing non-existent image: {e}")
        print("✅ Error handling works correctly!")

def main():
    """Main function to run all examples"""
    print("🎨 Sketch-to-Image Inference Examples")
    print("=" * 50)
    
    # Check if model file exists
    model_path = "output_data/checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        print("Please update the model_path in the examples to point to your trained model.")
        print("Common locations:")
        print("  - output_data/checkpoints/best_model.pth")
        print("  - output_data/checkpoints/final_model.pth")
        print("  - checkpoints/best_model.pth")
        return
    
    # Run examples
    example_single_image()
    example_batch_processing()
    example_custom_processing()
    example_error_handling()
    
    print("\n" + "=" * 50)
    print("🎉 All examples completed!")
    print("\nTo use these examples:")
    print("1. Update the file paths to match your setup")
    print("2. Ensure you have a trained model checkpoint")
    print("3. Prepare your sketch images")
    print("4. Run: python example_inference.py")

if __name__ == "__main__":
    main() 