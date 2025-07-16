import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Important: Make sure model.py is in the same directory as this script
# so we can import the UNetGenerator class.
from model import UNetGenerator

# --- 1. Configuration ---
# Path to your trained model checkpoint
MODEL_PATH = 'output_data/checkpoints/final_model.pth' 
# Path to your input sketch image
INPUT_IMAGE_PATH = 'C:\\Users\\Sachidanand\\Downloads\\aaf7cff2798325d5893623929f445308.jpg'  # <--- CHANGE THIS to your image file
# Path to save the output image
OUTPUT_IMAGE_PATH = 'output_data/generated_photo.png'
# The number of filters the generator was trained with (e.g., 16 for the tiny model, 32 for the next size up)
NUM_FILTERS = 16 # <--- IMPORTANT: MATCH THIS to the model you trained
# Image size the model was trained on
IMAGE_SIZE = 256

# --- 2. Function to Load the Model ---
def load_model(model_path, num_filters, device):
    """
    Loads the generator model from a checkpoint.
    """
    # Initialize the model structure
    # The model must have the same architecture as the one saved in the checkpoint
    model = UNetGenerator(input_channels=3, output_channels=3, num_filters=num_filters)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load the state dictionary into the model
    # The checkpoint saves the entire trainer state, so we need to extract the generator's state
    model.load_state_dict(checkpoint['generator_state_dict'])
    
    # Set the model to evaluation mode
    # This disables layers like Dropout and BatchNorm's training behavior
    model.eval()
    
    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    
    print(f"Model loaded from {model_path} and set to evaluation mode.")
    return model

# --- 3. Function to Preprocess the Input Image ---
def preprocess_image(image_path, image_size):
    """
    Loads and preprocesses an image for the model.
    """
    # Open the image
    image = Image.open(image_path).convert('RGB')
    
    # Define the transformations
    # These should be the same as the validation/test transforms used during training
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Apply the transform and add a batch dimension (B, C, H, W)
    return transform(image).unsqueeze(0)

# --- 4. Function to Generate and Save the Output ---
def generate_image(model, input_tensor, output_path, device):
    """
    Generates an image from the input tensor and saves it.
    """
    # Move the input tensor to the same device as the model
    input_tensor = input_tensor.to(device)
    
    # Generate the output
    with torch.no_grad(): # We don't need to calculate gradients for inference
        output_tensor = model(input_tensor)
        
    # Denormalize the output tensor to bring it from [-1, 1] to [0, 1] range
    output_tensor = (output_tensor + 1) / 2.0
    
    # Remove the batch dimension and convert to a PIL image
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())
    
    # Save the image
    output_image.save(output_path)
    print(f"Generated image saved to {output_path}")
    return output_image

# --- 5. Main Execution Block ---
if __name__ == '__main__':
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check if model and input image exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
    elif not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: Input image not found at {INPUT_IMAGE_PATH}")
    else:
        # Load the model
        generator = load_model(MODEL_PATH, NUM_FILTERS, device)
        
        # Preprocess the input image
        input_tensor = preprocess_image(INPUT_IMAGE_PATH, IMAGE_SIZE)
        
        # Generate and save the output image
        generated_image = generate_image(generator, input_tensor, OUTPUT_IMAGE_PATH, device)
        
        # Display the input and output images side-by-side
        try:
            input_display = Image.open(INPUT_IMAGE_PATH).resize((IMAGE_SIZE, IMAGE_SIZE))
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(input_display)
            ax[0].set_title('Input Sketch')
            ax[0].axis('off')
            
            ax[1].imshow(generated_image)
            ax[1].set_title('Generated Photo')
            ax[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not display images. Make sure you have a graphical backend. Error: {e}")