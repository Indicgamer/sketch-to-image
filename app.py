import os
import io
import base64
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import torch
from model import create_models, create_small_models
from torchvision import transforms
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class SketchToImageAPI:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Auto-detect model architecture
        use_small_model = self.detect_model_architecture(model_path)
        print(f"Auto-detected model architecture: {'Small' if use_small_model else 'Full'}")
        
        # Store model type for later reference
        self._is_small_model = use_small_model
        
        # Create only the generator using the appropriate model size
        if use_small_model:
            self.generator, _ = create_small_models(self.device)
        else:
            self.generator, _ = create_models(self.device)
        
        # Load trained model
        self.load_model(model_path)
        
        # Use the exact same transforms as training
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def detect_model_architecture(self, model_path):
        """Auto-detect whether the model uses small or full architecture"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if 'generator_state_dict' in checkpoint:
                state_dict = checkpoint['generator_state_dict']
            else:
                state_dict = checkpoint
            
            first_layer_weight = state_dict['down1.0.weight']
            num_filters = first_layer_weight.shape[0]
            
            if num_filters == 32:
                return True  # Small model
            elif num_filters == 64:
                return False  # Full model
            else:
                print(f"Warning: Unknown architecture with {num_filters} filters")
                return False
                
        except Exception as e:
            print(f"Warning: Could not auto-detect model architecture: {e}")
            return False
    
    def load_model(self, model_path):
        """Load trained generator model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'generator_state_dict' in checkpoint:
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
            else:
                self.generator.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    def preprocess_sketch(self, image):
        """Preprocess sketch image"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif hasattr(image, 'convert'):
            image = image.convert('RGB')
        
        sketch_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return sketch_tensor
    
    def generate_image(self, sketch_tensor):
        """Generate image from sketch tensor"""
        self.generator.eval()
        with torch.no_grad():
            generated = self.generator(sketch_tensor)
            generated = (generated + 1) / 2
            generated = torch.clamp(generated, 0, 1)
        return generated
    
    def tensor_to_image(self, tensor):
        """Convert tensor to PIL image"""
        image = tensor.squeeze(0).cpu().numpy()
        image = (image * 255).astype('uint8')
        image = image.transpose(1, 2, 0)
        return Image.fromarray(image)
    
    def process_image(self, image):
        """Process a single sketch image and return the generated image"""
        sketch_tensor = self.preprocess_sketch(image)
        generated_tensor = self.generate_image(sketch_tensor)
        generated_image = self.tensor_to_image(generated_tensor)
        return generated_image

# Initialize the model (you can change the model path here)
model_path = "output_data/checkpoints2/best_model (2).pth"
if not os.path.exists(model_path):
    model_path = "output_data/checkpoints/best_model.pth"

try:
    sketch_api = SketchToImageAPI(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sketch_api = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if sketch_api is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get the uploaded file
        if 'sketch' not in request.files:
            return jsonify({'error': 'No sketch file uploaded'}), 400
        
        file = request.files['sketch']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get settings from form data
        auto_enhance = request.form.get('auto_enhance', 'false').lower() == 'true'
        high_quality = request.form.get('high_quality', 'false').lower() == 'true'
        save_comparison = request.form.get('save_comparison', 'true').lower() == 'true'
        
        # Process the image
        image = Image.open(file.stream)
        generated_image = sketch_api.process_image(image)
        
        # Apply post-processing if requested
        if auto_enhance:
            generated_image = enhance_image(generated_image)
        
        # Convert images to base64 for display
        def image_to_base64(img):
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95 if high_quality else 85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
        
        # Resize original image for display
        original_resized = image.convert('RGB').resize((256, 256))
        
        original_b64 = image_to_base64(original_resized)
        generated_b64 = image_to_base64(generated_image)
        
        # Determine model type
        model_type = "Small Model" if hasattr(sketch_api, '_is_small_model') and sketch_api._is_small_model else "Full Model"
        
        return jsonify({
            'success': True,
            'original': original_b64,
            'generated': generated_b64,
            'model_type': model_type,
            'settings': {
                'auto_enhance': auto_enhance,
                'high_quality': high_quality,
                'save_comparison': save_comparison
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def enhance_image(image):
    """Simple image enhancement using PIL"""
    from PIL import ImageEnhance
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.05)
    
    # Enhance color
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.05)
    
    return image

@app.route('/batch_generate', methods=['POST'])
def batch_generate():
    if sketch_api is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        files = request.files.getlist('sketches')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        results = []
        for file in files:
            if file.filename == '':
                continue
                
            try:
                image = Image.open(file.stream)
                generated_image = sketch_api.process_image(image)
                
                # Convert to base64
                def image_to_base64(img):
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    return f"data:image/jpeg;base64,{img_str}"
                
                original_resized = image.convert('RGB').resize((256, 256))
                
                results.append({
                    'filename': file.filename,
                    'original': image_to_base64(original_resized),
                    'generated': image_to_base64(generated_image),
                    'success': True
                })
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'success': False
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len([r for r in results if r['success']])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': sketch_api is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
