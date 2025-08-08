# Sketch-to-Image Web Interface

A feature-rich web interface for the sketch-to-image generation model using Flask. Transform your sketches into realistic images with advanced AI technology.

## 🌟 Features

### Core Functionality
- **Single Image Processing**: Upload and process individual sketches
- **Batch Processing**: Process multiple images simultaneously
- **Real-time Preview**: See your uploaded sketches before processing
- **Auto Model Detection**: Automatically detects and loads the correct model architecture

### Advanced Features
- **Image Gallery**: View and manage all generated images
- **Side-by-side Comparison**: Interactive slider to compare original vs generated
- **Zoom & Modal View**: Click images to view them in full screen with zoom controls
- **Download Options**: Download individual images or entire batches
- **Progress Tracking**: Real-time progress bars for batch operations
- **Image Enhancement**: Optional auto-enhancement for better results

### User Interface
- **Responsive Design**: Works perfectly on desktop and mobile devices
- **Tabbed Interface**: Organized tabs for different functions
- **Drag & Drop Upload**: Easy file upload with drag and drop support
- **Advanced Settings**: Customizable processing options
- **Error Handling**: User-friendly error messages and loading states

## 🚀 Quick Start

### Option 1: Using the Launcher (Recommended)
```bash
python launch_web.py
```

The launcher will:
- Check for required dependencies
- Auto-detect available models
- Open your browser automatically
- Provide helpful status information

### Option 2: Direct Flask Launch
```bash
python app.py
```

Then open your browser to: `http://localhost:5000`

## 📋 Requirements

Install dependencies:
```bash
pip install flask torch torchvision Pillow
```

## 🎯 Usage Guide

### Single Image Mode
1. **Upload**: Click the upload area or drag & drop an image
2. **Configure**: Adjust settings in the Advanced Settings panel
3. **Generate**: Click "Generate Image" button
4. **View Results**: See original and generated images side by side
5. **Download**: Use download buttons to save results

### Batch Processing Mode
1. **Upload Multiple**: Select multiple images or drag & drop a folder
2. **Review List**: Check the file list and remove unwanted files
3. **Process**: Click "Process All Images" 
4. **Monitor Progress**: Watch the progress bar
5. **View Results**: Browse generated images in the results grid

### Gallery Mode
- **View All**: See all generated images in a grid layout
- **Download All**: Bulk download all generated images
- **Clear Gallery**: Remove all images from gallery

### Compare Mode
- **Interactive Slider**: Drag the slider to compare original vs generated
- **Real-time Comparison**: See differences instantly
- **Full Screen**: Click to view comparison in modal

## ⚙️ Advanced Settings

### Processing Options
- **Auto-enhance**: Automatically improve image quality
- **High Quality**: Use higher quality JPEG compression
- **Save Comparison**: Include side-by-side comparison images

### Image Controls
- **Zoom Controls**: Zoom in/out on images in modal view
- **Download Options**: Individual or batch downloads
- **Gallery Management**: Clear gallery or download all

## 🔧 Configuration

### Model Path
The app automatically tries to load models in this order:
1. `output_data/checkpoints2/best_model (2).pth` (Small model)
2. `output_data/checkpoints/best_model.pth` (Full model)

To use a custom model path:
```bash
python launch_web.py --model-path path/to/your/model.pth
```

### Server Options
```bash
# Custom host and port
python launch_web.py --host 0.0.0.0 --port 8080

# Enable debug mode
python launch_web.py --debug

# Disable auto-browser opening
python launch_web.py --no-browser
```

## 🌐 API Endpoints

### Image Generation
- **POST** `/generate` - Generate image from single sketch
  - Form data: `sketch` (image file)
  - Optional: `auto_enhance`, `high_quality`, `save_comparison`

### Batch Processing
- **POST** `/batch_generate` - Process multiple sketches
  - Form data: `sketches` (multiple image files)

### Health Check
- **GET** `/health` - Check server and model status

## 📱 Supported Formats

### Input Formats
- JPG/JPEG
- PNG
- BMP
- GIF (first frame)
- TIFF
- WebP

### Output Format
- High-quality JPEG images
- Base64 encoded for web display
- Downloadable as files

## 🎨 User Interface Features

### Keyboard Shortcuts
- **Escape**: Close modal view
- **Click outside modal**: Close modal

### Interactive Elements
- **Drag & Drop**: Upload files anywhere in upload areas
- **Progress Bars**: Real-time processing progress
- **Loading States**: Visual feedback during processing
- **Error Messages**: Clear error descriptions
- **Success Notifications**: Confirmation messages

### Mobile Responsiveness
- **Adaptive Layout**: Single column on mobile devices
- **Touch-friendly**: Large buttons and touch targets
- **Responsive Images**: Properly scaled on all devices

## 🛠️ Troubleshooting

### Common Issues

**Model Not Loading**
- Ensure model files exist in the expected directories
- Check console output for detailed error messages
- Verify PyTorch compatibility

**Upload Errors**
- Check file format is supported
- Ensure file size is under 16MB limit
- Try refreshing the page

**Generation Failures**
- Check available memory (especially for batch processing)
- Ensure stable internet connection
- Monitor server logs for detailed errors

### Performance Tips
- **CPU vs GPU**: GPU processing is faster if available
- **Batch Size**: Process smaller batches if memory is limited
- **Image Size**: Larger images take more time to process
- **Network**: Stable connection important for upload/download

## 🔒 Security Notes

### Model Loading
- Uses `weights_only=False` for PyTorch model loading
- Only use trusted model files
- Model files should be from verified sources

### File Uploads
- 16MB maximum file size limit
- Only image files are accepted
- Files are processed in memory (not stored on disk)

## 🚀 Development

### File Structure
```
├── app.py                 # Flask application
├── launch_web.py         # Launcher script
├── templates/
│   └── index.html        # Web interface
├── model.py              # Model definitions
├── inference.py          # Command-line inference
└── output_data/          # Model files
    ├── checkpoints/      # Full models
    └── checkpoints2/     # Small models
```

### Customization
- **Styling**: Edit CSS in `templates/index.html`
- **Features**: Modify JavaScript in the same file
- **Backend**: Update Flask routes in `app.py`
- **Models**: Configure model paths and parameters

## 📄 License

This project is part of the Sketch-to-Image generation system. Use responsibly and ensure compliance with AI ethics guidelines.
