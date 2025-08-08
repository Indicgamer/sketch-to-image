#!/usr/bin/env python3
"""
Sketch-to-Image Web Interface Launcher

This script starts the web interface for the sketch-to-image generation model.
It provides a simple way to launch the Flask server with appropriate configuration.
"""

import os
import sys
import argparse
import webbrowser
import time
from threading import Timer

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['flask', 'torch', 'torchvision', 'PIL']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install flask torch torchvision Pillow")
        return False
    
    print("✅ All required packages are installed")
    return True

def find_model_files():
    """Find available model files"""
    possible_paths = [
        "output_data/checkpoints2/best_model (2).pth",
        "output_data/checkpoints/best_model.pth",
        "output_data/checkpoints2/final_model (1).pth",
        "output_data/checkpoints/final_model.pth"
    ]
    
    found_models = []
    for path in possible_paths:
        if os.path.exists(path):
            found_models.append(path)
    
    return found_models

def open_browser(url, delay=2):
    """Open the web browser after a delay"""
    def open_url():
        try:
            webbrowser.open(url)
            print(f"🌐 Opened {url} in your default browser")
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please manually open: {url}")
    
    timer = Timer(delay, open_url)
    timer.start()

def main():
    parser = argparse.ArgumentParser(description='Launch Sketch-to-Image Web Interface')
    parser.add_argument('--host', default='localhost', help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--model-path', help='Path to model file (auto-detected if not specified)')
    
    args = parser.parse_args()
    
    print("🎨 Sketch-to-Image Web Interface Launcher")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Find model files
    if args.model_path:
        if not os.path.exists(args.model_path):
            print(f"❌ Model file not found: {args.model_path}")
            sys.exit(1)
        print(f"✅ Using specified model: {args.model_path}")
    else:
        models = find_model_files()
        if not models:
            print("❌ No model files found!")
            print("Please make sure you have trained models in:")
            print("  - output_data/checkpoints/")
            print("  - output_data/checkpoints2/")
            sys.exit(1)
        
        print(f"✅ Found {len(models)} model file(s):")
        for model in models:
            print(f"  - {model}")
        print(f"🔄 Will use: {models[0]}")
    
    # Set environment variables
    if args.model_path:
        os.environ['MODEL_PATH'] = args.model_path
    
    # Import and run the Flask app
    try:
        from app import app
        
        url = f"http://{args.host}:{args.port}"
        print(f"\n🚀 Starting web server...")
        print(f"📍 URL: {url}")
        print(f"💡 Press Ctrl+C to stop the server")
        
        if not args.no_browser:
            print(f"🌐 Opening browser in 2 seconds...")
            open_browser(url)
        
        print("\n" + "=" * 50)
        
        # Start the Flask app
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    except ImportError as e:
        print(f"❌ Error importing Flask app: {e}")
        print("Make sure app.py is in the current directory")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
