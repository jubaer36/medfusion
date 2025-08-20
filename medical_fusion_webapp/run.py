#!/usr/bin/env python3
"""
Simple launcher script for the Medical Image Fusion Web Application
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import flask
        import torch
        import cv2
        import PIL
        import numpy
        import matplotlib
        import seaborn
        import skimage
        print("✅ All dependencies are available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_models():
    """Check if model checkpoints are available."""
    model_paths = [
        '../checkpoints_resnet/resnet_fusion_best.pt',
        '../checkpoints_wavelet/wavelet_fusion_best.pt'
    ]
    
    available_models = []
    for path in model_paths:
        if os.path.exists(path):
            available_models.append(path)
            print(f"✅ Found model: {path}")
        else:
            print(f"⚠️  Model not found: {path}")
    
    if not available_models:
        print("❌ No trained models found. Please train models first.")
        return False
    
    return True

def main():
    """Main launcher function."""
    print("🚀 Medical Image Fusion Web Application Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check models
    if not check_models():
        print("\n⚠️  Warning: Some models are missing, but the app will still run with available methods.")
    
    # Start the application
    print("\n🌐 Starting Flask application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()