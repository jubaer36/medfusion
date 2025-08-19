#!/usr/bin/env python3
"""
Quick verification that all webapp components are working
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    print("Testing imports...")
    
    # Test basic imports
    import torch
    import numpy as np
    import cv2
    print("✅ Basic ML libraries")
    
    # Test Flask
    import flask
    print("✅ Flask")
    
    # Test our modules
    from fusion_manager import FusionManager
    print("✅ FusionManager")
    
    from utils.image_processing import load_image_from_path
    print("✅ Image processing utils")
    
    from utils.metrics import evaluate_fusion_comprehensive
    print("✅ Metrics utils")
    
    # Test models
    from models.resnet_fusion import ResNetFusionNet
    from models.wavelet_fusion import WaveletFusionNet, WaveletFusionNetSpatial
    print("✅ Fusion models")
    
    # Test fusion methods
    from fusion_methods.deep_learning import ResNetFusion, WaveletFusion, SpatialWaveletFusion
    from fusion_methods.traditional import AverageFusion
    print("✅ Fusion methods")
    
    print("\n🎉 All components successfully imported!")
    print("The webapp should be ready to run.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
