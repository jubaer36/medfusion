#!/usr/bin/env python3
"""
Detailed debugging script for fusion methods
"""

import sys
import os
sys.path.append('/mnt/Work/ML/trainable-multimodal-fusion/medical_fusion_webapp')

import torch
from fusion_manager import FusionManager

def main():
    print("🔍 Debugging Fusion Methods Loading")
    print("=" * 60)
    
    # Change to webapp directory for correct relative paths
    original_dir = os.getcwd()
    print(f"Original directory: {original_dir}")
    
    webapp_dir = '/mnt/Work/ML/trainable-multimodal-fusion/medical_fusion_webapp'
    os.chdir(webapp_dir)
    print(f"Changed to webapp directory: {os.getcwd()}")
    
    # Check if model files exist from webapp directory
    model_paths = [
        '../checkpoints_resnet/resnet_fusion_best.pt',
        '../checkpoints_wavelet/wavelet_fusion_best.pt', 
        '../checkpoints_enhanced/wavelet_fusion_spatial_best.pt'
    ]
    
    print("\n📁 Checking model file paths:")
    for path in model_paths:
        exists = os.path.exists(path)
        abs_path = os.path.abspath(path)
        print(f"  {'✅' if exists else '❌'} {path} -> {abs_path}")
    
    print("\n🚀 Initializing FusionManager...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        fusion_manager = FusionManager(device)
        
        print(f"\n📋 Initialized methods: {len(fusion_manager.fusion_methods)}")
        for key, method in fusion_manager.fusion_methods.items():
            info = method.get_info()
            status = "✅" if info['is_available'] else "❌"
            print(f"  {status} {key}: {info['name']}")
        
        # Get available and recommended methods
        available = fusion_manager.get_available_methods()
        recommended = fusion_manager.get_recommended_methods()
        
        print(f"\n🌟 Available methods: {len(available)}")
        print(f"🌟 Recommended methods: {len(recommended)}")
        print(f"Recommended: {recommended}")
        
        # Check specific methods
        has_resnet = 'resnet' in available
        has_wavelet1 = 'wavelet_option1' in available
        has_wavelet2 = 'wavelet_option2' in available
        
        print(f"\n🎯 Deep Learning Methods Status:")
        print(f"  ResNet: {'✅' if has_resnet else '❌'}")
        print(f"  Wavelet Option 1: {'✅' if has_wavelet1 else '❌'}")
        print(f"  Wavelet Option 2: {'✅' if has_wavelet2 else '❌'}")
        
        return has_resnet and has_wavelet1
        
    except Exception as e:
        print(f"❌ Error initializing FusionManager: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)
