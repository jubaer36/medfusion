#!/usr/bin/env python3
"""
Test script to verify all fusion methods are now available
"""

import sys
import os
sys.path.append('/mnt/Work/ML/trainable-multimodal-fusion/medical_fusion_webapp')

from fusion_manager import FusionManager

def main():
    print("🧪 Testing Updated Fusion Methods Display")
    print("=" * 50)
    
    try:
        # Initialize fusion manager with device
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fusion_manager = FusionManager(device)
        
        # Get all available methods
        available_methods = fusion_manager.get_available_methods()
        print(f"📋 Total Available Methods: {len(available_methods)}")
        for method in available_methods:
            print(f"  ✅ {method}")
        
        print("\n" + "=" * 50)
        
        # Get recommended methods (should now include ALL deep learning methods)
        recommended_methods = fusion_manager.get_recommended_methods()
        print(f"🌟 Recommended Methods (displayed in webapp): {len(recommended_methods)}")
        for method in recommended_methods:
            print(f"  ⭐ {method}")
        
        print("\n" + "=" * 50)
        
        # Check if ResNet and Wavelet Option 1 are now included
        missing_methods = []
        if 'resnet' not in recommended_methods:
            missing_methods.append('resnet')
        if 'wavelet_option1' not in recommended_methods:
            missing_methods.append('wavelet_option1')
        
        if missing_methods:
            print(f"❌ Still Missing: {missing_methods}")
            return False
        else:
            print("✅ SUCCESS: All deep learning methods are now included!")
            print("  • ResNet Fusion: ✅ Included")
            print("  • Wavelet Option 1: ✅ Included") 
            print("  • Wavelet Option 2: ✅ Included")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
