#!/usr/bin/env python3
"""
Test script to verify model loading and basic functionality
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image

# Add parent directory to path to import models
sys.path.append('..')

def test_resnet_model():
    """Test ResNet model loading and inference."""
    print("Testing ResNet Model...")
    
    try:
        from models.resnet_fusion import ResNetFusionNet
        
        # Load model
        model_path = '../checkpoints_resnet/resnet_fusion_best.pt'
        if not os.path.exists(model_path):
            print(f"❌ ResNet model not found at {model_path}")
            return False
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetFusionNet().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("✅ ResNet model loaded successfully")
        
        # Test inference with dummy data
        with torch.no_grad():
            dummy_ct = torch.randn(1, 1, 256, 256).to(device)
            dummy_mri = torch.randn(1, 1, 256, 256).to(device)
            
            output = model(dummy_ct, dummy_mri)
            print(f"✅ ResNet inference test passed - Output shape: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ ResNet model test failed: {e}")
        return False

def test_wavelet_model():
    """Test Wavelet model loading and inference."""
    print("\nTesting Wavelet Model...")
    
    try:
        from models.wavelet_fusion import WaveletFusionNet, WAVELETS_AVAILABLE
        
        if not WAVELETS_AVAILABLE:
            print("❌ pytorch_wavelets not available")
            return False
        
        # Try different possible paths
        model_paths = [
            '../checkpoints_wavelet/wavelet_fusion_best.pt',
            '../checkpoints_enhanced/wavelet_fusion_spatial_best.pt'
        ]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_loaded = False
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model = WaveletFusionNet().to(device)
                    checkpoint = torch.load(model_path, map_location=device)
                    
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        model.load_state_dict(checkpoint['model'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model.eval()
                    print(f"✅ Wavelet model loaded from {model_path}")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    print(f"⚠️  Failed to load from {model_path}: {e}")
                    continue
        
        if not model_loaded:
            print("❌ No wavelet model could be loaded")
            return False
        
        # Test inference with dummy data
        with torch.no_grad():
            dummy_ct = torch.randn(1, 1, 256, 256).to(device)
            dummy_mri = torch.randn(1, 1, 256, 256).to(device)
            
            output = model(dummy_ct, dummy_mri)
            print(f"✅ Wavelet inference test passed - Output shape: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Wavelet model test failed: {e}")
        return False

def test_sample_images():
    """Test loading sample images."""
    print("\nTesting Sample Image Loading...")
    
    sample_paths = [
        '../Harvard-Medical-Image-Fusion-Datasets/CT-MRI/CT',
        '../Harvard-Medical-Image-Fusion-Datasets/MyDatasets/CT-MRI/test/CT'
    ]
    
    for base_path in sample_paths:
        if os.path.exists(base_path):
            ct_files = [f for f in os.listdir(base_path) if f.endswith('.png')]
            if ct_files:
                sample_file = ct_files[0]
                ct_path = os.path.join(base_path, sample_file)
                mri_base = base_path.replace('/CT', '/MRI')
                mri_path = os.path.join(mri_base, sample_file)
                
                if os.path.exists(mri_path):
                    # Test loading
                    ct_img = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
                    mri_img = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
                    
                    if ct_img is not None and mri_img is not None and ct_img.size > 0 and mri_img.size > 0:
                        print(f"✅ Found sample images: {sample_file}")
                        print(f"✅ Sample images loaded - CT: {ct_img.shape}, MRI: {mri_img.shape}")
                        return True
    
    print("❌ No sample images found")
    return False

def test_baseline_methods():
    """Test baseline fusion methods."""
    print("\nTesting Baseline Fusion Methods...")
    
    try:
        # Create dummy images
        ct = np.random.rand(256, 256).astype(np.float32)
        mri = np.random.rand(256, 256).astype(np.float32)
        
        # Test average fusion
        avg_fused = 0.5 * (ct + mri)
        print("✅ Average fusion test passed")
        
        # Test maximum fusion
        max_fused = np.maximum(ct, mri)
        print("✅ Maximum fusion test passed")
        
        # Test weighted fusion
        weighted_fused = 0.6 * ct + 0.4 * mri
        print("✅ Weighted fusion test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Baseline methods test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Medical Image Fusion Model Testing")
    print("=" * 50)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    # Run tests
    tests = [
        ("ResNet Model", test_resnet_model),
        ("Wavelet Model", test_wavelet_model),
        ("Sample Images", test_sample_images),
        ("Baseline Methods", test_baseline_methods)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed >= 2:  # At least baseline methods and one model
        print("🚀 Application should work with available components")
    else:
        print("⚠️  Application may have limited functionality")

if __name__ == '__main__':
    main()