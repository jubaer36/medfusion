#!/usr/bin/env python3
"""
Test Script for New Wavelet Models in Web Application
====================================================

This script tests the integration of the new wavelet fusion models
into the web application framework.
"""

import sys
import os
import torch
import numpy as np
import cv2

# Add the webapp directory to the path
sys.path.append(os.path.dirname(__file__))

from fusion_manager import FusionManager
from utils.image_processing import load_image_from_path


def test_fusion_models():
    """Test all fusion models with sample data."""
    
    print("=" * 60)
    print("TESTING FUSION MODELS IN WEB APPLICATION")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize fusion manager
    fusion_manager = FusionManager(device)
    
    # Get available methods
    available_methods = fusion_manager.get_available_methods()
    all_methods = fusion_manager.get_all_methods()
    
    print(f"\nTotal methods: {len(all_methods)}")
    print(f"Available methods: {len(available_methods)}")
    
    print("\n" + "-" * 40)
    print("METHOD STATUS:")
    print("-" * 40)
    
    for key, info in all_methods.items():
        status = "✅ AVAILABLE" if info['is_available'] else "❌ UNAVAILABLE"
        print(f"{key:20s} | {status}")
        if not info['is_available']:
            print(f"{'':20s} | Error: {info.get('error_message', 'Unknown')}")
        else:
            print(f"{'':20s} | {info['description'][:60]}...")
    
    # Test with sample images if any methods are available
    if available_methods:
        print(f"\n" + "-" * 40)
        print("TESTING WITH SAMPLE IMAGES:")
        print("-" * 40)
        
        # Create synthetic test images
        test_ct = np.random.rand(256, 256).astype(np.float32)
        test_mri = np.random.rand(256, 256).astype(np.float32)
        
        print(f"Test image shape: {test_ct.shape}")
        print(f"CT range: [{test_ct.min():.3f}, {test_ct.max():.3f}]")
        print(f"MRI range: [{test_mri.min():.3f}, {test_mri.max():.3f}]")
        
        # Test only available methods
        available_keys = list(available_methods.keys())
        
        try:
            results = fusion_manager.fuse_images(test_ct, test_mri, available_keys)
            
            print(f"\nFusion Results:")
            for method_key, result in results.items():
                if result.get('fused_image') is not None:
                    fused_img = result['fused_image']
                    metrics = result.get('metrics', {})
                    print(f"  {method_key:20s} | ✅ SUCCESS | Shape: {fused_img.shape} | Range: [{fused_img.min():.3f}, {fused_img.max():.3f}]")
                    if metrics:
                        ssim_avg = metrics.get('ssim_average', 'N/A')
                        psnr_avg = metrics.get('psnr_average', 'N/A')
                        print(f"  {'':20s} | SSIM: {ssim_avg} | PSNR: {psnr_avg}")
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"  {method_key:20s} | ❌ FAILED | {error}")
        
        except Exception as e:
            print(f"Error during fusion testing: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("\n⚠️  No methods available for testing!")
    
    # Test specific wavelet models if available
    print(f"\n" + "-" * 40)
    print("WAVELET MODELS SPECIFIC TEST:")
    print("-" * 40)
    
    wavelet_keys = ['wavelet_option1', 'wavelet_option2']
    for key in wavelet_keys:
        if key in available_methods:
            method_info = available_methods[key]
            print(f"✅ {key}: {method_info['name']}")
            print(f"   Description: {method_info['description']}")
        else:
            if key in all_methods:
                error = all_methods[key].get('error_message', 'Unknown error')
                print(f"❌ {key}: {error}")
            else:
                print(f"❌ {key}: Not configured")
    
    print(f"\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def test_real_images():
    """Test with real medical images if available."""
    
    # Look for sample images
    sample_paths = [
        '../Harvard-Medical-Image-Fusion-Datasets/MyDatasets/CT-MRI/test/CT/',
        '../Harvard-Medical-Image-Fusion-Datasets/CT-MRI/',
        '../test_images/'
    ]
    
    for path in sample_paths:
        full_path = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(full_path):
            ct_files = [f for f in os.listdir(full_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if ct_files:
                print(f"\nFound sample images in: {path}")
                print(f"Sample files: {ct_files[:3]}")
                return full_path
    
    print("\nNo real medical images found for testing")
    return None


if __name__ == "__main__":
    test_fusion_models()
    
    # Test with real images if available
    image_path = test_real_images()
    if image_path:
        print("\nReal image testing could be implemented here...")
