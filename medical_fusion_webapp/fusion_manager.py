"""
Fusion Manager
==============

Central manager for coordinating different fusion methods and their evaluation.
"""

import os
import torch
from typing import Dict, List, Any, Optional
import numpy as np

from fusion_methods.base import FusionMethod
from fusion_methods.deep_learning import ResNetFusion, WaveletFusion, SpatialWaveletFusion
from fusion_methods.traditional import DWTPCAFusion
from utils.metrics import evaluate_fusion_comprehensive
from utils.image_processing import validate_image_pair


class FusionManager:
    """Manager class for all fusion methods."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.fusion_methods: Dict[str, FusionMethod] = {}
        self._initialize_methods()
    
    def _initialize_methods(self):
        """Initialize all available fusion methods."""
        
        # Deep learning methods
        model_configs = [
            {
                'class': ResNetFusion,
                'path': '../checkpoints_resnet/resnet_fusion_best.pt',
                'key': 'resnet'
            },
            {
                'class': WaveletFusion,
                'path': '../checkpoints_wavelet/wavelet_fusion_best.pt',
                'key': 'wavelet_option1'
            },
            {
                'class': SpatialWaveletFusion,
                'path': '../checkpoints_enhanced/wavelet_fusion_spatial_best.pt',
                'key': 'wavelet_option2'
            }
        ]
        
        for config in model_configs:
            try:
                method = config['class'](config['path'], self.device)
                self.fusion_methods[config['key']] = method
            except Exception as e:
                print(f"Failed to initialize {config['key']}: {e}")
        
        # Traditional methods (only DWT-PCA which is implemented in parent directory)
        traditional_methods = [
            ('dwt_pca', DWTPCAFusion())
        ]
        
        for key, method in traditional_methods:
            self.fusion_methods[key] = method
    
    def get_available_methods(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available fusion methods."""
        available = {}
        for key, method in self.fusion_methods.items():
            info = method.get_info()
            if info['is_available']:
                available[key] = info
        return available
    
    def get_all_methods(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all fusion methods (including unavailable ones)."""
        all_methods = {}
        for key, method in self.fusion_methods.items():
            all_methods[key] = method.get_info()
        return all_methods
    
    def fuse_images(self, ct: np.ndarray, mri: np.ndarray, 
                   methods: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Fuse images using specified methods.
        
        Args:
            ct: CT image array
            mri: MRI image array
            methods: List of method keys to use (None for all available)
            
        Returns:
            Dictionary with fusion results and metrics for each method
        """
        # Validate input images
        if not validate_image_pair(ct, mri):
            raise ValueError("Invalid image pair provided")
        
        # Use all available methods if none specified
        if methods is None:
            methods = list(self.get_available_methods().keys())
        
        results = {}
        
        for method_key in methods:
            if method_key not in self.fusion_methods:
                continue
            
            method = self.fusion_methods[method_key]
            
            # Skip unavailable methods
            if not method.is_available:
                results[method_key] = {
                    'method_info': method.get_info(),
                    'fused_image': None,
                    'metrics': {},
                    'error': method.error_message
                }
                continue
            
            try:
                # Perform fusion
                fused = method.fuse(ct, mri)
                
                if fused is not None:
                    # Evaluate fusion quality
                    metrics = evaluate_fusion_comprehensive(fused, ct, mri)
                    
                    results[method_key] = {
                        'method_info': method.get_info(),
                        'fused_image': fused,
                        'metrics': metrics,
                        'error': None
                    }
                else:
                    results[method_key] = {
                        'method_info': method.get_info(),
                        'fused_image': None,
                        'metrics': {},
                        'error': 'Fusion failed'
                    }
                    
            except Exception as e:
                results[method_key] = {
                    'method_info': method.get_info(),
                    'fused_image': None,
                    'metrics': {},
                    'error': str(e)
                }
        
        return results
    
    def add_custom_method(self, key: str, method: FusionMethod):
        """Add a custom fusion method."""
        self.fusion_methods[key] = method
    
    def remove_method(self, key: str):
        """Remove a fusion method."""
        if key in self.fusion_methods:
            del self.fusion_methods[key]
    
    def get_method_categories(self) -> Dict[str, List[str]]:
        """Get methods organized by category."""
        categories = {
            'Deep Learning': [],
            'Traditional': [],
            'Unavailable': []
        }
        
        for key, method in self.fusion_methods.items():
            info = method.get_info()
            
            if not info['is_available']:
                categories['Unavailable'].append(key)
            elif 'Deep' in info['name'] or 'ResNet' in info['name'] or 'Wavelet' in info['name']:
                categories['Deep Learning'].append(key)
            else:
                categories['Traditional'].append(key)
        
        return categories
    
    def get_recommended_methods(self) -> List[str]:
        """Get list of recommended methods for comparison."""
        recommended = []
        
        # Add all available deep learning methods (priority order)
        dl_priority = ['wavelet_option2', 'wavelet_option1', 'resnet']
        for method in dl_priority:
            if method in self.fusion_methods and self.fusion_methods[method].is_available:
                recommended.append(method)
        
        # Add DWT-PCA method (implemented in parent directory)
        if 'dwt_pca' in self.fusion_methods and self.fusion_methods['dwt_pca'].is_available:
            recommended.append('dwt_pca')
        
        return recommended