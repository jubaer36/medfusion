"""
Deep Learning-based Fusion Methods
==================================

Implementation of neural network-based fusion methods including ResNet and Wavelet models.
"""

import numpy as np
import torch
from typing import Optional

from .base import ModelBasedFusion
from models.resnet_fusion import ResNetFusionNet
from models.wavelet_fusion import WaveletFusionNet, WaveletFusionNetSpatial, WAVELETS_AVAILABLE


class ResNetFusion(ModelBasedFusion):
    """ResNet-based deep learning fusion method."""
    
    def __init__(self, model_path: str, device: torch.device):
        super().__init__(
            name="ResNet Deep Fusion",
            description="Deep learning fusion using ResNet50 backbone with multi-level feature extraction",
            model_path=model_path,
            device=device
        )
    
    def create_model(self):
        """Create ResNet fusion model."""
        return ResNetFusionNet()
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images using ResNet model."""
        if not self.is_available:
            return None
        
        try:
            with torch.no_grad():
                ct_tensor = self.prepare_tensor(ct)
                mri_tensor = self.prepare_tensor(mri)
                fused_tensor = self.model(ct_tensor, mri_tensor)
                return np.clip(self.tensor_to_numpy(fused_tensor), 0, 1)
        except Exception as e:
            print(f"ResNet fusion error: {e}")
            return None


class WaveletFusion(ModelBasedFusion):
    """Simple wavelet-based fusion method."""
    
    def __init__(self, model_path: str, device: torch.device):
        super().__init__(
            name="Wavelet Fusion",
            description="Trainable wavelet fusion with learnable frequency-band weights",
            model_path=model_path,
            device=device
        )
        
        # Check if wavelets are available
        if not WAVELETS_AVAILABLE:
            self.is_available = False
            self.error_message = "pytorch_wavelets package not available"
    
    def create_model(self):
        """Create wavelet fusion model."""
        return WaveletFusionNet()
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images using wavelet model."""
        if not self.is_available:
            return None
        
        try:
            with torch.no_grad():
                ct_tensor = self.prepare_tensor(ct)
                mri_tensor = self.prepare_tensor(mri)
                fused_tensor = self.model(ct_tensor, mri_tensor)
                return np.clip(self.tensor_to_numpy(fused_tensor), 0, 1)
        except Exception as e:
            print(f"Wavelet fusion error: {e}")
            return None


class SpatialWaveletFusion(ModelBasedFusion):
    """Spatial-adaptive wavelet fusion method."""
    
    def __init__(self, model_path: str, device: torch.device):
        super().__init__(
            name="Spatial-Adaptive Wavelet",
            description="Advanced wavelet fusion with CNN-based spatial masks for location-aware fusion",
            model_path=model_path,
            device=device
        )
        
        # Check if wavelets are available
        if not WAVELETS_AVAILABLE:
            self.is_available = False
            self.error_message = "pytorch_wavelets package not available"
    
    def create_model(self):
        """Create spatial wavelet fusion model."""
        return WaveletFusionNetSpatial()
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images using spatial-adaptive wavelet model."""
        if not self.is_available:
            return None
        
        try:
            with torch.no_grad():
                ct_tensor = self.prepare_tensor(ct)
                mri_tensor = self.prepare_tensor(mri)
                fused_tensor, masks = self.model(ct_tensor, mri_tensor)
                return np.clip(self.tensor_to_numpy(fused_tensor), 0, 1)
        except Exception as e:
            print(f"Spatial wavelet fusion error: {e}")
            return None