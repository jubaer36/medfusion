"""
Deep Learning-based Fusion Methods
==================================

Implementation of neural network-based fusion methods including ResNet, Wavelet, and FATFusion models.
"""

import numpy as np
import torch
from typing import Optional
import sys
import os

from .base import ModelBasedFusion
from models.resnet_fusion import ResNetFusionNet
from models.wavelet_fusion import WaveletFusionNet, WaveletFusionNetSpatial, WAVELETS_AVAILABLE

# Add FATFusion path to import
sys.path.append(os.path.join(os.path.dirname(__file__), '../../FATFusion'))

try:
    from Networks.networks import MODEL as FATFusionModel
    FATFUSION_AVAILABLE = True
except ImportError:
    FATFUSION_AVAILABLE = False

# Add MATR path to import
sys.path.append(os.path.join(os.path.dirname(__file__), '../../MATR-main'))

try:
    from Networks.net import MODEL as MATRModel
    MATR_AVAILABLE = True
except ImportError:
    MATR_AVAILABLE = False


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
            name="Wavelet Fusion (Option 1)",
            description="Trainable wavelet fusion with learnable global frequency-band weights (α_L=0.857, α_LH=0.838, α_HL=0.500, α_HH=0.500)",
            model_path=model_path,
            device=device
        )
        
        # Check if wavelets are available
        if not WAVELETS_AVAILABLE:
            self.is_available = False
            self.error_message = "pytorch_wavelets package not available"
    
    def create_model(self):
        """Create wavelet fusion model."""
        return WaveletFusionNet(wave='haar')  # Match training configuration
    
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
            name="Spatial-Adaptive Wavelet (Option 2)",
            description="Advanced wavelet fusion with CNN-based spatial masks for location-aware fusion. Uses enhanced loss function (L1 + SSIM + Gradient).",
            model_path=model_path,
            device=device
        )
        
        # Check if wavelets are available
        if not WAVELETS_AVAILABLE:
            self.is_available = False
            self.error_message = "pytorch_wavelets package not available"
    
    def create_model(self):
        """Create spatial-adaptive wavelet fusion model."""
        return WaveletFusionNetSpatial(wave='haar')  # Match training configuration
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images using spatial-adaptive wavelet model."""
        if not self.is_available:
            return None
        
        try:
            with torch.no_grad():
                ct_tensor = self.prepare_tensor(ct)
                mri_tensor = self.prepare_tensor(mri)
                fused_tensor = self.model(ct_tensor, mri_tensor)
                return np.clip(self.tensor_to_numpy(fused_tensor), 0, 1)
        except Exception as e:
            print(f"Spatial wavelet fusion error: {e}")
            return None


class FATFusion(ModelBasedFusion):
    """FATFusion: Feature Adaptive Transformer for Medical Image Fusion."""
    
    def __init__(self, model_path: str, device: torch.device):
        super().__init__(
            name="FATFusion Transformer",
            description="Feature Adaptive Transformer fusion using Swin Transformer blocks with channel attention for PET-MRI and CT-MRI fusion",
            model_path=model_path,
            device=device
        )
        
        # Check if FATFusion is available
        if not FATFUSION_AVAILABLE:
            self.is_available = False
            self.error_message = "FATFusion networks not available"
    
    def create_model(self):
        """Create FATFusion model."""
        return FATFusionModel(in_channel=1)  # Use default parameters like in Test.py
    
    def fuse(self, image1: np.ndarray, image2: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images using FATFusion model."""
        if not self.is_available:
            return None
        
        try:
            with torch.no_grad():
                # Prepare input tensors
                image1_tensor = self.prepare_tensor(image1)
                image2_tensor = self.prepare_tensor(image2)
                
                # FATFusion expects two input images
                fused_tensor = self.model(image1_tensor, image2_tensor)
                return np.clip(self.tensor_to_numpy(fused_tensor), 0, 1)
        except Exception as e:
            print(f"FATFusion error: {e}")
            return None


class MATRFusion(ModelBasedFusion):
    """MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer."""
    
    def __init__(self, model_path: str, device: torch.device):
        super().__init__(
            name="MATR Transformer",
            description="Multimodal Medical Image Fusion via Multiscale Adaptive Transformer for PET-MRI fusion",
            model_path=model_path,
            device=device
        )
        
        # Check if MATR is available
        if not MATR_AVAILABLE:
            self.is_available = False
            self.error_message = "MATR networks not available"
    
    def create_model(self):
        """Create MATR model."""
        return MATRModel(in_channel=2)  # MATR takes 2-channel input (concatenated images)
    
    def prepare_input(self, image1: np.ndarray, image2: np.ndarray) -> torch.Tensor:
        """Prepare concatenated input for MATR model."""
        # Convert to tensors
        image1_tensor = torch.from_numpy(image1).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        image2_tensor = torch.from_numpy(image2).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Concatenate along channel dimension
        input_tensor = torch.cat([image1_tensor, image2_tensor], dim=1)  # [1, 2, H, W]
        return input_tensor.to(self.device)
    
    def fuse(self, image1: np.ndarray, image2: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images using MATR model."""
        if not self.is_available:
            return None
        
        try:
            with torch.no_grad():
                # Prepare concatenated input tensor
                input_tensor = self.prepare_input(image1, image2)
                
                # MATR takes concatenated input
                fused_tensor = self.model(input_tensor)
                return np.clip(self.tensor_to_numpy(fused_tensor), 0, 1)
        except Exception as e:
            print(f"MATR fusion error: {e}")
            return None