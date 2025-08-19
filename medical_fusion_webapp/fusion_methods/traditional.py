"""
Traditional Fusion Methods
==========================

Implementation of classical image fusion techniques that don't require training.
"""

import numpy as np
from typing import Optional

from .base import TraditionalFusion


class AverageFusion(TraditionalFusion):
    """Simple pixel-wise average fusion."""
    
    def __init__(self):
        super().__init__(
            name="Average Fusion",
            description="Simple pixel-wise averaging of CT and MRI images"
        )
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images using pixel-wise averaging."""
        try:
            return 0.5 * (ct + mri)
        except Exception as e:
            print(f"Average fusion error: {e}")
            return None


class MaximumFusion(TraditionalFusion):
    """Pixel-wise maximum selection fusion."""
    
    def __init__(self):
        super().__init__(
            name="Maximum Fusion",
            description="Pixel-wise maximum selection between CT and MRI images"
        )
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images using pixel-wise maximum."""
        try:
            return np.maximum(ct, mri)
        except Exception as e:
            print(f"Maximum fusion error: {e}")
            return None


class WeightedFusion(TraditionalFusion):
    """Weighted linear combination fusion."""
    
    def __init__(self, ct_weight: float = 0.6):
        super().__init__(
            name=f"Weighted Fusion ({ct_weight:.1f})",
            description=f"Weighted linear combination (CT: {ct_weight:.1f}, MRI: {1-ct_weight:.1f})"
        )
        self.ct_weight = ct_weight
        self.mri_weight = 1.0 - ct_weight
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images using weighted combination."""
        try:
            return self.ct_weight * ct + self.mri_weight * mri
        except Exception as e:
            print(f"Weighted fusion error: {e}")
            return None


class GradientBasedFusion(TraditionalFusion):
    """Gradient-based fusion selecting pixels with higher gradients."""
    
    def __init__(self):
        super().__init__(
            name="Gradient-Based Fusion",
            description="Selects pixels based on local gradient magnitude"
        )
    
    def _calculate_gradient(self, image: np.ndarray) -> np.ndarray:
        """Calculate gradient magnitude using Sobel operators."""
        import cv2
        
        # Convert to uint8 for gradient calculation
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Calculate gradients
        grad_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate magnitude and normalize
        gradient = np.sqrt(grad_x**2 + grad_y**2) / 255.0
        return gradient
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images based on gradient magnitude."""
        try:
            # Calculate gradients
            ct_grad = self._calculate_gradient(ct)
            mri_grad = self._calculate_gradient(mri)
            
            # Create selection mask (1 where CT has higher gradient, 0 otherwise)
            mask = (ct_grad >= mri_grad).astype(np.float32)
            
            # Fuse based on gradient
            fused = mask * ct + (1 - mask) * mri
            
            return fused
        except Exception as e:
            print(f"Gradient-based fusion error: {e}")
            return None


class LaplacianPyramidFusion(TraditionalFusion):
    """Laplacian pyramid-based fusion."""
    
    def __init__(self, levels: int = 4):
        super().__init__(
            name="Laplacian Pyramid",
            description=f"Multi-resolution fusion using {levels}-level Laplacian pyramid"
        )
        self.levels = levels
    
    def _build_gaussian_pyramid(self, image: np.ndarray) -> list:
        """Build Gaussian pyramid."""
        import cv2
        
        pyramid = [image.copy()]
        for i in range(self.levels):
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid
    
    def _build_laplacian_pyramid(self, image: np.ndarray) -> list:
        """Build Laplacian pyramid."""
        import cv2
        
        gaussian_pyramid = self._build_gaussian_pyramid(image)
        laplacian_pyramid = []
        
        for i in range(len(gaussian_pyramid) - 1):
            # Expand the smaller image
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1], 
                               dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
            # Calculate Laplacian
            laplacian = gaussian_pyramid[i] - expanded
            laplacian_pyramid.append(laplacian)
        
        # Add the smallest Gaussian level
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid
    
    def _reconstruct_from_laplacian(self, laplacian_pyramid: list) -> np.ndarray:
        """Reconstruct image from Laplacian pyramid."""
        import cv2
        
        image = laplacian_pyramid[-1]  # Start with smallest level
        
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            # Expand and add Laplacian
            expanded = cv2.pyrUp(image, 
                               dstsize=(laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))
            image = expanded + laplacian_pyramid[i]
        
        return image
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images using Laplacian pyramid."""
        try:
            # Build Laplacian pyramids
            ct_pyramid = self._build_laplacian_pyramid(ct)
            mri_pyramid = self._build_laplacian_pyramid(mri)
            
            # Fuse each level (simple average for now)
            fused_pyramid = []
            for ct_level, mri_level in zip(ct_pyramid, mri_pyramid):
                fused_level = 0.5 * (ct_level + mri_level)
                fused_pyramid.append(fused_level)
            
            # Reconstruct fused image
            fused = self._reconstruct_from_laplacian(fused_pyramid)
            
            return np.clip(fused, 0, 1)
        except Exception as e:
            print(f"Laplacian pyramid fusion error: {e}")
            return None