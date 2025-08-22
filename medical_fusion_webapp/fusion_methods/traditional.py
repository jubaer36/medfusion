"""
Traditional Fusion Methods
==========================

Implementation of classical image fusion techniques that don't require training.
"""

import numpy as np
import cv2
import pywt
from sklearn.decomposition import PCA
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

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


class DWTPCAFusion(TraditionalFusion):
    """DWT-PCA based medical image fusion."""
    
    def __init__(self):
        super().__init__(
            name="DWT-PCA Fusion",
            description="Discrete Wavelet Transform with Principal Component Analysis fusion method for medical images"
        )
        self.wavelet = 'db4'
        self.mode = 'symmetric'
    
    def dwt_decomposition(self, image):
        """Perform 2-level DWT decomposition."""
        coeffs1 = pywt.dwt2(image, self.wavelet, mode=self.mode)
        cA1, (cH1, cV1, cD1) = coeffs1
        coeffs2 = pywt.dwt2(cA1, self.wavelet, mode=self.mode)
        cA2, (cH2, cV2, cD2) = coeffs2
        return {
            'cA2': cA2,
            'cH2': cH2, 'cV2': cV2, 'cD2': cD2,
            'cH1': cH1, 'cV1': cV1, 'cD1': cD1
        }
    
    def principal_component_averaging(self, coeffs1, coeffs2, block_size=8):
        """Apply PCA-based fusion to wavelet coefficients."""
        if coeffs1.shape != coeffs2.shape:
            min_h = min(coeffs1.shape[0], coeffs2.shape[0])
            min_w = min(coeffs1.shape[1], coeffs2.shape[1])
            coeffs1 = coeffs1[:min_h, :min_w]
            coeffs2 = coeffs2[:min_h, :min_w]
        
        h, w = coeffs1.shape
        fused_coeffs = np.zeros_like(coeffs1)
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block1 = coeffs1[i:i+block_size, j:j+block_size]
                block2 = coeffs2[i:i+block_size, j:j+block_size]
                
                data = np.column_stack([block1.flatten(), block2.flatten()])
                
                try:
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(data)
                    eigenvals = pca.explained_variance_
                    
                    # Check for valid eigenvalues and avoid division by zero
                    if len(eigenvals) >= 2 and eigenvals[0] + eigenvals[1] > 1e-10:
                        total_var = eigenvals[0] + eigenvals[1]
                        w1 = eigenvals[0] / total_var
                        w2 = eigenvals[1] / total_var
                        
                        # Ensure weights are valid numbers
                        if np.isnan(w1) or np.isnan(w2) or np.isinf(w1) or np.isinf(w2):
                            w1, w2 = 0.5, 0.5
                    else:
                        w1, w2 = 0.5, 0.5
                    
                    fused_block = w1 * block1 + w2 * block2
                    
                    # Final safety check for the fused block
                    if np.any(np.isnan(fused_block)) or np.any(np.isinf(fused_block)):
                        fused_block = 0.5 * (block1 + block2)
                        
                except Exception as e:
                    fused_block = 0.5 * (block1 + block2)
                
                fused_coeffs[i:i+block_size, j:j+block_size] = fused_block
        
        # Handle remaining blocks
        if h % block_size != 0 or w % block_size != 0:
            remaining_h = h - (h // block_size) * block_size
            remaining_w = w - (w // block_size) * block_size
            
            if remaining_h > 0:
                fused_coeffs[-remaining_h:, :] = 0.5 * (
                    coeffs1[-remaining_h:, :] + coeffs2[-remaining_h:, :]
                )
            if remaining_w > 0:
                fused_coeffs[:, -remaining_w:] = 0.5 * (
                    coeffs1[:, -remaining_w:] + coeffs2[:, -remaining_w:]
                )
        
        return fused_coeffs
    
    def maximum_selection_fusion(self, coeffs1, coeffs2):
        """Apply maximum selection rule for detail coefficients."""
        if coeffs1.shape != coeffs2.shape:
            min_h = min(coeffs1.shape[0], coeffs2.shape[0])
            min_w = min(coeffs1.shape[1], coeffs2.shape[1])
            coeffs1 = coeffs1[:min_h, :min_w]
            coeffs2 = coeffs2[:min_h, :min_w]
        
        mask = np.abs(coeffs1) >= np.abs(coeffs2)
        fused_coeffs = np.where(mask, coeffs1, coeffs2)
        return fused_coeffs
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images using DWT-PCA method."""
        try:
            # Ensure images are in proper format and size
            if ct.shape != mri.shape:
                h = min(ct.shape[0], mri.shape[0])
                w = min(ct.shape[1], mri.shape[1])
                ct = cv2.resize(ct, (w, h))
                mri = cv2.resize(mri, (w, h))
            
            # Crop to nearest multiple of 4 for both dimensions
            h, w = ct.shape
            h4, w4 = h - (h % 4), w - (w % 4)
            ct = ct[:h4, :w4]
            mri = mri[:h4, :w4]
            
            # Perform DWT decomposition
            mri_coeffs = self.dwt_decomposition(mri)
            ct_coeffs = self.dwt_decomposition(ct)
            
            # Fuse approximation coefficients using PCA
            fused_cA2 = self.principal_component_averaging(
                mri_coeffs['cA2'], ct_coeffs['cA2']
            )
            
            # Fuse detail coefficients using maximum selection
            fused_cH2 = self.maximum_selection_fusion(
                mri_coeffs['cH2'], ct_coeffs['cH2']
            )
            fused_cV2 = self.maximum_selection_fusion(
                mri_coeffs['cV2'], ct_coeffs['cV2']
            )
            fused_cD2 = self.maximum_selection_fusion(
                mri_coeffs['cD2'], ct_coeffs['cD2']
            )
            
            fused_cH1 = self.maximum_selection_fusion(
                mri_coeffs['cH1'], ct_coeffs['cH1']
            )
            fused_cV1 = self.maximum_selection_fusion(
                mri_coeffs['cV1'], ct_coeffs['cV1']
            )
            fused_cD1 = self.maximum_selection_fusion(
                mri_coeffs['cD1'], ct_coeffs['cD1']
            )
            
            # Reconstruct from level 2
            coeffs_level2 = (fused_cA2, (fused_cH2, fused_cV2, fused_cD2))
            reconstructed_cA1 = pywt.idwt2(coeffs_level2, self.wavelet, mode=self.mode)
            
            # Ensure all level 1 coefficients have the same size
            target_h, target_w = fused_cH1.shape
            if reconstructed_cA1.shape != (target_h, target_w):
                # Use interpolation that preserves valid values
                reconstructed_cA1 = cv2.resize(reconstructed_cA1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                
                # Check and fix any NaN/inf values that might arise from resizing
                if np.any(np.isnan(reconstructed_cA1)) or np.any(np.isinf(reconstructed_cA1)):
                    # Fallback to simple resizing without interpolation issues
                    from scipy import ndimage
                    zoom_h = target_h / reconstructed_cA1.shape[0]
                    zoom_w = target_w / reconstructed_cA1.shape[1]
                    reconstructed_cA1 = ndimage.zoom(reconstructed_cA1, (zoom_h, zoom_w), order=1)
            
            # Reconstruct final image
            coeffs_level1 = (reconstructed_cA1, (fused_cH1, fused_cV1, fused_cD1))
            fused_image = pywt.idwt2(coeffs_level1, self.wavelet, mode=self.mode)
            
            # Ensure the output is clean before clipping
            fused_image = np.nan_to_num(fused_image, nan=0.5, posinf=1.0, neginf=0.0)
            fused_image = np.clip(fused_image, 0, 1)
            
            return fused_image
            
        except Exception as e:
            print(f"DWT-PCA fusion error: {e}")
            return None