"""
Evaluation Metrics for Image Fusion
===================================

Comprehensive set of metrics for evaluating fusion quality.
"""

import numpy as np
import cv2
from typing import Dict, Any
from skimage.metrics import structural_similarity as ssim


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index."""
    try:
        result = ssim(img1, img2, data_range=1.0)
        return float(result)
    except Exception:
        return 0.0


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    try:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        result = 20 * np.log10(1.0 / np.sqrt(mse))
        return float(result)
    except Exception:
        return 0.0


def calculate_entropy(img: np.ndarray) -> float:
    """Calculate image entropy."""
    try:
        img_uint8 = (img * 255).astype(np.uint8)
        hist, _ = np.histogram(img_uint8, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        result = -np.sum(hist * np.log2(hist))
        return float(result)
    except Exception:
        return 0.0


def calculate_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Calculate gradient magnitude using Sobel operators."""
    try:
        img_uint8 = (img * 255).astype(np.uint8)
        grad_x = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(grad_x**2 + grad_y**2) / 255.0
    except Exception:
        return np.zeros_like(img)


def calculate_edge_preservation(fused: np.ndarray, ct: np.ndarray, mri: np.ndarray) -> float:
    """Calculate edge preservation metric."""
    try:
        grad_fused = calculate_gradient_magnitude(fused)
        grad_ct = calculate_gradient_magnitude(ct)
        grad_mri = calculate_gradient_magnitude(mri)
        grad_max = np.maximum(grad_ct, grad_mri)
        
        correlation = np.corrcoef(grad_fused.flatten(), grad_max.flatten())[0, 1]
        result = correlation if not np.isnan(correlation) else 0.0
        return float(result)
    except Exception:
        return 0.0


def calculate_mutual_information(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate mutual information between two images."""
    try:
        img1_uint8 = (img1 * 255).astype(np.uint8)
        img2_uint8 = (img2 * 255).astype(np.uint8)
        
        hist_2d, _, _ = np.histogram2d(img1_uint8.flatten(), img2_uint8.flatten(), bins=256)
        hist_2d = hist_2d / hist_2d.sum()
        
        hist_1 = hist_2d.sum(axis=1)
        hist_2 = hist_2d.sum(axis=0)
        
        mi = 0.0
        for i in range(256):
            for j in range(256):
                if hist_2d[i, j] > 0 and hist_1[i] > 0 and hist_2[j] > 0:
                    mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (hist_1[i] * hist_2[j]))
        
        return float(mi)
    except Exception:
        return 0.0


def calculate_standard_deviation(img: np.ndarray) -> float:
    """Calculate standard deviation of image."""
    try:
        return float(np.std(img))
    except Exception:
        return 0.0


def calculate_mean_gradient(img: np.ndarray) -> float:
    """Calculate mean gradient magnitude."""
    try:
        grad = calculate_gradient_magnitude(img)
        return float(np.mean(grad))
    except Exception:
        return 0.0


def evaluate_fusion_comprehensive(fused: np.ndarray, ct: np.ndarray, mri: np.ndarray) -> Dict[str, Any]:
    """
    Comprehensive evaluation of fusion quality.
    
    Args:
        fused: Fused image
        ct: Original CT image
        mri: Original MRI image
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    if fused is None:
        return {}
    
    metrics = {}
    
    # Structural similarity
    metrics['SSIM_CT'] = calculate_ssim(fused, ct)
    metrics['SSIM_MRI'] = calculate_ssim(fused, mri)
    metrics['SSIM_Avg'] = (metrics['SSIM_CT'] + metrics['SSIM_MRI']) / 2
    
    # Peak signal-to-noise ratio
    metrics['PSNR_CT'] = calculate_psnr(fused, ct)
    metrics['PSNR_MRI'] = calculate_psnr(fused, mri)
    metrics['PSNR_Avg'] = (metrics['PSNR_CT'] + metrics['PSNR_MRI']) / 2
    
    # Edge preservation
    metrics['Edge_Preservation'] = calculate_edge_preservation(fused, ct, mri)
    
    # Information content
    metrics['Entropy'] = calculate_entropy(fused)
    metrics['Entropy_CT'] = calculate_entropy(ct)
    metrics['Entropy_MRI'] = calculate_entropy(mri)
    
    # Mutual information
    metrics['MI_CT'] = calculate_mutual_information(fused, ct)
    metrics['MI_MRI'] = calculate_mutual_information(fused, mri)
    metrics['MI_Avg'] = (metrics['MI_CT'] + metrics['MI_MRI']) / 2
    
    # Statistical measures
    metrics['Std_Dev'] = calculate_standard_deviation(fused)
    metrics['Mean_Gradient'] = calculate_mean_gradient(fused)
    
    # Basic statistics
    metrics['Mean'] = float(np.mean(fused))
    metrics['Min'] = float(np.min(fused))
    metrics['Max'] = float(np.max(fused))
    
    return metrics


def get_metric_descriptions() -> Dict[str, str]:
    """Get descriptions of all available metrics."""
    return {
        'SSIM_Avg': 'Average Structural Similarity Index (0-1, higher better)',
        'PSNR_Avg': 'Average Peak Signal-to-Noise Ratio (dB, higher better)',
        'Edge_Preservation': 'Edge preservation correlation (0-1, higher better)',
        'Entropy': 'Information entropy (bits, higher better)',
        'MI_Avg': 'Average Mutual Information (bits, higher better)',
        'Std_Dev': 'Standard deviation (measure of contrast)',
        'Mean_Gradient': 'Average gradient magnitude (edge strength)'
    }