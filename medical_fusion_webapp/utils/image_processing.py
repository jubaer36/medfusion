"""
Image Processing Utilities
==========================

Utility functions for image preprocessing and postprocessing.
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional


def preprocess_image(image_file, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Preprocess uploaded image file.
    
    Args:
        image_file: Uploaded image file
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed image as numpy array with values in [0, 1]
    """
    # Read image
    image = Image.open(image_file).convert('L')  # Convert to grayscale
    
    # Resize
    image = image.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    return img_array


def load_image_from_path(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        target_size: Optional target size (height, width)
        
    Returns:
        Image as numpy array with values in [0, 1]
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize if requested
    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img


def array_to_base64(img_array: np.ndarray) -> str:
    """
    Convert numpy array to base64 string for web display.
    
    Args:
        img_array: Image array with values in [0, 1]
        
    Returns:
        Base64 encoded image string
    """
    # Convert to 8-bit
    img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    
    # Convert to PIL Image
    img_pil = Image.fromarray(img_uint8, mode='L')
    
    # Convert to base64
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        img: Input image
        
    Returns:
        Normalized image
    """
    img_min = img.min()
    img_max = img.max()
    
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    else:
        return img


def enhance_contrast(img: np.ndarray, alpha: float = 1.2, beta: float = 0.0) -> np.ndarray:
    """
    Enhance image contrast.
    
    Args:
        img: Input image
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (0-100)
        
    Returns:
        Enhanced image
    """
    enhanced = alpha * img + beta
    return np.clip(enhanced, 0, 1)


def apply_histogram_equalization(img: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to enhance contrast.
    
    Args:
        img: Input image with values in [0, 1]
        
    Returns:
        Equalized image
    """
    # Convert to uint8
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Apply histogram equalization
    equalized = cv2.equalizeHist(img_uint8)
    
    # Convert back to float
    return equalized.astype(np.float32) / 255.0


def resize_image(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        img: Input image
        target_size: Target size (height, width)
        
    Returns:
        Resized image
    """
    return cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)


def validate_image_pair(ct: np.ndarray, mri: np.ndarray) -> bool:
    """
    Validate that CT and MRI images are compatible for fusion.
    
    Args:
        ct: CT image
        mri: MRI image
        
    Returns:
        True if images are compatible, False otherwise
    """
    # Check if both images exist
    if ct is None or mri is None:
        return False
    
    # Check if shapes match
    if ct.shape != mri.shape:
        return False
    
    # Check if images have valid values
    if not (0 <= ct.min() and ct.max() <= 1):
        return False
    
    if not (0 <= mri.min() and mri.max() <= 1):
        return False
    
    return True