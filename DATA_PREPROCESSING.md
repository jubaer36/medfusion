# Data Preprocessing Pipeline for Medical Image Fusion

This document provides comprehensive documentation of the data preprocessing pipeline used in the medical image fusion project, including dataset organization, preprocessing steps, augmentation strategies, and quality control measures.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset Structure](#dataset-structure)
- [Preprocessing Steps](#preprocessing-steps)
- [Data Augmentation](#data-augmentation)
- [Quality Control](#quality-control)
- [Implementation Details](#implementation-details)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

The data preprocessing pipeline is designed to prepare CT-MRI image pairs for fusion training and evaluation. The pipeline ensures data quality, consistency, and proper formatting for both deep learning and traditional fusion methods.

### Key Objectives

1. **Data Quality**: Ensure high-quality, artifact-free images
2. **Consistency**: Standardize image formats and properties
3. **Alignment**: Maintain spatial alignment between CT-MRI pairs
4. **Augmentation**: Increase dataset diversity for better training
5. **Validation**: Verify data integrity and pairing accuracy

## 📁 Dataset Structure

### Original Dataset Organization
```
OriginalDataset/
├── CT-MRI/
│   ├── CT/                    # 573 CT images
│   │   ├── 16003.png
│   │   ├── 16004.png
│   │   └── ...
│   └── MRI/                   # 573 MRI images
│       ├── 16003.png
│       ├── 16004.png
│       └── ...
├── Dataset.md                 # Dataset documentation
└── split_info.json           # Split metadata
```

### Processed Dataset Organization
```
Dataset/
├── train/                     # Training data (70%)
│   ├── CT/                    # 400 CT images
│   └── MRI/                   # 400 MRI images
├── val/                       # Validation data (20%)
│   ├── CT/                    # 115 CT images
│   └── MRI/                   # 115 MRI images
└── test/                      # Test data (10%)
    ├── CT/                    # 58 CT images
    └── MRI/                   # 58 MRI images
```

### Dataset Statistics
- **Total Images**: 1,146 (573 CT + 573 MRI)
- **Image Format**: PNG (8-bit grayscale)
- **Image Size**: 256×256 pixels
- **Storage**: ~117 MB
- **Split Ratio**: 70% train, 20% validation, 10% test

## 🔧 Preprocessing Steps

### 1. Image Loading and Validation

#### Loading Process
```python
def load_and_validate_images(ct_path: str, mri_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate CT-MRI image pair."""
    # Load images
    ct_img = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
    mri_img = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
    
    # Validate loading
    if ct_img is None or mri_img is None:
        raise ValueError(f"Could not load images: {ct_path}, {mri_path}")
    
    # Validate pairing
    if ct_img.shape != mri_img.shape:
        raise ValueError(f"Image size mismatch: CT {ct_img.shape}, MRI {mri_img.shape}")
    
    return ct_img, mri_img
```

#### Validation Checks
- **File Existence**: Verify both CT and MRI files exist
- **Image Format**: Ensure valid image format (PNG, JPG, etc.)
- **Size Matching**: Verify CT and MRI have identical dimensions
- **Data Integrity**: Check for corrupted or empty images

### 2. Image Normalization

#### Normalization Process
```python
def normalize_images(ct_img: np.ndarray, mri_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize images to [0, 1] range."""
    # Convert to float32
    ct_img = ct_img.astype(np.float32)
    mri_img = mri_img.astype(np.float32)
    
    # Normalize to [0, 1]
    if ct_img.max() > 1.0:
        ct_img /= 255.0
    if mri_img.max() > 1.0:
        mri_img /= 255.0
    
    return ct_img, mri_img
```

#### Normalization Strategies
1. **Min-Max Normalization**: `(x - min) / (max - min)`
2. **Standard Normalization**: `(x - mean) / std`
3. **Unit Range**: `x / 255.0` (for 8-bit images)

### 3. Image Resizing

#### Resizing Process
```python
def resize_images(ct_img: np.ndarray, mri_img: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Resize images to target size."""
    # Resize CT image
    ct_resized = cv2.resize(ct_img, target_size, interpolation=cv2.INTER_AREA)
    
    # Resize MRI image
    mri_resized = cv2.resize(mri_img, target_size, interpolation=cv2.INTER_AREA)
    
    return ct_resized, mri_resized
```

#### Resizing Considerations
- **Interpolation Method**: INTER_AREA for downsampling, INTER_CUBIC for upsampling
- **Aspect Ratio**: Maintain or modify aspect ratio based on requirements
- **Quality Preservation**: Choose appropriate interpolation method

### 4. Contrast Enhancement

#### Histogram Matching
```python
def match_histograms(ct_img: np.ndarray, mri_img: np.ndarray) -> np.ndarray:
    """Match CT histogram to MRI histogram."""
    from skimage.exposure import match_histograms
    
    # Match CT histogram to MRI
    ct_matched = match_histograms(ct_img, mri_img)
    
    return ct_matched
```

#### Contrast Adjustment
```python
def enhance_contrast(img: np.ndarray, alpha: float = 1.2, beta: float = 0.0) -> np.ndarray:
    """Enhance image contrast."""
    enhanced = alpha * img + beta
    return np.clip(enhanced, 0, 1)
```

### 5. Noise Reduction

#### Gaussian Filtering
```python
def apply_gaussian_filter(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian filter for noise reduction."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=sigma)
```

#### Bilateral Filtering
```python
def apply_bilateral_filter(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filter for edge-preserving smoothing."""
    img_uint8 = (img * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
    return filtered.astype(np.float32) / 255.0
```

## 🎨 Data Augmentation

### 1. Geometric Transformations

#### Rotation
```python
def random_rotation(ct_img: np.ndarray, mri_img: np.ndarray, max_angle: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random rotation to image pair."""
    angle = np.random.uniform(-max_angle, max_angle)
    
    # Get rotation matrix
    h, w = ct_img.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    ct_rotated = cv2.warpAffine(ct_img, rotation_matrix, (w, h))
    mri_rotated = cv2.warpAffine(mri_img, rotation_matrix, (w, h))
    
    return ct_rotated, mri_rotated
```

#### Flipping
```python
def random_flip(ct_img: np.ndarray, mri_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random flipping to image pair."""
    # Horizontal flip
    if np.random.random() > 0.5:
        ct_img = cv2.flip(ct_img, 1)
        mri_img = cv2.flip(mri_img, 1)
    
    # Vertical flip
    if np.random.random() > 0.5:
        ct_img = cv2.flip(ct_img, 0)
        mri_img = cv2.flip(mri_img, 0)
    
    return ct_img, mri_img
```

#### Scaling
```python
def random_scaling(ct_img: np.ndarray, mri_img: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random scaling to image pair."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    
    h, w = ct_img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    ct_scaled = cv2.resize(ct_img, (new_w, new_h))
    mri_scaled = cv2.resize(mri_img, (new_w, new_h))
    
    # Crop or pad to original size
    if scale > 1.0:
        # Crop center
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        ct_scaled = ct_scaled[start_h:start_h+h, start_w:start_w+w]
        mri_scaled = mri_scaled[start_h:start_h+h, start_w:start_w+w]
    else:
        # Pad with zeros
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        ct_scaled = np.pad(ct_scaled, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        mri_scaled = np.pad(mri_scaled, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    return ct_scaled, mri_scaled
```

### 2. Intensity Transformations

#### Brightness Adjustment
```python
def random_brightness(img: np.ndarray, brightness_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Apply random brightness adjustment."""
    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    return np.clip(img * brightness, 0, 1)
```

#### Contrast Adjustment
```python
def random_contrast(img: np.ndarray, contrast_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Apply random contrast adjustment."""
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    mean = np.mean(img)
    return np.clip((img - mean) * contrast + mean, 0, 1)
```

#### Gamma Correction
```python
def random_gamma(img: np.ndarray, gamma_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """Apply random gamma correction."""
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    return np.power(img, gamma)
```

### 3. Noise Addition

#### Gaussian Noise
```python
def add_gaussian_noise(img: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, noise_level, img.shape)
    return np.clip(img + noise, 0, 1)
```

#### Salt and Pepper Noise
```python
def add_salt_pepper_noise(img: np.ndarray, noise_prob: float = 0.01) -> np.ndarray:
    """Add salt and pepper noise to image."""
    noisy = img.copy()
    
    # Salt noise
    salt_mask = np.random.random(img.shape) < noise_prob / 2
    noisy[salt_mask] = 1.0
    
    # Pepper noise
    pepper_mask = np.random.random(img.shape) < noise_prob / 2
    noisy[pepper_mask] = 0.0
    
    return noisy
```

## 🔍 Quality Control

### 1. Image Quality Assessment

#### Sharpness Measurement
```python
def calculate_sharpness(img: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance."""
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return float(np.var(laplacian))
```

#### Contrast Measurement
```python
def calculate_contrast(img: np.ndarray) -> float:
    """Calculate image contrast using standard deviation."""
    return float(np.std(img))
```

#### Brightness Measurement
```python
def calculate_brightness(img: np.ndarray) -> float:
    """Calculate image brightness using mean value."""
    return float(np.mean(img))
```

### 2. Pairing Validation

#### Spatial Alignment Check
```python
def validate_spatial_alignment(ct_img: np.ndarray, mri_img: np.ndarray) -> bool:
    """Validate spatial alignment between CT-MRI pair."""
    # Check size matching
    if ct_img.shape != mri_img.shape:
        return False
    
    # Check for significant differences in anatomy
    diff = np.abs(ct_img - mri_img)
    if np.mean(diff) > 0.5:  # Threshold for misalignment
        return False
    
    return True
```

#### Intensity Distribution Check
```python
def validate_intensity_distribution(ct_img: np.ndarray, mri_img: np.ndarray) -> bool:
    """Validate intensity distribution of CT-MRI pair."""
    # Check for reasonable intensity ranges
    ct_mean = np.mean(ct_img)
    mri_mean = np.mean(mri_img)
    
    # CT typically has higher intensity values
    if ct_mean < 0.1 or ct_mean > 0.9:
        return False
    
    if mri_mean < 0.1 or mri_mean > 0.9:
        return False
    
    return True
```

### 3. Data Integrity Checks

#### File Integrity
```python
def check_file_integrity(file_path: str) -> bool:
    """Check file integrity."""
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        return img is not None and img.size > 0
    except Exception:
        return False
```

#### Dataset Completeness
```python
def check_dataset_completeness(ct_dir: str, mri_dir: str) -> Dict[str, Any]:
    """Check dataset completeness."""
    ct_files = set(os.listdir(ct_dir))
    mri_files = set(os.listdir(mri_dir))
    
    # Find missing pairs
    missing_ct = mri_files - ct_files
    missing_mri = ct_files - mri_files
    
    return {
        'total_pairs': len(ct_files & mri_files),
        'missing_ct': len(missing_ct),
        'missing_mri': len(missing_mri),
        'completeness': len(ct_files & mri_files) / max(len(ct_files), len(mri_files))
    }
```

## 💻 Implementation Details

### Dataset Class Implementation

```python
class CTMRIDataset(Dataset):
    """Dataset class for CT-MRI image pairs."""
    
    def __init__(self, ct_dir: str, mri_dir: str, transform=None, target_size: Tuple[int, int] = (256, 256)):
        self.ct_dir = Path(ct_dir)
        self.mri_dir = Path(mri_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Find matching pairs
        ct_files = sorted(list(self.ct_dir.glob('*.png')))
        mri_files = sorted(list(self.mri_dir.glob('*.png')))
        
        ct_names = {f.stem: f for f in ct_files}
        mri_names = {f.stem: f for f in mri_files}
        
        self.pairs = []
        for name in ct_names.keys():
            if name in mri_names:
                self.pairs.append((ct_names[name], mri_names[name]))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        ct_path, mri_path = self.pairs[idx]
        
        # Load images
        ct_img = cv2.imread(str(ct_path), cv2.IMREAD_GRAYSCALE)
        mri_img = cv2.imread(str(mri_path), cv2.IMREAD_GRAYSCALE)
        
        # Normalize
        ct_img = ct_img.astype(np.float32) / 255.0
        mri_img = mri_img.astype(np.float32) / 255.0
        
        # Resize
        ct_img = cv2.resize(ct_img, self.target_size, interpolation=cv2.INTER_AREA)
        mri_img = cv2.resize(mri_img, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Convert to tensors
        ct_tensor = torch.from_numpy(ct_img).unsqueeze(0)
        mri_tensor = torch.from_numpy(mri_img).unsqueeze(0)
        
        # Apply transforms
        if self.transform:
            ct_tensor, mri_tensor = self.transform(ct_tensor, mri_tensor)
        
        return ct_tensor, mri_tensor, ct_path.name
```

### Transform Implementation

```python
class MedicalImageTransform:
    """Transform class for medical image augmentation."""
    
    def __init__(self, 
                 rotation_range: float = 10.0,
                 flip_prob: float = 0.5,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 noise_level: float = 0.01):
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_level = noise_level
    
    def __call__(self, ct_tensor: torch.Tensor, mri_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply transforms to image pair."""
        # Convert to numpy
        ct_img = ct_tensor.squeeze().numpy()
        mri_img = mri_tensor.squeeze().numpy()
        
        # Apply transforms
        ct_img, mri_img = self._apply_transforms(ct_img, mri_img)
        
        # Convert back to tensors
        ct_tensor = torch.from_numpy(ct_img).unsqueeze(0)
        mri_tensor = torch.from_numpy(mri_img).unsqueeze(0)
        
        return ct_tensor, mri_tensor
    
    def _apply_transforms(self, ct_img: np.ndarray, mri_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply individual transforms."""
        # Random rotation
        if np.random.random() > 0.5:
            ct_img, mri_img = random_rotation(ct_img, mri_img, self.rotation_range)
        
        # Random flip
        if np.random.random() > self.flip_prob:
            ct_img, mri_img = random_flip(ct_img, mri_img)
        
        # Random brightness
        if np.random.random() > 0.5:
            ct_img = random_brightness(ct_img, self.brightness_range)
            mri_img = random_brightness(mri_img, self.brightness_range)
        
        # Random contrast
        if np.random.random() > 0.5:
            ct_img = random_contrast(ct_img, self.contrast_range)
            mri_img = random_contrast(mri_img, self.contrast_range)
        
        # Random noise
        if np.random.random() > 0.5:
            ct_img = add_gaussian_noise(ct_img, self.noise_level)
            mri_img = add_gaussian_noise(mri_img, self.noise_level)
        
        return ct_img, mri_img
```

## 🚀 Usage Examples

### Basic Usage

```python
# Initialize dataset
dataset = CTMRIDataset(
    ct_dir='Dataset/train/CT',
    mri_dir='Dataset/train/MRI',
    target_size=(256, 256)
)

# Create data loader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate through dataset
for ct_batch, mri_batch, keys in dataloader:
    print(f"Batch shape: CT {ct_batch.shape}, MRI {mri_batch.shape}")
    # Process batch...
```

### With Augmentation

```python
# Initialize transform
transform = MedicalImageTransform(
    rotation_range=10.0,
    flip_prob=0.5,
    brightness_range=(0.8, 1.2),
    contrast_range=(0.8, 1.2),
    noise_level=0.01
)

# Initialize dataset with transform
dataset = CTMRIDataset(
    ct_dir='Dataset/train/CT',
    mri_dir='Dataset/train/MRI',
    transform=transform,
    target_size=(256, 256)
)

# Create data loader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### Quality Control

```python
# Check dataset completeness
completeness = check_dataset_completeness('Dataset/train/CT', 'Dataset/train/MRI')
print(f"Dataset completeness: {completeness['completeness']:.2%}")

# Validate image pairs
for ct_path, mri_path in dataset.pairs[:10]:
    ct_img = cv2.imread(str(ct_path), cv2.IMREAD_GRAYSCALE)
    mri_img = cv2.imread(str(mri_path), cv2.IMREAD_GRAYSCALE)
    
    # Validate alignment
    is_aligned = validate_spatial_alignment(ct_img, mri_img)
    print(f"Pair {ct_path.name}: {'✓' if is_aligned else '✗'}")
```

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues**
   ```python
   # Reduce batch size
   batch_size = 2  # Instead of 4
   
   # Use smaller image size
   target_size = (128, 128)  # Instead of (256, 256)
   ```

2. **File Loading Errors**
   ```python
   # Check file paths
   assert os.path.exists(ct_path), f"CT file not found: {ct_path}"
   assert os.path.exists(mri_path), f"MRI file not found: {mri_path}"
   
   # Check file integrity
   assert check_file_integrity(ct_path), f"CT file corrupted: {ct_path}"
   assert check_file_integrity(mri_path), f"MRI file corrupted: {mri_path}"
   ```

3. **Size Mismatch**
   ```python
   # Ensure size matching
   if ct_img.shape != mri_img.shape:
       # Resize to match
       target_size = (min(ct_img.shape[0], mri_img.shape[0]), 
                      min(ct_img.shape[1], mri_img.shape[1]))
       ct_img = cv2.resize(ct_img, target_size)
       mri_img = cv2.resize(mri_img, target_size)
   ```

### Performance Optimization

1. **Efficient Loading**
   ```python
   # Use multiple workers
   dataloader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True)
   
   # Preload data
   dataset = CTMRIDataset(ct_dir, mri_dir, preload=True)
   ```

2. **Memory Management**
   ```python
   # Clear cache
   torch.cuda.empty_cache()
   
   # Use gradient checkpointing
   torch.utils.checkpoint.checkpoint(model, input)
   ```

3. **Data Caching**
   ```python
   # Cache processed data
   cache_dir = 'cache/processed_data'
   os.makedirs(cache_dir, exist_ok=True)
   
   # Save processed data
   torch.save(processed_data, f'{cache_dir}/batch_{i}.pt')
   ```

## 📚 References

1. **Image Processing**
   - Gonzalez, R., et al. "Digital image processing." Pearson 2018
   - Sonka, M., et al. "Image processing, analysis, and machine vision." Cengage Learning 2014

2. **Data Augmentation**
   - Shorten, C., et al. "A survey on image data augmentation for deep learning." Journal of Big Data 2019
   - Perez, L., et al. "The effectiveness of data augmentation in image classification using deep learning." arXiv 2017

3. **Medical Image Processing**
   - Bankman, I. "Handbook of medical image processing and analysis." Academic Press 2008
   - Rangayyan, R. "Biomedical image analysis." CRC Press 2004

---

**Last Updated**: January 2025
**Version**: 2.0.0
**Maintainer**: Data Processing Team
