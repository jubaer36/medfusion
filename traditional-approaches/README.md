# Traditional Approaches for Medical Image Fusion

This folder contains traditional (non-deep learning) methods for multimodal medical image fusion, specifically designed for CT-MRI image fusion using signal processing and mathematical techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Methods Implemented](#methods-implemented)
- [Mathematical Foundations](#mathematical-foundations)
- [Implementation Details](#implementation-details)
- [Performance Analysis](#performance-analysis)
- [Usage Instructions](#usage-instructions)
- [Configuration](#configuration)
- [Dependencies](#dependencies)

## 🎯 Overview

Traditional fusion methods rely on mathematical transformations and signal processing techniques to combine information from different medical imaging modalities. These methods are computationally efficient, interpretable, and provide good baseline performance for comparison with deep learning approaches.

### Key Advantages of Traditional Methods

1. **Computational Efficiency**: Fast processing without GPU requirements
2. **Interpretability**: Clear mathematical foundations
3. **Robustness**: Stable performance across different datasets
4. **Baseline Performance**: Good reference for comparison

## 🔬 Methods Implemented

### 1. DWT-PCA Fusion (`dwt_pca.ipynb`)

#### Method Overview
- **Technique**: Discrete Wavelet Transform (DWT) + Principal Component Analysis (PCA)
- **Decomposition**: Two-level wavelet decomposition
- **Fusion Strategy**: PCA for approximation coefficients, maximum selection for detail coefficients
- **Performance**: SSIM: 0.5094, PSNR: 17.24 dB

#### Mathematical Foundation

**1. Wavelet Decomposition**
```
Level 1: Image → (cA1, cH1, cV1, cD1)
Level 2: cA1 → (cA2, cH2, cV2, cD2)
```

Where:
- `cA`: Approximation coefficients (low-frequency)
- `cH`: Horizontal detail coefficients
- `cV`: Vertical detail coefficients  
- `cD`: Diagonal detail coefficients

**2. PCA-Based Fusion for Approximation Coefficients**
```python
def principal_component_averaging(coeffs1, coeffs2, block_size=8):
    # For each block
    data = np.column_stack([block1.flatten(), block2.flatten()])
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    # Calculate weights from eigenvalues
    eigenvals = pca.explained_variance_
    w1 = eigenvals[0] / (eigenvals[0] + eigenvals[1])
    w2 = eigenvals[1] / (eigenvals[0] + eigenvals[1])
    
    # Weighted fusion
    fused_block = w1 * block1 + w2 * block2
```

**3. Maximum Selection for Detail Coefficients**
```python
def maximum_selection_fusion(coeffs1, coeffs2):
    mask = np.abs(coeffs1) >= np.abs(coeffs2)
    fused_coeffs = np.where(mask, coeffs1, coeffs2)
    return fused_coeffs
```

#### Implementation Details

**Class Structure:**
```python
class MedicalImageFusion:
    def __init__(self):
        self.wavelet = 'db4'  # Daubechies wavelet
        self.mode = 'symmetric'
    
    def dwt_decomposition(self, image):
        # Two-level DWT decomposition
        coeffs1 = pywt.dwt2(image, self.wavelet, mode=self.mode)
        cA1, (cH1, cV1, cD1) = coeffs1
        coeffs2 = pywt.dwt2(cA1, self.wavelet, mode=self.mode)
        cA2, (cH2, cV2, cD2) = coeffs2
        return {'cA2': cA2, 'cH2': cH2, 'cV2': cV2, 'cD2': cD2,
                'cH1': cH1, 'cV1': cV1, 'cD1': cD1}
    
    def fuse_images(self, ct_img, mri_img):
        # DWT decomposition
        ct_coeffs = self.dwt_decomposition(ct_img)
        mri_coeffs = self.dwt_decomposition(mri_img)
        
        # Fuse coefficients
        fused_coeffs = {}
        for key in ct_coeffs.keys():
            if key == 'cA2':  # Approximation coefficients
                fused_coeffs[key] = self.principal_component_averaging(
                    ct_coeffs[key], mri_coeffs[key]
                )
            else:  # Detail coefficients
                fused_coeffs[key] = self.maximum_selection_fusion(
                    ct_coeffs[key], mri_coeffs[key]
                )
        
        # Inverse DWT reconstruction
        fused_img = self.reconstruct_image(fused_coeffs)
        return fused_img
```

#### Performance Metrics
- **SSIM**: 0.5094 ± 0.0958
- **PSNR**: 17.24 ± 1.99 dB
- **Edge Preservation**: 0.7377 ± 0.1224
- **Mutual Information**: 1.4008 ± 0.3936 bits
- **Entropy**: 5.6452 ± 1.0147 bits

### 2. Traditional Wavelet Fusion (`wavelet-fusion.ipynb`)

#### Method Overview
- **Technique**: Multi-resolution wavelet-based fusion
- **Wavelet Types**: Haar, Daubechies (db4), Biorthogonal
- **Fusion Strategies**: Multiple fusion rules
- **Performance**: SSIM: 0.5953, PSNR: 17.68 dB

#### Fusion Strategies

**1. Maximum Selection Rule**
```python
def max_fusion(ct_band, mr_band):
    """Choose coefficient with larger absolute value"""
    ct_abs = torch.abs(ct_band)
    mr_abs = torch.abs(mr_band)
    mask = ct_abs >= mr_abs
    return torch.where(mask, ct_band, mr_band)
```

**2. Simple Averaging**
```python
def average_fusion(ct_band, mr_band):
    """Simple averaging of coefficients"""
    return 0.5 * (ct_band + mr_band)
```

**3. Weighted Averaging**
```python
def weighted_fusion(ct_band, mr_band, weight):
    """Weighted averaging with fixed weights"""
    return weight * ct_band + (1.0 - weight) * mr_band
```

**4. Activity-Based Fusion**
```python
def activity_fusion(ct_band, mr_band, window_size=3):
    """Activity measure based fusion using local variance"""
    def local_variance(x):
        # Compute local variance using unfold
        patches = x.unfold(2, window_size, 1).unfold(3, window_size, 1)
        patches_flat = patches.contiguous().view(B, C, H, W, -1)
        variance = torch.var(patches_flat, dim=-1, keepdim=False)
        return variance
    
    ct_activity = local_variance(ct_band)
    mr_activity = local_variance(mr_band)
    
    mask = ct_activity >= mr_activity
    return torch.where(mask, ct_band, mr_band)
```

#### Implementation Details

**Class Structure:**
```python
class TraditionalWaveletFusion(nn.Module):
    def __init__(self, wave='haar', fusion_method='max'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave)
        self.idwt = DWTInverse(wave=wave)
        self.fusion_method = fusion_method
        self.alpha_low = 0.6   # CT weight for low-frequency
        self.alpha_high = 0.5  # Balanced for high-frequency
    
    def forward(self, ct, mr):
        # Wavelet decomposition
        ct_low, ct_high = self.dwt(ct)
        mr_low, mr_high = self.dwt(mr)
        
        # Extract orientation bands
        ct_lh = ct_high[0][:, :, 0, :, :]
        ct_hl = ct_high[0][:, :, 1, :, :]
        ct_hh = ct_high[0][:, :, 2, :, :]
        mr_lh = mr_high[0][:, :, 0, :, :]
        mr_hl = mr_high[0][:, :, 1, :, :]
        mr_hh = mr_high[0][:, :, 2, :, :]
        
        # Apply fusion rules
        if self.fusion_method == 'max':
            fused_low = self._weighted_fusion(ct_low, mr_low, self.alpha_low)
            fused_lh = self._max_fusion(ct_lh, mr_lh)
            fused_hl = self._max_fusion(ct_hl, mr_hl)
            fused_hh = self._max_fusion(ct_hh, mr_hh)
        # ... other fusion methods
        
        # Reconstruct fused image
        fused_high = torch.stack([fused_lh, fused_hl, fused_hh], dim=2)
        fused = self.idwt((fused_low, [fused_high]))
        return fused
```

#### Wavelet Types Comparison

| Wavelet Type | Characteristics | Best For |
|--------------|----------------|----------|
| **Haar** | Simple, fast | Quick processing |
| **Daubechies (db4)** | Good frequency localization | General purpose |
| **Biorthogonal** | Linear phase | Symmetric features |

#### Performance Metrics
- **SSIM**: 0.5953 ± 0.0865
- **PSNR**: 17.68 ± 2.22 dB
- **Edge Preservation**: 0.9469 ± 0.0159
- **Mutual Information**: 1.5387 ± 0.1960 bits
- **Entropy**: 5.2310 ± 0.8794 bits

## 🧮 Mathematical Foundations

### Wavelet Transform Theory

**1. Continuous Wavelet Transform (CWT)**
```
W(a,b) = (1/√a) ∫ f(t) ψ*((t-b)/a) dt
```

Where:
- `a`: Scale parameter
- `b`: Translation parameter
- `ψ`: Mother wavelet function

**2. Discrete Wavelet Transform (DWT)**
```
W[j,k] = Σ f[n] ψ[j,k][n]
```

Where:
- `j`: Scale level
- `k`: Translation index

**3. Multi-Resolution Analysis**
```
V₀ = V₁ ⊕ W₁ = V₂ ⊕ W₂ ⊕ W₁ = ...
```

Where:
- `Vⱼ`: Approximation space at level j
- `Wⱼ`: Detail space at level j

### Principal Component Analysis (PCA)

**1. Covariance Matrix**
```
C = (1/n) XᵀX
```

**2. Eigenvalue Decomposition**
```
C = PΛPᵀ
```

**3. Principal Components**
```
Y = XP
```

### Fusion Quality Metrics

**1. Structural Similarity Index (SSIM)**
```
SSIM(x,y) = (2μₓμᵧ + c₁)(2σₓᵧ + c₂) / ((μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂))
```

**2. Peak Signal-to-Noise Ratio (PSNR)**
```
PSNR = 20 * log₁₀(MAX_I / √MSE)
```

**3. Mutual Information (MI)**
```
MI(X,Y) = Σ p(x,y) * log₂(p(x,y) / (p(x) * p(y)))
```

## 📊 Performance Analysis

### Comparative Performance

| Method | SSIM | PSNR (dB) | Edge Preservation | MI (bits) | Entropy (bits) |
|--------|------|-----------|-------------------|-----------|----------------|
| **Traditional Wavelet** | **0.5953** | **17.68** | **0.9469** | **1.5387** | 5.2310 |
| **DWT-PCA** | 0.5094 | 17.24 | 0.7377 | 1.4008 | 5.6452 |

### Performance Insights

1. **Best Overall**: Traditional Wavelet Fusion
2. **Best Edge Preservation**: Traditional Wavelet (0.9469)
3. **Best Information Transfer**: Traditional Wavelet (1.5387 bits MI)
4. **Most Consistent**: Traditional Wavelet (lower variance)

### Clinical Interpretation

- **SSIM > 0.5**: Acceptable structural preservation
- **Edge Preservation > 0.9**: Excellent detail retention
- **MI > 1.4**: Good information fusion
- **Entropy 5-6 bits**: Optimal information content

## 🚀 Usage Instructions

### Running DWT-PCA Fusion

```bash
# Open notebook
jupyter notebook dwt_pca.ipynb

# Update dataset paths
ct_dir = '../Dataset/CT-MRI/CT'
mri_dir = '../Dataset/CT-MRI/MRI'

# Run fusion
fusion = MedicalImageFusion()
fused_image = fusion.fuse_images(ct_img, mri_img)
```

### Running Traditional Wavelet Fusion

```bash
# Open notebook
jupyter notebook wavelet-fusion.ipynb

# Configure parameters
cfg = Config(
    ct_dir='../Dataset/CT-MRI/CT',
    mri_dir='../Dataset/CT-MRI/MRI',
    output_dir='results/wavelet_fusion',
    wave='haar',                    # 'haar', 'db4', 'biorthogonal'
    fusion_method='max'            # 'max', 'average', 'weighted', 'activity'
)

# Run fusion
wavelet_fusion = TraditionalWaveletFusion(wave='haar', fusion_method='max')
fused_image = wavelet_fusion(ct_tensor, mri_tensor)
```

### Batch Processing

```python
# Load dataset
ct_files = sorted(glob.glob('../Dataset/CT-MRI/CT/*.png'))
mri_files = sorted(glob.glob('../Dataset/CT-MRI/MRI/*.png'))

# Initialize fusion method
fusion = MedicalImageFusion()

# Process all pairs
results = []
for ct_path, mri_path in zip(ct_files, mri_files):
    # Load images
    ct_img = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
    mri_img = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
    
    # Normalize
    ct_img = ct_img.astype(np.float32) / 255.0
    mri_img = mri_img.astype(np.float32) / 255.0
    
    # Fuse
    fused_img = fusion.fuse_images(ct_img, mri_img)
    
    # Evaluate
    metrics = evaluate_fusion_quality(fused_img, ct_img, mri_img)
    results.append(metrics)
```

## ⚙️ Configuration

### Dataset Paths
Update the following paths in the notebooks:

```python
# For dwt_pca.ipynb
ct_dir = '../Dataset/CT-MRI/CT'
mri_dir = '../Dataset/CT-MRI/MRI'

# For wavelet-fusion.ipynb
cfg.ct_dir = '../Dataset/CT-MRI/CT'
cfg.mri_dir = '../Dataset/CT-MRI/MRI'
```

### Fusion Parameters
Adjust fusion parameters based on your requirements:

```python
# Wavelet type
wave = 'haar'  # 'haar', 'db4', 'biorthogonal'

# Fusion method
fusion_method = 'max'  # 'max', 'average', 'weighted', 'activity'

# Image resizing (optional)
resize = (256, 256)  # or None for original size

# DWT-PCA specific
block_size = 8  # PCA block size
wavelet = 'db4'  # Wavelet type
```

### Quality Metrics Configuration
```python
# Metrics to calculate
metrics_config = {
    'ssim': True,
    'psnr': True,
    'entropy': True,
    'mutual_information': True,
    'edge_preservation': True,
    'correlation': True
}
```

## 📦 Dependencies

### Required Python Packages
```bash
# Core packages
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.4.0
scipy>=1.7.0
scikit-image>=0.18.0
scikit-learn>=1.0.0

# Wavelet processing
pywavelets>=1.3.0
torch>=1.9.0
pytorch-wavelets>=1.3.0

# Data handling
pandas>=1.3.0
seaborn>=0.11.0
```

### Installation Commands
```bash
# Install core packages
pip install numpy opencv-python matplotlib scipy scikit-image scikit-learn

# Install wavelet packages
pip install pywavelets pytorch-wavelets

# Install PyTorch (for differentiable wavelets)
conda install pytorch torchvision -c pytorch
```

## 🔧 Troubleshooting

### Common Issues

1. **File not found errors**
   ```python
   # Ensure dataset paths are correct
   assert os.path.exists(ct_dir), f"CT directory not found: {ct_dir}"
   assert os.path.exists(mri_dir), f"MRI directory not found: {mri_dir}"
   ```

2. **Memory errors**
   ```python
   # Reduce batch size or image resolution
   batch_size = 1  # Instead of 4
   resize = (128, 128)  # Instead of (256, 256)
   ```

3. **PyTorch Wavelets installation**
   ```bash
   # Install with specific version
   pip install pytorch-wavelets==1.3.0
   ```

4. **CUDA errors**
   ```python
   # The code automatically falls back to CPU if CUDA is not available
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

### Performance Optimization

1. **Memory Optimization**
   - Process images in smaller batches
   - Use appropriate data types (float32 vs float64)
   - Clear intermediate variables

2. **Speed Optimization**
   - Use vectorized operations
   - Optimize wavelet transform parameters
   - Cache frequently used computations

3. **Quality Optimization**
   - Adjust fusion parameters
   - Try different wavelet types
   - Optimize block sizes for PCA

## 📚 References

1. **Wavelet Transform**
   - Mallat, S. "A wavelet tour of signal processing." Academic Press 1999
   - Daubechies, I. "Ten lectures on wavelets." SIAM 1992

2. **Image Fusion**
   - Li, S., et al. "Pixel-level image fusion: A survey of the state of the art." Information Fusion 2017
   - Pajares, G., et al. "A wavelet-based image fusion tutorial." Pattern Recognition 2004

3. **Principal Component Analysis**
   - Jolliffe, I. "Principal component analysis." Springer 2002
   - Abdi, H., et al. "Principal component analysis." Wiley Interdisciplinary Reviews 2010

4. **Quality Metrics**
   - Wang, Z., et al. "Image quality assessment: from error visibility to structural similarity." TIP 2004
   - Hossny, M., et al. "Image fusion: A survey of the state of the art." Information Fusion 2013

## 🤝 Contributing

We welcome contributions to improve the traditional methods:

1. **New Fusion Strategies**: Implement novel fusion rules
2. **Wavelet Types**: Add support for new wavelet families
3. **Quality Metrics**: Develop new evaluation metrics
4. **Optimization**: Improve computational efficiency
5. **Documentation**: Enhance code documentation

### Development Guidelines

1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings
3. Include mathematical explanations
4. Write unit tests
5. Update documentation

---

**Last Updated**: January 2025
**Version**: 2.0.0
**Maintainer**: Traditional Methods Team