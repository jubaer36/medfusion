# Traditional Approaches for Medical Image Fusion

This folder contains traditional (non-deep learning) methods for multimodal medical image fusion, specifically designed for CT-MRI image fusion.

## Folder Structure

```
traditional-approaches/
├── dwt_pca.ipynb              # DWT + PCA based fusion method
├── wavelet-fusion.ipynb       # Traditional wavelet fusion with multiple strategies
├── checkpoints/               # Directory for saving model checkpoints
├── results/                   # Directory for fusion results
│   ├── dwt_pca/              # Results from DWT-PCA method
│   └── wavelet_fusion/       # Results from wavelet fusion method
└── README.md                 # This file
```

## Methods Implemented

### 1. DWT-PCA Fusion (`dwt_pca.ipynb`)
- **Method**: Discrete Wavelet Transform (DWT) with Principal Component Analysis (PCA)
- **Features**:
  - Two-level DWT decomposition
  - PCA-based fusion for approximation coefficients
  - Maximum selection for detail coefficients
  - Comprehensive quality metrics evaluation
- **Output Naming**: `fused_image_dwt_pca_{filename}.png`

### 2. Traditional Wavelet Fusion (`wavelet-fusion.ipynb`)
- **Method**: Multi-resolution wavelet-based fusion
- **Fusion Strategies**:
  - Maximum selection rule
  - Simple averaging
  - Weighted averaging (CT=0.6, MRI=0.4 for low frequency)
  - Activity-based fusion
- **Wavelet Types**: Haar, Daubechies (db4), Biorthogonal
- **Output Naming**: `fused_image_wavelet_fusion_{method}_{filename}.png`

## Dataset Structure

The methods expect the dataset to be organized as follows:

```
Dataset/
└── CT-MRI/
    ├── CT/
    │   ├── 16003.png
    │   ├── 16004.png
    │   └── ...
    └── MRI/
        ├── 16003.png
        ├── 16004.png
        └── ...
```

## Usage Instructions

### Running DWT-PCA Fusion

1. Open `dwt_pca.ipynb` in Jupyter notebook
2. Update the dataset paths if necessary (default: `../Dataset/CT-MRI/CT` and `../Dataset/CT-MRI/MRI`)
3. Run all cells sequentially
4. Results will be saved in `results/dwt_pca/` with filenames like:
   - `fused_image_dwt_pca_16003.png`
   - `metrics_dwt_pca_16003.txt`

### Running Wavelet Fusion

1. Open `wavelet-fusion.ipynb` in Jupyter notebook
2. Configure the parameters in the `Config` class:
   ```python
   cfg = Config(
       ct_dir='../Dataset/CT-MRI/CT',
       mri_dir='../Dataset/CT-MRI/MRI',
       output_dir='results/wavelet_fusion',
       wave='haar',                    # 'haar', 'db4', 'biorthogonal'
       fusion_method='max'            # 'max', 'average', 'weighted', 'activity'
   )
   ```
3. Run all cells sequentially
4. Results will be saved in `results/wavelet_fusion/` with filenames like:
   - `fused_image_wavelet_fusion_max_16003.png`
   - `fused_image_wavelet_fusion_average_16003.png`

## Quality Metrics

Both methods calculate comprehensive quality metrics:

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MSE**: Mean Squared Error
- **SD**: Standard Deviation
- **SNR**: Signal-to-Noise Ratio
- **Correlation**: Pearson correlation coefficient
- **NCC**: Normalized Cross Correlation
- **MD**: Maximum Difference
- **NAE**: Normalized Absolute Error
- **MI**: Mutual Information (wavelet fusion only)
- **Edge Variance**: Edge preservation measure (wavelet fusion only)

## Output Files

### Fused Images
- Format: PNG (8-bit grayscale)
- Naming: `fused_image_{method}_{filename}.png`
- Location: `results/{method}/`

### Metrics Files
- Format: Text file with key-value pairs
- Naming: `metrics_{method}_{filename}.txt`
- Location: `results/{method}/`

## Dependencies

### Required Python Packages
- numpy
- opencv-python (cv2)
- matplotlib
- scipy
- scikit-image
- scikit-learn
- torch
- pytorch-wavelets
- pandas
- seaborn

### Installation
```bash
pip install numpy opencv-python matplotlib scipy scikit-image scikit-learn torch pytorch-wavelets pandas seaborn
```

## Configuration

### Dataset Paths
Update the following paths in the notebooks if your dataset is located elsewhere:

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
```

## Troubleshooting

### Common Issues

1. **File not found errors**: Ensure dataset paths are correct relative to the notebook location
2. **Memory errors**: Reduce batch size or image resolution
3. **PyTorch Wavelets installation**: Install with `pip install pytorch-wavelets`
4. **CUDA errors**: The code automatically falls back to CPU if CUDA is not available

### Path Resolution
- Notebooks use relative paths (`../Dataset/`) assuming the project structure
- Adjust paths if you've moved the notebooks or reorganized the project

## Performance Notes

- **DWT-PCA**: Good for preserving complementary information, computationally efficient
- **Wavelet Fusion**: Flexible with multiple strategies, good for detail preservation
- **Recommended**: Use 'max' fusion method for medical images to preserve both CT bone details and MRI soft tissue information

## Citation

If you use these methods in your research, please cite the original papers and this implementation appropriately.
