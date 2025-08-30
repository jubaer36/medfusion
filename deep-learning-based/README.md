# Deep Learning Based Medical Image Fusion

This folder contains deep learning-based methods for multimodal medical image fusion, specifically designed for CT-MRI image fusion using neural networks.

## Folder Structure

```
deep-learning-based/
├── resnet_based_fusion.ipynb          # ResNet-based feature fusion method
├── trainable-wavelet-fusion.ipynb     # Trainable wavelet fusion with spatial adaptation
├── checkpoints/                       # Model checkpoints directory
│   ├── intermediate/                  # Intermediate checkpoints (every N epochs)
│   │   ├── resnet_fusion/            # ResNet intermediate checkpoints
│   │   └── trainable_wavelet_fusion/ # Wavelet intermediate checkpoints
│   └── final/                        # Final best models
│       ├── resnet_fusion/            # ResNet best model
│       └── trainable_wavelet_fusion/ # Wavelet best model
├── results/                           # Directory for fusion results
│   ├── resnet_fusion/                # Results from ResNet fusion method
│   └── trainable_wavelet_fusion/     # Results from trainable wavelet fusion
└── README.md                         # This file
```

## Methods Implemented

### 1. ResNet-Based Fusion (`resnet_based_fusion.ipynb`)
- **Method**: ResNet feature extraction with trainable fusion layers
- **Architecture**: Pretrained ResNet backbone with custom fusion modules
- **Features**:
  - Feature-level fusion using deep neural networks
  - Multi-scale feature extraction and fusion
  - Comprehensive loss function (SSIM + Gradient + Perceptual)
  - End-to-end trainable fusion pipeline
- **Output Naming**: Model checkpoints saved as `resnet_fusion_best.pt`

### 2. Trainable Wavelet Fusion (`trainable-wavelet-fusion.ipynb`)
- **Method**: Wavelet transform with trainable spatial adaptation masks
- **Architecture**: CNN-based fusion masks for wavelet coefficients
- **Features**:
  - Differentiable wavelet transforms
  - Spatial-adaptive fusion weights
  - Multi-resolution processing
  - Enhanced loss functions for medical imaging
- **Output Naming**:
  - Best model: `wavelet_fusion_spatial_best.pt`
  - Epoch checkpoints: `wavelet_fusion_spatial_epoch_X.pt`
  - Final model: `wavelet_fusion_spatial_final.pt`
  - Training history: `training_history.pt`

## Dataset Structure

Both methods expect the dataset to be organized as follows:

```
Dataset/
├── train/CT/     # Training CT images
├── train/MRI/    # Training MRI images
├── val/CT/       # Validation CT images
├── val/MRI/      # Validation MRI images
├── test/CT/      # Test CT images
├── test/MRI/     # Test MRI images
└── split_info.json  # Dataset split metadata
```

## Usage Instructions

### Running ResNet-Based Fusion

1. Open `resnet_based_fusion.ipynb` in Jupyter notebook
2. Update the Config class if needed:
   ```python
   @dataclass
   class Config:
       ct_dir: str = '../Dataset/train/CT'
       mri_dir: str = '../Dataset/train/MRI'
       ct_val_dir: str = '../Dataset/val/CT'
       mri_val_dir: str = '../Dataset/val/MRI'
       save_dir: str = 'results/resnet_fusion'
   ```
3. Run all cells sequentially
4. Model checkpoints will be saved in `results/resnet_fusion/`

### Running Trainable Wavelet Fusion

1. Open `trainable-wavelet-fusion.ipynb` in Jupyter notebook
2. Update the dataset path:
   ```python
   dataset = CTMRIDataset('../Dataset/train/CT', '../Dataset/train/MRI')
   ```
3. Run all cells sequentially
4. Model checkpoints and training history will be saved in `results/trainable_wavelet_fusion/`

## Model Architecture

### ResNet Fusion Network
- **Backbone**: Pretrained ResNet-50 (frozen feature layers)
- **Fusion Modules**: Custom convolutional layers for feature fusion
- **Decoder**: Multi-scale upsampling with skip connections
- **Output**: Fused image at original resolution

### Trainable Wavelet Fusion Network
- **Wavelet Transform**: 1-level DWT/IDWT (Haar wavelets)
- **Fusion Masks**: CNN-generated spatial adaptation masks
- **Architecture**: Encoder-decoder with wavelet domain processing
- **Loss Function**: MSE + SSIM + Gradient loss

## Training Configuration

### ResNet Fusion
- **Batch Size**: 4
- **Learning Rate**: 2e-4 (with layer-wise decay)
- **Epochs**: 100
- **Optimizer**: Adam
- **Loss Weights**: SSIM=1.0, Gradient=0.5, Perceptual=0.1

### Trainable Wavelet Fusion
- **Batch Size**: 2
- **Learning Rate**: Adaptive (with scheduler)
- **Epochs**: 20
- **Optimizer**: Adam
- **Loss Function**: Multi-term loss for image quality

## Output Files

### Model Checkpoints
- **ResNet Intermediate**: `checkpoints/intermediate/resnet_fusion/resnet_fusion_epoch_XXX.pt`
- **ResNet Best**: `checkpoints/final/resnet_fusion/resnet_fusion_best.pt`
- **Wavelet Intermediate**: `checkpoints/intermediate/trainable_wavelet_fusion/wavelet_fusion_spatial_epoch_X.pt`
- **Wavelet Best**: `checkpoints/final/trainable_wavelet_fusion/wavelet_fusion_spatial_best.pt`
- **Wavelet Final**: `checkpoints/final/trainable_wavelet_fusion/wavelet_fusion_spatial_final.pt`
- **Wavelet History**: `checkpoints/final/trainable_wavelet_fusion/training_history.pt`

## Dependencies

### Required Python Packages
- torch
- torchvision
- pytorch-wavelets
- opencv-python
- numpy
- matplotlib
- scikit-image
- pillow
- tqdm

### Installation
```bash
pip install torch torchvision pytorch-wavelets opencv-python numpy matplotlib scikit-image pillow tqdm
```

## Configuration

### Dataset Paths
Update the following paths in the notebooks if your dataset is located elsewhere:

```python
# For resnet_based_fusion.ipynb
cfg.ct_dir = '../Dataset/CT-MRI/CT'
cfg.mri_dir = '../Dataset/CT-MRI/MRI'

# For trainable-wavelet-fusion.ipynb
dataset = CTMRIDataset('../Dataset/CT-MRI/CT', '../Dataset/CT-MRI/MRI')
```

### Training Parameters
Adjust training parameters based on your requirements:

```python
# ResNet Fusion
cfg.batch_size = 4
cfg.epochs = 100
cfg.lr = 2e-4

# Wavelet Fusion
batch_size = 2
num_epochs = 20
```

## Performance Notes

- **ResNet Fusion**: Better for capturing semantic features, requires more compute
- **Wavelet Fusion**: Efficient with spatial adaptation, good for detail preservation
- **Recommended**: Use ResNet for high-quality fusion, Wavelet for faster training

## Citation

If you use these methods in your research, please cite the original papers and this implementation appropriately.
