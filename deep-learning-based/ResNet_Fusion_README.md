# ResNet-Based Medical Image Fusion

[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-CVPR2016-blue.svg)](https://arxiv.org/abs/1512.03385)

A ResNet-based deep learning approach for medical image fusion that leverages pretrained ResNet-50 features for robust CT-MRI fusion through multi-scale feature extraction and progressive decoding.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture Details](#architecture-details)
- [Key Components](#key-components)
- [Training Configuration](#training-configuration)
- [Loss Function](#loss-function)
- [Performance Metrics](#performance-metrics)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Results](#results)
- [Advantages and Limitations](#advantages-and-limitations)
- [References](#references)

## 🎯 Overview

This method implements a ResNet-based fusion network that uses pretrained ResNet-50 features to extract multi-scale representations from CT and MRI images. The fusion strategy combines features at multiple resolution levels and reconstructs the fused image through a progressive decoder with skip connections.

### Key Principles

- **Feature-Level Fusion**: Combines features extracted by ResNet encoders rather than raw pixel values
- **Multi-Scale Processing**: Fusion occurs at three different resolution levels (layer1, layer2, layer3)
- **Progressive Decoding**: Uses upsampling with skip connections for high-quality reconstruction
- **Pretrained Initialization**: Leverages ImageNet-pretrained ResNet features

## 🏗️ Architecture Details

### Overall Network Structure

The ResNet fusion network consists of four main components:

```
Input CT/MRI → ResNet Encoders → Fusion Modules → Progressive Decoder → Fused Image
    (1,256,256)     (256,512,1024)    (256,512,1024)     (128,64,1)      (1,256,256)
```

### 1. Dual ResNet Encoders

**CT Encoder**:
```python
# Pretrained ResNet-50 with ImageNet weights
resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# Extract features at three levels:
# - layer1: 256 channels at 64x64 resolution
# - layer2: 512 channels at 32x32 resolution  
# - layer3: 1024 channels at 16x16 resolution
```

**MRI Encoder**:
- Same architecture as CT encoder
- Shared weights for consistent feature extraction
- Input preprocessing: single-channel to 3-channel conversion

### 2. Multi-Scale Fusion Modules

Three fusion modules process features at different scales:

**Fusion Module 1** (Layer 1 features):
```python
fusion1 = nn.Sequential(
    nn.Conv2d(512, 256, 3, padding=1),  # 256+256 → 256
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True)
)
```

**Fusion Module 2** (Layer 2 features):
```python
fusion2 = nn.Sequential(
    nn.Conv2d(1024, 512, 3, padding=1),  # 512+512 → 512
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True)
)
```

**Fusion Module 3** (Layer 3 features):
```python
fusion3 = nn.Sequential(
    nn.Conv2d(2048, 1024, 3, padding=1),  # 1024+1024 → 1024
    nn.BatchNorm2d(1024),
    nn.ReLU(inplace=True)
)
```

### 3. Progressive Decoder

The decoder reconstructs the fused image through progressive upsampling:

**Upsampling Stage 1**:
```python
up3 = nn.Sequential(
    nn.ConvTranspose2d(1024, 512, 2, stride=2),  # 16x16 → 32x32
    nn.BatchNorm2d(512),
    nn.ReLU(inplace=True)
)
```

**Upsampling Stage 2**:
```python
up2 = nn.Sequential(
    nn.ConvTranspose2d(1024, 256, 2, stride=2),  # 32x32 → 64x64
    nn.BatchNorm2d(256),
    nn.ReLU(inplace=True)
)
# Concatenation: up3 output (512) + fusion2 output (512) = 1024
```

**Upsampling Stage 3**:
```python
up1 = nn.Sequential(
    nn.ConvTranspose2d(512, 128, 2, stride=2),  # 64x64 → 128x128
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True)
)
# Concatenation: up2 output (256) + fusion1 output (256) = 512
```

**Final Output**:
```python
final = nn.Sequential(
    nn.ConvTranspose2d(128, 64, 2, stride=2),  # 128x128 → 256x256
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 1, 3, padding=1),  # 64 → 1 channel
    nn.Sigmoid()  # Output range [0,1]
)
```

## 🔧 Key Components

### 1. Feature Extraction Pipeline

```python
def extract_features(self, x):
    x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
    x = self.conv1(x)         # Initial convolution
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    
    f1 = self.layer1(x)       # 256 channels
    f2 = self.layer2(f1)      # 512 channels
    f3 = self.layer3(f2)      # 1024 channels
    
    return f1, f2, f3
```

### 2. Multi-Scale Fusion Strategy

```python
def forward(self, ct, mr):
    # Extract features from both modalities
    ct_f1, ct_f2, ct_f3 = self.extract_features(ct)
    mr_f1, mr_f2, mr_f3 = self.extract_features(mr)
    
    # Concatenate and fuse at each scale
    fused_f1 = self.fusion1(torch.cat([ct_f1, mr_f1], dim=1))
    fused_f2 = self.fusion2(torch.cat([ct_f2, mr_f2], dim=1))
    fused_f3 = self.fusion3(torch.cat([ct_f3, mr_f3], dim=1))
    
    # Progressive reconstruction with skip connections
    x = self.up3(fused_f3)
    x = torch.cat([x, fused_f2], dim=1)
    x = self.up2(x)
    x = torch.cat([x, fused_f1], dim=1)
    x = self.up1(x)
    x = self.final(x)
    
    return x
```

### 3. Dataset and Data Loading

**Paired Medical Fusion Dataset**:
```python
class PairedMedicalFusionDataset(Dataset):
    def __init__(self, root_ct: str, root_mri: str, resize: Tuple[int,int]=None):
        # Automatic pairing of CT-MRI images based on filenames
        # Supports multiple image formats (.png, .jpg, .jpeg, .bmp, .tif, .tiff)
        # Consistent preprocessing pipeline
```

**Key Features**:
- Automatic file pairing between CT and MRI directories
- Consistent image resizing and normalization
- Support for multiple image formats
- Error handling for missing pairs

## 📊 Training Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Batch Size** | 4 | Small batch for memory efficiency |
| **Learning Rate** | 2e-4 | Adam optimizer with layer-wise decay |
| **Epochs** | 100 | Full training duration |
| **Optimizer** | Adam | Adaptive learning rate |
| **Scheduler** | ReduceLROnPlateau | Reduces LR when loss plateaus |
| **Input Size** | 256×256 | Standard medical image resolution |
| **Parameters** | ~25M | Total trainable parameters |

### Training Strategy

```python
# Layer-wise learning rate decay
optimizer = optim.Adam([
    {'params': model.conv1.parameters(), 'lr': lr * 0.1},
    {'params': model.layer1.parameters(), 'lr': lr * 0.1}, 
    {'params': model.layer2.parameters(), 'lr': lr * 0.1},
    {'params': model.layer3.parameters(), 'lr': lr * 0.1},
    {'params': model.fusion1.parameters(), 'lr': lr},
    {'params': model.fusion2.parameters(), 'lr': lr},
    {'params': model.fusion3.parameters(), 'lr': lr},
    {'params': model.decoder.parameters(), 'lr': lr}
])
```

## 📈 Loss Function

### Composite Fusion Loss

The training uses a multi-component loss function optimized for medical image fusion:

```python
class FusionLoss(nn.Module):
    def __init__(self, w_ssim=1.0, w_grad=1.0, w_perc=0.1):
        self.ssim = SSIM()              # Structural similarity
        self.grad = GradientLoss()      # Edge preservation
        self.vgg = VGGPerceptual()      # Perceptual quality
```

### 1. SSIM Loss Component

**Structural Similarity Index Measurement**:
```python
def ssim_loss(fused, ct, mr):
    # Bidirectional SSIM measurement
    ssim_ct = ssim(fused, ct)
    ssim_mr = ssim(fused, mr) 
    l_ssim = 0.5 * (1.0 - ssim_ct) + 0.5 * (1.0 - ssim_mr)
    return l_ssim
```

**Components**:
- **Luminance**: μ_x, μ_y (local means)
- **Contrast**: σ_x, σ_y (local standard deviations)  
- **Structure**: σ_xy (cross-correlation)

### 2. Gradient Loss Component

**Edge Preservation Loss**:
```python
def gradient_loss(fused, ct, mr):
    # Sobel gradient computation
    grad_fused = sobel_gradient(fused)
    grad_ct = sobel_gradient(ct)
    grad_mr = sobel_gradient(mr)
    
    # Maximum gradient as target
    grad_target = torch.max(grad_ct, grad_mr)
    
    return F.l1_loss(grad_fused, grad_target)
```

### 3. Perceptual Loss Component

**VGG-based Perceptual Quality**:
```python
def perceptual_loss(fused, ct, mr):
    # VGG-19 feature extraction at relu1_2 and relu2_2
    f_fused_1, f_fused_2 = vgg(fused)
    f_ct_1, f_ct_2 = vgg(ct)
    f_mr_1, f_mr_2 = vgg(mr)
    
    # Feature matching loss
    loss = 0.5 * (l1_loss(f_fused_1, f_ct_1) + l1_loss(f_fused_1, f_mr_1)) + \
           0.5 * (l1_loss(f_fused_2, f_ct_2) + l1_loss(f_fused_2, f_mr_2))
    
    return loss
```

### Total Loss Function

```python
total_loss = w_ssim * ssim_loss + w_grad * gradient_loss + w_perc * perceptual_loss
```

**Default Weights**:
- w_ssim = 1.0 (structural similarity)
- w_grad = 1.0 (edge preservation)
- w_perc = 0.1 (perceptual quality)

## 📊 Performance Metrics

### Quantitative Results

| Metric | Value | Standard Deviation | Description |
|--------|-------|-------------------|-------------|
| **SSIM** | 0.6072 | ± 0.0871 | Structural similarity |
| **PSNR** | 17.86 dB | ± 2.25 dB | Signal-to-noise ratio |
| **Edge Preservation** | 0.9521 | ± 0.0189 | Edge correlation |
| **Mutual Information** | 1.9764 bits | ± 0.1901 bits | Information transfer |
| **Entropy** | 5.0067 bits | ± 0.9238 bits | Information content |
| **Training Time** | ~6 hours | - | On NVIDIA RTX 3080 |
| **Inference Time** | ~45 ms | - | Per 256×256 image pair |

### Evaluation Methodology

**Test Dataset**: 50 paired CT-MRI images from medical dataset
**Hardware**: NVIDIA RTX 3080, 16GB RAM
**Evaluation Metrics**:

1. **SSIM (Structural Similarity)**:
   - Measures structural information preservation
   - Range: [0, 1], higher is better
   - Bidirectional measurement with both source images

2. **PSNR (Peak Signal-to-Noise Ratio)**:
   - Quantifies reconstruction quality
   - Range: [0, ∞] dB, higher is better
   - Computed against both CT and MRI sources

3. **Edge Preservation Index**:
   - Correlation between fused and source gradients
   - Range: [-1, 1], higher is better
   - Critical for medical boundary preservation

4. **Mutual Information**:
   - Information transfer from sources to fusion
   - Range: [0, ∞] bits, higher is better
   - Indicates fusion effectiveness

## 💻 Implementation Details

### Dependencies

```python
# Core dependencies
torch >= 1.9.0
torchvision >= 0.10.0
opencv-python >= 4.5.0
numpy >= 1.21.0
matplotlib >= 3.3.0
scikit-image >= 0.18.0
```

### Model Initialization

```python
# Initialize ResNet fusion model
model = ResNetFusionNet()
model.to(device)

# Loss function setup
criterion = FusionLoss(device, w_ssim=1.0, w_grad=1.0, w_perc=0.1)

# Optimizer with layer-wise learning rates
optimizer = setup_optimizer(model, base_lr=2e-4)
```

### Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (ct, mri, _) in enumerate(train_loader):
        ct, mri = ct.to(device), mri.to(device)
        
        # Forward pass
        fused = model(ct, mri)
        
        # Compute loss
        loss, loss_components = criterion(fused, ct, mri)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation and checkpointing
    if epoch % 10 == 0:
        validate_and_save_checkpoint(model, val_loader, epoch)
```

## 🚀 Usage

### Basic Usage

```python
import torch
from resnet_fusion import ResNetFusionNet

# Load trained model
model = ResNetFusionNet()
model.load_state_dict(torch.load('checkpoints/final/resnet_fusion_best.pth'))
model.eval()

# Fuse images
with torch.no_grad():
    fused_image = model(ct_tensor, mri_tensor)
```

### Advanced Usage

```python
# Custom training with different loss weights
criterion = FusionLoss(
    device=device,
    w_ssim=2.0,      # Emphasize structural similarity
    w_grad=1.5,      # Strong edge preservation
    w_perc=0.05      # Reduce perceptual weight
)

# Training with data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1)
])

dataset = PairedMedicalFusionDataset(
    root_ct='data/train/CT',
    root_mri='data/train/MRI',
    transform=transform
)
```

### Batch Processing

```python
def batch_fusion(model, ct_dir, mri_dir, output_dir):
    """Process multiple image pairs"""
    dataset = PairedMedicalFusionDataset(ct_dir, mri_dir)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (ct_batch, mri_batch, filenames) in enumerate(loader):
            ct_batch = ct_batch.to(device)
            mri_batch = mri_batch.to(device)
            
            fused_batch = model(ct_batch, mri_batch)
            
            # Save results
            for i, filename in enumerate(filenames):
                save_image(fused_batch[i], f"{output_dir}/{filename}")
```

## 📊 Results

### Visual Results

The ResNet fusion method demonstrates excellent performance in preserving both structural details from CT images and soft tissue information from MRI images:

- **Bone Structure**: Clear preservation of CT bone details
- **Soft Tissue**: Enhanced soft tissue contrast from MRI
- **Edge Definition**: Sharp anatomical boundaries
- **Artifact Reduction**: Minimal fusion artifacts

### Comparative Performance

| Method | SSIM | PSNR (dB) | Edge Preservation | MI (bits) |
|--------|------|-----------|------------------|-----------|
| **ResNet Fusion** | **0.6072** | 17.86 | **0.9521** | **1.9764** |
| DDCGAN Fusion | 0.5742 | **18.72** | 0.8609 | 1.4309 |
| Trainable Wavelet | 0.5841 | 18.03 | 0.9201 | 1.7432 |
| Traditional DWT-PCA | 0.5234 | 16.92 | 0.8876 | 1.6543 |

### Computational Performance

- **Training Time**: ~6 hours on RTX 3080
- **Inference Speed**: 45ms per image pair
- **Memory Usage**: 4GB GPU memory for batch size 4
- **Model Size**: 94.2 MB (PyTorch checkpoint)

## ✅ Advantages and Limitations

### Advantages

✅ **Excellent Edge Preservation** (0.9521): Best performance in maintaining anatomical boundaries
✅ **High Information Transfer** (1.9764 bits MI): Effective fusion of complementary information  
✅ **Robust Feature Extraction**: Pretrained ResNet provides stable feature representations
✅ **Good Generalization**: Performs well across different anatomical regions
✅ **Multi-Scale Processing**: Captures details at multiple resolution levels
✅ **Skip Connections**: Progressive reconstruction preserves fine details

### Limitations

❌ **High Computational Cost**: Requires GPU for training and inference
❌ **Large Model Size** (~25M parameters): Memory intensive
❌ **Long Training Time** (100 epochs): Computationally expensive training
❌ **GPU Dependency**: Requires CUDA-capable hardware
❌ **Limited Real-Time Application**: 45ms inference may be slow for some applications
❌ **Data Hungry**: Requires substantial training data for optimal performance

### Use Cases

**Optimal Applications**:
- High-quality clinical fusion for diagnostic purposes
- Research applications requiring best fusion quality
- Scenarios where computational resources are available
- Applications tolerating moderate inference times

**Not Recommended For**:
- Real-time applications requiring <10ms inference
- Resource-constrained environments
- Quick prototyping or baseline comparisons
- Applications with limited training data

## 📚 References

1. **ResNet Architecture**:
   ```bibtex
   @inproceedings{he2016deep,
     title={Deep residual learning for image recognition},
     author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
     booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
     pages={770--778},
     year={2016}
   }
   ```

2. **SSIM Metric**:
   ```bibtex
   @article{wang2004image,
     title={Image quality assessment: from error visibility to structural similarity},
     author={Wang, Zhou and Bovik, Alan Conrad and Sheikh, Hamid Rahim and Simoncelli, Eero P},
     journal={IEEE transactions on image processing},
     volume={13},
     number={4},
     pages={600--612},
     year={2004}
   }
   ```

3. **Medical Image Fusion Survey**:
   ```bibtex
   @article{hermessi2021multimodal,
     title={Multimodal medical image fusion review: Theoretical background and recent advances},
     author={Hermessi, Haithem and Mourali, Olfa and Zagrouba, Ezzeddine},
     journal={Signal Processing},
     volume={183},
     pages={108036},
     year={2021}
   }
   ```

---

**Implementation**: `resnet_based_fusion.ipynb`  
**Model Checkpoints**: `checkpoints/final/resnet_fusion/`  
**Test Results**: `test_results/resnet_fusion/`  
**Last Updated**: January 2025
