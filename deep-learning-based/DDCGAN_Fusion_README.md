# DDCGAN-Based Medical Image Fusion

[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-GAN-blue.svg)](https://arxiv.org/abs/1406.2661)

A Generative Adversarial Network (GAN) approach for medical image fusion using a dual-discriminator architecture specifically designed for CT-MRI fusion tasks.

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

DDCGAN (Dual-Discriminator Conditional GAN) is a generative approach to medical image fusion that uses adversarial training to learn optimal fusion strategies. The method employs a generator network to produce fused images and dual discriminators to ensure both individual modality fidelity and fusion quality.

### Key Principles

- **Adversarial Training**: Generator and discriminators compete to improve fusion quality
- **Dual Discriminator**: Separate evaluation of CT and MRI fidelity
- **Conditional Generation**: Generator conditioned on both input modalities
- **Feature Matching**: Stabilizes training through intermediate feature alignment

## 🏗️ Architecture Details

### Overall Network Structure

```
Input (CT+MRI) → Generator → Fused Image
                     ↓
              [Discriminator_CT] ← CT Image
              [Discriminator_MRI] ← MRI Image
                     ↑
                Real/Fake Classification
```

### 1. Generator Architecture

**Encoder-Decoder Structure**:
```python
Generator:
Input: (2, 256, 256) - Concatenated CT+MRI
├── Encoder (5 blocks)
│   ├── Block1: 2→48 channels, stride=1
│   ├── Block2: 48→48 channels, stride=2  # Downsample
│   ├── Block3: 48→48 channels, stride=1
│   ├── Block4: 48→48 channels, stride=2  # Downsample  
│   └── Block5: 48→48 channels, stride=1
├── Bottleneck: 48 channels at 64×64
└── Decoder (5 layers)
    ├── ConvTranspose2d: 48→128, stride=1
    ├── ConvTranspose2d: 128→64, stride=2  # Upsample
    ├── ConvTranspose2d: 64→32, stride=1
    ├── ConvTranspose2d: 32→16, stride=2   # Upsample
    └── Conv2d: 16→1, Tanh activation
Output: (1, 256, 256) - Fused image
```

**Building Blocks**:
```python
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel=3):
        super().__init__()
        padding = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
```

**Complete Generator**:
```python
class Generator(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, feat=48):
        super().__init__()
        self.encoder = Encoder(in_channels, feat)
        self.decoder = Decoder(feat, out_channels)
    
    def forward(self, x):
        # x shape: (batch, 2, 256, 256)
        h = self.encoder(x)        # (batch, 48, 64, 64)
        out = self.decoder(h)      # (batch, 1, 256, 256)
        return out
```

### 2. Dual Discriminator Architecture

**Individual Discriminators for CT and MRI**:
```python
Discriminator:
Input: (1, 256, 256) - Individual modality
├── Layer1: 1→64 channels, stride=2    # 256→128
├── Layer2: 64→128 channels, stride=2  # 128→64
├── Layer3: 128→256 channels, stride=2 # 64→32
├── Layer4: 256→512 channels, stride=2 # 32→16
├── AdaptiveAvgPool2d(1)              # Global pooling
└── Linear: 512→1                     # Real/Fake score
Output: Scalar real/fake probability
```

**Implementation**:
```python
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, return_features=False):
        super().__init__()
        self.return_features = return_features
        
        # Feature layers
        layers = []
        layers += [nn.Conv2d(in_channels, 64, 4, 2, 1), 
                   nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(64, 128, 4, 2, 1), 
                   nn.BatchNorm2d(128), 
                   nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(128, 256, 4, 2, 1), 
                   nn.BatchNorm2d(256), 
                   nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(256, 512, 4, 2, 1), 
                   nn.BatchNorm2d(512), 
                   nn.LeakyReLU(0.2, inplace=True)]
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )
```

### 3. Training Architecture

**DDcGAN Trainer Class**:
```python
class DDcGANTrainer:
    def __init__(self, config):
        # Initialize networks
        self.generator = Generator(in_channels=2, out_channels=1, feat=48)
        self.disc_ct = Discriminator(in_channels=1, return_features=True)
        self.disc_mri = Discriminator(in_channels=1, return_features=True)
        
        # Optimizers with different learning rates
        self.optim_g = optim.Adam(self.generator.parameters(), 
                                  lr=2e-4, betas=(0.5, 0.999))
        self.optim_d_ct = optim.Adam(self.disc_ct.parameters(), 
                                     lr=1e-4, betas=(0.5, 0.999))
        self.optim_d_mri = optim.Adam(self.disc_mri.parameters(), 
                                      lr=1e-4, betas=(0.5, 0.999))
```

## 🔧 Key Components

### 1. Feature Matching Loss

Stabilizes training by matching intermediate discriminator features:

```python
def feature_matching_loss(fake_features, real_features):
    """
    Feature matching between generated and real images
    at intermediate discriminator layers
    """
    loss = 0.0
    for fake_feat, real_feat in zip(fake_features, real_features):
        loss += F.mse_loss(fake_feat.mean(0), real_feat.mean(0))
    return loss
```

### 2. Fusion Quality Loss

Ensures fused image preserves information from both modalities:

```python
def fusion_quality_loss(fused, ct, mri, lambda_ct=1.0, lambda_mri=1.0):
    """
    Fusion quality based on correlation with source images
    """
    # SSIM-based correlation losses
    ssim_ct = structural_similarity(fused, ct)
    ssim_mri = structural_similarity(fused, mri)
    
    # Information preservation loss
    loss = lambda_ct * (1.0 - ssim_ct) + lambda_mri * (1.0 - ssim_mri)
    
    return loss
```

### 3. Adversarial Loss

Standard GAN loss with dual discriminators:

```python
def adversarial_loss(real_scores, fake_scores, loss_type='lsgan'):
    """
    Adversarial loss for generator and discriminators
    """
    if loss_type == 'lsgan':
        # Least squares GAN loss for stability
        real_loss = F.mse_loss(real_scores, torch.ones_like(real_scores))
        fake_loss = F.mse_loss(fake_scores, torch.zeros_like(fake_scores))
        d_loss = 0.5 * (real_loss + fake_loss)
        g_loss = F.mse_loss(fake_scores, torch.ones_like(fake_scores))
    else:
        # Standard binary cross entropy
        real_loss = F.binary_cross_entropy_with_logits(
            real_scores, torch.ones_like(real_scores))
        fake_loss = F.binary_cross_entropy_with_logits(
            fake_scores, torch.zeros_like(fake_scores))
        d_loss = 0.5 * (real_loss + fake_loss)
        g_loss = F.binary_cross_entropy_with_logits(
            fake_scores, torch.ones_like(fake_scores))
    
    return d_loss, g_loss
```

## 📊 Training Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Batch Size** | 8 | Optimal for GAN stability |
| **Epochs** | 200 | Extended training for convergence |
| **Generator LR** | 2e-4 | Higher rate for generator |
| **Discriminator LR** | 1e-4 | Lower rate for discriminators |
| **Optimizer** | Adam | β₁=0.5, β₂=0.999 |
| **Loss Type** | LSGAN | Least squares for stability |
| **Feature Dim** | 48 | Generator feature channels |
| **Parameters** | ~1M | Compact model size |

### Training Strategy

**Two-Phase Training**:
1. **Discriminator Update**: Train both discriminators on real/fake pairs
2. **Generator Update**: Train generator to fool discriminators and preserve fusion quality

```python
def train_step(self, ct_batch, mri_batch):
    batch_size = ct_batch.size(0)
    
    # Concatenate inputs for generator
    inputs = torch.cat([ct_batch, mri_batch], dim=1)
    
    # === Train Discriminators ===
    self.optim_d_ct.zero_grad()
    self.optim_d_mri.zero_grad()
    
    # Generate fake images
    fake_images = self.generator(inputs).detach()
    
    # Discriminator CT
    real_ct_scores, real_ct_features = self.disc_ct(ct_batch)
    fake_ct_scores, fake_ct_features = self.disc_ct(fake_images)
    d_ct_loss, _ = adversarial_loss(real_ct_scores, fake_ct_scores)
    
    # Discriminator MRI  
    real_mri_scores, real_mri_features = self.disc_mri(mri_batch)
    fake_mri_scores, fake_mri_features = self.disc_mri(fake_images)
    d_mri_loss, _ = adversarial_loss(real_mri_scores, fake_mri_scores)
    
    # Update discriminators
    (d_ct_loss + d_mri_loss).backward()
    self.optim_d_ct.step()
    self.optim_d_mri.step()
    
    # === Train Generator ===
    self.optim_g.zero_grad()
    
    # Generate new fake images
    fake_images = self.generator(inputs)
    
    # Adversarial losses
    fake_ct_scores, fake_ct_features = self.disc_ct(fake_images)
    fake_mri_scores, fake_mri_features = self.disc_mri(fake_images)
    
    _, g_adv_ct = adversarial_loss(None, fake_ct_scores)
    _, g_adv_mri = adversarial_loss(None, fake_mri_scores)
    
    # Feature matching losses
    fm_ct = feature_matching_loss(fake_ct_features, real_ct_features)
    fm_mri = feature_matching_loss(fake_mri_features, real_mri_features)
    
    # Fusion quality loss
    fusion_loss = fusion_quality_loss(fake_images, ct_batch, mri_batch)
    
    # Total generator loss
    g_loss = g_adv_ct + g_adv_mri + 10.0 * (fm_ct + fm_mri) + 20.0 * fusion_loss
    
    # Update generator
    g_loss.backward()
    self.optim_g.step()
    
    return {
        'd_ct_loss': d_ct_loss.item(),
        'd_mri_loss': d_mri_loss.item(), 
        'g_loss': g_loss.item(),
        'fusion_loss': fusion_loss.item()
    }
```

## 📈 Loss Function

### Complete Loss Formulation

**Generator Loss**:
```
L_G = L_adv_CT + L_adv_MRI + λ_fm(L_fm_CT + L_fm_MRI) + λ_fusion * L_fusion

Where:
- L_adv_CT: Adversarial loss for CT discriminator
- L_adv_MRI: Adversarial loss for MRI discriminator  
- L_fm_CT: Feature matching loss for CT features
- L_fm_MRI: Feature matching loss for MRI features
- L_fusion: Fusion quality preservation loss
- λ_fm = 10.0: Feature matching weight
- λ_fusion = 20.0: Fusion quality weight
```

**Discriminator Loss**:
```
L_D_CT = E[log(D_CT(CT))] + E[log(1 - D_CT(G(CT,MRI)))]
L_D_MRI = E[log(D_MRI(MRI))] + E[log(1 - D_MRI(G(CT,MRI)))]
```

### Loss Components Detail

1. **Adversarial Loss**: Standard GAN objective for realistic image generation
2. **Feature Matching**: Matches intermediate feature statistics for stability
3. **Fusion Quality**: Preserves information from both source modalities
4. **LSGAN Formulation**: Uses least squares loss for training stability

## 📊 Performance Metrics

### Quantitative Results

| Metric | Value | Standard Deviation | Description |
|--------|-------|-------------------|-------------|
| **SSIM** | 0.5742 | ± 0.0781 | Structural similarity |
| **PSNR** | 18.72 dB | ± 1.98 dB | **Best PSNR performance** |
| **Edge Preservation** | 0.8609 | ± 0.0366 | Edge correlation |
| **Mutual Information** | 1.4309 bits | ± 0.2138 bits | Information transfer |
| **Entropy** | 5.7079 bits | ± 0.7349 bits | Information content |
| **Training Time** | ~12 hours | - | On NVIDIA RTX 3080 |
| **Inference Time** | ~25 ms | - | Per 256×256 image pair |

### Training Stability Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Generator Convergence** | ~150 epochs | Stable generation quality |
| **Discriminator Balance** | 0.6-0.7 accuracy | Optimal competition |
| **Mode Collapse** | None observed | Stable diverse generation |
| **Training Loss** | Convergent | Smooth convergence curves |

## 💻 Implementation Details

### Dependencies

```python
# Core dependencies
torch >= 1.9.0
torchvision >= 0.10.0
opencv-python >= 4.5.0
numpy >= 1.21.0
matplotlib >= 3.3.0
kornia >= 0.6.0  # Advanced image processing
tqdm >= 4.60.0   # Progress tracking
```

### Model Initialization

```python
# Initialize DDCGAN trainer
config = {
    'batch_size': 8,
    'lr_g': 2e-4,
    'lr_d': 1e-4,
    'epochs': 200,
    'device': 'cuda',
    'loss_type': 'lsgan',
    'feature_matching_weight': 10.0,
    'fusion_quality_weight': 20.0
}

trainer = DDcGANTrainer(config)
```

### Dataset Preparation

```python
class CTMRIDataset(Dataset):
    def __init__(self, dataset_path, img_size=(256,256)):
        self.ct_dir = Path(dataset_path) / "CT"
        self.mri_dir = Path(dataset_path) / "MRI"
        self.img_size = img_size
        
        # Pair CT and MRI files
        self.ct_files = sorted([p for p in self.ct_dir.iterdir() 
                               if p.suffix.lower() in ['.png','.jpg','.jpeg']])
        self.mri_files = sorted([p for p in self.mri_dir.iterdir() 
                                if p.suffix.lower() in ['.png','.jpg','.jpeg']])
        
    def __getitem__(self, idx):
        ct = self._load_image(self.ct_files[idx])
        mri = self._load_image(self.mri_files[idx])
        
        # Convert to tensors in range [-1, 1]
        ct_tensor = torch.from_numpy(ct).unsqueeze(0) * 2.0 - 1.0
        mri_tensor = torch.from_numpy(mri).unsqueeze(0) * 2.0 - 1.0
        
        return ct_tensor, mri_tensor
```

## 🚀 Usage

### Basic Fusion

```python
import torch
from ddcgan_fusion import DDcGANTrainer

# Load trained model
trainer = DDcGANTrainer(config)
trainer.load_checkpoint('checkpoints/final/ddcgan_fusion_best.pth')

# Fuse images
ct_tensor = load_ct_image('path/to/ct.png')
mri_tensor = load_mri_image('path/to/mri.png')

with torch.no_grad():
    # Concatenate inputs
    inputs = torch.cat([ct_tensor, mri_tensor], dim=1)
    fused_image = trainer.generator(inputs)
    
    # Convert from [-1,1] to [0,1]
    fused_image = (fused_image + 1.0) / 2.0
```

### Training from Scratch

```python
# Setup data loader
dataset = CTMRIDataset('data/train', img_size=(256, 256))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Training loop
trainer = DDcGANTrainer(config)

for epoch in range(config['epochs']):
    epoch_losses = []
    
    for batch_idx, (ct_batch, mri_batch) in enumerate(dataloader):
        ct_batch = ct_batch.to(device)
        mri_batch = mri_batch.to(device)
        
        # Training step
        losses = trainer.train_step(ct_batch, mri_batch)
        epoch_losses.append(losses)
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: {losses}")
    
    # Save checkpoint
    if epoch % 20 == 0:
        trainer.save_checkpoint(f'checkpoints/ddcgan_epoch_{epoch}.pth')
```

### Advanced Configuration

```python
# Custom loss weights for different fusion requirements
config_medical = {
    'batch_size': 4,                    # Smaller batch for memory
    'lr_g': 1e-4,                      # Lower LR for stability
    'lr_d': 5e-5,                      # Even lower for discriminators
    'epochs': 300,                     # Extended training
    'feature_matching_weight': 15.0,    # Higher for feature preservation
    'fusion_quality_weight': 25.0,     # Higher for medical accuracy
    'gradient_clip': 1.0,              # Gradient clipping
    'lr_decay_factor': 0.5,            # LR decay every 50 epochs
    'save_frequency': 10               # Save every 10 epochs
}
```

## 📊 Results

### Visual Results

DDCGAN demonstrates strong generative capabilities:

- **Realistic Generation**: Produces visually plausible fused images
- **High PSNR**: Best pixel-level reconstruction quality (18.72 dB)
- **Smooth Textures**: GAN training produces smooth, artifact-free textures
- **Balanced Fusion**: Good preservation of both CT and MRI characteristics

### Comparative Performance

| Method | SSIM | PSNR (dB) | Edge Preservation | MI (bits) | Model Size |
|--------|------|-----------|------------------|-----------|------------|
| **DDCGAN Fusion** | 0.5742 | **18.72** | 0.8609 | 1.4309 | **1M** |
| ResNet Fusion | **0.6072** | 17.86 | **0.9521** | **1.9764** | 25M |
| Trainable Wavelet | 0.5841 | 18.03 | 0.9201 | 1.7432 | 2M |
| Traditional DWT-PCA | 0.5234 | 16.92 | 0.8876 | 1.6543 | 0 |

### Training Progression

**Generator Loss Curve**:
- Epochs 1-50: Initial convergence, high loss
- Epochs 50-100: Rapid improvement, stabilization
- Epochs 100-150: Fine-tuning, slow improvement  
- Epochs 150-200: Convergence, stable performance

**Discriminator Balance**:
- Maintains ~60-70% accuracy throughout training
- Indicates healthy adversarial competition
- No mode collapse or discriminator overpowering

## ✅ Advantages and Limitations

### Advantages

✅ **Best PSNR Performance** (18.72 dB): Highest pixel-level reconstruction quality
✅ **Small Model Size** (~1M parameters): Memory efficient and fast inference  
✅ **Generative Approach**: Learns to generate realistic fused images
✅ **Good Texture Quality**: Smooth, artifact-free outputs
✅ **Fast Inference** (25ms): Suitable for real-time applications
✅ **Training Stability**: LSGAN formulation provides stable training

### Limitations

❌ **Moderate SSIM** (0.5742): Lower structural similarity preservation
❌ **Edge Preservation** (0.8609): Weaker than ResNet-based methods
❌ **Information Transfer** (1.4309 bits MI): Lower mutual information
❌ **Long Training Time** (~12 hours): GAN training requires patience
❌ **Training Complexity**: Requires careful hyperparameter tuning
❌ **Mode Collapse Risk**: Potential GAN training instability

### Use Cases

**Optimal Applications**:
- Applications prioritizing visual quality and PSNR
- Real-time fusion with resource constraints
- Scenarios requiring smooth, artifact-free outputs
- Research into generative fusion approaches

**Not Recommended For**:
- Applications requiring maximum structural preservation
- Edge-critical medical diagnostics
- Scenarios with limited training time
- Applications needing deterministic outputs

## 📚 References

1. **Generative Adversarial Networks**:
   ```bibtex
   @article{goodfellow2014generative,
     title={Generative adversarial nets},
     author={Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua},
     journal={Advances in neural information processing systems},
     volume={27},
     year={2014}
   }
   ```

2. **Least Squares GAN**:
   ```bibtex
   @inproceedings{mao2017least,
     title={Least squares generative adversarial networks},
     author={Mao, Xudong and Li, Qing and Xie, Haoran and Lau, Raymond YK and Wang, Zhen and Paul Smolley, Stephen},
     booktitle={Proceedings of the IEEE international conference on computer vision},
     pages={2794--2802},
     year={2017}
   }
   ```

3. **Medical Image Fusion with GANs**:
   ```bibtex
   @article{ma2020ddcgan,
     title={DDcGAN: A dual-discriminator conditional generative adversarial network for multi-resolution image fusion},
     author={Ma, Jiayi and Yu, Wei and Liang, Pengwei and Li, Chang and Jiang, Junjun},
     journal={IEEE Transactions on Image Processing},
     volume={29},
     pages={4980--4995},
     year={2020}
   }
   ```

---

**Implementation**: `DDCGAN.ipynb`  
**Model Checkpoints**: `checkpoints/final/ddcgan_fusion/`  
**Test Results**: `test_results/ddcgan_fusion/`  
**Last Updated**: January 2025
