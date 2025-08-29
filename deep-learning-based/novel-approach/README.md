# Medical Image Fusion with Transfer Learning

## 🏥 Overview

This project implements a state-of-the-art **Medical Image Fusion** system that combines CT (Computed Tomography) and MRI (Magnetic Resonance Imaging) scans using deep learning and transfer learning techniques. The system leverages pretrained computer vision models to create enhanced medical images that preserve the complementary information from both modalities.

## 🎯 Key Features

- **Dual-Modality Fusion**: Combines CT and MRI images for comprehensive medical imaging
- **Transfer Learning**: Utilizes pretrained ResNet34 backbones for efficient training
- **Attention Mechanisms**: Implements spatial and channel attention for optimal feature fusion
- **Medical-Specific Processing**: Includes CT windowing, contrast enhancement, and medical image preprocessing
- **Comprehensive Evaluation**: Extensive metrics including SSIM, PSNR, Mutual Information, Edge Preservation Index
- **Production-Ready**: Complete pipeline from training to inference with benchmarking

## 🏗️ Architecture

### Model Components

#### 1. **Dual Encoder Design**
```
CT Input (3-channel) → ResNet34 Encoder → Multi-scale Features
MRI Input (1-channel) → Adapted ResNet34 Encoder → Multi-scale Features
```

#### 2. **Attention-Based Fusion**
- **Spatial Attention**: Focuses on important regions in feature maps
- **Channel Attention**: Emphasizes informative feature channels
- **Multi-Scale Fusion**: Combines features at 4 different resolutions (64, 128, 256, 512 channels)

#### 3. **U-Net Style Decoder**
- **Skip Connections**: Preserves spatial information from encoder
- **Progressive Upsampling**: Transposed convolutions for resolution recovery
- **Medical Attention Refinement**: Final attention layer optimized for medical content

### Transfer Learning Strategy

#### Phase 1: Pre-training (Natural Images)
```python
# Train on visible-infrared pairs (LLVIP dataset)
backbone = models.resnet34(pretrained=True)  # ImageNet weights
# Learn general fusion patterns and feature extraction
# Output: fusion_model_fixed.pth
```

#### Phase 2: Medical Fine-tuning
```python
# Load pretrained fusion model
checkpoint = torch.load('fusion_model_fixed.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Fine-tune for medical domain
config = {
    'lr': 1e-5,  # 10x lower than pre-training
    'freeze_encoder_epochs': 5,  # Freeze encoders initially
    'medical_specific_loss': True  # Domain-adapted loss functions
}
# Output: medical_fusion_finetuned.pth
```

## 📊 Evaluation Metrics

The system provides comprehensive quantitative evaluation:

### Core Metrics
- **SSIM (Structural Similarity Index)**: Measures structural preservation
- **PSNR (Peak Signal-to-Noise Ratio)**: Quantifies reconstruction quality
- **Mutual Information (MI)**: Measures information transfer between modalities
- **Edge Preservation Index (EPI)**: Evaluates detail preservation
- **Fusion Quality Index (FQI)**: Overall fusion performance metric

### Medical-Specific Analysis
- **Information Content**: Entropy analysis for information preservation
- **Edge Analysis**: Specialized edge detection and preservation metrics
- **Attention Visualization**: Visual inspection of learned attention patterns
- **Performance Benchmarking**: Speed, memory usage, and throughput analysis

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd trainable-multimodal-fusion/novel-approach

# Install dependencies
pip install torch torchvision numpy matplotlib scikit-image opencv-python
pip install seaborn pandas tqdm nibabel pydicom
```

### Data Preparation

```bash
# Organize your data
Dataset/
├── CT-MRI/
│   ├── CT/          # CT scan images
│   └── MRI/         # MRI scan images
```

### Training Pipeline

#### Step 1: Pre-train on Natural Images
```python
# initial_model.ipynb
python -c "from initial_model import main; main()"
# Output: fusion_model_fixed.pth
```

#### Step 2: Fine-tune for Medical Images
```python
# finetuning.ipynb
python -c "from finetuning import main; main()"
# Output: medical_fusion_finetuned.pth
```

#### Step 3: Comprehensive Testing
```python
# testmedfusion.ipynb
python -c "from testmedfusion import main_testing_pipeline; main_testing_pipeline()"
```

## 📁 Project Structure

```
novel-approach/
├── 📓 initial_model.ipynb          # Pre-training on natural images
├── 📓 finetuning.ipynb             # Medical domain fine-tuning
├── 📓 testmedfusion.ipynb          # Comprehensive evaluation suite
├── 📓 claude.ipynb                 # Additional experiments
├── 📓 fd.ipynb                     # Feature development
├── 🧠 fusion_model_fixed.pth       # Pretrained fusion model
├── 🧠 medical_fusion_finetuned.pth # Fine-tuned medical model
├── 📊 medical_fusion_results.png   # Training results visualization
├── 📊 medical_training_curves.png  # Training curves
├── 📄 medical_fusion_evaluation.json # Evaluation metrics
├── 🗂️ Dataset/                     # Data directory
│   └── CT-MRI/
│       ├── CT/                     # CT images
│       └── MRI/                    # MRI images
├── 🗂️ results/                     # Fusion result images
├── 🗂️ test_results/                 # Evaluation outputs
└── 🗂️ LLVIP/                       # Pre-training dataset
```

## 🔧 Configuration

### Training Configuration
```python
config = {
    # Pre-training
    'batch_size': 8,
    'lr': 1e-4,
    'num_epochs': 50,
    
    # Fine-tuning
    'batch_size': 4,
    'lr': 1e-5,
    'num_epochs': 20,
    'freeze_encoder_epochs': 5,
    
    # Medical-specific
    'ct_window_center': 40,
    'ct_window_width': 400,
    'enhance_contrast': True,
}
```

### Loss Function Weights
```python
loss_weights = {
    'content': 15.0,      # Medical content preservation
    'structure': 8.0,     # Anatomical structure
    'detail': 5.0,        # Diagnostic details
    'consistency': 3.0,   # Cross-modality consistency
    'gradient': 2.0,      # Edge preservation
    'medical': 10.0,      # Medical-specific loss
}
```

## 📈 Results and Outputs

### Generated Files
- **Visualizations**: `performance_analysis.png`, `correlation_matrix.png`
- **Metrics**: `detailed_metrics.csv`, `summary_statistics.json`
- **Analysis**: `edge_analysis.png`, `attention_visualization.png`
- **Benchmarks**: `performance_benchmark.csv`

### Performance Metrics (Example)
```
FUSION QUALITY METRICS
SSIM_CT      | Mean: 0.8234 ± 0.0456 | Range: [0.7123, 0.9456]
SSIM_MRI     | Mean: 0.7989 ± 0.0523 | Range: [0.6789, 0.9234]
FQI          | Mean: 0.7567 ± 0.0345 | Range: [0.6890, 0.8345]
EPI          | Mean: 0.8123 ± 0.0289 | Range: [0.7456, 0.8890]
```

## 🧠 Model Architecture Details

### ResNet34 Backbone Adaptation
```python
# CT Encoder (3-channel input)
self.ct_stem = nn.Sequential(
    backbone.conv1,      # 7x7 conv, stride 2
    backbone.bn1,        # Batch normalization
    backbone.relu,       # ReLU activation
    backbone.maxpool,    # 3x3 maxpool, stride 2
)
self.ct_layer1 = backbone.layer1  # 64 channels
self.ct_layer2 = backbone.layer2  # 128 channels
self.ct_layer3 = backbone.layer3  # 256 channels
self.ct_layer4 = backbone.layer4  # 512 channels

# MRI Encoder (1-channel input adaptation)
mri_backbone = models.resnet34(pretrained=True)
old_conv1 = mri_backbone.conv1
mri_backbone.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
# Initialize with averaged RGB weights
with torch.no_grad():
    mri_backbone.conv1.weight = nn.Parameter(old_conv1.weight.mean(dim=1, keepdim=True))
```

### Attention Mechanisms
```python
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.conv(x)
        return x * self.sigmoid(attention)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

## 🔬 Technical Insights

### Transfer Learning Benefits
1. **Feature Reuse**: Low-level features (edges, textures) transfer well
2. **Training Efficiency**: 10x faster convergence with pretrained weights
3. **Better Generalization**: Pretrained models provide better gradient flow
4. **Domain Adaptation**: Medical-specific fine-tuning preserves general vision capabilities

### Medical Image Challenges Addressed
- **Modality Differences**: CT (structural) vs MRI (soft tissue) characteristics
- **Intensity Variations**: Different intensity scales and ranges
- **Anatomical Preservation**: Maintaining diagnostic information integrity
- **Noise Characteristics**: Different noise patterns in medical imaging

## 🤝 Contributing

### Development Guidelines
1. **Code Style**: Follow PEP 8 conventions
2. **Documentation**: Add docstrings to all functions and classes
3. **Testing**: Include unit tests for new functionality
4. **Version Control**: Use descriptive commit messages

### Adding New Features
1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes
4. **Test** thoroughly
5. **Submit** a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **Torchvision** for pretrained model implementations
- **Medical Imaging Community** for datasets and research insights
- **Open Source Contributors** for various libraries used

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the wiki for detailed guides
- **Community**: Join discussions in the GitHub Discussions tab

---

**🏥 Ready to revolutionize medical imaging with AI-powered fusion!**
