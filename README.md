# Trainable Multimodal Medical Image Fusion

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive framework for multimodal medical image fusion, combining deep learning and traditional approaches for CT-MRI image fusion. This project provides state-of-the-art methods for fusing complementary information from different medical imaging modalities to enhance diagnostic capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methods Implemented](#methods-implemented)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Evaluation Metrics](#evaluation-metrics)
- [Results and Performance](#results-and-performance)
- [Usage Examples](#usage-examples)
- [Method-Specific Documentation](#method-specific-documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This project implements and compares multiple approaches for multimodal medical image fusion, specifically focusing on CT-MRI fusion. The framework includes:

**Deep Learning Methods:**
- DDCGAN-based Fusion
- ResNet Attention Fine-tuning
- ResNet-based Feature Fusion
- Trainable Wavelet Fusion
- MATR (Multiscale Adaptive Transformer)
- FATFusion (Functional–Anatomical Transformer)

**Traditional Methods:**
- DWT-PCA Fusion
- Traditional Wavelet Fusion with multiple strategies

The project provides a unified platform for training, testing, and comparing different fusion approaches with comprehensive evaluation metrics.

## ✨ Key Features

- 🔬 **Multiple Fusion Paradigms**: Both deep learning and traditional approaches
- 📊 **Comprehensive Evaluation**: SSIM, PSNR, Entropy, MI, and other metrics
- 🧠 **State-of-the-art Models**: Implementation of recent research papers
- 📈 **Batch Processing**: Automated comparison across multiple image pairs
- 🎨 **Visualization**: Rich plotting and comparison tools
- 🔧 **Modular Design**: Easy to extend with new fusion methods
- 📚 **Educational**: Well-documented code with detailed explanations

## 📁 Project Structure

```
trainable-multimodal-fusion/
├── Dataset/                          # Medical image dataset
│   ├── train/                        # Training data (70%)
│   │   ├── CT/                       # CT images
│   │   └── MRI/                      # MRI images
│   ├── val/                          # Validation data (20%)
│   │   ├── CT/                       # CT images
│   │   └── MRI/                      # MRI images
│   └── test/                         # Test data (10%)
│       ├── CT/                       # CT images
│       └── MRI/                      # MRI images
├── deep-learning-based/               # Deep learning fusion methods
│   ├── resnet_based_fusion.ipynb     # ResNet feature fusion
│   ├── trainable-wavelet-fusion.ipynb # Trainable wavelet fusion
│   ├── DDCGAN_fusion_test.ipynb     # DDCGAN implementation
│   ├── resnet_attention_finetuning_medical_model.ipynb
│   ├── MATR-main/                    # MATR implementation
│   ├── checkpoints/                  # Model checkpoints
│   │   ├── intermediate/             # Intermediate checkpoints
│   │   └── final/                    # Final best models
│   ├── results/                      # Fusion results
│   ├── test_results/                  # Test set results
│   ├── utils/                        # Utility functions
│   │   ├── metrics.py               # Evaluation metrics
│   │   └── image_processing.py      # Image processing utilities
│   └── README.md                    # Deep learning methods guide
├── traditional-approaches/           # Traditional fusion methods
│   ├── dwt_pca.ipynb               # DWT-PCA fusion
│   ├── wavelet-fusion.ipynb        # Traditional wavelet fusion
│   ├── results/                    # Fusion results
│   └── README.md                   # Traditional methods guide
├── results/                         # Comparative analysis results
│   └── comparison/                 # Fusion comparison images
├── comprehensive_fusion_comparison.ipynb # Main comparison notebook
├── preprocessing.ipynb              # Data preprocessing pipeline
├── dataset_split.ipynb             # Dataset splitting utility
├── batch_analysis_results.txt      # Batch analysis results
├── conda_env_setup.md              # Environment setup guide
└── README.md                       # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.4+ (recommended for GPU acceleration)
- conda or pip package manager

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jubaer36/medfusion.git
   cd medfusion
   ```

2. **Create conda environment:**
   ```bash
   conda create -n medfusion python=3.10 -y
   conda activate medfusion
   ```

3. **Install dependencies:**
   ```bash
   # Core dependencies
   conda install -c conda-forge numpy matplotlib pandas scikit-learn scikit-image pillow pywavelets scipy seaborn pip

   # PyTorch (choose appropriate version)
   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

   # Additional packages
   pip install flask==2.3.3 opencv-python==4.8.1.78 pytorch-wavelets==1.3.0 Werkzeug==2.3.7

   # Jupyter support (optional)
   conda install notebook jupyterlab
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch, torchvision, cv2, numpy, matplotlib, scipy, sklearn"
   ```

For detailed setup instructions, see [`conda_env_setup.md`](conda_env_setup.md).

## 🎯 Quick Start

### 1. Dataset Preparation

Ensure your dataset is organized as:
```
Dataset/
├── train/
│   ├── CT/     # Training CT images
│   └── MRI/    # Training MRI images (paired with CT)
├── val/
│   ├── CT/     # Validation CT images
│   └── MRI/    # Validation MRI images
└── test/
    ├── CT/     # Test CT images
    └── MRI/    # Test MRI images
```

### 2. Run Comprehensive Comparison

Open the main comparison notebook:
```bash
jupyter notebook comprehensive_fusion_comparison.ipynb
```

This notebook provides:
- Batch processing of all test image pairs
- Comparison of all implemented methods
- Comprehensive evaluation metrics
- Visual comparison of fusion results

### 3. Individual Method Testing

**Deep Learning Methods:**
```bash
# ResNet-based Fusion
jupyter notebook deep-learning-based/resnet_based_fusion.ipynb

# Trainable Wavelet Fusion
jupyter notebook deep-learning-based/trainable-wavelet-fusion.ipynb

# DDCGAN Fusion
jupyter notebook deep-learning-based/DDCGAN_fusion_test.ipynb
```

**Traditional Methods:**
```bash
# DWT-PCA Fusion
jupyter notebook traditional-approaches/dwt_pca.ipynb

# Traditional Wavelet Fusion
jupyter notebook traditional-approaches/wavelet-fusion.ipynb
```

## 🔬 Methods Implemented

### Deep Learning Approaches

#### 1. ResNet-Based Fusion
- **Architecture**: Pretrained ResNet backbone with custom fusion layers
- **Features**: Multi-scale feature extraction, end-to-end training
- **Loss Function**: SSIM + Gradient + Perceptual loss
- **Use Case**: High-quality semantic feature fusion
- **Performance**: SSIM: 0.6072, PSNR: 17.86 dB

#### 2. Trainable Wavelet Fusion
- **Architecture**: CNN-based fusion masks in wavelet domain
- **Features**: Differentiable wavelet transforms, spatial adaptation
- **Training**: 20 epochs with adaptive learning rate
- **Use Case**: Detail preservation with spatial awareness
- **Performance**: SSIM: 0.6072, PSNR: 17.86 dB

#### 3. DDCGAN Fusion
- **Architecture**: Encoder-decoder with adversarial training
- **Features**: Generative approach to fusion
- **Training**: GAN-based training with discriminator
- **Use Case**: Realistic fused image generation
- **Performance**: SSIM: 0.5742, PSNR: 18.72 dB

#### 4. ResNet Attention Fine-tuning
- **Architecture**: Dual attention mechanism (spatial + channel)
- **Features**: Attention-based feature fusion
- **Training**: Fine-tuned on medical images
- **Use Case**: Focused feature extraction
- **Performance**: SSIM: 0.5581, PSNR: 16.67 dB

#### 5. MATR (Multiscale Adaptive Transformer)
- **Paper**: IEEE TIP 2022
- **Architecture**: Transformer-based multiscale fusion
- **Features**: Adaptive attention mechanisms
- **Citation**: Tang et al. (2022)
- **Performance**: SSIM: 0.6072, PSNR: 17.86 dB

### Traditional Approaches

#### 1. DWT-PCA Fusion
- **Method**: Discrete Wavelet Transform + Principal Component Analysis
- **Features**: Two-level decomposition, PCA for approximation coefficients
- **Fusion Rule**: Maximum selection for detail coefficients
- **Performance**: SSIM: 0.5094, PSNR: 17.24 dB

#### 2. Traditional Wavelet Fusion
- **Wavelet Types**: Haar, Daubechies (db4), Biorthogonal
- **Fusion Strategies**:
  - Maximum selection
  - Simple averaging
  - Weighted averaging (CT:0.6, MRI:0.4)
  - Activity-based fusion
- **Performance**: SSIM: 0.5953, PSNR: 17.68 dB

## 📊 Dataset

### Medical Image Dataset
- **Modalities**: CT and MRI
- **Format**: PNG (256×256, 8-bit grayscale)
- **Size**: 573 paired images
- **Total Images**: 1,146 (573 CT + 573 MRI)
- **Storage**: ~117 MB
- **Split**: 70% train, 20% validation, 10% test

### Data Structure
Images are perfectly aligned pairs representing the same anatomical regions, making them ideal for:
- Pixel-wise fusion evaluation
- Feature-level fusion research
- Deep learning model training
- Multimodal analysis studies

## 🔧 Data Preprocessing

### Image Preprocessing Pipeline

1. **Loading and Validation**
   ```python
   # Load images
   ct_img = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
   mri_img = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
   
   # Validate pairing
   assert ct_img.shape == mri_img.shape
   ```

2. **Normalization**
   ```python
   # Normalize to [0, 1]
   ct_img = ct_img.astype(np.float32) / 255.0
   mri_img = mri_img.astype(np.float32) / 255.0
   ```

3. **Resizing (if needed)**
   ```python
   # Resize to standard size
   target_size = (256, 256)
   ct_img = cv2.resize(ct_img, target_size, interpolation=cv2.INTER_AREA)
   mri_img = cv2.resize(mri_img, target_size, interpolation=cv2.INTER_AREA)
   ```

4. **Data Augmentation (for training)**
   ```python
   # Random rotation, flipping, scaling
   transforms = [
       RandomRotation(degrees=10),
       RandomHorizontalFlip(p=0.5),
       RandomVerticalFlip(p=0.5)
   ]
   ```

### Dataset Splitting
- **Random Seed**: 42 (for reproducibility)
- **Train/Val/Test Split**: 70%/20%/10%
- **Stratified**: Ensures balanced distribution across splits

## 📈 Evaluation Metrics

### Primary Metrics

#### 1. Structural Similarity Index (SSIM)
**Formula**: 
```
SSIM(x,y) = (2μₓμᵧ + c₁)(2σₓᵧ + c₂) / ((μₓ² + μᵧ² + c₁)(σₓ² + σᵧ² + c₂))
```
- **Range**: [0, 1] (higher is better)
- **Interpretation**: Measures structural similarity
- **Typical Values**: >0.7 good, >0.8 excellent

#### 2. Peak Signal-to-Noise Ratio (PSNR)
**Formula**: 
```
PSNR = 20 * log₁₀(MAX_I / √MSE)
```
- **Range**: [0, ∞] dB (higher is better)
- **Interpretation**: Measures image quality
- **Typical Values**: >20 dB acceptable, >30 dB good

#### 3. Edge Preservation
**Formula**: 
```
EP = corrcoef(∇F, max(∇CT, ∇MRI))
```
- **Range**: [0, 1] (higher is better)
- **Interpretation**: Measures edge detail preservation
- **Typical Values**: >0.8 good preservation

#### 4. Mutual Information (MI)
**Formula**: 
```
MI(X,Y) = Σ p(x,y) * log₂(p(x,y) / (p(x) * p(y)))
```
- **Range**: [0, ∞] bits (higher is better)
- **Interpretation**: Measures information content
- **Typical Values**: >1.5 bits good information transfer

#### 5. Entropy
**Formula**: 
```
H(X) = -Σ p(x) * log₂(p(x))
```
- **Range**: [0, 8] bits (higher is better)
- **Interpretation**: Measures information content
- **Typical Values**: >5 bits good information content

### Secondary Metrics

- **Mean Squared Error (MSE)**: Lower is better
- **Standard Deviation**: Measures contrast
- **Mean Gradient**: Measures edge strength
- **Correlation Coefficient**: Measures linear relationship

## 📊 Results and Performance

### Comprehensive Comparison Results

Based on testing across 58 test image pairs:

| Method | SSIM | PSNR (dB) | Edge Preservation | MI (bits) | Entropy (bits) |
|--------|------|-----------|-------------------|-----------|----------------|
| **MATR** | **0.6072** | **17.86** | **0.9521** | **1.9764** | 5.0067 |
| **ResNet Fusion** | **0.6072** | **17.86** | **0.9521** | **1.9764** | 5.0067 |
| **Trainable Wavelet** | **0.6072** | **17.86** | **0.9521** | **1.9764** | 5.0067 |
| **Traditional Wavelet** | 0.5953 | 17.68 | 0.9469 | 1.5387 | 5.2310 |
| **DDCGAN** | 0.5742 | 18.72 | 0.8609 | 1.4309 | 5.7079 |
| **ResNet Attention** | 0.5581 | 16.67 | 0.8969 | 1.4420 | 5.9903 |
| **DWT-PCA** | 0.5094 | 17.24 | 0.7377 | 1.4008 | 5.6452 |

### Performance Analysis

1. **Best Overall Performance**: MATR, ResNet Fusion, and Trainable Wavelet show identical performance
2. **Best Edge Preservation**: MATR-based methods (0.9521)
3. **Best Information Transfer**: MATR-based methods (1.9764 bits MI)
4. **Best PSNR**: DDCGAN (18.72 dB)
5. **Most Consistent**: Traditional Wavelet (lowest variance)

### Clinical Interpretation

- **High SSIM (>0.6)**: Good structural preservation
- **High Edge Preservation (>0.9)**: Excellent detail retention
- **High MI (>1.5)**: Good information fusion
- **Balanced Entropy (5-6 bits)**: Optimal information content

## 💡 Usage Examples

### Basic Fusion Pipeline
```python
import torch
from fusion_methods import ResNetFusion, WaveletFusion

# Load images
ct_image = load_image('Dataset/test/CT/00001.png')
mri_image = load_image('Dataset/test/MRI/00001.png')

# Initialize fusion method
fusion_model = ResNetFusion()
fused_image = fusion_model.fuse(ct_image, mri_image)

# Evaluate
metrics = evaluate_fusion(ct_image, mri_image, fused_image)
print(f"SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.4f}")
```

### Batch Processing
```python
from comprehensive_comparison import FusionComparison

# Initialize comparison framework
comparison = FusionComparison()

# Run batch analysis
results = comparison.run_batch_analysis(
    methods=['resnet', 'wavelet', 'dwt_pca', 'traditional'],
    num_samples=58
)

# Generate comparison report
comparison.generate_report(results)
```

### Custom Fusion Method
```python
class CustomFusionMethod:
    def __init__(self):
        self.model = self._build_model()
    
    def fuse(self, ct_image, mri_image):
        # Implement your fusion logic
        fused = self.model(ct_image, mri_image)
        return fused
    
    def _build_model(self):
        # Define your model architecture
        pass
```

## 📚 Method-Specific Documentation

### Deep Learning Methods

#### [ResNet-Based Fusion](deep-learning-based/README.md#resnet-based-fusion)
- Detailed architecture explanation
- Training procedure
- Loss function components
- Performance analysis

#### [Trainable Wavelet Fusion](deep-learning-based/README.md#trainable-wavelet-fusion)
- Wavelet domain processing
- Spatial adaptation mechanisms
- Training configuration
- Results interpretation

#### [DDCGAN Fusion](deep-learning-based/README.md#ddcgan-fusion)
- Generator-discriminator architecture
- Adversarial training process
- Loss function design
- Performance characteristics

#### [MATR Implementation](deep-learning-based/MATR-main/README.md)
- Transformer-based architecture
- Multiscale attention mechanisms
- Training procedure
- Citation and references

### Traditional Methods

#### [DWT-PCA Fusion](traditional-approaches/README.md#dwt-pca-fusion)
- Wavelet decomposition process
- PCA-based coefficient fusion
- Implementation details
- Performance analysis

#### [Traditional Wavelet Fusion](traditional-approaches/README.md#traditional-wavelet-fusion)
- Multiple fusion strategies
- Wavelet type selection
- Quality metrics
- Comparative analysis

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-fusion-method`
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

### Adding New Fusion Methods

1. Create a new class inheriting from `BaseFusionMethod`
2. Implement the `fuse()` method
3. Add evaluation metrics
4. Update the comparison framework
5. Add documentation and examples

### Code Style Guidelines

- Follow PEP 8 for Python code
- Add comprehensive docstrings
- Include type hints
- Write unit tests
- Update documentation

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{medfusion2025,
  title={Trainable Multimodal Medical Image Fusion},
  author={Jubaer Ahmed},
  year={2025},
  publisher={GitHub},
  url={https://github.com/jubaer36/medfusion}
}
```

### Related Papers

**MATR:**
```bibtex
@ARTICLE{Tang_2022_MATR,
  author={Tang, Wei and He, Fazhi and Liu, Yu and Duan, Yansong},
  journal={IEEE Transactions on Image Processing},
  title={MATR: Multimodal Medical Image Fusion via Multiscale Adaptive Transformer},
  year={2022},
  volume={31},
  pages={5134-5149},
  doi={10.1109/TIP.2022.3193288}
}
```

**FATFusion:**
```bibtex
@ARTICLE{Tang_2024_FATFusion,
  author={Tang, Wei and He, Fazhi},
  journal={Information Processing & Management},
  title={FATFusion: A Functional–Anatomical Transformer for Medical Image Fusion},
  year={2024},
  volume={61},
  number={4},
  pages={103687},
  doi={10.1016/j.ipm.2024.103687}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Harvard Medical School AANLIB dataset
- **Research Papers**: MATR and FATFusion implementations
- **Community**: Open-source computer vision and deep learning communities
- **Contributors**: All developers who contributed to this project

## 🔗 Related Resources

- [Medical Image Fusion Survey](https://example.com/survey)
- [Deep Learning for Medical Imaging](https://example.com/dl-medical)
- [Wavelet Transform Tutorial](https://example.com/wavelet-tutorial)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Last Updated**: January 2025
**Version**: 2.0.0
**Maintainer**: Jubaer Ahmed