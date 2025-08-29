# Medical Image Fusion

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
- [Results and Evaluation](#results-and-evaluation)
- [Usage Examples](#usage-examples)
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
│   └── CT-MRI/                      # CT-MRI paired images
│       ├── CT/                      # CT images
│       └── MRI/                     # MRI images
├── deep-learning-based/             # Deep learning fusion methods
│   ├── resnet_based_fusion.ipynb    # ResNet feature fusion
│   ├── trainable-wavelet-fusion.ipynb # Trainable wavelet fusion
│   ├── DDCGAN_fusion_test.ipynb     # DDCGAN implementation
│   ├── resnet_attention_finetuning_medical_model.ipynb
│   ├── FATFusion/                   # FATFusion implementation
│   ├── MATR-main/                   # MATR implementation
│   ├── checkpoints/                 # Model checkpoints
│   ├── results/                     # Fusion results
│   └── README.md                    # Deep learning methods guide
├── traditional-approaches/          # Traditional fusion methods
│   ├── dwt_pca.ipynb               # DWT-PCA fusion
│   ├── wavelet-fusion.ipynb        # Traditional wavelet fusion
│   ├── results/                    # Fusion results
│   └── README.md                   # Traditional methods guide
├── results/                         # Comparative analysis results
│   └── comparison/                 # Fusion comparison images
├── comprehensive_fusion_comparison.ipynb # Main comparison notebook
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
Dataset/CT-MRI/
├── CT/     # CT images
└── MRI/    # MRI images (paired with CT)
```

### 2. Run Comprehensive Comparison

Open the main comparison notebook:
```bash
jupyter notebook comprehensive_fusion_comparison.ipynb
```

This notebook provides:
- Batch processing of 5 random image pairs
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

#### 2. Trainable Wavelet Fusion
- **Architecture**: CNN-based fusion masks in wavelet domain
- **Features**: Differentiable wavelet transforms, spatial adaptation
- **Training**: 20 epochs with adaptive learning rate
- **Use Case**: Detail preservation with spatial awareness

#### 3. MATR (Multiscale Adaptive Transformer)
- **Paper**: IEEE TIP 2022
- **Architecture**: Transformer-based multiscale fusion
- **Features**: Adaptive attention mechanisms
- **Citation**: Tang et al. (2022)

#### 4. FATFusion (Functional–Anatomical Transformer)
- **Paper**: IPM 2024
- **Architecture**: Functional-anatomical transformer
- **Features**: Specialized for medical imaging
- **Citation**: Tang & He (2024)

### Traditional Approaches

#### 1. DWT-PCA Fusion
- **Method**: Discrete Wavelet Transform + Principal Component Analysis
- **Features**: Two-level decomposition, PCA for approximation coefficients
- **Fusion Rule**: Maximum selection for detail coefficients

#### 2. Traditional Wavelet Fusion
- **Wavelet Types**: Haar, Daubechies (db4), Biorthogonal
- **Fusion Strategies**:
  - Maximum selection
  - Simple averaging
  - Weighted averaging (CT:0.6, MRI:0.4)
  - Activity-based fusion

## 📊 Dataset

### Medical Image Dataset
- **Modalities**: CT and MRI
- **Format**: PNG (256×256, 8-bit grayscale)
- **Size**: 573 paired images
- **Total Images**: 1,146 (573 CT + 573 MRI)
- **Storage**: ~117 MB

### Data Structure
Images are perfectly aligned pairs representing the same anatomical regions, making them ideal for:
- Pixel-wise fusion evaluation
- Feature-level fusion research
- Deep learning model training
- Multimodal analysis studies

For detailed dataset information, see [`Dataset/Dataset.md`](Dataset/Dataset.md).

## 📈 Results and Evaluation

### Evaluation Metrics
- **SSIM**: Structural Similarity Index
- **PSNR**: Peak Signal-to-Noise Ratio
- **MSE**: Mean Squared Error
- **Entropy**: Information content measure
- **MI**: Mutual Information
- **Correlation**: Pearson correlation coefficient
- **Edge Variance**: Edge preservation measure

### Sample Results
The project includes batch analysis results comparing all methods across multiple image pairs. Results are saved in:
- `batch_analysis_results.txt`: Tabular comparison
- `results/comparison/`: Visual comparison images
- Individual method results in respective folders

## 💡 Usage Examples

### Basic Fusion Pipeline
```python
import torch
from fusion_methods import ResNetFusion, WaveletFusion

# Load images
ct_image = load_image('Dataset/CT-MRI/CT/16003.png')
mri_image = load_image('Dataset/CT-MRI/MRI/16003.png')

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
    methods=['resnet', 'wavelet', 'ddt_pca', 'traditional'],
    num_samples=5
)

# Generate comparison report
comparison.generate_report(results)
```

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

## 📞 Contact

- **Author**: Jubaer Ahmed
- **GitHub**: [@jubaer36](https://github.com/jubaer36)
- **Repository**: [medfusion](https://github.com/jubaer36/medfusion)

---

**Last Updated**: August 30, 2025
**Version**: 1.0.0</content>
<parameter name="filePath">/mnt/Work/ML/trainable-multimodal-fusion/README.md
