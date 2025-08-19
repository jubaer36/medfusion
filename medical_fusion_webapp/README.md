# Medical Image Fusion Web Application

A comprehensive, modular web application for comparing different medical image fusion methods, specifically designed for CT and MRI image fusion.

## Features

### 🔬 Multiple Fusion Methods

**Deep Learning Methods:**
- **ResNet Deep Fusion**: Advanced deep learning model using ResNet50 backbone for multi-level feature extraction and fusion
- **Wavelet Fusion**: Trainable wavelet fusion with learnable frequency-band weights (placeholder for future models)
- **Spatial-Adaptive Wavelet**: Advanced wavelet fusion with CNN-based spatial masks (placeholder for future models)

**Traditional Methods:**
- **Average Fusion**: Simple pixel-wise averaging baseline
- **Maximum Fusion**: Pixel-wise maximum selection
- **Weighted Fusion**: Weighted linear combination with configurable weights
- **Gradient-Based Fusion**: Selection based on local gradient magnitude
- **Laplacian Pyramid**: Multi-resolution fusion using Laplacian pyramid decomposition

### 📊 Comprehensive Evaluation
- **SSIM (Structural Similarity Index)**: Measures structural similarity with source images
- **PSNR (Peak Signal-to-Noise Ratio)**: Quantifies reconstruction quality
- **Edge Preservation**: Evaluates how well edges are maintained
- **Entropy**: Measures information content in fused images
- **Mutual Information**: Quantifies information shared between fused and source images
- **Statistical Measures**: Standard deviation, mean gradient, and basic statistics

### 🎨 Interactive Interface
- Drag-and-drop file upload
- Real-time image preview
- Sample image loading for testing
- Comparative visualization with performance charts
- Method categorization and availability status
- Responsive design for mobile and desktop

### 🔧 Modular Architecture
- Easy addition of new fusion methods
- Pluggable evaluation metrics
- Extensible model support
- Clean separation of concerns

## Installation

1. **Clone the repository and navigate to the webapp directory:**
```bash
cd medical_fusion_webapp
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Ensure model checkpoints are available:**
   - ResNet model: `../checkpoints_resnet/resnet_fusion_best.pt`
   - Wavelet model: `../checkpoints_wavelet/wavelet_fusion_best.pt` (optional)

## Usage

### Starting the Application

**Option 1: Using the launcher script (Recommended)**
```bash
python start_app.py
```

**Option 2: Direct execution**
```bash
python app.py
```

**Option 3: Using the run script**
```bash
python run.py
```

The application will start on `http://localhost:5000`

### Using the Interface

1. **Upload Images:**
   - Click "Choose CT Image" to upload a CT scan
   - Click "Choose MRI Image" to upload an MRI scan
   - Or click "Load Sample Images" to use test data

2. **Process Images:**
   - Click "Process & Compare Fusion Methods"
   - Wait for processing to complete

3. **View Results:**
   - Compare original images with fusion results
   - Analyze quantitative metrics
   - View performance comparison chart

### Supported Image Formats
- PNG, JPEG, JPG, BMP, TIFF
- Grayscale or color (automatically converted to grayscale)
- Recommended size: 256x256 pixels (automatically resized)

## Model Architecture

### ResNet Fusion Model
```
Input: CT + MRI (1x256x256 each)
├── Feature Extraction (ResNet50 backbone)
│   ├── Layer 1: 256 channels
│   ├── Layer 2: 512 channels
│   └── Layer 3: 1024 channels
├── Multi-level Fusion
│   ├── Fusion Layer 1: 512→256 channels
│   ├── Fusion Layer 2: 1024→512 channels
│   └── Fusion Layer 3: 2048→1024 channels
└── Decoder (Upsampling + Skip Connections)
    └── Output: Fused Image (1x256x256)
```

### Wavelet Fusion Model
```
Input: CT + MRI (1x256x256 each)
├── Wavelet Decomposition (Haar)
│   ├── Low-frequency components
│   └── High-frequency components (LH, HL, HH)
├── Learnable Fusion Weights
│   ├── α_low: Low-frequency weight
│   ├── α_lh: LH detail weight
│   ├── α_hl: HL detail weight
│   └── α_hh: HH detail weight
└── Inverse Wavelet Transform
    └── Output: Fused Image (1x256x256)
```

## API Endpoints

### POST /upload
Upload and process CT/MRI image pair.

**Request:**
- `ct_image`: CT image file
- `mri_image`: MRI image file

**Response:**
```json
{
  "ct_image": "base64_encoded_ct",
  "mri_image": "base64_encoded_mri",
  "results": {
    "ResNet Model": {
      "image": "base64_encoded_result",
      "metrics": {
        "SSIM_Avg": 0.85,
        "PSNR_Avg": 25.3,
        "Edge_Preservation": 0.92,
        "Entropy": 7.2
      }
    }
  },
  "comparison_plot": "path/to/plot.png"
}
```

### GET /sample
Load sample images for testing.

**Response:**
```json
{
  "ct_image": "base64_encoded_sample_ct",
  "mri_image": "base64_encoded_sample_mri",
  "message": "Loaded sample: filename.png"
}
```

## Performance Metrics

### SSIM (Structural Similarity Index)
- **Range**: 0-1 (higher is better)
- **Interpretation**: Measures structural similarity between fused and source images
- **Good values**: > 0.8

### PSNR (Peak Signal-to-Noise Ratio)
- **Range**: 0-∞ dB (higher is better)
- **Interpretation**: Quantifies reconstruction quality
- **Good values**: > 20 dB

### Edge Preservation
- **Range**: 0-1 (higher is better)
- **Interpretation**: Correlation between fused and source image gradients
- **Good values**: > 0.8

### Entropy
- **Range**: 0-8 bits (higher is better)
- **Interpretation**: Information content in the fused image
- **Good values**: > 6 bits

## Troubleshooting

### Common Issues

1. **Model not loading:**
   - Ensure checkpoint files exist in the correct paths
   - Check CUDA availability for GPU acceleration

2. **Out of memory errors:**
   - Reduce image size
   - Use CPU instead of GPU

3. **Slow processing:**
   - Enable GPU acceleration
   - Reduce image resolution

### Error Messages

- **"Please upload both CT and MRI images"**: Both image files are required
- **"Model not available"**: Checkpoint file not found or failed to load
- **"Processing failed"**: Internal error during fusion process

## Development

### Project Structure
```
medical_fusion_webapp/
├── app.py                    # Main Flask application
├── fusion_manager.py         # Central fusion method coordinator
├── start_app.py             # Application launcher
├── models/                  # Model definitions
│   ├── resnet_fusion.py     # ResNet-based fusion model
│   └── wavelet_fusion.py    # Wavelet-based fusion models
├── fusion_methods/          # Fusion method implementations
│   ├── base.py             # Abstract base classes
│   ├── deep_learning.py    # Neural network methods
│   └── traditional.py      # Classical methods
├── utils/                   # Utility functions
│   ├── metrics.py          # Evaluation metrics
│   └── image_processing.py # Image preprocessing utilities
├── templates/
│   └── index.html          # Web interface
├── static/
│   └── results/            # Generated plots and results
├── uploads/                # Temporary upload storage
├── requirements.txt        # Python dependencies
├── ADDING_NEW_METHODS.md   # Guide for extending the application
└── README.md              # This file
```

### Adding New Fusion Methods

The application uses a modular architecture that makes it easy to add new fusion methods. See [ADDING_NEW_METHODS.md](ADDING_NEW_METHODS.md) for a comprehensive guide.

**Quick Example:**
```python
# 1. Create your method class
class MyCustomFusion(TraditionalFusion):
    def __init__(self):
        super().__init__(
            name="My Custom Method",
            description="Description of what this method does"
        )
    
    def fuse(self, ct, mri):
        # Your fusion logic here
        return fused_image

# 2. Register in fusion_manager.py
traditional_methods = [
    # ... existing methods
    ('my_custom', MyCustomFusion()),
]
```

**Supported Method Types:**
- **Traditional Methods**: Non-learning algorithms (always available)
- **Deep Learning Methods**: Neural network-based methods (require trained models)

**Features:**
- Automatic availability detection
- Built-in error handling
- Comprehensive evaluation metrics
- Web interface integration

## License

This project is part of a medical image fusion research framework. Please cite appropriately if used in academic work.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify model checkpoints are available
3. Ensure all dependencies are installed correctly