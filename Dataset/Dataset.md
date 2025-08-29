# Medical Image Fusion Dataset

## Overview

This dataset contains paired Computed Tomography (CT) and Magnetic Resonance Imaging (MRI) medical images specifically designed for multimodal medical image fusion research and development. The dataset is structured to facilitate the development and evaluation of fusion algorithms that combine complementary information from different imaging modalities.

## Dataset Structure

```
Dataset/
└── CT-MRI/
    ├── CT/
    │   ├── 16003.png
    │   ├── 16004.png
    │   ├── ...
    │   └── 45389.png
    └── MRI/
        ├── 16003.png
        ├── 16004.png
        ├── ...
        └── 45389.png
```

## Image Specifications

### Format
- **File Format**: PNG (Portable Network Graphics)
- **Color Mode**: Grayscale
- **Bit Depth**: 8-bit (256 intensity levels)
- **Channels**: Single channel (monochrome)

### Resolution
- **Dimensions**: 256 × 256 pixels
- **Aspect Ratio**: 1:1 (square images)

### File Properties
- **Compression**: Lossless PNG compression
- **Interlacing**: Non-interlaced
- **Color Space**: Grayscale with 8-bit depth

## Dataset Statistics

### Size Information
- **Total Images**: 1,146 (573 CT + 573 MRI)
- **Paired Images**: 573 pairs (each CT image has a corresponding MRI image)
- **CT Images**: 573 files
- **MRI Images**: 573 files

### Storage Requirements
- **CT Directory**: ~59 MB
- **MRI Directory**: ~58 MB
- **Total Dataset Size**: ~117 MB
- **Average File Size**: ~102 KB per image

## File Naming Convention

Images follow a consistent naming pattern:
- **Format**: `{patient_id}{slice_number}.png`
- **Patient IDs**: Range from 16000-45000+ (various patient identifiers)
- **Slice Numbers**: 01-89 (sequential slice numbers)
- **Example**: `16003.png` represents patient 160 with slice 03

### Naming Pattern Details
- Images are paired by filename (same filename in both CT/ and MRI/ directories)
- Sequential numbering within patient scans
- No gaps in numbering for complete scan sequences

## Usage Guidelines

### Multimodal Fusion
- Each CT-MRI pair represents the same anatomical region
- Images are perfectly aligned for fusion tasks
- Suitable for pixel-wise and feature-level fusion methods

### Research Applications
- Medical image fusion algorithm development
- Multimodal feature extraction
- Deep learning model training and validation
- Image registration and alignment studies

### Data Loading
```python
import os
from PIL import Image
import numpy as np

# Load paired images
ct_path = "Dataset/CT-MRI/CT/16003.png"
mri_path = "Dataset/CT-MRI/MRI/16003.png"

ct_image = np.array(Image.open(ct_path))
mri_image = np.array(Image.open(mri_path))

print(f"CT image shape: {ct_image.shape}")
print(f"MRI image shape: {mri_image.shape}")
```

**Last Updated**: August 29, 2025
**Dataset Version**: 1.0
**Total Files**: 1,146 images (573 pairs)
