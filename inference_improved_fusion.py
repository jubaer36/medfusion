from improved_fusion import fuse_pair
import matplotlib.pyplot as plt
import numpy as np

# Test fusion
fused = fuse_pair(
    model_path='runs/pretrained_fusion/best.pt',
    ct_path='Patient Data/p1/ct.jpg',
    mri_path='Patient Data/p1/mri.jpg',
    img_size=256,
    device='cuda'
)

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(plt.imread('Patient Data/p1/ct.jpg'), cmap='gray')
axes[0].set_title('CT')
axes[1].imshow(plt.imread('Patient Data/p1/mri.jpg'), cmap='gray')  
axes[1].set_title('MRI')
axes[2].imshow(fused, cmap='gray')
axes[2].set_title('Fused')
axes[3].imshow(plt.imread('Patient Data/p1/fusion.jpg'), cmap='gray')
axes[3].set_title('Ground Truth')
plt.tight_layout()
plt.savefig('fusion_comparison.png', dpi=150)
plt.show()