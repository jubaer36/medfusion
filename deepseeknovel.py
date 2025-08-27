# Training Configuration
TOTAL_EPOCHS = 100
STAGE1_EPOCHS = 30
PRETRAINED_EPOCHS = 20
COMPARISON_EPOCHS = 15
BATCH_SIZE = 8  # Increased batch size for better training

print("🎯 TRAINING CONFIGURATION")
print("=" * 50)
print(f"Total Pipeline Epochs:     {TOTAL_EPOCHS}")
print(f"Stage 1 Pre-training:      {STAGE1_EPOCHS}")
print(f"Pretrained Fine-tuning:    {PRETRAINED_EPOCHS}")
print(f"Configuration Comparison:  {COMPARISON_EPOCHS}")
print(f"Batch Size:                {BATCH_SIZE}")
print("=" * 50)

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Memory management
def clear_memory():
    """Clear GPU memory when needed"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Medical Image Dataset with Updated Structure
class MedicalImageDataset(Dataset):
    """Dataset for medical images organized by patient directories."""
    
    def __init__(self, root_dir: str, mode: str = 'train', 
                 resize: Tuple[int, int] = (256, 256),
                 file_exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
        self.root_dir = root_dir
        self.resize = resize
        self.mode = mode
        
        # Get all patient directories
        patient_dirs = sorted([d for d in glob.glob(os.path.join(root_dir, 'p*')) if os.path.isdir(d)])
        
        # Split into train/val
        train_patients, val_patients = train_test_split(patient_dirs, test_size=0.2, random_state=42)
        self.patient_dirs = train_patients if mode == 'train' else val_patients
        
        # Collect valid image pairs
        self.image_pairs = []
        for patient_dir in self.patient_dirs:
            ct_path = os.path.join(patient_dir, 'ct.jpg')
            mri_path = os.path.join(patient_dir, 'mri.jpg')
            
            # Check if both images exist
            if os.path.exists(ct_path) and os.path.exists(mri_path):
                self.image_pairs.append((ct_path, mri_path))
        
        if not self.image_pairs:
            raise RuntimeError("No valid image pairs found.")
        
        # Define augmentations
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)
            ])
        else:
            self.transform = None
        
        print(f"Found {len(self.image_pairs)} {mode} image pairs")
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int):
        ct_path, mri_path = self.image_pairs[idx]
        
        # Load images
        ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
        mri = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
        
        if ct is None or mri is None:
            ct = np.zeros((256, 256), dtype=np.uint8) if ct is None else ct
            mri = np.zeros((256, 256), dtype=np.uint8) if mri is None else mri
        
        # Resize if specified
        if self.resize is not None:
            H, W = self.resize
            ct = cv2.resize(ct, (W, H), interpolation=cv2.INTER_AREA)
            mri = cv2.resize(mri, (W, H), interpolation=cv2.INTER_AREA)
        
        # Medical image specific normalization
        ct = self.normalize_ct(ct)
        mri = self.normalize_mri(mri)
        
        # Convert to tensors (1, H, W)
        ct_tensor = torch.from_numpy(ct).unsqueeze(0).float()
        mri_tensor = torch.from_numpy(mri).unsqueeze(0).float()
        
        # Apply augmentations for training
        if self.transform and self.mode == 'train':
            # Stack for joint transformation
            stacked = torch.cat([ct_tensor, mri_tensor], dim=0)
            stacked = self.transform(stacked)
            ct_tensor, mri_tensor = stacked[0:1], stacked[1:2]
        
        return {
            'ct': ct_tensor,
            'mri': mri_tensor,
            'patient_id': os.path.basename(os.path.dirname(ct_path))
        }
    
    def normalize_ct(self, image: np.ndarray) -> np.ndarray:
        """CT-specific normalization (Hounsfield units approximation)"""
        image = image.astype(np.float32)
        # Clip to reasonable CT range and normalize
        image = np.clip(image, 0, 255)
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        return image
    
    def normalize_mri(self, image: np.ndarray) -> np.ndarray:
        """MRI-specific normalization"""
        image = image.astype(np.float32)
        # MRI typically has different intensity characteristics
        if image.max() > 1.0:
            image /= 255.0
        # Normalize to zero mean, unit variance
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        return image
    
    
    
# Simplified Hybrid CNN + Transformer Autoencoder
class SimplifiedHybridAutoencoder(nn.Module):
    """Simplified hybrid autoencoder with reduced parameters."""
    
    def __init__(self, img_size=256, in_channels=1, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        
        # Encoder
        self.encoder = SimplifiedHybridEncoder(
            img_size, in_channels, embed_dim, num_heads, num_layers
        )
        
        # Decoder
        self.decoder = SimplifiedDecoder(embed_dim, in_channels)
        
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

class SimplifiedHybridEncoder(nn.Module):
    """Simplified encoder with reduced complexity."""
    
    def __init__(self, img_size=256, in_channels=1, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        
        # CNN Feature Extractor
        self.cnn_extractor = SimplifiedCNNFeatureExtractor(in_channels)
        
        # Transformer component
        self.transformer = SimplifiedVisionTransformer(
            img_size, in_channels, embed_dim, num_heads, num_layers
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(256 + embed_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # CNN features
        cnn_features = self.cnn_extractor(x)
        
        # Transformer features
        transformer_features = self.transformer(x)
        
        # Resize transformer features to match CNN
        transformer_features = F.interpolate(
            transformer_features, size=cnn_features.shape[2:], mode='bilinear', align_corners=False
        )
        
        # Fuse features
        fused = torch.cat([cnn_features, transformer_features], dim=1)
        fused = self.fusion(fused)
        
        return fused

class SimplifiedCNNFeatureExtractor(nn.Module):
    """Simplified CNN feature extractor."""
    
    def __init__(self, in_channels=1):
        super().__init__()
        
        base_channels = 32  # Reduced from 64
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(base_channels, base_channels * 2, 5, padding=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(base_channels * 4, 256, 3, padding=1),  # Fixed output channels
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
    def forward(self, x):
        return self.conv_layers(x)

class SimplifiedVisionTransformer(nn.Module):
    """Simplified Vision Transformer."""
    
    def __init__(self, img_size=256, in_channels=1, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        
        self.patch_size = 16
        self.num_patches = (img_size // self.patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Simplified transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim * 4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Reshape back to (B, embed_dim, H/patch, W/patch)
        H_patch = int(self.num_patches ** 0.5)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, H_patch, H_patch)
        
        return x

class SimplifiedDecoder(nn.Module):
    """Simplified decoder."""
    
    def __init__(self, input_dim, output_channels):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, output_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.decoder(x)

# Enhanced Loss Functions with Anatomical Consistency
class MedicalImageLoss(nn.Module):
    """Loss function with anatomical consistency for medical images."""
    
    def __init__(self, w_recon=1.0, w_ssim=0.5, w_perceptual=0.1, w_grad=0.5):
        super().__init__()
        self.w_recon = w_recon
        self.w_ssim = w_ssim
        self.w_perceptual = w_perceptual
        self.w_grad = w_grad
        
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIM()
        self.perceptual_loss = PerceptualLoss()
        
    def forward(self, reconstructed, original, modalities=None):
        # Reconstruction loss
        recon_loss = self.mse_loss(reconstructed, original)
        
        # SSIM loss
        ssim_loss = 1.0 - self.ssim_loss(reconstructed, original)
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(reconstructed, original)
        
        # Gradient loss for anatomical consistency
        grad_loss = 0
        if modalities is not None:
            grad_loss = self.gradient_consistency_loss(reconstructed, modalities)
        
        # Combined loss
        total_loss = (self.w_recon * recon_loss +
                     self.w_ssim * ssim_loss + 
                     self.w_perceptual * perceptual_loss +
                     self.w_grad * grad_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'ssim_loss': ssim_loss,
            'perceptual_loss': perceptual_loss,
            'grad_loss': grad_loss
        }
    
    def gradient_consistency_loss(self, fused, modalities):
        """Encourage anatomical consistency through gradient preservation."""
        loss = 0
        for modality in modalities:
            # Calculate gradients
            grad_fused_x = torch.abs(fused[:, :, :, 1:] - fused[:, :, :, :-1])
            grad_modality_x = torch.abs(modality[:, :, :, 1:] - modality[:, :, :, :-1])
            
            grad_fused_y = torch.abs(fused[:, :, 1:, :] - fused[:, :, :-1, :])
            grad_modality_y = torch.abs(modality[:, :, 1:, :] - modality[:, :, :-1, :])
            
            # Gradient difference loss
            loss += F.l1_loss(grad_fused_x, grad_modality_x) + F.l1_loss(grad_fused_y, grad_modality_y)
        
        return loss / len(modalities)

# Training Functions with Validation
def train_with_validation(model, train_loader, val_loader, num_epochs, lr=1e-4, device='cuda'):
    """Train model with validation and early stopping."""
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = MedicalImageLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            ct_images = batch['ct'].to(device)
            mri_images = batch['mri'].to(device)
            
            optimizer.zero_grad()
            
            # Train on both modalities
            ct_output = model(ct_images)
            ct_loss = criterion(ct_output, ct_images, [ct_images, mri_images])
            
            mri_output = model(mri_images)
            mri_loss = criterion(mri_output, mri_images, [ct_images, mri_images])
            
            total_loss = ct_loss['total_loss'] + mri_loss['total_loss']
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_losses.append(total_loss.item())
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {total_loss.item():.6f}')
        
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                ct_images = batch['ct'].to(device)
                mri_images = batch['mri'].to(device)
                
                ct_output = model(ct_images)
                ct_loss = criterion(ct_output, ct_images, [ct_images, mri_images])
                
                mri_output = model(mri_images)
                mri_loss = criterion(mri_output, mri_images, [ct_images, mri_images])
                
                total_loss = ct_loss['total_loss'] + mri_loss['total_loss']
                epoch_val_losses.append(total_loss.item())
        
        avg_val_loss = np.mean(epoch_val_losses)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Clear memory only at epoch end
        clear_memory()
    
    return train_losses, val_losses


# Add these classes before the MedicalImageLoss class

class SSIM(nn.Module):
    """Structural Similarity Index Measure."""
    
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.C1 = C1
        self.C2 = C2
        
        # Create Gaussian window
        gauss = cv2.getGaussianKernel(window_size, window_size/6)
        gauss = gauss @ gauss.T
        window = torch.from_numpy(gauss.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window)
    
    def _filter(self, x):
        # Ensure window is on the same device as input
        window = self.window.to(x.device)
        return F.conv2d(x, window, padding=self.window_size//2, groups=x.size(1))
    
    def forward(self, x, y):
        mu_x = self._filter(x)
        mu_y = self._filter(y)
        mu_x_sq = mu_x * mu_x
        mu_y_sq = mu_y * mu_y
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = self._filter(x * x) - mu_x_sq
        sigma_y_sq = self._filter(y * y) - mu_y_sq
        sigma_xy = self._filter(x * y) - mu_xy
        
        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2))
        
        return ssim_map.mean()

class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained VGG features."""
    
    def __init__(self):
        super().__init__()
        # Use ResNet features instead of VGG for medical images
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.features.eval()
    
    def forward(self, x, y):
        # Convert grayscale to RGB
        x_rgb = x.repeat(1, 3, 1, 1)
        y_rgb = y.repeat(1, 3, 1, 1)
        
        # Extract features
        x_features = self.features(x_rgb)
        y_features = self.features(y_rgb)
        
        return F.mse_loss(x_features, y_features)

# Also add this function for the clinical evaluation metrics
def ssim(img1, img2):
    """Structural similarity index measure."""
    # Convert to numpy for easier calculation
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    # Ensure images are 2D
    if img1.ndim > 2:
        img1 = img1.squeeze()
    if img2.ndim > 2:
        img2 = img2.squeeze()
    
    # Calculate SSIM using skimage if available
    try:
        from skimage.metrics import structural_similarity as ssim_func
        return ssim_func(img1, img2, data_range=1.0)
    except ImportError:
        # Fallback implementation
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma2_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)

# Clinical Evaluation Metrics
def calculate_clinical_metrics(original_ct, original_mri, fused_image):
    """Calculate metrics relevant for medical image fusion."""
    
    metrics = {}
    
    # Mutual Information (MI)
    metrics['mi_ct'] = mutual_information(original_ct, fused_image)
    metrics['mi_mri'] = mutual_information(original_mri, fused_image)
    
    # Edge preservation
    metrics['edge_preservation'] = edge_based_metric(original_ct, original_mri, fused_image)
    
    # Structural similarity
    metrics['ssim_ct'] = ssim(original_ct, fused_image)
    metrics['ssim_mri'] = ssim(original_mri, fused_image)
    
    return metrics

def mutual_information(img1, img2):
    """Calculate mutual information between two images."""
    # Convert to numpy for easier calculation
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    # Calculate histogram and joint histogram
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=20)
    
    # Convert to probability
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # Calculate mutual information
    mi = 0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
    
    return mi

def edge_based_metric(img1, img2, fused):
    """Edge preservation metric for medical images."""
    # Calculate gradients
    grad_x1 = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    grad_y1 = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
    grad1 = np.sqrt(grad_x1**2 + grad_y1**2)
    
    grad_x2 = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
    grad_y2 = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
    grad2 = np.sqrt(grad_x2**2 + grad_y2**2)
    
    grad_xf = cv2.Sobel(fused, cv2.CV_64F, 1, 0, ksize=3)
    grad_yf = cv2.Sobel(fused, cv2.CV_64F, 0, 1, ksize=3)
    gradf = np.sqrt(grad_xf**2 + grad_yf**2)
    
    # Edge preservation coefficient
    Q_AB = np.sum(grad1 * gradf) / np.sum(grad1**2)
    Q_BA = np.sum(grad2 * gradf) / np.sum(grad2**2)
    
    return (Q_AB + Q_BA) / 2



# Main execution with updated paths
if __name__ == "__main__":
    # Create datasets
    root_dir = "Patient Data"  # Update this path if needed
    
    train_dataset = MedicalImageDataset(root_dir=root_dir, mode='train')
    val_dataset = MedicalImageDataset(root_dir=root_dir, mode='val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Initialize model
    model = SimplifiedHybridAutoencoder().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (Target: 10-20M)")
    
    # Train model
    train_losses, val_losses = train_with_validation(
        model, train_loader, val_loader, STAGE1_EPOCHS, device=device
    )
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_curve.png')
    plt.show()
    
    # Evaluate on test set (using val as test for this example)
    test_dataset = MedicalImageDataset(root_dir=root_dir, mode='val')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    test_metrics = []
    
    with torch.no_grad():
        for batch in test_loader:
            ct_images = batch['ct'].to(device)
            mri_images = batch['mri'].to(device)
            
            # Generate fused image
            fused_ct = model(ct_images)
            fused_mri = model(mri_images)
            
            # Calculate metrics
            for i in range(ct_images.shape[0]):
                metrics = calculate_clinical_metrics(
                    ct_images[i].cpu().numpy(),
                    mri_images[i].cpu().numpy(),
                    fused_ct[i].cpu().numpy()
                )
                test_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in test_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in test_metrics])
    
    print("Clinical Evaluation Metrics:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")