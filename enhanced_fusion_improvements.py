# Enhanced Trainable Wavelet Fusion - Improvement Strategies
# Multiple approaches to significantly improve performance over baseline methods

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
import numpy as np

# =============================================================================
# IMPROVEMENT 1: Spatial-Adaptive Fusion Network
# =============================================================================
class SpatialAdaptiveWaveletFusion(nn.Module):
    """Spatially-adaptive fusion weights using lightweight CNN"""
    def __init__(self, wave='haar', channels=64):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave)
        self.idwt = DWTInverse(wave=wave)
        
        # Spatial attention network for fusion weights
        self.fusion_net = nn.Sequential(
            # Input: concatenated CT+MRI features
            nn.Conv2d(2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//2, 4, 3, padding=1),  # 4 output channels for 4 bands
            nn.Sigmoid()  # Ensure weights in [0,1]
        )
        
    def forward(self, ct, mr):
        # Get wavelet decompositions
        ct_low, ct_high = self.dwt(ct)
        mr_low, mr_high = self.dwt(mr)
        
        # Extract detail bands
        ct_lh, ct_hl, ct_hh = ct_high[0][:,0:1], ct_high[0][:,1:2], ct_high[0][:,2:3]
        mr_lh, mr_hl, mr_hh = mr_high[0][:,0:1], mr_high[0][:,1:2], mr_high[0][:,2:3]
        
        # Generate spatial fusion weights
        # Concatenate original images for context
        fusion_input = torch.cat([ct, mr], dim=1)  # (B, 2, H, W)
        weights = self.fusion_net(fusion_input)  # (B, 4, H, W)
        
        # Downsample weights for low-frequency and detail bands
        w_low = F.adaptive_avg_pool2d(weights[:, 0:1], ct_low.shape[-2:])
        w_lh = F.adaptive_avg_pool2d(weights[:, 1:2], ct_lh.shape[-2:])
        w_hl = F.adaptive_avg_pool2d(weights[:, 2:3], ct_hl.shape[-2:])
        w_hh = F.adaptive_avg_pool2d(weights[:, 3:4], ct_hh.shape[-2:])
        
        # Spatially-adaptive fusion
        fused_low = w_low * ct_low + (1 - w_low) * mr_low
        fused_lh = w_lh * ct_lh + (1 - w_lh) * mr_lh
        fused_hl = w_hl * ct_hl + (1 - w_hl) * mr_hl
        fused_hh = w_hh * ct_hh + (1 - w_hh) * mr_hh
        
        # Reconstruct
        fused_high = torch.cat([fused_lh, fused_hl, fused_hh], dim=1)
        fused = self.idwt((fused_low, [fused_high]))
        
        return fused, weights  # Return weights for analysis

# =============================================================================
# IMPROVEMENT 2: Multi-Scale Wavelet Fusion
# =============================================================================
class MultiScaleWaveletFusion(nn.Module):
    """Multi-level wavelet decomposition with learnable fusion"""
    def __init__(self, wave='haar', levels=2):
        super().__init__()
        self.levels = levels
        self.dwt = DWTForward(J=levels, wave=wave)
        self.idwt = DWTInverse(wave=wave)
        
        # Learnable parameters for each level and band
        self.fusion_weights = nn.ParameterDict()
        for level in range(levels):
            self.fusion_weights[f'level_{level}_lh'] = nn.Parameter(torch.tensor(0.0))
            self.fusion_weights[f'level_{level}_hl'] = nn.Parameter(torch.tensor(0.0))
            self.fusion_weights[f'level_{level}_hh'] = nn.Parameter(torch.tensor(0.0))
        
        # Low-frequency fusion weight
        self.fusion_weights['low'] = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, ct, mr):
        # Multi-level decomposition
        ct_low, ct_highs = self.dwt(ct)
        mr_low, mr_highs = self.dwt(mr)
        
        # Fuse low-frequency
        alpha_low = torch.sigmoid(self.fusion_weights['low'])
        fused_low = alpha_low * ct_low + (1 - alpha_low) * mr_low
        
        # Fuse each level of detail coefficients
        fused_highs = []
        for level in range(self.levels):
            ct_level = ct_highs[level]  # (B, 3, H, W)
            mr_level = mr_highs[level]
            
            # Extract LH, HL, HH
            ct_lh, ct_hl, ct_hh = ct_level[:,0:1], ct_level[:,1:2], ct_level[:,2:3]
            mr_lh, mr_hl, mr_hh = mr_level[:,0:1], mr_level[:,1:2], mr_level[:,2:3]
            
            # Get fusion weights for this level
            alpha_lh = torch.sigmoid(self.fusion_weights[f'level_{level}_lh'])
            alpha_hl = torch.sigmoid(self.fusion_weights[f'level_{level}_hl'])
            alpha_hh = torch.sigmoid(self.fusion_weights[f'level_{level}_hh'])
            
            # Fuse
            fused_lh = alpha_lh * ct_lh + (1 - alpha_lh) * mr_lh
            fused_hl = alpha_hl * ct_hl + (1 - alpha_hl) * mr_hl
            fused_hh = alpha_hh * ct_hh + (1 - alpha_hh) * mr_hh
            
            fused_level = torch.cat([fused_lh, fused_hl, fused_hh], dim=1)
            fused_highs.append(fused_level)
        
        # Reconstruct
        fused = self.idwt((fused_low, fused_highs))
        return fused

# =============================================================================
# IMPROVEMENT 3: Enhanced Loss Function
# =============================================================================
class EnhancedFusionLoss(nn.Module):
    """Enhanced loss with better weighting and additional terms"""
    def __init__(self, device, w_ssim=1.0, w_grad=1.0, w_perc=1.0, w_tv=0.1, w_contrast=0.5):
        super().__init__()
        from torchvision.models import vgg19
        
        self.w_ssim = w_ssim
        self.w_grad = w_grad
        self.w_perc = w_perc
        self.w_tv = w_tv
        self.w_contrast = w_contrast
        
        # Improved perceptual loss with multiple layers
        vgg = vgg19(pretrained=True).features.to(device).eval()
        self.vgg_layers = {
            'relu1_1': vgg[:2],
            'relu2_1': vgg[:7],
            'relu3_1': vgg[:12],
            'relu4_1': vgg[:21]
        }
        
        for param in vgg.parameters():
            param.requires_grad = False
    
    def ssim_loss(self, fused, src):
        """Improved SSIM calculation"""
        def ssim(img1, img2, window_size=11, size_average=True):
            channel = img1.size(1)
            window = self.create_window(window_size, channel).to(img1.device)
            
            mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
            mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
            
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            
            if size_average:
                return ssim_map.mean()
            else:
                return ssim_map.mean(1).mean(1).mean(1)
        
        return 1 - ssim(fused, src)
    
    def perceptual_loss(self, fused, ct, mr):
        """Multi-layer perceptual loss"""
        def get_features(x):
            features = {}
            x = x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x  # Convert to RGB
            for name, layer in self.vgg_layers.items():
                x = layer(x)
                features[name] = x
            return features
        
        fused_features = get_features(fused)
        ct_features = get_features(ct)
        mr_features = get_features(mr)
        
        loss = 0
        for name in self.vgg_layers.keys():
            # Loss with both sources
            loss += F.mse_loss(fused_features[name], ct_features[name])
            loss += F.mse_loss(fused_features[name], mr_features[name])
        
        return loss / (2 * len(self.vgg_layers))
    
    def tv_loss(self, x):
        """Total variation loss for smoothness"""
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    
    def contrast_loss(self, fused, ct, mr):
        """Preserve local contrast"""
        def local_contrast(x):
            kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                                 dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
            return F.conv2d(x, kernel, padding=1)
        
        fused_contrast = local_contrast(fused)
        ct_contrast = local_contrast(ct)
        mr_contrast = local_contrast(mr)
        
        # Target contrast should be maximum of source contrasts
        target_contrast = torch.max(torch.abs(ct_contrast), torch.abs(mr_contrast))
        return F.mse_loss(torch.abs(fused_contrast), target_contrast)
    
    def create_window(self, window_size, channel):
        """Create Gaussian window for SSIM"""
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, fused, ct, mr):
        # SSIM loss
        l_ssim = 0.5 * (self.ssim_loss(fused, ct) + self.ssim_loss(fused, mr))
        
        # Gradient loss
        l_grad = self.gradient_loss(fused, ct, mr)
        
        # Enhanced perceptual loss
        l_perc = self.perceptual_loss(fused, ct, mr)
        
        # Total variation loss
        l_tv = self.tv_loss(fused)
        
        # Contrast preservation loss
        l_contrast = self.contrast_loss(fused, ct, mr)
        
        total = (self.w_ssim * l_ssim + 
                self.w_grad * l_grad + 
                self.w_perc * l_perc + 
                self.w_tv * l_tv + 
                self.w_contrast * l_contrast)
        
        return total, {
            "ssim": l_ssim.item(),
            "grad": l_grad.item(), 
            "perc": l_perc.item(),
            "tv": l_tv.item(),
            "contrast": l_contrast.item()
        }
    
    def gradient_loss(self, fused, ct, mr):
        """Gradient preservation loss"""
        def sobel_filter(x):
            kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
            ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
            
            gx = F.conv2d(x, kx, padding=1)
            gy = F.conv2d(x, ky, padding=1)
            return torch.sqrt(gx**2 + gy**2 + 1e-8)
        
        grad_fused = sobel_filter(fused)
        grad_ct = sobel_filter(ct)
        grad_mr = sobel_filter(mr)
        grad_target = torch.max(grad_ct, grad_mr)
        
        return F.mse_loss(grad_fused, grad_target)

# =============================================================================
# IMPROVEMENT 4: Training Configuration
# =============================================================================
class ImprovedConfig:
    """Enhanced training configuration"""
    # Data
    ct_dir: str = 'Harvard-Medical-Image-Fusion-Datasets/CT-MRI/CT'
    mri_dir: str = 'Harvard-Medical-Image-Fusion-Datasets/CT-MRI/MRI'
    save_dir: str = 'checkpoints_enhanced'
    resize: tuple = (256, 256)  # Fixed size for consistency
    
    # Training
    batch_size: int = 8  # Increased batch size
    epochs: int = 100    # More epochs
    lr: float = 1e-4     # Lower learning rate
    lr_schedule: bool = True  # Use learning rate scheduling
    val_split: float = 0.2    # More validation data
    
    # Model
    model_type: str = 'spatial_adaptive'  # 'spatial_adaptive', 'multi_scale', 'original'
    wave: str = 'db4'    # Better wavelet than Haar
    levels: int = 2      # For multi-scale
    
    # Loss weights (rebalanced)
    w_ssim: float = 1.0
    w_grad: float = 2.0      # Increased gradient importance
    w_perc: float = 1.0      # Increased perceptual importance
    w_tv: float = 0.1        # Smoothness
    w_contrast: float = 0.5  # Contrast preservation
    
    # Augmentation
    use_augmentation: bool = True
    flip_prob: float = 0.5
    rotation_degrees: int = 10

# =============================================================================
# IMPROVEMENT 5: Data Augmentation
# =============================================================================
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class AugmentedPairedDataset(Dataset):
    """Dataset with augmentation for better generalization"""
    def __init__(self, base_dataset, config):
        self.base_dataset = base_dataset
        self.config = config
        
        if config.use_augmentation:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=config.flip_prob),
                transforms.RandomRotation(degrees=config.rotation_degrees, fill=0),
                transforms.RandomCrop(config.resize, padding=10, fill=0),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        ct, mr, key = self.base_dataset[idx]
        
        if self.transform is not None:
            # Apply same transform to both images
            seed = torch.random.initial_seed()
            
            torch.manual_seed(seed)
            ct = self.transform(ct)
            
            torch.manual_seed(seed)
            mr = self.transform(mr)
        
        return ct, mr, key

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
def create_improved_model(config):
    """Factory function to create improved models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.model_type == 'spatial_adaptive':
        model = SpatialAdaptiveWaveletFusion(wave=config.wave).to(device)
    elif config.model_type == 'multi_scale':
        model = MultiScaleWaveletFusion(wave=config.wave, levels=config.levels).to(device)
    else:
        # Original model with improvements
        from your_original_file import WaveletFusionNet
        model = WaveletFusionNet(wave=config.wave).to(device)
    
    # Enhanced loss function
    loss_fn = EnhancedFusionLoss(
        device, 
        w_ssim=config.w_ssim,
        w_grad=config.w_grad, 
        w_perc=config.w_perc,
        w_tv=config.w_tv,
        w_contrast=config.w_contrast
    ).to(device)
    
    # Better optimizer with scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    
    if config.lr_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    else:
        scheduler = None
    
    return model, loss_fn, optimizer, scheduler

print("Enhanced Wavelet Fusion Models Created!")
print("Key improvements:")
print("1. Spatial-adaptive fusion weights")
print("2. Multi-scale wavelet decomposition")
print("3. Enhanced loss function with multiple terms")
print("4. Better training configuration")
print("5. Data augmentation for generalization")
