#!/usr/bin/env python3
"""
Improved Medical Image Fusion (PyTorch)
=======================================

Key improvements for better metrics:
1) Fixed inference bug (was creating duplicate models)
2) Enhanced loss function with pixel consistency term
3) Better normalization strategy 
4) Improved fusion architecture with residual connections
5) More robust training with warmup and better augmentations

"""
from __future__ import annotations
import os, math, json, random, argparse, time
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utility: Seeding & Small Ops
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------------
# I/O: Simple image loader
# -----------------------------
def load_grayscale_image(path: str) -> np.ndarray:
    """Load an image as float32 [H,W] in range [0,1]. Supports standard 8/16-bit PNG/JPG/TIFF.
    If you have NIfTI, adapt here (e.g., nibabel)."""
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('L')  # to single channel
    arr = np.array(img).astype(np.float32)
    # Normalize to [0,1] per-file range to preserve contrast across different bit-depths
    if arr.max() > 0:
        arr = (arr - arr.min()) / max(1e-6, (arr.max() - arr.min()))
    return arr

# -----------------------------
# Enhanced Normalization (Medical-aware)
# -----------------------------

def normalize_ct_hu(x: torch.Tensor, window: Tuple[int,int]=(-1000, 400)) -> torch.Tensor:
    """More robust CT normalization with better contrast preservation"""
    # Percentile-based windowing per-image with better percentiles
    q1 = torch.quantile(x.view(x.size(0), -1), 0.005, dim=1, keepdim=True)
    q99 = torch.quantile(x.view(x.size(0), -1), 0.995, dim=1, keepdim=True)
    q1 = q1.view(-1,1,1,1)
    q99 = q99.view(-1,1,1,1)
    x = (x - q1) / (q99 - q1 + 1e-8)
    return x.clamp(0, 1)

def normalize_mri_zscore(x: torch.Tensor) -> torch.Tensor:
    """Enhanced MRI normalization with outlier rejection"""
    # Remove extreme outliers first
    q1 = torch.quantile(x.view(x.size(0), -1), 0.01, dim=1, keepdim=True).view(-1,1,1,1)
    q99 = torch.quantile(x.view(x.size(0), -1), 0.99, dim=1, keepdim=True).view(-1,1,1,1)
    x_clipped = torch.clamp(x, q1, q99)
    
    # Z-score on clipped data
    mean = x_clipped.mean(dim=(2,3), keepdim=True)
    std = x_clipped.std(dim=(2,3), keepdim=True) + 1e-8
    x = (x - mean) / std
    
    # More conservative min-max to [0,1]
    minv = torch.quantile(x.view(x.size(0), -1), 0.01, dim=1, keepdim=True).view(-1,1,1,1)
    maxv = torch.quantile(x.view(x.size(0), -1), 0.99, dim=1, keepdim=True).view(-1,1,1,1)
    x = (x - minv) / (maxv - minv + 1e-8)
    return x.clamp(0, 1)

# -----------------------------
# Enhanced Augmentations
# -----------------------------
class RandomGamma:
    def __init__(self, p=0.4, gamma_range=(0.85,1.15)):  # Less aggressive
        self.p = p
        self.gamma_range = gamma_range
    def __call__(self, x: torch.Tensor):
        if random.random() < self.p:
            g = random.uniform(*self.gamma_range)
            x = x.clamp(0,1) ** g
        return x

class RandomAffine2D:
    def __init__(self, degrees=8, translate=0.03, scale=(0.95,1.05), p=0.6):  # Less aggressive
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.p = p
    def __call__(self, x: torch.Tensor):
        if random.random() >= self.p:
            return x
        B, C, H, W = x.shape
        angle = random.uniform(-self.degrees, self.degrees)
        tx = random.uniform(-self.translate, self.translate)
        ty = random.uniform(-self.translate, self.translate)
        sc = random.uniform(*self.scale)
        theta = torch.tensor([
            [sc*math.cos(math.radians(angle)), -sc*math.sin(math.radians(angle)), tx],
            [sc*math.sin(math.radians(angle)),  sc*math.cos(math.radians(angle)), ty]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(B,1,1)
        grid = F.affine_grid(theta, size=x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

class RandomFlip:
    def __init__(self, p_h=0.5, p_v=0.2):  # Reduce vertical flip
        self.p_h = p_h
        self.p_v = p_v
    def __call__(self, x: torch.Tensor):
        if random.random() < self.p_h:
            x = torch.flip(x, dims=[3])
        if random.random() < self.p_v:
            x = torch.flip(x, dims=[2])
        return x

class AdditiveGaussianNoise:
    def __init__(self, p=0.25, sigma=(0.0, 0.01)):  # Less aggressive noise
        self.p = p
        self.sigma = sigma
    def __call__(self, x: torch.Tensor):
        if random.random() < self.p:
            s = random.uniform(*self.sigma)
            x = x + s * torch.randn_like(x)
            x = x.clamp(0,1)
        return x

# -----------------------------
# Dataset (unchanged)
# -----------------------------
class MedicalFusionDataset(Dataset):
    def __init__(self, csv_path: str, split: str, img_size: int = 256, random_crop: Optional[int]=None):
        import csv
        self.items = []
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r['split'].lower() == split.lower():
                    self.items.append((r['ct_path'], r['mri_path'], r.get('patient_id','')))    
        if len(self.items) == 0:
            raise ValueError(f"No items for split={split} in {csv_path}")
        self.img_size = img_size
        self.random_crop = random_crop

    def _resize_or_pad(self, arr: np.ndarray, size: int) -> np.ndarray:
        h,w = arr.shape
        s = max(h,w)
        pad_h_top = (s - h)//2
        pad_h_bottom = s - h - pad_h_top
        pad_w_left = (s - w)//2
        pad_w_right = s - w - pad_w_left
        arr = np.pad(arr, ((pad_h_top,pad_h_bottom),(pad_w_left,pad_w_right)), mode='edge')
        img = Image.fromarray((arr*255).astype(np.uint8))
        img = img.resize((size,size), resample=Image.BILINEAR)
        return np.array(img).astype(np.float32)/255.0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ct_path, mri_path, pid = self.items[idx]
        ct = load_grayscale_image(ct_path)
        mri = load_grayscale_image(mri_path)
        ct = self._resize_or_pad(ct, self.img_size)
        mri = self._resize_or_pad(mri, self.img_size)
        ct = torch.from_numpy(ct).unsqueeze(0)
        mri = torch.from_numpy(mri).unsqueeze(0)
        if self.random_crop is not None and self.random_crop < self.img_size:
            H = W = self.img_size
            ch = cw = self.random_crop
            y0 = random.randint(0, H - ch)
            x0 = random.randint(0, W - cw)
            ct = ct[:, y0:y0+ch, x0:x0+cw]
            mri = mri[:, y0:y0+ch, x0:x0+cw]
        return {'ct': ct.float(), 'mri': mri.float(), 'pid': pid}

# -----------------------------
# Pretrained Encoder-based Model
# -----------------------------
import torchvision.models as models

class PretrainedEncoder(nn.Module):
    """ResNet-18 based encoder adapted for grayscale medical images"""
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pretrained ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Adapt first layer for grayscale input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Initialize grayscale conv1 by averaging RGB weights
            self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Extract ResNet blocks
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels  
        self.layer3 = resnet.layer3  # 256 channels
        
        # Skip layer4 to keep features at reasonable resolution
        
    def forward(self, x):
        # Initial conv + pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f1 = x  # 64 channels, H/2 x W/2
        
        x = self.maxpool(x)
        f2 = self.layer1(x)  # 64 channels, H/4 x W/4
        f3 = self.layer2(f2)  # 128 channels, H/8 x W/8
        f4 = self.layer3(f3)  # 256 channels, H/16 x W/16
        
        return [f2, f3, f4]  # Skip f1 as it's too high-res

class EnhancedSEFusion(nn.Module):
    """Enhanced channel attention fusion with spatial attention"""
    def __init__(self, ch, r=8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch*2, max(ch//r, 8), 1),  # Ensure min 8 channels
            nn.ReLU(inplace=True),
            nn.Conv2d(max(ch//r, 8), ch*2, 1),
            nn.Sigmoid()
        )
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, a, b):
        # Channel attention
        x = torch.cat([a,b], dim=1)
        ca = self.channel_att(x)
        wa, wb = torch.chunk(ca, 2, dim=1)
        
        # Spatial attention on mean-pooled features
        sa_input = torch.cat([a.mean(1, keepdim=True), b.mean(1, keepdim=True)], dim=1)
        sa = self.spatial_att(sa_input)
        
        # Combine both attentions
        fused = wa*a + wb*b
        fused = fused * sa
        
        return fused

class FusionDecoder(nn.Module):
    """Decoder adapted for ResNet feature dimensions"""
    def __init__(self):
        super().__init__()
        # Upsample from 256 -> 128 channels
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),  # 128 + 128 from skip
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Upsample from 128 -> 64 channels  
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),  # 64 + 64 from skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling to original resolution
        self.up3 = nn.ConvTranspose2d(64, 32, 4, stride=4)  # 4x upsampling
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, f4, f3, f2):
        # f4: 256 ch, H/16 x W/16
        # f3: 128 ch, H/8 x W/8  
        # f2: 64 ch, H/4 x W/4
        
        x = self.up1(f4)  # 128 ch, H/8 x W/8
        x = torch.cat([x, f3], dim=1)  # 256 channels
        x = self.conv1(x)  # 128 channels
        
        x = self.up2(x)  # 64 ch, H/4 x W/4
        x = torch.cat([x, f2], dim=1)  # 128 channels
        x = self.conv2(x)  # 64 channels
        
        x = self.up3(x)  # 32 ch, H x W
        x = self.final(x)  # 1 ch, H x W
        
        return x

class PretrainedFusionNet(nn.Module):
    """Medical image fusion network with pretrained ResNet encoders"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.enc_ct = PretrainedEncoder(pretrained=pretrained)
        self.enc_mri = PretrainedEncoder(pretrained=pretrained)
        
        # Fusion modules for each level
        self.fuse2 = EnhancedSEFusion(64)   # H/4 level
        self.fuse3 = EnhancedSEFusion(128)  # H/8 level  
        self.fuse4 = EnhancedSEFusion(256)  # H/16 level
        
        self.decoder = FusionDecoder()
        
        # Freeze early layers of pretrained encoders initially
        if pretrained:
            self._freeze_early_layers()
    
    def _freeze_early_layers(self):
        """Freeze early conv layers, unfreeze later layers for medical domain adaptation"""
        for name, param in self.named_parameters():
            if 'conv1' in name or 'bn1' in name or 'layer1' in name:
                param.requires_grad = False
                
    def unfreeze_all(self):
        """Unfreeze all parameters for fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
        
    def forward(self, ct, mri):
        # Extract multi-scale features
        ct_feats = self.enc_ct(ct)  # [f2, f3, f4]
        mri_feats = self.enc_mri(mri)
        
        # Fuse at each scale
        f2_fused = self.fuse2(ct_feats[0], mri_feats[0])
        f3_fused = self.fuse3(ct_feats[1], mri_feats[1]) 
        f4_fused = self.fuse4(ct_feats[2], mri_feats[2])
        
        # Decode to final image
        output = self.decoder(f4_fused, f3_fused, f2_fused)
        
        return output

# For backward compatibility, use the pretrained version by default
EnhancedFusionNet = PretrainedFusionNet

# -----------------------------
# Enhanced Loss Function
# -----------------------------
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, c1=0.01**2, c2=0.03**2):
        super().__init__()
        self.window_size = window_size
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.register_buffer('window_base', self._create_window(window_size), persistent=True)

    def _create_window(self, size, sigma=1.5):
        gauss = torch.tensor([math.exp(-(x - size//2)**2/(2*sigma**2)) for x in range(size)], dtype=torch.float32)
        gauss = gauss / gauss.sum()
        window = gauss[:, None] @ gauss[None, :]
        window = window / window.sum()
        return window.view(1, 1, size, size)

    def _filter(self, x, window):
        ch = x.shape[1]
        w = window.expand(ch, 1, self.window_size, self.window_size)
        w = w.to(dtype=x.dtype, device=x.device)
        return F.conv2d(x, w, padding=self.window_size // 2, groups=ch)

    def forward(self, x, y):
        window = self.window_base
        mu_x = self._filter(x, window)
        mu_y = self._filter(y, window)
        sigma_x = self._filter(x * x, window) - mu_x * mu_x
        sigma_y = self._filter(y * y, window) - mu_y * mu_y
        sigma_xy = self._filter(x * y, window) - mu_x * mu_y
        c1 = torch.tensor(self.c1, dtype=x.dtype, device=x.device)
        c2 = torch.tensor(self.c2, dtype=x.dtype, device=x.device)
        num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        ssim_map = num / (den + 1e-12)
        ssim_map = torch.clamp(ssim_map, -1.0 + 1e-6, 1.0 - 1e-6)
        return 1.0 - ssim_map.mean()

class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer('kx_base', kx.view(1, 1, 3, 3), persistent=True)
        self.register_buffer('ky_base', ky.view(1, 1, 3, 3), persistent=True)

    def forward(self, x):
        kx = self.kx_base.to(dtype=x.dtype, device=x.device)
        ky = self.ky_base.to(dtype=x.dtype, device=x.device)
        gx = F.conv2d(x, kx, padding=1, groups=x.shape[1])
        gy = F.conv2d(x, ky, padding=1, groups=x.shape[1])
        eps = torch.tensor(1e-6, dtype=x.dtype, device=x.device)
        g = torch.sqrt(gx * gx + gy * gy + eps)
        return g

class EnhancedFusionLoss(nn.Module):
    """Enhanced loss with pixel consistency and better weighting"""
    def __init__(self, w_ssim=0.5, w_grad=0.25, w_contrast=0.1, w_pixel=0.15):
        super().__init__()
        self.ssim = SSIMLoss()
        self.grad = SobelGrad()
        self.w_ssim = float(w_ssim)
        self.w_grad = float(w_grad)
        self.w_contrast = float(w_contrast)
        self.w_pixel = float(w_pixel)

    def local_std(self, x, k=7):
        pad = k // 2
        mean = F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
        mean_sq = F.avg_pool2d(x * x, kernel_size=k, stride=1, padding=pad)
        var = (mean_sq - mean * mean).clamp(min=0.0)
        eps = torch.tensor(1e-6, dtype=x.dtype, device=x.device)
        return torch.sqrt(var + eps)

    def forward(self, fused, ct, mri):
        # 1) SSIM to each source
        l_ssim = 0.5 * (self.ssim(fused, ct) + self.ssim(fused, mri))

        # 2) Gradient/edge preservation
        g_f = self.grad(fused)
        g_c = self.grad(ct)
        g_m = self.grad(mri)
        g_max = torch.maximum(g_c, g_m)
        l_grad = F.l1_loss(g_f, g_max)

        # 3) Contrast preservation
        cs_f = self.local_std(fused)
        cs_src = 0.5 * (self.local_std(ct) + self.local_std(mri))
        l_contrast = F.relu(cs_src - cs_f).mean()

        # 4) NEW: Pixel intensity consistency - fused should be between source intensities
        src_max = torch.maximum(ct, mri)
        src_min = torch.minimum(ct, mri)
        l_pixel = F.relu(fused - src_max).mean() + F.relu(src_min - fused).mean()

        # Combine losses
        w_ssim = torch.tensor(self.w_ssim, dtype=fused.dtype, device=fused.device)
        w_grad = torch.tensor(self.w_grad, dtype=fused.dtype, device=fused.device)
        w_contrast = torch.tensor(self.w_contrast, dtype=fused.dtype, device=fused.device)
        w_pixel = torch.tensor(self.w_pixel, dtype=fused.dtype, device=fused.device)

        total = w_ssim * l_ssim + w_grad * l_grad + w_contrast * l_contrast + w_pixel * l_pixel
        return total

# -----------------------------
# Validation metrics (unchanged)
# -----------------------------
@torch.no_grad()
def entropy(x: torch.Tensor, bins=256) -> torch.Tensor:
    x = x.clamp(0,1)
    x = (x* (bins-1)).long()
    B = x.size(0)
    H = torch.zeros(B, bins, device=x.device)
    for b in range(B):
        hist = torch.bincount(x[b,0].view(-1), minlength=bins).float()
        p = hist / hist.sum().clamp_min(1.0)
        H[b] = -(p * (p+1e-9).log()).nan_to_num()
    return H.sum(dim=1)

@torch.no_grad()
def mutual_information(x: torch.Tensor, y: torch.Tensor, bins=64) -> torch.Tensor:
    xh = (x.clamp(0,1) * (bins-1)).long()
    yh = (y.clamp(0,1) * (bins-1)).long()
    B = x.size(0)
    mi = []
    for b in range(B):
        xv = xh[b,0].view(-1)
        yv = yh[b,0].view(-1)
        joint = torch.zeros(bins, bins, device=x.device)
        joint.index_put_((xv, yv), torch.ones_like(xv, dtype=torch.float32), accumulate=True)
        joint /= joint.sum().clamp_min(1.0)
        px = joint.sum(dim=1, keepdim=True)
        py = joint.sum(dim=0, keepdim=True)
        pij = joint + 1e-12
        mi_b = (pij * (pij/(px@py + 1e-12)).log()).sum()
        mi.append(mi_b)
    return torch.stack(mi, dim=0)

@torch.no_grad()
def ssim_to_sources(fused, ct, mri):
    ssim = SSIMLoss()
    s_ct = 1 - ssim(fused, ct)  # Direct comparison, sources already normalized
    s_mr = 1 - ssim(fused, mri)
    return s_ct, s_mr

@torch.no_grad()
def gradient_corr(fused, ct, mri):
    sob = SobelGrad()
    gf = sob(fused).view(fused.size(0), -1)
    gc = sob(ct).view(ct.size(0), -1)
    gm = sob(mri).view(mri.size(0), -1)
    def corr(a,b):
        a = a - a.mean(dim=1, keepdim=True)
        b = b - b.mean(dim=1, keepdim=True)
        num = (a*b).sum(dim=1)
        den = a.norm(dim=1)*b.norm(dim=1) + 1e-6
        return num/den
    return corr(gf,gc), corr(gf,gm)

# -----------------------------
# Training with Warmup
# -----------------------------
@dataclass
class Config:
    csv: str
    outdir: str = 'runs/exp'
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4  # Lower LR for pretrained model
    warmup_epochs: int = 5
    unfreeze_epoch: int = 20  # When to unfreeze early layers
    img_size: int = 256
    random_crop: Optional[int] = None
    num_workers: int = 8
    seed: int = 42
    base_channels: int = 48  # Not used in pretrained model but kept for compatibility
    grad_clip: float = 1.0
    device: str = 'cuda'  # Force CUDA usage
    pretrained: bool = True

def make_loaders(cfg: Config):
    train_ds = MedicalFusionDataset(cfg.csv, split='train', img_size=cfg.img_size, random_crop=cfg.random_crop)
    val_ds   = MedicalFusionDataset(cfg.csv, split='val', img_size=cfg.img_size, random_crop=None)

    def collate(batch):
        ct = torch.stack([b['ct'] for b in batch], dim=0)
        mri = torch.stack([b['mri'] for b in batch], dim=0)
        pid = [b['pid'] for b in batch]
        return {'ct': ct, 'mri': mri, 'pid': pid}

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate)
    return train_loader, val_loader

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def train(cfg: Config):
    seed_everything(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)
    
    # Force CUDA and check availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but required for training")
    
    device = torch.device('cuda')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = PretrainedFusionNet(pretrained=cfg.pretrained).to(device)
    print(f"Model params: {count_params(model)/1e6:.2f}M")
    print(f"Pretrained encoders: {cfg.pretrained}")

    # Different learning rates for pretrained vs new layers
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only trainable params
            if 'enc_ct' in name or 'enc_mri' in name:
                pretrained_params.append(param)
            else:
                new_params.append(param)
    
    # Use different LRs: lower for pretrained, higher for new layers
    param_groups = [
        {'params': pretrained_params, 'lr': cfg.lr * 0.1},  # 10x lower for pretrained
        {'params': new_params, 'lr': cfg.lr}  # Normal LR for fusion/decoder
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-2, betas=(0.9, 0.999))
    scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_epochs, cfg.epochs, cfg.lr)

    criterion = EnhancedFusionLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    train_loader, val_loader = make_loaders(cfg)

    best_val = float('inf')
    best_path = os.path.join(cfg.outdir, 'best.pt')

    aug_affine = RandomAffine2D()
    aug_flip = RandomFlip()
    aug_gamma = RandomGamma()
    aug_noise = AdditiveGaussianNoise()

    for epoch in range(1, cfg.epochs+1):
        model.train()
        current_lr = scheduler.step(epoch-1)
        
        # Unfreeze early layers after specified epoch
        if epoch == cfg.unfreeze_epoch:
            print(f"Unfreezing all layers at epoch {epoch}")
            model.unfreeze_all()
            # Update optimizer param groups
            optimizer.param_groups[0]['params'] = list(model.parameters())
        
        t0 = time.time()
        tr_loss = 0.0
        
        for batch in train_loader:
            ct = batch['ct'].to(device)
            mri = batch['mri'].to(device)

            # Enhanced normalization first
            ct = normalize_ct_hu(ct)
            mri = normalize_mri_zscore(mri)

            # Lighter augmentation
            cat = torch.cat([ct, mri], dim=0)
            cat = aug_flip(cat)
            cat = aug_affine(cat)
            ct, mri = cat.chunk(2, dim=0)
            ct = aug_gamma(ct)
            mri = aug_gamma(mri)
            ct = aug_noise(ct)
            mri = aug_noise(mri)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                fused = model(ct, mri)
                loss = criterion(fused, ct, mri)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item()*ct.size(0)
        
        tr_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_logs = {
            'entropy': [], 'mi_ct': [], 'mi_mri': [],
            'ssim_ct': [], 'ssim_mri': [], 'gcorr_ct': [], 'gcorr_mri': []
        }
        with torch.no_grad():
            for batch in val_loader:
                ct = normalize_ct_hu(batch['ct'].to(device))
                mri = normalize_mri_zscore(batch['mri'].to(device))
                fused = model(ct, mri)
                loss = criterion(fused, ct, mri)
                val_loss += loss.item()*ct.size(0)
                # Metrics
                val_logs['entropy'].append(entropy(fused).cpu())
                val_logs['mi_ct'].append(mutual_information(fused, ct).cpu())
                val_logs['mi_mri'].append(mutual_information(fused, mri).cpu())
                s_ct, s_mr = ssim_to_sources(fused, ct, mri)
                val_logs['ssim_ct'].append(s_ct.cpu().unsqueeze(0))
                val_logs['ssim_mri'].append(s_mr.cpu().unsqueeze(0))
                gc_ct, gc_mr = gradient_corr(fused, ct, mri)
                val_logs['gcorr_ct'].append(gc_ct.cpu().unsqueeze(0))
                val_logs['gcorr_mri'].append(gc_mr.cpu().unsqueeze(0))
        val_loss /= len(val_loader.dataset)
        for k in list(val_logs.keys()):
            val_logs[k] = torch.cat(val_logs[k], dim=0).mean().item()

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__, 'epoch': epoch}, best_path)

        log = {
            'epoch': epoch,
            'train_loss': tr_loss,
            'val_loss': val_loss,
            **val_logs,
            'lr': current_lr,
            'time_sec': time.time()-t0,
            'best_val': best_val,
        }
        print(json.dumps(log))
        with open(os.path.join(cfg.outdir, 'log.jsonl'), 'a') as f:
            f.write(json.dumps(log)+'\n')
        # Save periodic checkpoints
        if epoch % 10 == 0:
            torch.save({'model': model.state_dict(), 'cfg': cfg.__dict__, 'epoch': epoch}, os.path.join(cfg.outdir, f'epoch_{epoch}.pt'))

    print(f"Training done. Best val loss={best_val:.6f}. Best model at {best_path}")

# -----------------------------
# FIXED Inference helper - This was the main bug!
# -----------------------------
@torch.no_grad()
def fuse_pair(model_path: str, ct_path: str, mri_path: str, img_size=256, device='cuda') -> np.ndarray:
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(model_path, map_location=device)
    cfgd = ckpt.get('cfg', {})
    pretrained = cfgd.get('pretrained', True)
    
    # Create model with same config as training
    model = PretrainedFusionNet(pretrained=False)  # Don't reload pretrained weights
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    
    def _prep(p):
        arr = load_grayscale_image(p)
        H = W = img_size
        h,w = arr.shape
        s = max(h,w)
        pad_h_top = (s - h)//2
        pad_h_bottom = s - h - pad_h_top
        pad_w_left = (s - w)//2
        pad_w_right = s - w - pad_w_left
        arr = np.pad(arr, ((pad_h_top,pad_h_bottom),(pad_w_left,pad_w_right)), mode='edge')
        img = Image.fromarray((arr*255).astype(np.uint8)).resize((H,W), Image.BILINEAR)
        t = torch.from_numpy(np.array(img).astype(np.float32)/255.).unsqueeze(0).unsqueeze(0)
        return t
    
    ct = _prep(ct_path).to(device)
    mri = _prep(mri_path).to(device)
    ct = normalize_ct_hu(ct)
    mri = normalize_mri_zscore(mri)
    
    # Use the loaded model for inference
    out = model(ct, mri)
    img = out.squeeze().cpu().numpy()
    return img

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, required=True, help='CSV with split,ct_path,mri_path,patient_id')
    p.add_argument('--outdir', type=str, default='runs/exp1')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--unfreeze-epoch', type=int, default=20)
    p.add_argument('--img-size', type=int, default=256)
    p.add_argument('--random-crop', type=int, default=None)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--base', type=int, default=48, help='Base channels (kept for compatibility)')
    p.add_argument('--no-pretrained', action='store_true', help='Disable pretrained encoders')
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(
        csv=args.csv, outdir=args.outdir, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, warmup_epochs=args.warmup_epochs, unfreeze_epoch=args.unfreeze_epoch,
        img_size=args.img_size, random_crop=args.random_crop, num_workers=args.workers, 
        seed=args.seed, base_channels=args.base, pretrained=not args.no_pretrained
    )
    train(cfg)

if __name__ == '__main__':
    main()