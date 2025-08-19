"""
Wavelet-based Medical Image Fusion Models
=========================================

Trainable wavelet fusion models for CT and MRI medical images.
Includes both simple and spatial-adaptive variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward, DWTInverse
    WAVELETS_AVAILABLE = True
except ImportError:
    WAVELETS_AVAILABLE = False
    print("Warning: pytorch_wavelets not available. Wavelet models will be disabled.")


class WaveletFusionNet(nn.Module):
    """Simple trainable wavelet fusion network."""
    
    def __init__(self, wave='haar'):
        super().__init__()
        if not WAVELETS_AVAILABLE:
            raise ImportError("pytorch_wavelets is required for wavelet fusion models")
            
        self.dwt = DWTForward(J=1, wave=wave)
        self.idwt = DWTInverse(wave=wave)
        
        # Learnable fusion weights
        self.alpha_low = nn.Parameter(torch.tensor(0.5))
        self.alpha_lh = nn.Parameter(torch.tensor(0.5))
        self.alpha_hl = nn.Parameter(torch.tensor(0.5))
        self.alpha_hh = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, ct, mr):
        # Wavelet decomposition
        ct_low, ct_high = self.dwt(ct)
        mr_low, mr_high = self.dwt(mr)
        
        # Extract high-frequency components
        ct_lh, ct_hl, ct_hh = ct_high[0][:,:,0,:,:], ct_high[0][:,:,1,:,:], ct_high[0][:,:,2,:,:]
        mr_lh, mr_hl, mr_hh = mr_high[0][:,:,0,:,:], mr_high[0][:,:,1,:,:], mr_high[0][:,:,2,:,:]
        
        # Fusion with learnable weights
        fused_low = self.alpha_low * ct_low + (1 - self.alpha_low) * mr_low
        fused_lh = self.alpha_lh * ct_lh + (1 - self.alpha_lh) * mr_lh
        fused_hl = self.alpha_hl * ct_hl + (1 - self.alpha_hl) * mr_hl
        fused_hh = self.alpha_hh * ct_hh + (1 - self.alpha_hh) * mr_hh
        
        # Reconstruct high-frequency components
        fused_high = torch.stack([fused_lh, fused_hl, fused_hh], dim=2)
        
        # Inverse wavelet transform
        fused = self.idwt((fused_low, [fused_high]))
        return fused


class WaveletFusionNetSpatial(nn.Module):
    """Spatial-adaptive wavelet fusion network with CNN-based masks."""
    
    def __init__(self, wave='haar'):
        super().__init__()
        if not WAVELETS_AVAILABLE:
            raise ImportError("pytorch_wavelets is required for wavelet fusion models")
            
        self.dwt = DWTForward(J=1, wave=wave)
        self.idwt = DWTInverse(wave=wave)
        
        # CNN for generating spatial fusion masks
        self.mask_net = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 4, 3, padding=1), 
            nn.Sigmoid()
        )

    def forward(self, ct, mr):
        # Ensure 4D tensors
        if ct.dim() == 3: 
            ct = ct.unsqueeze(0)
        if mr.dim() == 3: 
            mr = mr.unsqueeze(0)

        # Wavelet decomposition
        ct_low, ct_high = self.dwt(ct)
        mr_low, mr_high = self.dwt(mr)

        # Extract high-frequency components
        ct_lh, ct_hl, ct_hh = ct_high[0][:,:,0:1,:,:], ct_high[0][:,:,1:2,:,:], ct_high[0][:,:,2:3,:,:]
        mr_lh, mr_hl, mr_hh = mr_high[0][:,:,0:1,:,:], mr_high[0][:,:,1:2,:,:], mr_high[0][:,:,2:3,:,:]
        
        # Squeeze extra dimension
        ct_lh, ct_hl, ct_hh = ct_lh.squeeze(2), ct_hl.squeeze(2), ct_hh.squeeze(2)
        mr_lh, mr_hl, mr_hh = mr_lh.squeeze(2), mr_hl.squeeze(2), mr_hh.squeeze(2)

        # Generate spatial masks
        low_stack = torch.cat([ct_low, mr_low], dim=1)
        masks_low = self.mask_net(low_stack)
        mL, mLH_small, mHL_small, mHH_small = masks_low[:,0:1], masks_low[:,1:2], masks_low[:,2:3], masks_low[:,3:4]
        
        # Resize high-frequency masks
        high_freq_size = (ct_lh.shape[2], ct_lh.shape[3])
        mLH = F.interpolate(mLH_small, size=high_freq_size, mode='bilinear', align_corners=False)
        mHL = F.interpolate(mHL_small, size=high_freq_size, mode='bilinear', align_corners=False)
        mHH = F.interpolate(mHH_small, size=high_freq_size, mode='bilinear', align_corners=False)

        # Spatial fusion
        fused_low = mL * ct_low + (1 - mL) * mr_low
        fused_lh = mLH * ct_lh + (1 - mLH) * mr_lh
        fused_hl = mHL * ct_hl + (1 - mHL) * mr_hl
        fused_hh = mHH * ct_hh + (1 - mHH) * mr_hh
        
        # Reconstruct
        fused_high = torch.stack([fused_lh, fused_hl, fused_hh], dim=2)
        fused = self.idwt((fused_low, [fused_high]))
        
        return fused, {'mL': mL, 'mLH': mLH, 'mHL': mHL, 'mHH': mHH}