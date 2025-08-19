"""
ResNet-based Medical Image Fusion Model
=======================================

Deep learning model using ResNet50 backbone for feature extraction and fusion
of CT and MRI medical images.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNetFusionNet(nn.Module):
    """ResNet-based feature fusion network for CT-MRI fusion."""
    
    def __init__(self):
        super().__init__()
        # Load pretrained ResNet50
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Feature extraction layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1      # 256 channels
        self.layer2 = resnet.layer2      # 512 channels  
        self.layer3 = resnet.layer3      # 1024 channels
        
        # Fusion layers for each feature level
        self.fusion1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),  # 256*2 -> 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),  # 512*2 -> 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.fusion3 = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1),  # 1024*2 -> 1024
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling layers
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 2, stride=2),  # 512+512 -> 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 2, stride=2),   # 256+256 -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def extract_features(self, x):
        """Extract multi-level features using ResNet backbone."""
        # Convert grayscale to RGB by repeating channels
        x = x.repeat(1, 3, 1, 1)
        
        # Initial conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        f1 = self.layer1(x)  # 256 channels
        f2 = self.layer2(f1) # 512 channels
        f3 = self.layer3(f2) # 1024 channels
        
        return f1, f2, f3
        
    def forward(self, ct, mr):
        """Forward pass for fusion."""
        # Extract features from both modalities
        ct_f1, ct_f2, ct_f3 = self.extract_features(ct)
        mr_f1, mr_f2, mr_f3 = self.extract_features(mr)
        
        # Fuse features at each level
        fused_f1 = self.fusion1(torch.cat([ct_f1, mr_f1], dim=1))
        fused_f2 = self.fusion2(torch.cat([ct_f2, mr_f2], dim=1))
        fused_f3 = self.fusion3(torch.cat([ct_f3, mr_f3], dim=1))
        
        # Upsample and skip connections
        x = self.up3(fused_f3)
        x = torch.cat([x, fused_f2], dim=1)
        x = self.up2(x)
        x = torch.cat([x, fused_f1], dim=1) 
        x = self.up1(x)
        x = self.final(x)
        
        return x