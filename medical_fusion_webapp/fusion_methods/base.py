"""
Base classes for fusion methods
===============================

Abstract base classes and interfaces for implementing fusion methods.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple


class FusionMethod(ABC):
    """Abstract base class for all fusion methods."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.is_available = True
        self.error_message = None
    
    @abstractmethod
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """
        Fuse CT and MRI images.
        
        Args:
            ct: CT image as numpy array (H, W) with values in [0, 1]
            mri: MRI image as numpy array (H, W) with values in [0, 1]
            
        Returns:
            Fused image as numpy array (H, W) with values in [0, 1], or None if failed
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this fusion method."""
        return {
            'name': self.name,
            'description': self.description,
            'is_available': self.is_available,
            'error_message': self.error_message
        }


class ModelBasedFusion(FusionMethod):
    """Base class for deep learning model-based fusion methods."""
    
    def __init__(self, name: str, description: str, model_path: str, device: torch.device):
        super().__init__(name, description)
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()
    
    @abstractmethod
    def create_model(self):
        """Create and return the model instance."""
        pass
    
    def _load_model(self):
        """Load the trained model."""
        try:
            import os
            if not os.path.exists(self.model_path):
                self.is_available = False
                self.error_message = f"Model checkpoint not found: {self.model_path}"
                return
            
            self.model = self.create_model().to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Handle shape mismatches for wavelet models
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    # Fix parameter shapes for wavelet models
                    model_state = self.model.state_dict()
                    for key in ['_a_low', '_a_lh', '_a_hl', '_a_hh']:
                        if key in state_dict and key in model_state:
                            saved_param = state_dict[key]
                            if saved_param.shape != model_state[key].shape:
                                if saved_param.numel() == 1:
                                    if len(model_state[key].shape) == 0:
                                        state_dict[key] = saved_param.squeeze()
                                    else:
                                        state_dict[key] = saved_param.reshape(model_state[key].shape)
                    self.model.load_state_dict(state_dict)
                else:
                    raise e
            
            self.model.eval()
            
        except Exception as e:
            self.is_available = False
            self.error_message = f"Failed to load model: {str(e)}"
    
    def prepare_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor format."""
        return torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to numpy array."""
        return tensor.squeeze().cpu().numpy()


class TraditionalFusion(FusionMethod):
    """Base class for traditional (non-learning) fusion methods."""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        # Traditional methods are always available
        self.is_available = True