# Adding New Fusion Methods

This guide explains how to add new fusion methods to the Medical Image Fusion Web Application.

## Architecture Overview

The application uses a modular architecture with the following components:

```
medical_fusion_webapp/
├── models/                    # Model definitions
├── fusion_methods/           # Fusion method implementations
│   ├── base.py              # Abstract base classes
│   ├── deep_learning.py     # Neural network methods
│   └── traditional.py       # Classical methods
├── utils/                   # Utility functions
├── fusion_manager.py        # Central coordinator
└── app.py                   # Flask web application
```

## Types of Fusion Methods

### 1. Traditional Methods (Non-learning)

Traditional methods don't require training and are always available.

**Example: Adding a new traditional method**

```python
# In fusion_methods/traditional.py

class MyCustomFusion(TraditionalFusion):
    """My custom fusion method."""
    
    def __init__(self, parameter=1.0):
        super().__init__(
            name="My Custom Fusion",
            description="Description of what this method does"
        )
        self.parameter = parameter
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Implement your fusion logic here."""
        try:
            # Your fusion algorithm
            fused = self.parameter * ct + (1 - self.parameter) * mri
            return fused
        except Exception as e:
            print(f"Custom fusion error: {e}")
            return None
```

### 2. Deep Learning Methods

Deep learning methods require trained models and inherit from `ModelBasedFusion`.

**Example: Adding a new deep learning method**

```python
# First, create your model in models/my_model.py

class MyCustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define your network architecture
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, ct, mri):
        # Concatenate inputs
        x = torch.cat([ct, mri], dim=1)
        x = F.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        return x

# Then, create the fusion method in fusion_methods/deep_learning.py

class MyCustomFusion(ModelBasedFusion):
    """My custom deep learning fusion method."""
    
    def __init__(self, model_path: str, device: torch.device):
        super().__init__(
            name="My Custom Deep Fusion",
            description="Deep learning fusion using my custom architecture",
            model_path=model_path,
            device=device
        )
    
    def create_model(self):
        """Create and return the model instance."""
        from models.my_model import MyCustomNet
        return MyCustomNet()
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images using the custom model."""
        if not self.is_available:
            return None
        
        try:
            with torch.no_grad():
                ct_tensor = self.prepare_tensor(ct)
                mri_tensor = self.prepare_tensor(mri)
                fused_tensor = self.model(ct_tensor, mri_tensor)
                return np.clip(self.tensor_to_numpy(fused_tensor), 0, 1)
        except Exception as e:
            print(f"Custom deep fusion error: {e}")
            return None
```

## Step-by-Step Guide

### Step 1: Implement Your Method

Choose the appropriate base class:
- `TraditionalFusion` for non-learning methods
- `ModelBasedFusion` for deep learning methods

### Step 2: Register the Method

Add your method to the `FusionManager` in `fusion_manager.py`:

```python
# In fusion_manager.py, _initialize_methods()

# For traditional methods
traditional_methods = [
    # ... existing methods
    ('my_custom', MyCustomFusion(parameter=1.5)),
]

# For deep learning methods
model_configs = [
    # ... existing configs
    {
        'class': MyCustomFusion,
        'path': '../checkpoints/my_custom_model.pt',
        'key': 'my_custom'
    }
]
```

### Step 3: Add Method Icon (Optional)

Update the icon mapping in `templates/index.html`:

```javascript
const methodIcons = {
    // ... existing icons
    'My Custom Fusion': 'fas fa-star',
    'My Custom Deep Fusion': 'fas fa-rocket'
};
```

### Step 4: Test Your Method

Create a test function in `test_models.py`:

```python
def test_my_custom_method():
    """Test my custom fusion method."""
    print("Testing My Custom Method...")
    
    try:
        # Create dummy images
        ct = np.random.rand(256, 256).astype(np.float32)
        mri = np.random.rand(256, 256).astype(np.float32)
        
        # Test fusion
        method = MyCustomFusion(parameter=1.5)
        fused = method.fuse(ct, mri)
        
        if fused is not None:
            print("✅ Custom method test passed")
            return True
        else:
            print("❌ Custom method returned None")
            return False
            
    except Exception as e:
        print(f"❌ Custom method test failed: {e}")
        return False
```

## Advanced Features

### Custom Evaluation Metrics

Add custom metrics in `utils/metrics.py`:

```python
def calculate_my_custom_metric(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate my custom quality metric."""
    try:
        # Your metric calculation
        return metric_value
    except Exception:
        return 0.0

# Update evaluate_fusion_comprehensive to include your metric
def evaluate_fusion_comprehensive(fused, ct, mri):
    # ... existing metrics
    metrics['My_Custom_Metric'] = calculate_my_custom_metric(fused, ct)
    return metrics
```

### Method Categories

Methods are automatically categorized based on their type and name. To customize categorization, modify `get_method_categories()` in `fusion_manager.py`.

### Method Recommendations

Update `get_recommended_methods()` in `fusion_manager.py` to include your method in the default comparison set.

## Best Practices

### 1. Error Handling
Always wrap your fusion logic in try-catch blocks and return `None` on failure.

### 2. Input Validation
Validate input images using `validate_image_pair()` from `utils/image_processing.py`.

### 3. Memory Management
For deep learning methods, use `torch.no_grad()` during inference to save memory.

### 4. Normalization
Ensure output images are in the range [0, 1] using `np.clip()`.

### 5. Documentation
Provide clear descriptions of what your method does and any parameters it uses.

## Example: Complete Custom Method

Here's a complete example of adding a frequency-domain fusion method:

```python
# In fusion_methods/traditional.py

class FrequencyDomainFusion(TraditionalFusion):
    """Fusion in frequency domain using FFT."""
    
    def __init__(self, low_freq_weight=0.5):
        super().__init__(
            name="Frequency Domain Fusion",
            description=f"FFT-based fusion with low-frequency weight: {low_freq_weight}"
        )
        self.low_freq_weight = low_freq_weight
    
    def fuse(self, ct: np.ndarray, mri: np.ndarray) -> Optional[np.ndarray]:
        """Fuse images in frequency domain."""
        try:
            # Convert to frequency domain
            ct_fft = np.fft.fft2(ct)
            mri_fft = np.fft.fft2(mri)
            
            # Create frequency mask (low frequencies in center)
            h, w = ct.shape
            center_h, center_w = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            mask = ((y - center_h)**2 + (x - center_w)**2) < (min(h, w) // 4)**2
            
            # Fuse: use CT for low frequencies, MRI for high frequencies
            fused_fft = np.where(mask, 
                                self.low_freq_weight * ct_fft + (1 - self.low_freq_weight) * mri_fft,
                                0.5 * (ct_fft + mri_fft))
            
            # Convert back to spatial domain
            fused = np.real(np.fft.ifft2(fused_fft))
            
            return np.clip(fused, 0, 1)
            
        except Exception as e:
            print(f"Frequency domain fusion error: {e}")
            return None
```

Then register it in `fusion_manager.py`:

```python
traditional_methods = [
    # ... existing methods
    ('frequency', FrequencyDomainFusion(low_freq_weight=0.6)),
]
```

## Testing Your Method

1. Run the test script: `python test_models.py`
2. Start the application: `python start_app.py`
3. Upload test images and verify your method appears in the results
4. Check that metrics are calculated correctly

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all imports use relative paths correctly
2. **Model Loading Fails**: Check model checkpoint path and format
3. **Method Not Appearing**: Verify registration in `fusion_manager.py`
4. **Runtime Errors**: Add proper error handling and logging

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export FLASK_DEBUG=1
python start_app.py
```

This will provide detailed error messages and stack traces.