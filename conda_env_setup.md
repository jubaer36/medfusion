# Conda Environment Setup for Medical Image Fusion Project

This guide will help you set up a conda environment with all the dependencies required to run the Medical Image Fusion Web Application and related notebooks.

## 1. Create a new conda environment:

```bash
conda create -n ml python=3.10
```

## 2. Activate the new environment:

```bash
conda activate ml
```

## 3. Install the conda packages:

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision torchaudio pytorch-cuda=12.4 tensorflow scikit-learn pandas matplotlib seaborn opencv numpy pillow imageio tqdm joblib scipy pywavelets scikit-image
```

## 4. Install the pip packages:

```bash
pip install art==6.5 blinker==1.9.0 click==8.2.1 easydict==1.13 flask==3.1.1 git-filter-repo==2.47.0 hf-xet==1.1.7 importlib-resources==6.5.2 itsdangerous==2.2.0 kornia==0.8.1 kornia-rs==0.1.9 nibabel==5.3.2 nvidia-ml-py==12.575.51 pycm==4.4 pydicom==3.0.1 pynvml==12.0.0 pytorch-wavelets==1.3.0 safetensors==0.6.2 scikit-multilearn==0.2.0 timm==1.0.19 jupyter ipywidgets
```

## 5. Additional Notes:

- This setup includes all packages identified from the codebase imports, including those for deep learning (PyTorch, TensorFlow), image processing (OpenCV, scikit-image, nibabel), data handling (pandas, joblib), and notebook support (jupyter, ipywidgets).
- Ensure CUDA 12.4 is compatible with your GPU drivers.
- For any missing packages, you can install them as needed using `conda install` or `pip install`.
