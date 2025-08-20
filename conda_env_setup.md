# Conda Environment Setup for Medical Image Fusion Project

This guide will help you set up a conda environment with all the dependencies required to run the Medical Image Fusion Web Application and related notebooks.

## 1. Create and Activate the Environment

```bash
conda create -n medfusion python=3.10 -y
conda activate medfusion
```

## 2. Install Core Dependencies

You can install most dependencies using conda for better compatibility and performance:

```bash
conda install -c conda-forge \
    numpy \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-image \
    pillow \
    pywavelets \
    scipy \
    seaborn \
    pip
```

## 3. Install PyTorch

Choose the appropriate command for your system (see https://pytorch.org/get-started/locally/ for more options):

```bash
# For CPU only
conda install pytorch torchvision cpuonly -c pytorch

# For CUDA (GPU) support, use the appropriate cudatoolkit version, e.g.:
# conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

## 4. Install Remaining Packages via pip

Some packages are only available via pip:

```bash
pip install flask==2.3.3 opencv-python==4.8.1.78 pytorch-wavelets==1.3.0 Werkzeug==2.3.7
```

## 5. (Optional) Jupyter Notebook Support

If you want to run the notebooks:

```bash
conda install notebook jupyterlab
```

## 6. Verify Installation

You can check that all dependencies are installed by running:

```bash
python -c "import flask, torch, torchvision, cv2, PIL, numpy, matplotlib, seaborn, skimage, pywt, sklearn, pandas, werkzeug, scipy"
```

---

**Note:**
- If you encounter issues with `opencv-python` or `pytorch-wavelets`, try installing them via pip after all conda installs.
- For GPU support, ensure your system has the correct CUDA drivers.
- This environment covers all packages found in the codebase and notebooks as of August 2025.
