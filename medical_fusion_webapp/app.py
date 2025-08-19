#!/usr/bin/env python3
"""
Medical Image Fusion Web Application
====================================

A Flask web application for comparing different medical image fusion methods
including ResNet-based fusion and Wavelet-based fusion models.

Features:
- Upload CT and MRI image pairs
- Compare multiple fusion methods
- Display quantitative metrics
- Interactive visualization
- Modular architecture for easy extension
"""

import os
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from fusion_manager import FusionManager
from utils.image_processing import preprocess_image, array_to_base64, load_image_from_path
from utils.metrics import get_metric_descriptions

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize fusion manager
fusion_manager = FusionManager(device)

# Helper functions for visualization

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def create_comparison_plot(results):
    """Create comparison plot of metrics."""
    # Extract method names and metrics
    methods = []
    metrics_data = []
    
    for method_key, result in results.items():
        if result.get('fused_image') is not None and result.get('metrics'):
            method_name = result['method_info']['name']
            methods.append(method_name)
            metrics_data.append(result['metrics'])
    
    if not methods:
        return None
    
    # Key metrics for comparison
    key_metrics = ['SSIM_Avg', 'PSNR_Avg', 'Edge_Preservation', 'Entropy']
    
    # Prepare data for plotting
    plot_data = []
    for i, method in enumerate(methods):
        for metric in key_metrics:
            if metric in metrics_data[i]:
                value = metrics_data[i][metric]
                # Normalize some metrics for better visualization
                if metric == 'PSNR_Avg':
                    value = min(value / 30, 1.0)  # Normalize to 0-1
                elif metric == 'Entropy':
                    value = value / 8.0  # Normalize to 0-1
                
                plot_data.append({
                    'Method': method,
                    'Metric': metric,
                    'Value': value
                })
    
    if not plot_data:
        return None
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Convert to DataFrame and pivot
    df = pd.DataFrame(plot_data)
    pivot_df = df.pivot(index='Method', columns='Metric', values='Value')
    
    # Create grouped bar chart
    ax = pivot_df.plot(kind='bar', figsize=(14, 8), width=0.8, 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    plt.title('Fusion Methods Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Fusion Methods', fontsize=12)
    plt.ylabel('Normalized Score (0-1)', fontsize=12)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save plot
    plot_path = 'static/results/comparison_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

# Routes
@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and fusion."""
    try:
        # Check if files were uploaded
        if 'ct_image' not in request.files or 'mri_image' not in request.files:
            return jsonify({'error': 'Please upload both CT and MRI images'}), 400
        
        ct_file = request.files['ct_image']
        mri_file = request.files['mri_image']
        
        if ct_file.filename == '' or mri_file.filename == '':
            return jsonify({'error': 'Please select both CT and MRI images'}), 400
        
        # Preprocess images
        ct_img = preprocess_image(ct_file)
        mri_img = preprocess_image(mri_file)
        
        # Get recommended fusion methods
        recommended_methods = fusion_manager.get_recommended_methods()
        
        # Perform fusion using all recommended methods
        fusion_results = fusion_manager.fuse_images(ct_img, mri_img, recommended_methods)
        
        # Prepare results for web response
        web_results = {}
        for method_key, result in fusion_results.items():
            method_info = result['method_info']
            
            if result['fused_image'] is not None:
                web_results[method_key] = {
                    'name': method_info['name'],
                    'description': method_info['description'],
                    'image': array_to_base64(result['fused_image']),
                    'metrics': convert_numpy_types(result['metrics']),
                    'available': True
                }
            else:
                web_results[method_key] = {
                    'name': method_info['name'],
                    'description': method_info['description'],
                    'image': None,
                    'metrics': {},
                    'available': False,
                    'error': result.get('error', 'Method not available')
                }
        
        # Create comparison plot
        plot_path = create_comparison_plot(fusion_results)
        
        # Prepare response
        response_data = {
            'ct_image': array_to_base64(ct_img),
            'mri_image': array_to_base64(mri_img),
            'results': web_results,
            'comparison_plot': plot_path,
            'method_info': {
                'total_methods': len(fusion_results),
                'available_methods': len([r for r in fusion_results.values() if r['fused_image'] is not None])
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/sample')
def load_sample():
    """Load sample images for testing."""
    try:
        # Look for sample images in the dataset
        sample_paths = [
            '../Harvard-Medical-Image-Fusion-Datasets/CT-MRI/CT',
            '../Harvard-Medical-Image-Fusion-Datasets/MyDatasets/CT-MRI/test/CT'
        ]
        
        ct_path = None
        mri_path = None
        
        for base_path in sample_paths:
            if os.path.exists(base_path):
                ct_files = [f for f in os.listdir(base_path) if f.endswith('.png')]
                
                # Find a valid image pair
                for sample_file in ct_files:
                    ct_candidate = os.path.join(base_path, sample_file)
                    mri_base = base_path.replace('/CT', '/MRI')
                    mri_candidate = os.path.join(mri_base, sample_file)
                    
                    if os.path.exists(mri_candidate):
                        # Test if images can be loaded and are not empty
                        try:
                            ct_test = load_image_from_path(ct_candidate, (256, 256))
                            mri_test = load_image_from_path(mri_candidate, (256, 256))
                            
                            if ct_test.size > 0 and mri_test.size > 0:
                                ct_path = ct_candidate
                                mri_path = mri_candidate
                                break
                        except:
                            continue
                
                if ct_path and mri_path:
                    break
        
        if ct_path and mri_path:
            # Load sample images
            ct_img = load_image_from_path(ct_path, (256, 256))
            mri_img = load_image_from_path(mri_path, (256, 256))
            
            return jsonify({
                'ct_image': array_to_base64(ct_img),
                'mri_image': array_to_base64(mri_img),
                'message': f'Loaded sample: {os.path.basename(ct_path)}'
            })
        else:
            return jsonify({'error': 'No valid sample images found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Failed to load sample: {str(e)}'}), 500

@app.route('/methods')
def get_methods():
    """Get information about available fusion methods."""
    try:
        all_methods = fusion_manager.get_all_methods()
        categories = fusion_manager.get_method_categories()
        recommended = fusion_manager.get_recommended_methods()
        
        return jsonify({
            'methods': all_methods,
            'categories': categories,
            'recommended': recommended,
            'metric_descriptions': get_metric_descriptions()
        })
    except Exception as e:
        return jsonify({'error': f'Failed to get methods info: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Medical Image Fusion Web Application...")
    print(f"📱 Device: {device}")
    
    # Display available methods
    available_methods = fusion_manager.get_available_methods()
    categories = fusion_manager.get_method_categories()
    
    print(f"📊 Available fusion methods: {len(available_methods)}")
    for category, methods in categories.items():
        if methods:
            print(f"  {category}: {len(methods)} methods")
    
    print("🌐 Starting Flask server...")
    print("🔗 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)