#!/usr/bin/env python3
"""
Create CSV dataset file from patient folder structure
"""
import os
import csv
import random
from pathlib import Path

def create_dataset_csv(data_root, output_csv, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Create CSV file from patient data structure:
    Patient Data/
    ├── p1/
    │   ├── ct.jpg
    │   ├── mri.jpg
    │   └── ...
    ├── p2/
    └── ...
    """
    data_root = Path(data_root)
    patients = []
    
    # Find all patient folders
    for patient_dir in sorted(data_root.iterdir()):
        if patient_dir.is_dir() and patient_dir.name.startswith('p'):
            ct_path = patient_dir / 'ct.jpg'
            mri_path = patient_dir / 'mri.jpg'
            
            if ct_path.exists() and mri_path.exists():
                patients.append({
                    'patient_id': patient_dir.name,
                    'ct_path': str(ct_path.absolute()),
                    'mri_path': str(mri_path.absolute())
                })
            else:
                print(f"Warning: Missing ct.jpg or mri.jpg in {patient_dir}")
    
    print(f"Found {len(patients)} patients with complete CT/MRI pairs")
    
    # Shuffle patients for random split
    random.seed(42)
    random.shuffle(patients)
    
    # Calculate split indices
    n_total = len(patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    print(f"Split: {n_train} train, {n_val} val, {n_test} test")
    
    # Assign splits
    for i, patient in enumerate(patients):
        if i < n_train:
            patient['split'] = 'train'
        elif i < n_train + n_val:
            patient['split'] = 'val'
        else:
            patient['split'] = 'test'
    
    # Write CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'ct_path', 'mri_path', 'patient_id'])
        writer.writeheader()
        writer.writerows(patients)
    
    print(f"Dataset CSV created at: {output_csv}")
    return output_csv

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='Patient Data', 
                        help='Path to Patient Data folder')
    parser.add_argument('--output', type=str, default='dataset.csv',
                        help='Output CSV file path')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    
    create_dataset_csv(
        args.data_root, 
        args.output,
        args.train_ratio,
        args.val_ratio, 
        args.test_ratio
    )