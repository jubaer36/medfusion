#!/usr/bin/env python3
"""
Check webapp method availability
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from fusion_manager import FusionManager
import torch

# Initialize fusion manager
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fm = FusionManager(device)

# Get methods
recommended = fm.get_recommended_methods()
available = fm.get_available_methods()

print("🔍 WEBAPP METHOD ANALYSIS")
print("=" * 50)

print(f"\n📊 RECOMMENDED METHODS ({len(recommended)}):")
for method in recommended:
    if method in available:
        info = available[method]
        print(f"  ✅ {method:20s} | {info['name']}")
    else:
        print(f"  ❌ {method:20s} | NOT AVAILABLE")

print(f"\n📋 ALL AVAILABLE METHODS ({len(available)}):")
for key, info in available.items():
    in_recommended = "🔹" if key in recommended else "⭕"
    print(f"  {in_recommended} {key:20s} | {info['name']}")

print(f"\n❗ MISSING FROM WEBAPP:")
missing = set(available.keys()) - set(recommended)
for method in missing:
    info = available[method]
    print(f"  ❌ {method:20s} | {info['name']}")

print(f"\n💡 SOLUTION:")
print("The webapp is only showing 'recommended' methods, not all available methods.")
print("ResNet and Wavelet Option 1 are available but not in recommended list!")
