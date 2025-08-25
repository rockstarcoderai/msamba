#!/usr/bin/env python3
"""
Script to analyze MOSI data structure from a single .pkl file.
"""

import pickle
import numpy as np
import os
from pathlib import Path
import argparse

def analyze_mosi_data(input_file):
    """
    Analyze the structure of MOSI data file.
    
    Args:
        input_file: Path to your MOSI .pkl file
    """
    print(f"🔍 Analyzing MOSI data from: {input_file}")
    print("=" * 60)
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"📊 Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"📋 Data keys: {list(data.keys())}")
        print("\n📈 Data structure analysis:")
        
        for key, value in data.items():
            print(f"\n🔑 Key: '{key}'")
            print(f"   Type: {type(value)}")
            
            if isinstance(value, np.ndarray):
                print(f"   Shape: {value.shape}")
                print(f"   Dtype: {value.dtype}")
                print(f"   Min: {value.min():.4f}, Max: {value.max():.4f}, Mean: {value.mean():.4f}")
                
                # Check if it looks like features
                if len(value.shape) == 3:
                    print(f"   Looks like features: {value.shape[0]} samples, {value.shape[1]} timesteps, {value.shape[2]} features")
                elif len(value.shape) == 2:
                    print(f"   Looks like 2D data: {value.shape[0]} samples, {value.shape[1]} features")
                elif len(value.shape) == 1:
                    print(f"   Looks like 1D data: {value.shape[0]} samples")
                    
            elif isinstance(value, list):
                print(f"   Length: {len(value)}")
                if len(value) > 0:
                    print(f"   First item type: {type(value[0])}")
                    if isinstance(value[0], (int, float, str)):
                        print(f"   Sample values: {value[:5]}")
                        
            elif isinstance(value, dict):
                print(f"   Dictionary with keys: {list(value.keys())}")
                # Recursively analyze nested dict
                for sub_key, sub_value in value.items():
                    print(f"     Sub-key '{sub_key}': {type(sub_value)}")
                    if isinstance(sub_value, np.ndarray):
                        print(f"       Shape: {sub_value.shape}")
                    elif isinstance(sub_value, list):
                        print(f"       Length: {len(sub_value)}")
                        
            else:
                print(f"   Value: {value}")
    
    elif isinstance(data, list):
        print(f"📋 List with {len(data)} items")
        if len(data) > 0:
            print(f"   First item type: {type(data[0])}")
            if isinstance(data[0], dict):
                print(f"   First item keys: {list(data[0].keys())}")
    
    else:
        print(f"   Value: {data}")
    
    print("\n" + "=" * 60)
    print("💡 Analysis complete! Check the output above to understand your data structure.")
    print("📝 Next steps:")
    print("   1. Run: python convert_mosi_data.py your_file.pkl --output_dir data/mosi")
    print("   2. The conversion script will automatically split into train/valid/test")
    print("   3. Then run: python train.py --dataset mosi --batch_size 8 --num_epochs 10")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MOSI data structure")
    parser.add_argument("input_file", help="Path to your MOSI .pkl file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ Error: File {args.input_file} not found!")
        exit(1)
    
    analyze_mosi_data(args.input_file)
