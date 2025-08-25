#!/usr/bin/env python3
"""
Script to convert MOSI data to the expected format for MSAmba.
"""

import pickle
import numpy as np
import os
from pathlib import Path
import argparse

def convert_mosi_data(input_file, output_dir):
    """
    Convert MOSI data to the expected format.
    
    Args:
        input_file: Path to your MOSI .pkl file
        output_dir: Directory to save converted data
    """
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data keys: {list(data.keys())}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data is already split (train/valid/test structure)
    if 'train' in data or 'valid' in data or 'test' in data:
        print("✅ Data is already split into train/valid/test!")
        print("Processing each split...")
        
        splits = ['train', 'valid', 'test']
        for split in splits:
            if split in data:
                split_data = data[split]
                output_file = output_dir / f"{split}.pkl"
                
                # Convert split data to expected format
                converted_data = convert_split_data(split_data)
                
                with open(output_file, 'wb') as f:
                    pickle.dump(converted_data, f)
                print(f"✅ Saved {split} data to {output_file}")
                print(f"   Samples: {len(converted_data['labels'])}")
                print(f"   Text shape: {converted_data['text'].shape}")
                print(f"   Audio shape: {converted_data['audio'].shape}")
                print(f"   Vision shape: {converted_data['vision'].shape}")
    else:
        # Data is not split, create splits
        print("Data not split, creating train/valid/test splits...")
        
        # Convert single data to expected format
        converted_data = convert_split_data(data)
        
        # Get total samples
        total_samples = len(converted_data['labels'])
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        # Split ratios: 70% train, 15% valid, 15% test
        train_end = int(0.7 * total_samples)
        valid_end = int(0.85 * total_samples)
        
        train_indices = indices[:train_end]
        valid_indices = indices[train_end:valid_end]
        test_indices = indices[valid_end:]
        
        splits = [
            ('train', train_indices),
            ('valid', valid_indices),
            ('test', test_indices)
        ]
        
        for split_name, split_indices in splits:
            split_data = {}
            for key, value in converted_data.items():
                if isinstance(value, np.ndarray):
                    split_data[key] = value[split_indices]
                elif isinstance(value, list):
                    split_data[key] = [value[i] for i in split_indices]
            
            output_file = output_dir / f"{split_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"Saved {split_name} data ({len(split_indices)} samples) to {output_file}")

def convert_split_data(split_data):
    """Convert a single split of data to expected format."""
    print(f"Converting split data with keys: {list(split_data.keys())}")
    
    # Expected format
    expected_format = {
        'text': None,      # BERT features [N, seq_len, 768]
        'audio': None,     # COVAREP features [N, seq_len, 74]
        'vision': None,    # FACET features [N, seq_len, 47]
        'labels': None,    # Sentiment scores [N]
        'ids': None        # Sample IDs [N]
    }
    
    # Key mappings for MOSI data
    key_mappings = {
        'text': ['text', 'text_bert', 'bert', 'language', 'word_embeddings'],
        'audio': ['audio', 'covarep', 'acoustic', 'audio_features'],
        'vision': ['vision', 'facet', 'visual', 'openface', 'visual_features'],
        'labels': ['regression_labels', 'labels', 'label', 'y', 'sentiment', 'sentiment_labels'],
        'ids': ['ids', 'id', 'video_ids', 'segment_ids', 'sample_ids']
    }
    
    # Map data
    for expected_key, possible_keys in key_mappings.items():
        for key in possible_keys:
            if key in split_data:
                expected_format[expected_key] = split_data[key]
                print(f"   Mapped '{key}' -> '{expected_key}'")
                break
    
    # Check what we found
    missing_keys = [key for key, value in expected_format.items() if value is None]
    if missing_keys:
        print(f"   Warning: Missing keys: {missing_keys}")
        print("   Creating dummy data for missing modalities...")
        
        # Get sample count from available data
        sample_count = None
        for value in expected_format.values():
            if value is not None:
                if isinstance(value, np.ndarray):
                    sample_count = value.shape[0]
                    break
                elif isinstance(value, list):
                    sample_count = len(value)
                    break
        
        if sample_count is None:
            raise ValueError("Could not determine sample count from data")
        
        # Create dummy data for missing modalities
        if 'text' not in expected_format or expected_format['text'] is None:
            expected_format['text'] = np.random.randn(sample_count, 50, 768)
            print("   Created dummy text features")
        
        if 'audio' not in expected_format or expected_format['audio'] is None:
            expected_format['audio'] = np.random.randn(sample_count, 50, 74)
            print("   Created dummy audio features")
        
        if 'vision' not in expected_format or expected_format['vision'] is None:
            expected_format['vision'] = np.random.randn(sample_count, 50, 47)
            print("   Created dummy vision features")
        
        if 'labels' not in expected_format or expected_format['labels'] is None:
            expected_format['labels'] = np.random.uniform(-3, 3, sample_count)
            print("   Created dummy labels")
        
        if 'ids' not in expected_format or expected_format['ids'] is None:
            expected_format['ids'] = list(range(sample_count))
            print("   Created dummy IDs")
    
    return expected_format

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MOSI data to MSAmba format")
    parser.add_argument("input_file", help="Path to your MOSI .pkl file")
    parser.add_argument("--output_dir", default="data/mosi", help="Output directory")
    
    args = parser.parse_args()
    
    convert_mosi_data(args.input_file, args.output_dir)
    print("Conversion complete!")
