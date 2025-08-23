# tests/__init__.py
"""Test suite for Enhanced MSAmba."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Test utilities
import torch
import numpy as np


def create_dummy_batch(batch_size=4, seq_len=50, device='cpu'):
    """Create dummy multimodal batch for testing."""
    return {
        'text_features': torch.randn(batch_size, seq_len, 768).to(device),
        'audio_features': torch.randn(batch_size, seq_len, 74).to(device), 
        'vision_features': torch.randn(batch_size, seq_len, 47).to(device),
        'attention_mask': torch.ones(batch_size, seq_len).to(device),
        'regression_targets': torch.randn(batch_size, 1).to(device),
        'classification_targets': torch.randint(0, 7, (batch_size,)).to(device),
    }


def assert_tensor_shape(tensor, expected_shape, name="tensor"):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"{name} shape {tensor.shape} != expected {expected_shape}"


def assert_no_nan_inf(tensor, name="tensor"):
    """Assert tensor contains no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf values"

