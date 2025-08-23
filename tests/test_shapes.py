
"""Test tensor shapes throughout the model."""

import torch
import pytest
from src.models import HierarchicalMSAmba
from src.utils.config import ModelConfig
from . import create_dummy_batch, assert_tensor_shape


class TestShapes:
    """Test tensor shapes in model components."""
    
    @pytest.fixture
    def config(self):
        return ModelConfig()
    
    @pytest.fixture
    def model(self, config):
        return HierarchicalMSAmba(config)
    
    @pytest.fixture
    def batch(self):
        return create_dummy_batch(batch_size=8, seq_len=100)
    
    def test_model_forward_shapes(self, model, batch):
        """Test model forward pass shapes."""
        model.eval()
        
        with torch.no_grad():
            outputs = model(**{k: v for k, v in batch.items() 
                             if k not in ['regression_targets', 'classification_targets']})
        
        batch_size = batch['text_features'].size(0)
        
        # Check output shapes
        assert 'regression' in outputs
        assert 'classification' in outputs
        
        assert_tensor_shape(outputs['regression'], (batch_size, 1), "regression output")
        assert_tensor_shape(outputs['classification'], (batch_size, 7), "classification output")
    
    def test_different_batch_sizes(self, model, config):
        """Test model with different batch sizes."""
        model.eval()
        
        for batch_size in [1, 4, 16, 32]:
            batch = create_dummy_batch(batch_size=batch_size, seq_len=75)
            
            with torch.no_grad():
                outputs = model(**{k: v for k, v in batch.items() 
                                 if k not in ['regression_targets', 'classification_targets']})
            
            assert outputs['regression'].size(0) == batch_size
            assert outputs['classification'].size(0) == batch_size
    
    def test_different_sequence_lengths(self, model, config):
        """Test model with different sequence lengths."""
        model.eval()
        
        for seq_len in [25, 50, 100, 200]:
            batch = create_dummy_batch(batch_size=4, seq_len=seq_len)
            
            with torch.no_grad():
                outputs = model(**{k: v for k, v in batch.items() 
                                 if k not in ['regression_targets', 'classification_targets']})
            
            # Output should be independent of sequence length
            assert_tensor_shape(outputs['regression'], (4, 1), f"regression output (seq_len={seq_len})")
            assert_tensor_shape(outputs['classification'], (4, 7), f"classification output (seq_len={seq_len})")

