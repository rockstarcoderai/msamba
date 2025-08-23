"""Test handling of missing modalities."""

import torch
import pytest
from src.models import HierarchicalMSAmba
from src.utils.config import ModelConfig
from . import assert_tensor_shape, assert_no_nan_inf


class TestMissingModalities:
    """Test model behavior with missing modalities."""
    
    @pytest.fixture
    def config(self):
        return ModelConfig()
    
    @pytest.fixture
    def model(self, config):
        return HierarchicalMSAmba(config)
    
    def test_missing_single_modality(self, model):
        """Test with one modality missing."""
        batch_size, seq_len = 4, 50
        
        # Create full batch
        full_batch = {
            'text_features': torch.randn(batch_size, seq_len, 768),
            'audio_features': torch.randn(batch_size, seq_len, 74),
            'vision_features': torch.randn(batch_size, seq_len, 47),
            'attention_mask': torch.ones(batch_size, seq_len),
        }
        
        # Test missing each modality
        for missing_mod in ['text_features', 'audio_features', 'vision_features']:
            batch = full_batch.copy()
            del batch[missing_mod]
            
            with torch.no_grad():
                outputs = model(**batch)
            
            # Should still produce valid outputs
            assert_tensor_shape(outputs['regression'], (batch_size, 1), 
                              f"regression with missing {missing_mod}")
            assert_tensor_shape(outputs['classification'], (batch_size, 7),
                              f"classification with missing {missing_mod}")
            
            assert_no_nan_inf(outputs['regression'], f"regression with missing {missing_mod}")
            assert_no_nan_inf(outputs['classification'], f"classification with missing {missing_mod}")
    
    def test_missing_multiple_modalities(self, model):
        """Test with multiple modalities missing."""
        batch_size, seq_len = 4, 50
        
        # Test with only text
        text_only = {
            'text_features': torch.randn(batch_size, seq_len, 768),
            'attention_mask': torch.ones(batch_size, seq_len),
        }
        
        with torch.no_grad():
            outputs = model(**text_only)
        
        assert_tensor_shape(outputs['regression'], (batch_size, 1), "text-only regression")
        assert_tensor_shape(outputs['classification'], (batch_size, 7), "text-only classification")
        assert_no_nan_inf(outputs['regression'], "text-only regression")
        assert_no_nan_inf(outputs['classification'], "text-only classification")
        
        # Test with only audio
        audio_only = {
            'audio_features': torch.randn(batch_size, seq_len, 74),
            'attention_mask': torch.ones(batch_size, seq_len),
        }
        
        with torch.no_grad():
            outputs = model(**audio_only)
        
        assert_tensor_shape(outputs['regression'], (batch_size, 1), "audio-only regression")
        assert_tensor_shape(outputs['classification'], (batch_size, 7), "audio-only classification")
        assert_no_nan_inf(outputs['regression'], "audio-only regression")
        assert_no_nan_inf(outputs['classification'], "audio-only classification")
    
    def test_attention_mask_handling(self, model):
        """Test proper attention mask handling with missing modalities."""
        batch_size, seq_len = 4, 50
        
        # Create batch with partial attention (some timesteps masked)
        batch = {
            'text_features': torch.randn(batch_size, seq_len, 768),
            'attention_mask': torch.cat([
                torch.ones(batch_size, seq_len // 2),
                torch.zeros(batch_size, seq_len - seq_len // 2)
            ], dim=1),
        }
        
        with torch.no_grad():
            outputs = model(**batch)
        
        # Should handle partial sequences correctly
        assert_tensor_shape(outputs['regression'], (batch_size, 1), "masked regression")
        assert_tensor_shape(outputs['classification'], (batch_size, 7), "masked classification")
        assert_no_nan_inf(outputs['regression'], "masked regression")
        assert_no_nan_inf(outputs['classification'], "masked classification")