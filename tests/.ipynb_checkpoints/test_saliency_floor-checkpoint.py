"""Test saliency floor constraints."""

import torch
import pytest
from src.models.saliency import SaliencyScorer
from src.utils.config import ModelConfig
from . import create_dummy_batch, assert_no_nan_inf


class TestSaliencyFloor:
    """Test saliency floor constraints."""
    
    @pytest.fixture
    def config(self):
        return ModelConfig()
    
    @pytest.fixture
    def saliency_scorer(self, config):
        return SaliencyScorer(
            d_model=config.d_model,
            emotion_classes=config.emotion_classes,
            floor_value=config.saliency_floor,
        )
    
    def test_floor_constraint(self, saliency_scorer):
        """Test that attention weights respect floor constraint."""
        batch_size, seq_len = 8, 100
        
        # Create inputs
        embeddings = torch.randn(batch_size, seq_len, saliency_scorer.d_model)
        
        # Compute saliency weights
        weights = saliency_scorer(embeddings)
        
        # Test floor constraint
        assert torch.all(weights >= saliency_scorer.floor_value), \
            f"Saliency weights below floor {saliency_scorer.floor_value}: {weights.min()}"
        
        # Test upper bound
        assert torch.all(weights <= 1.0), f"Saliency weights above 1.0: {weights.max()}"
        
        # Test no NaN/Inf
        assert_no_nan_inf(weights, "saliency weights")
    
    def test_residual_bypass(self, saliency_scorer):
        """Test residual bypass functionality."""
        batch_size, seq_len = 4, 50
        
        # Create inputs
        embeddings = torch.randn(batch_size, seq_len, saliency_scorer.d_model)
        
        # Apply saliency pruning
        pruned_embeddings = saliency_scorer.apply_pruning(embeddings)
        
        # Should preserve original shape
        assert pruned_embeddings.shape == embeddings.shape
        
        # Should not be identical (unless floor is 1.0)
        if saliency_scorer.floor_value < 1.0:
            assert not torch.allclose(pruned_embeddings, embeddings, atol=1e-6)
        
        assert_no_nan_inf(pruned_embeddings, "pruned embeddings")
    
    def test_different_floor_values(self, config):
        """Test different floor values."""
        batch_size, seq_len = 4, 50
        embeddings = torch.randn(batch_size, seq_len, config.d_model)
        
        for floor_value in [0.1, 0.25, 0.5, 0.75]:
            scorer = SaliencyScorer(
                d_model=config.d_model,
                emotion_classes=config.emotion_classes,
                floor_value=floor_value,
            )
            
            weights = scorer(embeddings)
            assert torch.all(weights >= floor_value), f"Floor {floor_value} violated"
            
            pruned = scorer.apply_pruning(embeddings)
            assert_no_nan_inf(pruned, f"pruned embeddings (floor={floor_value})")
