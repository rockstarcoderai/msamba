"""Test EMA memory horizon constraints."""

import torch
import pytest
from src.models.memory import EMAMemory
from src.utils.config import ModelConfig
from . import create_dummy_batch, assert_no_nan_inf


class TestMemoryHorizon:
    """Test EMA memory horizon constraints."""
    
    @pytest.fixture
    def config(self):
        return ModelConfig()
    
    @pytest.fixture
    def memory(self, config):
        return EMAMemory(
            d_model=config.d_model,
            memory_size=config.memory_size,
            tau=config.memory_tau,
            drop_memory=config.drop_memory,
        )
    
    def test_per_clip_scope(self, memory):
        """Test that memory is scoped per clip."""
        batch_size, seq_len = 4, 50
        
        # Simulate two separate clips
        clip1_features = torch.randn(batch_size, seq_len, memory.d_model)
        clip2_features = torch.randn(batch_size, seq_len, memory.d_model)
        
        # Process first clip
        memory.training = False  # Disable drop-memory for testing
        output1 = memory(clip1_features, reset_memory=True)
        
        # Check memory was initialized
        assert memory.memory_bank is not None
        memory_after_clip1 = memory.memory_bank.clone()
        
        # Process second clip (should reset)
        output2 = memory(clip2_features, reset_memory=True)
        memory_after_clip2 = memory.memory_bank.clone()
        
        # Memory should be different after reset
        assert not torch.allclose(memory_after_clip1, memory_after_clip2, atol=1e-6)
        
        assert_no_nan_inf(output1, "clip1 output")
        assert_no_nan_inf(output2, "clip2 output")
    
    def test_ema_decay(self, memory):
        """Test EMA decay behavior."""
        batch_size, seq_len = 2, 25
        
        # Initialize with first input
        memory.training = False
        x1 = torch.ones(batch_size, seq_len, memory.d_model)
        output1 = memory(x1, reset_memory=True)
        
        initial_memory = memory.memory_bank.clone()
        
        # Update with second input
        x2 = torch.zeros(batch_size, seq_len, memory.d_model)  # Different values
        output2 = memory(x2, reset_memory=False)
        
        updated_memory = memory.memory_bank.clone()
        
        # Memory should have changed according to EMA
        assert not torch.allclose(initial_memory, updated_memory, atol=1e-6)
        
        # Check EMA property: new_memory = tau * old_memory + (1-tau) * new_value
        # Since new values are zeros, memory should decay toward zero
        assert torch.all(torch.abs(updated_memory) < torch.abs(initial_memory))
    
    def test_drop_memory_regularization(self, memory):
        """Test drop-memory regularization during training."""
        batch_size, seq_len = 4, 30
        memory.training = True  # Enable training mode
        
        x = torch.randn(batch_size, seq_len, memory.d_model)
        
        # Run multiple times to test stochastic behavior
        outputs = []
        for _ in range(10):
            memory.reset()
            output = memory(x, reset_memory=True)
            outputs.append(output)
        
        # Outputs should vary due to drop-memory (with high probability)
        outputs_tensor = torch.stack(outputs)
        variance = outputs_tensor.var(dim=0).mean()
        
        # Should have some variance (not deterministic)
        assert variance > 1e-6, "Drop-memory should introduce variance"
        
        for i, output in enumerate(outputs):
            assert_no_nan_inf(output, f"output {i}")
