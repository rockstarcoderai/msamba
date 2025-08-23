"""Test SGSM stability constraints."""

import torch
import pytest
from src.models.sgsm import SGSM
from src.utils.config import ModelConfig
from . import create_dummy_batch, assert_no_nan_inf


class TestSGSMStability:
    """Test SGSM stability constraints."""
    
    @pytest.fixture
    def config(self):
        return ModelConfig()
    
    @pytest.fixture  
    def sgsm(self, config):
        return SGSM(
            d_model=config.d_model,
            vad_dim=config.vad_dim,
            polarity_dim=config.polarity_dim,
            alpha=config.sgsm_alpha,
            dropout=config.sgsm_drop,
        )
    
    def test_gamma_beta_bounds(self, sgsm):
        """Test gamma and beta parameter bounds."""
        batch = create_dummy_batch(batch_size=8, seq_len=50)
        
        # Create dummy sentiment features
        vad_features = torch.randn(8, 50, 3)  # VAD
        polarity_features = torch.randn(8, 50, 1)  # Polarity
        
        gamma, beta = sgsm.compute_modulation(vad_features, polarity_features)
        
        # Test bounds
        assert torch.all(gamma >= 1 - sgsm.alpha), "Gamma lower bound violated"
        assert torch.all(gamma <= 1 + sgsm.alpha), "Gamma upper bound violated"
        
        assert torch.all(beta >= -sgsm.alpha), "Beta lower bound violated"
        assert torch.all(beta <= sgsm.alpha), "Beta upper bound violated"
        
        # Test no NaN/Inf
        assert_no_nan_inf(gamma, "gamma")
        assert_no_nan_inf(beta, "beta")
    
    def test_zero_initialization(self, config):
        """Test zero initialization of SGSM parameters."""
        sgsm = SGSM(
            d_model=config.d_model,
            vad_dim=config.vad_dim,
            polarity_dim=config.polarity_dim,
            alpha=config.sgsm_alpha,
            dropout=config.sgsm_drop,
        )
        
        # Check that raw parameters are zero-initialized
        assert torch.allclose(sgsm.gamma_proj.weight, torch.zeros_like(sgsm.gamma_proj.weight), atol=1e-6)
        assert torch.allclose(sgsm.beta_proj.weight, torch.zeros_like(sgsm.beta_proj.weight), atol=1e-6)
        
        # Check that biases are zero if present
        if sgsm.gamma_proj.bias is not None:
            assert torch.allclose(sgsm.gamma_proj.bias, torch.zeros_like(sgsm.gamma_proj.bias), atol=1e-6)
        if sgsm.beta_proj.bias is not None:
            assert torch.allclose(sgsm.beta_proj.bias, torch.zeros_like(sgsm.beta_proj.bias), atol=1e-6)
    
    def test_gradient_clipping_compatibility(self, sgsm):
        """Test compatibility with gradient clipping."""
        batch_size, seq_len = 4, 50
        
        # Create inputs that require gradients
        vad_features = torch.randn(batch_size, seq_len, 3, requires_grad=True)
        polarity_features = torch.randn(batch_size, seq_len, 1, requires_grad=True)
        x = torch.randn(batch_size, seq_len, sgsm.d_model, requires_grad=True)
        
        # Forward pass
        output = sgsm(x, vad_features, polarity_features)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist and are finite
        for param in sgsm.parameters():
            if param.grad is not None:
                assert_no_nan_inf(param.grad, f"gradient for {param}")
                assert param.grad.norm() < 1000, f"Gradient too large: {param.grad.norm()}"

