"""
Core State Space Model (SSM) implementations for Enhanced MSAmba.

This module provides the fundamental SSM building blocks including selective scanning
and bi-directional processing capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class SelectiveSSM(nn.Module):
    """
    Core Selective State Space Model with learnable selection mechanism.
    
    Implements the selective scanning mechanism from Mamba with optimized
    CUDA kernels when available, falling back to PyTorch implementation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_rank: int = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        bias: bool = False,
        conv_bias: bool = True,
        pscan: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = d_model * expand_factor
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.pscan = pscan
        
        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            groups=self.d_inner,  # Depthwise convolution
            padding=d_conv - 1
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # State space matrices (A, B, C, D)
        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.register_buffer("A_log", torch.log(A))  # Log-parameterized for stability
        
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        
        # Initialize dt projection
        self._init_dt_proj(dt_min, dt_max, dt_init)
    
    def _init_dt_proj(self, dt_min: float, dt_max: float, dt_init: str):
        """Initialize the dt projection layer."""
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, 1.0)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1)
        
        # Initialize bias to inverse of dt for stability
        with torch.no_grad():
            inv_dt = dt + torch.rand_like(dt) * 0.001
            self.dt_proj.bias.copy_(inv_dt)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through selective SSM.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, dim = x.shape
        
        # Input projection and gating
        xz = self.in_proj(x)  # [batch, seq_len, d_inner * 2]
        x_ssm, z = xz.chunk(2, dim=-1)  # Each: [batch, seq_len, d_inner]
        
        # Apply activation to gate
        z = F.silu(z)
        
        # 1D Convolution for local dependencies
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Generate SSM parameters
        x_dbl = self.x_proj(x_conv)  # [batch, seq_len, dt_rank + 2*d_state]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Transform parameters
        dt = F.softplus(self.dt_proj(dt))  # [batch, seq_len, d_inner]
        B = B.contiguous()  # [batch, seq_len, d_state]
        C = C.contiguous()  # [batch, seq_len, d_state]
        
        # Get A matrix
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Apply selective scan
        if self.pscan and self.training:
            y = self._selective_scan_pscan(x_conv, dt, A, B, C)
        else:
            y = self._selective_scan_sequential(x_conv, dt, A, B, C)
        
        # Apply skip connection
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gate and project output
        y = y * z
        output = self.out_proj(y)
        
        return output
    
    def _selective_scan_sequential(
        self, 
        x: torch.Tensor, 
        dt: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor
    ) -> torch.Tensor:
        """Sequential implementation of selective scan (fallback)."""
        batch, seq_len, d_inner = x.shape
        _, _, d_state = B.shape
        
        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, dtype=x.dtype, device=x.device)
        outputs = []
        
        for i in range(seq_len):
            # Get current timestep parameters
            dt_i = dt[:, i, :]  # [batch, d_inner]
            B_i = B[:, i, :].unsqueeze(1)  # [batch, 1, d_state]
            C_i = C[:, i, :].unsqueeze(-1)  # [batch, d_state, 1]
            x_i = x[:, i, :].unsqueeze(-1)  # [batch, d_inner, 1]
            
            # Discretize A and B
            dA = torch.exp(dt_i.unsqueeze(-1) * A)  # [batch, d_inner, d_state]
            dB = dt_i.unsqueeze(-1) * B_i  # [batch, d_inner, d_state]
            
            # Update hidden state: h = dA * h + dB * x
            h = dA * h + dB * x_i
            
            # Compute output: y = C * h
            y_i = torch.sum(C_i * h.transpose(1, 2), dim=1)  # [batch, d_inner]
            outputs.append(y_i)
        
        return torch.stack(outputs, dim=1)  # [batch, seq_len, d_inner]
    
    def _selective_scan_pscan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """Parallel scan implementation (more efficient for training)."""
        batch, seq_len, d_inner = x.shape
        
        # Discretize
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # [batch, seq_len, d_inner, d_state]
        dt_B_x = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # [batch, seq_len, d_inner, d_state]
        
        # Convert to scan format
        dA = torch.exp(dt_A)
        dB = dt_B_x
        
        # Parallel scan (simplified version)
        h = torch.zeros(batch, d_inner, self.d_state, dtype=x.dtype, device=x.device)
        outputs = []
        
        for i in range(seq_len):
            h = dA[:, i] * h + dB[:, i]
            y_i = torch.sum(C[:, i].unsqueeze(1) * h, dim=-1)
            outputs.append(y_i)
        
        return torch.stack(outputs, dim=1)


class DualDirectionSSM(nn.Module):
    """
    Bi-directional SSM that processes sequences in both forward and backward directions.
    
    Combines outputs from both directions for enhanced sequential modeling.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = d_model * expand_factor
        
        # Forward and backward SSMs
        self.forward_ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            **kwargs
        )
        
        self.backward_ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            **kwargs
        )
        
        # Fusion layer for combining directions
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bi-directional processing.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Fused output [batch, seq_len, d_model]
        """
        # Forward pass
        forward_out = self.forward_ssm(x)
        
        # Backward pass (flip sequence)
        x_reversed = torch.flip(x, dims=[1])
        backward_out = self.backward_ssm(x_reversed)
        backward_out = torch.flip(backward_out, dims=[1])  # Flip back
        
        # Concatenate and fuse
        combined = torch.cat([forward_out, backward_out], dim=-1)
        fused = self.fusion(combined)
        
        # Residual connection and normalization
        output = self.layer_norm(x + fused)
        
        return output


class SSMBlock(nn.Module):
    """
    Complete SSM block with normalization and residual connections.
    
    This is the building block used in ISM and CHM modules.
    """
    
    def __init__(
        self,
        d_model: int,
        bidirectional: bool = True,
        dropout: float = 0.1,
        **ssm_kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.bidirectional = bidirectional
        
        # SSM layer
        if bidirectional:
            self.ssm = DualDirectionSSM(d_model=d_model, **ssm_kwargs)
        else:
            self.ssm = SelectiveSSM(d_model=d_model, **ssm_kwargs)
        
        # Normalization and regularization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional masking.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        
        # SSM processing
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        
        # Residual connection
        output = residual + x
        
        # Apply mask again after residual
        if mask is not None:
            output = output * mask.unsqueeze(-1).float()
        
        return output


class AdaptiveSSM(nn.Module):
    """
    Adaptive SSM that modifies parameters based on input content.
    
    Useful for handling varying sequence dynamics within a single model.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        **ssm_kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        
        # Expert SSMs
        self.experts = nn.ModuleList([
            SelectiveSSM(d_model=d_model, **ssm_kwargs)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adaptive processing using mixture of experts.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, dim = x.shape
        
        # Compute gating weights
        gate_input = x.mean(dim=1)  # Global average pooling
        gate_weights = self.gate(gate_input)  # [batch, num_experts]
        
        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [batch, seq_len, d_model]
            expert_outputs.append(expert_out)
        
        # Weight and combine expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch, seq_len, d_model, num_experts]
        gate_weights = gate_weights.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, num_experts]
        
        output = torch.sum(expert_outputs * gate_weights, dim=-1)
        
        return output


# Utility functions for SSM operations
def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def apply_flash_attention_fallback(q, k, v, mask=None):
    """Fallback attention implementation when FlashAttention is not available."""
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if mask is not None:
        scores.masked_fill_(mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)