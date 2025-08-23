"""
Intra-Modal Sequential Mamba (ISM) for Enhanced MSAmba.

Implements bi-directional selective scanning with Global-Local Context Extractor (GLCE)
for modeling sequential information within each modality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math

from .ssm_core import SelectiveSSM, DualDirectionSSM, SSMBlock


class GlobalLocalContextExtractor(nn.Module):
    """
    Global-Local Context Extractor (GLCE) for capturing multi-scale patterns.
    
    Combines global context via attention with local context via convolution
    to provide rich contextual representations for SSM processing.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        conv_kernel_sizes: list = [3, 5, 7],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.conv_kernel_sizes = conv_kernel_sizes
        
        # Global context via multi-head attention
        self.global_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Local context via multi-scale convolutions
        self.local_convs = nn.ModuleList()
        for kernel_size in conv_kernel_sizes:
            conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model // len(conv_kernel_sizes),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=d_model // len(conv_kernel_sizes)  # Depthwise conv for efficiency
            )
            self.local_convs.append(conv)
        
        # Fusion layers
        self.global_proj = nn.Linear(d_model, d_model // 2)
        self.local_proj = nn.Linear(d_model, d_model // 2)
        self.fusion = nn.Linear(d_model, d_model)
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract global and local context.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Context-enhanced tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        residual = x
        
        # Global context via self-attention
        if mask is not None:
            # Convert mask to attention mask format
            attn_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            attn_mask = ~attn_mask.bool()  # Invert for MultiheadAttention
        else:
            attn_mask = None
        
        global_context, _ = self.global_attention(x, x, x, key_padding_mask=mask if mask is not None else None)
        global_context = self.global_proj(global_context)
        
        # Local context via multi-scale convolutions
        x_transposed = x.transpose(1, 2)  # [batch, d_model, seq_len]
        local_features = []
        
        for conv in self.local_convs:
            local_feat = conv(x_transposed)  # [batch, d_model//num_convs, seq_len]
            local_feat = F.gelu(local_feat)
            local_features.append(local_feat)
        
        # Concatenate local features
        local_context = torch.cat(local_features, dim=1)  # [batch, d_model, seq_len]
        local_context = local_context.transpose(1, 2)  # [batch, seq_len, d_model]
        local_context = self.local_proj(local_context)
        
        # Fuse global and local context
        combined_context = torch.cat([global_context, local_context], dim=-1)
        fused_context = self.fusion(combined_context)
        
        # Apply mask if provided
        if mask is not None:
            fused_context = fused_context * mask.unsqueeze(-1).float()
        
        # Residual connection and normalization
        output = self.layer_norm(residual + self.dropout(fused_context))
        
        return output


class ISMBlock(nn.Module):
    """
    Intra-Modal Sequential Mamba Block.
    
    Combines GLCE with bi-directional selective scanning for comprehensive
    intra-modal sequence modeling.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        num_heads: int = 8,
        conv_kernel_sizes: list = [3, 5, 7],
        dropout: float = 0.1,
        use_bidirectional: bool = True,
        use_glce: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_bidirectional = use_bidirectional
        self.use_glce = use_glce
        
        # Global-Local Context Extractor
        if use_glce:
            self.glce = GlobalLocalContextExtractor(
                d_model=d_model,
                num_heads=num_heads,
                conv_kernel_sizes=conv_kernel_sizes,
                dropout=dropout
            )
        
        # SSM processing
        self.ssm_block = SSMBlock(
            d_model=d_model,
            bidirectional=use_bidirectional,
            dropout=dropout,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor
        )
        
        # Additional processing layers
        self.ffn = FeedForwardNetwork(d_model, dropout=dropout)
        
        # Normalization layers
        self.pre_ssm_norm = nn.LayerNorm(d_model)
        self.pre_ffn_norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process input through ISM block.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional sequence mask [batch, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Global-Local Context Extraction
        if self.use_glce:
            x = self.glce(x, mask=mask)
        
        # SSM processing with residual connection
        residual = x
        x_norm = self.pre_ssm_norm(x)
        ssm_out = self.ssm_block(x_norm, mask=mask)
        x = residual + ssm_out
        
        # Feed-forward network with residual connection
        residual = x
        x_norm = self.pre_ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out
        
        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        
        # Return attention weights if using GLCE and requested
        attention_weights = None
        if return_attention and self.use_glce:
            # This would need to be implemented in GLCE to return attention weights
            pass
        
        return x, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network with SwiGLU activation.
    
    Uses SwiGLU (Swish-Gated Linear Unit) for better performance.
    """
    
    def __init__(self, d_model: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        
        hidden_dim = d_model * ff_mult
        
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU FFN."""
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        hidden = gate * up  # Gating
        output = self.down_proj(hidden)
        return self.dropout(output)


class MultiModalISM(nn.Module):
    """
    Multi-modal ISM that processes multiple modalities in parallel.
    
    Each modality gets its own ISM processing while maintaining
    consistent architecture across modalities.
    """
    
    def __init__(
        self,
        modalities: list,
        d_model: int,
        num_layers: int = 4,
        **ism_kwargs
    ):
        super().__init__()
        
        self.modalities = modalities
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Create ISM blocks for each modality
        self.modality_isms = nn.ModuleDict()
        for modality in modalities:
            layers = nn.ModuleList([
                ISMBlock(d_model=d_model, **ism_kwargs)
                for _ in range(num_layers)
            ])
            self.modality_isms[modality] = layers
        
        # Modality-specific input projections
        self.input_projections = nn.ModuleDict()
        for modality in modalities:
            # These would be set based on actual input dimensions
            # For now, assume all inputs are projected to d_model
            self.input_projections[modality] = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process multiple modalities through their respective ISMs.
        
        Args:
            inputs: Dict of input tensors {modality: [batch, seq_len, dim]}
            masks: Optional dict of masks {modality: [batch, seq_len]}
            
        Returns:
            Dict of processed features {modality: [batch, seq_len, d_model]}
        """
        outputs = {}
        
        for modality, input_tensor in inputs.items():
            if modality not in self.modalities:
                continue
            
            # Project input to model dimension
            x = self.input_projections[modality](input_tensor)
            
            # Get mask for this modality
            mask = masks.get(modality) if masks else None
            
            # Process through ISM layers
            for ism_layer in self.modality_isms[modality]:
                x, _ = ism_layer(x, mask=mask)
            
            outputs[modality] = x
        
        return outputs


class AdaptiveISM(nn.Module):
    """
    Adaptive ISM that modifies processing based on input characteristics.
    
    Dynamically adjusts parameters like attention heads, conv kernels
    based on sequence properties.
    """
    
    def __init__(
        self,
        d_model: int,
        base_num_heads: int = 8,
        base_kernel_sizes: list = [3, 5, 7],
        adaptation_dim: int = 64,
        **ism_kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.base_num_heads = base_num_heads
        self.base_kernel_sizes = base_kernel_sizes
        
        # Adaptation network
        self.adaptation_net = nn.Sequential(
            nn.Linear(d_model, adaptation_dim),
            nn.ReLU(),
            nn.Linear(adaptation_dim, 3)  # [num_heads_scale, kernel_scale, depth_scale]
        )
        
        # Multiple ISM configurations
        self.light_ism = ISMBlock(d_model=d_model, num_heads=4, **ism_kwargs)
        self.standard_ism = ISMBlock(d_model=d_model, num_heads=8, **ism_kwargs)
        self.heavy_ism = ISMBlock(d_model=d_model, num_heads=16, **ism_kwargs)
        
        # Gating network for ISM selection
        self.ism_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Adaptive ISM processing.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional sequence mask [batch, seq_len]
            
        Returns:
            Processed tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute sequence representation for adaptation
        if mask is not None:
            # Masked average
            mask_expanded = mask.unsqueeze(-1).float()
            seq_repr = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Global average
            seq_repr = x.mean(dim=1)  # [batch, d_model]
        
        # Get adaptation parameters
        adaptation_params = torch.sigmoid(self.adaptation_net(seq_repr))  # [batch, 3]
        
        # Get ISM selection weights
        ism_weights = self.ism_gate(seq_repr)  # [batch, 3]
        
        # Apply different ISM configurations
        light_out, _ = self.light_ism(x, mask=mask)
        standard_out, _ = self.standard_ism(x, mask=mask)
        heavy_out, _ = self.heavy_ism(x, mask=mask)
        
        # Weight and combine outputs
        ism_weights = ism_weights.unsqueeze(1).unsqueeze(3)  # [batch, 1, 1, 3]
        ism_outputs = torch.stack([light_out, standard_out, heavy_out], dim=-1)  # [batch, seq, d_model, 3]
        
        adaptive_output = torch.sum(ism_outputs * ism_weights, dim=-1)
        
        return adaptive_output
    
    def get_adaptation_stats(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Get statistics about adaptation decisions."""
        with torch.no_grad():
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                seq_repr = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                seq_repr = x.mean(dim=1)
            
            adaptation_params = torch.sigmoid(self.adaptation_net(seq_repr))
            ism_weights = self.ism_gate(seq_repr)
            
            return {
                'avg_num_heads_scale': float(adaptation_params[:, 0].mean()),
                'avg_kernel_scale': float(adaptation_params[:, 1].mean()),
                'avg_depth_scale': float(adaptation_params[:, 2].mean()),
                'light_ism_usage': float(ism_weights[:, 0].mean()),
                'standard_ism_usage': float(ism_weights[:, 1].mean()),
                'heavy_ism_usage': float(ism_weights[:, 2].mean())
            }