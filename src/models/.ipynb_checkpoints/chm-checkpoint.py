# src/models/chm.py
"""
Cross-Modal Hybrid Mamba (CHM) block implementation.
Handles cross-modal interaction with centralized conditioning and optional self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from .ssm_core import SelectiveSSM
from .memory import EMAMemory

class CHMBlock(nn.Module):
    """
    Cross-Modal Hybrid Mamba block.
    Implements centralized cross-modal interaction with optional self-attention and EMA memory.
    """
    
    def __init__(
        self,
        dim: int,
        heads: int = 1,
        use_self_attn: bool = True,
        central_modality: str = "text",
        use_memory: bool = True,
        memory_tau: float = 0.9,
        drop_memory: float = 0.2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.heads = heads
        self.use_self_attn = use_self_attn
        self.central_modality = central_modality
        self.use_memory = use_memory
        
        # Cross-modal SSM for fusion
        self.cross_modal_ssm = SelectiveSSM(dim)
        
        # Optional self-attention with small dimension
        if use_self_attn and heads > 0:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True
            )
            self.attn_norm = nn.LayerNorm(dim)
        
        # Dimension alignment projections
        self.input_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        
        # Class token extraction
        self.cross_modal_cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # EMA Memory
        if use_memory:
            self.memory = EMAMemory(
                memory_dim=dim,
                tau=memory_tau,
                drop_memory_prob=drop_memory
            )
        
        # Normalization layers
        self.input_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        clip_ids: Optional[List[str]] = None,
        segment_times: Optional[List[float]] = None,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform cross-modal interaction.
        
        Args:
            modality_features: Dict mapping modality names to features (B, L, D)
            clip_ids: List of clip identifiers for memory management
            segment_times: List of segment timestamps
            training: Whether in training mode
            
        Returns:
            Tuple of (cross_modal_cls_token, cross_modal_features)
        """
        batch_size = next(iter(modality_features.values())).size(0)
        device = next(iter(modality_features.values())).device
        
        # Get central modality
        if self.central_modality not in modality_features:
            # Fallback to first available modality
            central_mod = next(iter(modality_features.keys()))
            warnings.warn(f"Central modality '{self.central_modality}' not found, using '{central_mod}'")
        else:
            central_mod = self.central_modality
        
        central_features = modality_features[central_mod]  # (B, L_central, D)
        
        # Memory conditioning if enabled
        memory_context = None
        if self.use_memory and clip_ids is not None:
            memory_states = []
            for clip_id in clip_ids:
                mem_state = self.memory.read_memory(clip_id)
                memory_states.append(mem_state)
            memory_context = torch.stack(memory_states, dim=0)  # (B, D)
        
        cross_modal_outputs = {}
        
        # Process each non-central modality
        for mod_name, features in modality_features.items():
            if mod_name == central_mod:
                continue
                
            # Concatenate with central modality
            B, L_mod, D = features.shape
            L_central = central_features.size(1)
            
            # Align sequence lengths by padding/truncating
            if L_mod != L_central:
                if L_mod < L_central:
                    # Pad modality features
                    pad_length = L_central - L_mod
                    features = F.pad(features, (0, 0, 0, pad_length))
                else:
                    # Truncate modality features
                    features = features[:, :L_central]
            
            # Concatenate along sequence dimension
            concat_features = torch.cat([features, central_features], dim=1)  # (B, 2*L, D)
            
            # Add memory context if available
            if memory_context is not None:
                # Expand memory to sequence length and add
                memory_expanded = memory_context.unsqueeze(1).expand(-1, concat_features.size(1), -1)
                concat_features = concat_features + 0.1 * memory_expanded
            
            # Normalize and project
            concat_features = self.input_norm(concat_features)
            concat_features = self.input_proj(concat_features)
            
            # Cross-modal SSM processing
            ssm_output = self.cross_modal_ssm(concat_features)
            
            # Optional self-attention refinement
            if self.use_self_attn and hasattr(self, 'self_attn'):
                attn_output, _ = self.self_attn(ssm_output, ssm_output, ssm_output)
                ssm_output = ssm_output + self.attn_norm(attn_output)
            
            # Output projection
            cross_modal_output = self.output_proj(ssm_output)
            cross_modal_outputs[f"{mod_name}_x_{central_mod}"] = cross_modal_output
        
        # Generate cross-modal CLS token by attending to all cross-modal outputs
        if cross_modal_outputs:
            # Stack all cross-modal features
            all_cross_modal = torch.stack(list(cross_modal_outputs.values()), dim=0)  # (N_pairs, B, L, D)
            cross_modal_pooled = all_cross_modal.mean(dim=(0, 2))  # (B, D) - mean over pairs and sequence
        else:
            # Fallback: use central modality mean
            cross_modal_pooled = central_features.mean(dim=1)  # (B, D)
        
        # Apply output normalization
        cross_modal_cls = self.output_norm(cross_modal_pooled)
        
        # Update memory if enabled
        if self.use_memory and clip_ids is not None:
            for i, clip_id in enumerate(clip_ids):
                segment_time = segment_times[i] if segment_times is not None else None
                self.memory.write_memory(
                    clip_id=clip_id,
                    new_state=cross_modal_cls[i],
                    segment_time=segment_time,
                    training=training
                )
        
        weights = None
        if self.use_memory:
            weights = memory_context
            
        return cross_modal_cls, cross_modal_outputs