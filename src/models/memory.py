"""
EMA Memory System for Enhanced MSAmba.

Provides temporal consistency through exponential moving average memory banks
with drop-memory regularization during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class EMAMemory(nn.Module):
    """
    Exponential Moving Average Memory Bank.
    
    Maintains a running average of representations for temporal consistency
    across clips with configurable decay and drop-memory regularization.
    """
    
    def __init__(
        self,
        memory_dim: int,
        memory_size: int = 32,
        tau: float = 0.9,
        drop_memory_prob: float = 0.2,
        temperature: float = 0.07,
        normalize: bool = True
    ):
        super().__init__()
        
        self.memory_dim = memory_dim
        self.memory_size = memory_size
        self.tau = tau
        self.drop_memory_prob = drop_memory_prob
        self.temperature = temperature
        self.normalize = normalize
        
        # Initialize memory bank
        self.register_buffer('memory_bank', torch.randn(memory_size, memory_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Normalize initial memory
        if normalize:
            self.memory_bank = F.normalize(self.memory_bank, dim=1)
    
    def forward(
        self,
        features: torch.Tensor,
        update_memory: bool = True,
        return_similarities: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process features through memory bank.
        
        Args:
            features: Input features [batch, seq_len, memory_dim]
            update_memory: Whether to update memory bank
            return_similarities: Whether to return memory similarities
            
        Returns:
            Tuple of (enhanced_features, similarities)
        """
        batch_size, seq_len, dim = features.shape
        
        # Normalize features if required
        if self.normalize:
            features_norm = F.normalize(features, dim=-1)
        else:
            features_norm = features
        
        # Reshape for memory operations
        flat_features = features_norm.view(-1, dim)  # [batch*seq_len, dim]
        
        # Compute similarities with memory bank
        memory_bank = self.memory_bank.clone().detach()
        if self.normalize:
            memory_bank = F.normalize(memory_bank, dim=1)
        
        similarities = torch.matmul(flat_features, memory_bank.T) / self.temperature  # [batch*seq_len, memory_size]
        memory_attention = F.softmax(similarities, dim=-1)
        
        # Retrieve from memory
        memory_retrieved = torch.matmul(memory_attention, memory_bank)  # [batch*seq_len, dim]
        
        # Combine with original features
        enhanced_flat = flat_features + memory_retrieved
        enhanced_features = enhanced_flat.view(batch_size, seq_len, dim)
        
        # Update memory bank during training
        if update_memory and self.training:
            self._update_memory(flat_features.detach())
        
        # Return similarities if requested
        similarities_out = None
        if return_similarities:
            similarities_out = similarities.view(batch_size, seq_len, self.memory_size)
        
        return enhanced_features, similarities_out
    
    def _update_memory(self, features: torch.Tensor):
        """Update memory bank with new features."""
        batch_size = features.size(0)
        
        # Apply drop-memory regularization
        if self.drop_memory_prob > 0 and self.training:
            keep_mask = torch.rand(batch_size, device=features.device) > self.drop_memory_prob
            features = features[keep_mask]
            if features.size(0) == 0:
                return
        
        # Update memory bank with EMA
        with torch.no_grad():
            # Get current memory pointer
            ptr = int(self.memory_ptr)
            
            # Update each feature
            for feat in features:
                # Normalize if required
                if self.normalize:
                    feat = F.normalize(feat.unsqueeze(0), dim=1).squeeze(0)
                
                # EMA update
                self.memory_bank[ptr] = self.tau * self.memory_bank[ptr] + (1 - self.tau) * feat
                
                # Advance pointer
                ptr = (ptr + 1) % self.memory_size
            
            # Update pointer
            self.memory_ptr[0] = ptr
    
    def clear_memory(self):
        """Clear memory bank (useful for new clips/episodes)."""
        with torch.no_grad():
            self.memory_bank = torch.randn_like(self.memory_bank)
            if self.normalize:
                self.memory_bank = F.normalize(self.memory_bank, dim=1)
            self.memory_ptr.zero_()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get statistics about memory bank for monitoring."""
        with torch.no_grad():
            memory = self.memory_bank
            if self.normalize:
                memory = F.normalize(memory, dim=1)
            
            # Compute pairwise similarities
            sim_matrix = torch.matmul(memory, memory.T)
            
            # Remove diagonal
            mask = ~torch.eye(self.memory_size, dtype=torch.bool, device=memory.device)
            off_diag_sims = sim_matrix[mask]
            
            return {
                'memory_norm_mean': float(torch.norm(memory, dim=1).mean()),
                'memory_norm_std': float(torch.norm(memory, dim=1).std()),
                'pairwise_sim_mean': float(off_diag_sims.mean()),
                'pairwise_sim_std': float(off_diag_sims.std()),
                'memory_ptr': int(self.memory_ptr),
                'memory_diversity': float(torch.det(torch.matmul(memory.T, memory)).clamp(min=1e-8).log())
            }


class ClipMemoryManager(nn.Module):
    """
    Manages separate memory banks for different clips/episodes.
    
    Maintains temporal consistency within clips while preventing
    cross-clip contamination.
    """
    
    def __init__(
        self,
        memory_dim: int,
        max_clips: int = 16,
        memory_size_per_clip: int = 32,
        **memory_kwargs
    ):
        super().__init__()
        
        self.memory_dim = memory_dim
        self.max_clips = max_clips
        self.memory_size_per_clip = memory_size_per_clip
        
        # Create memory banks for each clip
        self.clip_memories = nn.ModuleDict()
        self.active_clips = []
        self.memory_kwargs = memory_kwargs
    
    def get_or_create_memory(self, clip_id: str) -> EMAMemory:
        """Get existing memory or create new one for clip."""
        if clip_id not in self.clip_memories:
            # Remove oldest clip if at capacity
            if len(self.clip_memories) >= self.max_clips:
                oldest_clip = self.active_clips.pop(0)
                del self.clip_memories[oldest_clip]
            
            # Create new memory
            self.clip_memories[clip_id] = EMAMemory(
                memory_dim=self.memory_dim,
                memory_size=self.memory_size_per_clip,
                **self.memory_kwargs
            )
            self.active_clips.append(clip_id)
        
        return self.clip_memories[clip_id]
    
    def forward(
        self,
        features: torch.Tensor,
        clip_id: str,
        update_memory: bool = True
    ) -> torch.Tensor:
        """
        Process features through clip-specific memory.
        
        Args:
            features: Input features [batch, seq_len, memory_dim]
            clip_id: Identifier for current clip
            update_memory: Whether to update memory
            
        Returns:
            Enhanced features [batch, seq_len, memory_dim]
        """
        memory = self.get_or_create_memory(clip_id)
        enhanced_features, _ = memory(features, update_memory=update_memory)
        return enhanced_features
    
    def clear_clip_memory(self, clip_id: str):
        """Clear memory for specific clip."""
        if clip_id in self.clip_memories:
            self.clip_memories[clip_id].clear_memory()
    
    def clear_all_memories(self):
        """Clear all clip memories."""
        for memory in self.clip_memories.values():
            memory.clear_memory()
    
    def get_memory_summary(self) -> Dict[str, Dict]:
        """Get summary statistics for all memories."""
        summary = {}
        for clip_id, memory in self.clip_memories.items():
            summary[clip_id] = memory.get_memory_stats()
        return summary


class HierarchicalMemory(nn.Module):
    """
    Hierarchical memory system with different memory banks for different levels.
    
    Maintains separate memories for:
    - Token-level features (short-term)
    - Segment-level features (medium-term)  
    - Clip-level features (long-term)
    """
    
    def __init__(
        self,
        memory_dim: int,
        token_memory_size: int = 64,
        segment_memory_size: int = 32,
        clip_memory_size: int = 16,
        **memory_kwargs
    ):
        super().__init__()
        
        self.token_memory = EMAMemory(
            memory_dim=memory_dim,
            memory_size=token_memory_size,
            tau=0.8,  # Faster decay for short-term
            **memory_kwargs
        )
        
        self.segment_memory = EMAMemory(
            memory_dim=memory_dim,
            memory_size=segment_memory_size,
            tau=0.9,  # Medium decay
            **memory_kwargs
        )
        
        self.clip_memory = EMAMemory(
            memory_dim=memory_dim,
            memory_size=clip_memory_size,
            tau=0.95,  # Slower decay for long-term
            **memory_kwargs
        )
        
        # Fusion layers
        self.token_fusion = nn.Linear(memory_dim * 2, memory_dim)
        self.segment_fusion = nn.Linear(memory_dim * 2, memory_dim)
        self.final_fusion = nn.Linear(memory_dim * 3, memory_dim)
    
    def forward(
        self,
        token_features: torch.Tensor,
        segment_features: torch.Tensor,
        clip_features: torch.Tensor,
        level: str = "all"
    ) -> Dict[str, torch.Tensor]:
        """
        Process features through hierarchical memory.
        
        Args:
            token_features: Token-level features [batch, seq_len, dim]
            segment_features: Segment-level features [batch, num_segments, dim]
            clip_features: Clip-level features [batch, dim]
            level: Which level to process ("token", "segment", "clip", "all")
            
        Returns:
            Dict of enhanced features for each level
        """
        results = {}
        
        if level in ["token", "all"]:
            # Process token-level features
            enhanced_tokens, _ = self.token_memory(token_features)
            token_combined = torch.cat([token_features, enhanced_tokens], dim=-1)
            results["token"] = self.token_fusion(token_combined)
        
        if level in ["segment", "all"]:
            # Process segment-level features
            enhanced_segments, _ = self.segment_memory(segment_features)
            segment_combined = torch.cat([segment_features, enhanced_segments], dim=-1)
            results["segment"] = self.segment_fusion(segment_combined)
        
        if level in ["clip", "all"]:
            # Process clip-level features
            clip_features_expanded = clip_features.unsqueeze(1)  # Add seq dimension
            enhanced_clip, _ = self.clip_memory(clip_features_expanded)
            results["clip"] = enhanced_clip.squeeze(1)  # Remove seq dimension
        
        # Final fusion if processing all levels
        if level == "all":
            # Aggregate all levels
            token_agg = results["token"].mean(dim=1)  # Pool tokens
            segment_agg = results["segment"].mean(dim=1)  # Pool segments
            clip_agg = results["clip"]
            
            all_combined = torch.cat([token_agg, segment_agg, clip_agg], dim=-1)
            results["fused"] = self.final_fusion(all_combined)
        
        return results
    
    def clear_all_memories(self):
        """Clear all memory banks."""
        self.token_memory.clear_memory()
        self.segment_memory.clear_memory()
        self.clip_memory.clear_memory()


class ContextualMemory(nn.Module):
    """
    Contextual memory that adapts based on current input context.
    
    Uses attention mechanisms to selectively retrieve relevant memories.
    """
    
    def __init__(
        self,
        memory_dim: int,
        context_dim: int,
        memory_size: int = 64,
        num_heads: int = 8,
        tau: float = 0.9,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.memory_dim = memory_dim
        self.context_dim = context_dim
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.tau = tau
        self.temperature = temperature
        
        # Memory bank and keys
        self.register_buffer('memory_bank', torch.randn(memory_size, memory_dim))
        self.register_buffer('memory_keys', torch.randn(memory_size, context_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        
        # Attention mechanism for memory retrieval
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=context_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Context projection
        self.context_proj = nn.Linear(context_dim, memory_dim)
        
        # Output fusion
        self.fusion = nn.Linear(memory_dim * 2, memory_dim)
        
        # Initialize memories
        self._init_memories()
    
    def _init_memories(self):
        """Initialize memory bank and keys."""
        nn.init.xavier_uniform_(self.memory_bank)
        nn.init.xavier_uniform_(self.memory_keys)
    
    def forward(
        self,
        features: torch.Tensor,
        context: torch.Tensor,
        update_memory: bool = True
    ) -> torch.Tensor:
        """
        Retrieve from contextual memory.
        
        Args:
            features: Input features [batch, seq_len, memory_dim]
            context: Context for memory retrieval [batch, seq_len, context_dim]
            update_memory: Whether to update memory
            
        Returns:
            Enhanced features [batch, seq_len, memory_dim]
        """
        batch_size, seq_len, _ = features.shape
        
        # Reshape context for attention
        flat_context = context.view(-1, self.context_dim).unsqueeze(1)  # [batch*seq_len, 1, context_dim]
        
        # Memory keys as key/value
        memory_keys_exp = self.memory_keys.unsqueeze(0).expand(batch_size * seq_len, -1, -1)  # [batch*seq_len, memory_size, context_dim]
        
        # Attention-based memory retrieval
        attended_keys, attn_weights = self.memory_attention(
            flat_context,  # query
            memory_keys_exp,  # key
            memory_keys_exp,  # value
        )
        
        # Project attended keys to memory space
        memory_context = self.context_proj(attended_keys.squeeze(1))  # [batch*seq_len, memory_dim]
        
        # Retrieve memory values using attention weights
        attn_weights_flat = attn_weights.squeeze(1)  # [batch*seq_len, memory_size]
        retrieved_memory = torch.matmul(attn_weights_flat, self.memory_bank)  # [batch*seq_len, memory_dim]
        
        # Combine context and retrieved memory
        combined_memory = memory_context + retrieved_memory
        
        # Reshape back
        combined_memory = combined_memory.view(batch_size, seq_len, self.memory_dim)
        
        # Fuse with original features
        fused_input = torch.cat([features, combined_memory], dim=-1)
        enhanced_features = self.fusion(fused_input)
        
        # Update memory if requested
        if update_memory and self.training:
            self._update_contextual_memory(features.detach(), context.detach())
        
        return enhanced_features
    
    def _update_contextual_memory(self, features: torch.Tensor, context: torch.Tensor):
        """Update contextual memory bank."""
        batch_size, seq_len, _ = features.shape
        
        # Sample a few items to update (to avoid overwhelming memory)
        num_updates = min(batch_size * seq_len, self.memory_size // 4)
        indices = torch.randperm(batch_size * seq_len)[:num_updates]
        
        flat_features = features.view(-1, self.memory_dim)[indices]
        flat_context = context.view(-1, self.context_dim)[indices]
        
        with torch.no_grad():
            ptr = int(self.memory_ptr)
            
            for feat, ctx in zip(flat_features, flat_context):
                # EMA update for both memory and keys
                self.memory_bank[ptr] = self.tau * self.memory_bank[ptr] + (1 - self.tau) * feat
                self.memory_keys[ptr] = self.tau * self.memory_keys[ptr] + (1 - self.tau) * ctx
                
                ptr = (ptr + 1) % self.memory_size
            
            self.memory_ptr[0] = ptr
    
    def clear_memory(self):
        """Clear contextual memory."""
        with torch.no_grad():
            self._init_memories()
            self.memory_ptr.zero_()


# Utility functions for memory management
def create_memory_mask(seq_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Create mask for variable-length sequences in memory operations."""
    batch_size = seq_lengths.size(0)
    mask = torch.arange(max_len, device=seq_lengths.device).expand(
        batch_size, max_len
    ) < seq_lengths.unsqueeze(1)
    return mask


def compute_memory_efficiency(memory_module: nn.Module) -> Dict[str, float]:
    """Compute memory usage efficiency metrics."""
    stats = {}
    
    if hasattr(memory_module, 'memory_bank'):
        memory_bank = memory_module.memory_bank
        
        # Compute diversity (how different are the memory slots)
        if memory_bank.dim() == 2:
            # Normalize for fair comparison
            normalized_memory = F.normalize(memory_bank, dim=1)
            similarity_matrix = torch.matmul(normalized_memory, normalized_memory.T)
            
            # Remove diagonal
            mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=memory_bank.device)
            off_diagonal = similarity_matrix[mask]
            
            stats['diversity_score'] = 1.0 - float(off_diagonal.mean())  # Higher is more diverse
            stats['redundancy_score'] = float((off_diagonal > 0.9).float().mean())  # Fraction of highly similar pairs
            stats['memory_norm_variance'] = float(torch.norm(memory_bank, dim=1).var())
        
        # Memory utilization (how much of the memory is being used effectively)
        if hasattr(memory_module, 'memory_ptr'):
            ptr = int(memory_module.memory_ptr)
            stats['utilization_ratio'] = min(ptr / memory_bank.size(0), 1.0)
    
    return stats


def regularize_memory_diversity(memory_bank: torch.Tensor, diversity_weight: float = 0.01) -> torch.Tensor:
    """Compute regularization loss to encourage memory diversity."""
    if memory_bank.size(0) < 2:
        return torch.tensor(0.0, device=memory_bank.device)
    
    # Normalize memory vectors
    normalized_memory = F.normalize(memory_bank, dim=1)
    
    # Compute pairwise similarities
    similarity_matrix = torch.matmul(normalized_memory, normalized_memory.T)
    
    # Remove diagonal and compute diversity loss
    mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=memory_bank.device)
    off_diagonal = similarity_matrix[mask]
    
    # Penalize high similarities (encourage diversity)
    diversity_loss = diversity_weight * torch.mean(torch.clamp(off_diagonal - 0.5, min=0) ** 2)
    
    return diversity_loss