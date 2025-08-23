"""
Emotion-Saliency Soft Pruning Module for Enhanced MSAmba.

This module implements continuous attention reweighting based on emotion saliency
with safety constraints to prevent information loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SaliencyScorer(nn.Module):
    """
    Computes emotion-based saliency scores for multimodal tokens.
    
    Uses continuous soft pruning with floor constraints to prevent hard dropping
    of potentially important information.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_emotions: int = 6,  # Basic emotions
        saliency_floor: float = 0.25,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_emotions = num_emotions
        self.saliency_floor = saliency_floor
        
        # Emotion detection head
        self.emotion_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_emotions)
        )
        
        # Saliency scoring head
        self.saliency_head = nn.Sequential(
            nn.Linear(input_dim + num_emotions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Ensures output in [0,1]
        )
        
        # Residual projection for bypass
        self.residual_proj = nn.Linear(input_dim, input_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with careful scaling."""
        for module in [self.emotion_detector, self.saliency_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)  # Small gain for stability
                    nn.init.zeros_(layer.bias)
        
        # Zero-init residual projection
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)
    
    def forward(
        self,
        modality_tokens: Dict[str, torch.Tensor],
        return_scores: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Apply emotion-guided saliency pruning.
        
        Args:
            modality_tokens: Dict of tensors [batch, seq_len, dim]
            return_scores: Whether to return saliency scores
            
        Returns:
            Tuple of (pruned_tokens, saliency_scores)
        """
        pruned_tokens = {}
        saliency_scores = {} if return_scores else None
        
        for modality, tokens in modality_tokens.items():
            batch_size, seq_len, dim = tokens.shape
            
            # Flatten for processing
            flat_tokens = tokens.view(-1, dim)  # [batch*seq, dim]
            
            # Detect emotions
            emotion_logits = self.emotion_detector(flat_tokens)  # [batch*seq, num_emotions]
            emotion_probs = F.softmax(emotion_logits, dim=-1)
            
            # Compute saliency scores
            combined_features = torch.cat([flat_tokens, emotion_probs], dim=-1)
            raw_scores = self.saliency_head(combined_features).squeeze(-1)  # [batch*seq]
            
            # Apply floor constraint: ensure minimum retention
            clamped_scores = torch.clamp(raw_scores, min=self.saliency_floor)
            
            # Reshape back
            saliency_weights = clamped_scores.view(batch_size, seq_len, 1)  # [batch, seq, 1]
            
            # Apply soft pruning with residual bypass
            residual_component = self.residual_proj(tokens)  # Learnable residual
            pruned = saliency_weights * tokens + (1 - saliency_weights) * residual_component
            
            pruned_tokens[modality] = pruned
            
            if return_scores:
                saliency_scores[modality] = saliency_weights.squeeze(-1)  # [batch, seq]
        
        return pruned_tokens, saliency_scores
    
    def get_emotion_predictions(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get emotion predictions for analysis."""
        flat_tokens = tokens.view(-1, tokens.size(-1))
        emotion_logits = self.emotion_detector(flat_tokens)
        return F.softmax(emotion_logits, dim=-1).view(*tokens.shape[:-1], -1)
    
    def analyze_saliency_distribution(self, modality_tokens: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """Analyze saliency score distributions for monitoring."""
        with torch.no_grad():
            _, scores = self.forward(modality_tokens, return_scores=True)
            
            analysis = {}
            for modality, score_tensor in scores.items():
                flat_scores = score_tensor.flatten()
                analysis[modality] = {
                    'mean': float(flat_scores.mean()),
                    'std': float(flat_scores.std()),
                    'min': float(flat_scores.min()),
                    'max': float(flat_scores.max()),
                    'below_floor': float((flat_scores <= self.saliency_floor + 1e-6).float().mean()),
                    'above_half': float((flat_scores > 0.5).float().mean())
                }
            
            return analysis


class EmotionAwareFusion(nn.Module):
    """
    Emotion-aware fusion module that combines saliency pruning with cross-modal attention.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        saliency_floor: float = 0.25
    ):
        super().__init__()
        
        self.saliency_scorer = SaliencyScorer(
            input_dim=input_dim,
            saliency_floor=saliency_floor
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, modality_tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply saliency pruning followed by cross-modal attention fusion.
        
        Args:
            modality_tokens: Dict of modality tensors [batch, seq_len, dim]
            
        Returns:
            Fused representation [batch, seq_len, dim]
        """
        # Apply saliency pruning
        pruned_tokens, _ = self.saliency_scorer(modality_tokens)
        
        # Concatenate all modalities
        all_tokens = torch.cat([pruned_tokens[mod] for mod in sorted(pruned_tokens.keys())], dim=1)
        
        # Self-attention across all pruned tokens
        attended, _ = self.cross_attention(all_tokens, all_tokens, all_tokens)
        
        # Residual connection and normalization
        output = self.layer_norm(all_tokens + self.dropout(attended))
        
        return output