"""
Sentiment-Guided State Modulation (SGSM) for Enhanced MSAmba.

This module provides FiLM-style conditioning using VAD (valence-arousal-dominance)
sentiment features to modulate SSM computations without directly editing state matrices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import warnings


class SGSM(nn.Module):
    """
    Sentiment-Guided State Modulation using FiLM conditioning.
    
    Modulates projections and MLPs around SSMs based on sentiment features
    from a frozen probe, avoiding direct state matrix manipulation for stability.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        sentiment_dim: int = 4,  # VAD + polarity
        alpha: float = 0.3,
        dropout: float = 0.2,
        use_kl_regularization: bool = True,
        kl_weight: float = 0.01
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.sentiment_dim = sentiment_dim
        self.alpha = alpha
        self.dropout = dropout
        self.use_kl_regularization = use_kl_regularization
        self.kl_weight = kl_weight
        
        # Sentiment probe (frozen during training)
        self.sentiment_probe = SentimentProbe(hidden_dim, sentiment_dim)
        
        # FiLM modulation networks
        self.gamma_net = nn.Sequential(
            nn.Linear(sentiment_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Tanh()  # Bounded output for stability
        )
        
        self.beta_net = nn.Sequential(
            nn.Linear(sentiment_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Tanh()  # Bounded output for stability
        )
        
        # Reference sentiment distribution for KL regularization
        self.register_buffer('reference_sentiment', torch.zeros(sentiment_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with zero-init for stability."""
        for net in [self.gamma_net, self.beta_net]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def freeze_sentiment_probe(self):
        """Freeze sentiment probe to prevent label leakage."""
        for param in self.sentiment_probe.parameters():
            param.requires_grad = False
    
    def unfreeze_sentiment_probe(self):
        """Unfreeze sentiment probe (for pre-training)."""
        for param in self.sentiment_probe.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        target_sentiment: Optional[torch.Tensor] = None,
        return_sentiment: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply sentiment-guided modulation.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            target_sentiment: Ground truth sentiment for KL loss [batch, sentiment_dim]
            return_sentiment: Whether to return predicted sentiment
            
        Returns:
            Tuple of (modulated_states, kl_loss, predicted_sentiment)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Predict sentiment (stop gradient to prevent label leakage)
        with torch.no_grad():
            predicted_sentiment = self.sentiment_probe(hidden_states.detach())
        
        # Pool sentiment across sequence (mean pooling)
        pooled_sentiment = predicted_sentiment.mean(dim=1)  # [batch, sentiment_dim]
        
        # Generate FiLM parameters
        gamma_raw = self.gamma_net(pooled_sentiment)  # [batch, hidden_dim]
        beta_raw = self.beta_net(pooled_sentiment)    # [batch, hidden_dim]
        
        # Apply bounded scaling: gamma = 1 + α·tanh(gamma_raw), beta = α·tanh(beta_raw)
        gamma = 1.0 + self.alpha * gamma_raw  # Already bounded by tanh
        beta = self.alpha * beta_raw          # Already bounded by tanh
        
        # Reshape for broadcasting
        gamma = gamma.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden_dim]
        beta = beta.unsqueeze(1).expand(-1, seq_len, -1)    # [batch, seq_len, hidden_dim]
        
        # Apply FiLM modulation: γ ⊙ x + β
        modulated_states = gamma * hidden_states + beta
        
        # Compute KL regularization loss
        kl_loss = None
        if self.use_kl_regularization and self.training:
            if target_sentiment is not None:
                # KL divergence between predicted and target sentiment
                pred_dist = F.log_softmax(pooled_sentiment, dim=-1)
                target_dist = F.softmax(target_sentiment, dim=-1)
                kl_loss = F.kl_div(pred_dist, target_dist, reduction='batchmean')
            else:
                # KL divergence from reference distribution (regularization)
                pred_dist = F.log_softmax(pooled_sentiment, dim=-1)
                ref_dist = F.softmax(self.reference_sentiment.unsqueeze(0).expand(batch_size, -1), dim=-1)
                kl_loss = F.kl_div(pred_dist, ref_dist, reduction='batchmean')
            
            kl_loss = self.kl_weight * kl_loss
        
        # Return values
        sentiment_output = predicted_sentiment if return_sentiment else None
        
        return modulated_states, kl_loss, sentiment_output
    
    def get_conditioning_params(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get sentiment conditioning parameters (gamma, beta) for external use.
        
        Args:
            hidden_states: Input tensor [batch, hidden_dim]
            
        Returns:
            Dictionary with 'gamma' and 'beta' parameters [batch, hidden_dim]
        """
        # Predict sentiment
        with torch.no_grad():
            predicted_sentiment = self.sentiment_probe(hidden_states.detach())
        
        # Generate FiLM parameters
        gamma_raw = self.gamma_net(predicted_sentiment)  # [batch, hidden_dim]
        beta_raw = self.beta_net(predicted_sentiment)    # [batch, hidden_dim]
        
        # Apply bounded scaling: gamma = 1 + α·tanh(gamma_raw), beta = α·tanh(beta_raw)
        gamma = 1.0 + self.alpha * gamma_raw  # Already bounded by tanh
        beta = self.alpha * beta_raw          # Already bounded by tanh
        
        return {
            'gamma': gamma,
            'beta': beta,
            'sentiment': predicted_sentiment
        }
    
    def get_modulation_stats(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """Get statistics about modulation parameters for monitoring."""
        with torch.no_grad():
            predicted_sentiment = self.sentiment_probe(hidden_states.detach())
            pooled_sentiment = predicted_sentiment.mean(dim=1)
            
            gamma_raw = self.gamma_net(pooled_sentiment)
            beta_raw = self.beta_net(pooled_sentiment)
            
            gamma = 1.0 + self.alpha * gamma_raw
            beta = self.alpha * beta_raw
            
            return {
                'gamma_mean': float(gamma.mean()),
                'gamma_std': float(gamma.std()),
                'gamma_min': float(gamma.min()),
                'gamma_max': float(gamma.max()),
                'beta_mean': float(beta.mean()),
                'beta_std': float(beta.std()),
                'beta_min': float(beta.min()),
                'beta_max': float(beta.max()),
                'sentiment_norm': float(torch.norm(pooled_sentiment, dim=-1).mean())
            }


class SentimentProbe(nn.Module):
    """
    Lightweight sentiment probe for extracting VAD + polarity features.
    
    This probe is frozen during main training to prevent label leakage.
    """
    
    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        
        self.probe = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, output_dim)
        )
        
        # Initialize with small weights
        for layer in self.probe:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Extract sentiment features.
        
        Args:
            hidden_states: [batch, seq_len, input_dim]
            
        Returns:
            Sentiment features [batch, seq_len, output_dim]
        """
        return self.probe(hidden_states)


class AdaptiveSGSM(nn.Module):
    """
    Adaptive SGSM that modulates α based on sentiment confidence.
    
    Uses higher modulation strength when sentiment predictions are more confident.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        sentiment_dim: int = 4,
        base_alpha: float = 0.3,
        max_alpha: float = 0.5,
        **kwargs
    ):
        super().__init__()
        
        self.base_alpha = base_alpha
        self.max_alpha = max_alpha
        
        # Base SGSM module
        self.sgsm = SGSM(hidden_dim, sentiment_dim, alpha=base_alpha, **kwargs)
        
        # Confidence estimation network
        self.confidence_net = nn.Sequential(
            nn.Linear(sentiment_dim, sentiment_dim // 2),
            nn.ReLU(),
            nn.Linear(sentiment_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        target_sentiment: Optional[torch.Tensor] = None,
        return_sentiment: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply adaptive sentiment modulation."""
        
        # Get base prediction
        with torch.no_grad():
            predicted_sentiment = self.sgsm.sentiment_probe(hidden_states.detach())
            pooled_sentiment = predicted_sentiment.mean(dim=1)
        
        # Estimate confidence
        confidence = self.confidence_net(pooled_sentiment)  # [batch, 1]
        
        # Adapt alpha based on confidence
        adaptive_alpha = self.base_alpha + confidence.squeeze(-1) * (self.max_alpha - self.base_alpha)
        
        # Update SGSM alpha (this is a bit hacky, but works for demonstration)
        original_alpha = self.sgsm.alpha
        batch_size = hidden_states.size(0)
        
        # Apply different alpha per sample (simplified - in practice you'd want to vectorize this)
        outputs = []
        kl_losses = []
        sentiments = []
        
        for i in range(batch_size):
            self.sgsm.alpha = float(adaptive_alpha[i])
            
            out, kl, sent = self.sgsm(
                hidden_states[i:i+1],
                target_sentiment[i:i+1] if target_sentiment is not None else None,
                return_sentiment
            )
            
            outputs.append(out)
            kl_losses.append(kl)
            sentiments.append(sent)
        
        # Restore original alpha
        self.sgsm.alpha = original_alpha
        
        # Combine outputs
        final_output = torch.cat(outputs, dim=0)
        final_kl = torch.stack([kl for kl in kl_losses if kl is not None]).mean() if any(kl is not None for kl in kl_losses) else None
        final_sentiment = torch.cat(sentiments, dim=0) if return_sentiment and sentiments[0] is not None else None
        
        return final_output, final_kl, final_sentiment