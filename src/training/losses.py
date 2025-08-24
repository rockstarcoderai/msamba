"""Loss functions for multimodal sentiment analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SentimentLoss(nn.Module):
    """Multi-scale sentiment prediction loss."""
    
    def __init__(
        self,
        regression_weight: float = 1.0,
        classification_weight: float = 1.0,
        label_smoothing: float = 0.1,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.label_smoothing = label_smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def focal_loss(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model predictions
                - regression: [B, 1] continuous sentiment scores
                - classification: [B, C] class logits
            targets: Ground truth labels
                - regression: [B, 1] continuous scores
                - classification: [B] class indices
        
        Returns:
            Dictionary of loss components
        """
        losses = {}
        total_loss = 0.0
        
        # Regression loss (MSE + MAE)
        if 'regression' in outputs and 'regression' in targets:
            reg_pred = outputs['regression']
            reg_target = targets['regression']
            
            mse = self.mse_loss(reg_pred, reg_target)
            mae = self.mae_loss(reg_pred, reg_target)
            
            reg_loss = 0.7 * mse + 0.3 * mae
            losses['regression'] = reg_loss
            total_loss += self.regression_weight * reg_loss
        
        # Classification loss (CE + Focal)
        if 'classification' in outputs and 'classification' in targets:
            cls_pred = outputs['classification']
            cls_target = targets['classification']
            
            ce = self.ce_loss(cls_pred, cls_target)
            focal = self.focal_loss(cls_pred, cls_target)
            
            cls_loss = 0.6 * ce + 0.4 * focal
            losses['classification'] = cls_loss
            total_loss += self.classification_weight * cls_loss
        
        losses['total'] = total_loss
        return losses


class ConsistencyLoss(nn.Module):
    """Cross-modal and temporal consistency regularization."""
    
    def __init__(
        self,
        modal_weight: float = 0.1,
        temporal_weight: float = 0.05,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.modal_weight = modal_weight
        self.temporal_weight = temporal_weight
        self.temperature = temperature
    
    def modal_consistency(
        self,
        embeddings: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encourage similar predictions across modalities."""
        modalities = list(embeddings.keys())
        if len(modalities) < 2:
            return torch.tensor(0.0, device=next(iter(embeddings.values())).device)
        
        total_loss = 0.0
        pairs = 0
        
        for i, mod_i in enumerate(modalities):
            for j, mod_j in enumerate(modalities[i+1:], i+1):
                emb_i = F.normalize(embeddings[mod_i], dim=-1)
                emb_j = F.normalize(embeddings[mod_j], dim=-1)
                
                # Cosine similarity loss
                sim = F.cosine_similarity(emb_i, emb_j, dim=-1)
                loss = 1 - sim.mean()
                
                total_loss += loss
                pairs += 1
        
        return total_loss / max(pairs, 1)
    
    def temporal_consistency(
        self,
        predictions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Smooth temporal transitions in predictions."""
        if predictions.size(1) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute temporal differences
        diff = predictions[:, 1:] - predictions[:, :-1]
        temporal_loss = (diff ** 2).mean()
        
        if mask is not None:
            # Only consider valid timesteps
            valid_mask = mask[:, 1:] * mask[:, :-1]
            if valid_mask.sum() > 0:
                temporal_loss = (diff ** 2 * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum()
        
        return temporal_loss
    
    def forward(
        self,
        embeddings: Optional[Dict[str, torch.Tensor]] = None,
        predictions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: Per-modality embeddings for modal consistency
            predictions: Temporal predictions for smoothness
            mask: Valid timestep mask
        
        Returns:
            Dictionary of consistency losses
        """
        losses = {}
        total_loss = 0.0
        
        # Modal consistency
        if embeddings is not None:
            modal_loss = self.modal_consistency(embeddings)
            losses['modal'] = modal_loss
            total_loss += self.modal_weight * modal_loss
        
        # Temporal consistency
        if predictions is not None:
            temporal_loss = self.temporal_consistency(predictions, mask)
            losses['temporal'] = temporal_loss
            total_loss += self.temporal_weight * temporal_loss
        
        losses['total'] = total_loss
        return losses


class KLRegularizer(nn.Module):
    """KL divergence regularization for preventing label leakage."""
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
        self.kl_div = nn.KLDivLoss(reduction='batchmean', log_target=False)
    
    def forward(
        self,
        probe_logits: torch.Tensor,
        uniform_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Regularize sentiment probe to prevent label leakage.
        
        Args:
            probe_logits: [B, num_classes] logits from sentiment probe
            uniform_prior: [num_classes] uniform distribution
        
        Returns:
            KL divergence loss
        """
        if uniform_prior is None:
            num_classes = probe_logits.size(-1)
            uniform_prior = torch.ones(num_classes, device=probe_logits.device) / num_classes
        
        probe_probs = F.softmax(probe_logits, dim=-1)
        kl_loss = self.kl_div(
            probe_probs.log(),
            uniform_prior.expand_as(probe_probs)
        )
        
        return self.weight * kl_loss


class MultimodalLoss(nn.Module):
    """Combined loss function for enhanced MSAmba."""
    
    def __init__(
        self,
        sentiment_config: Optional[Dict] = None,
        consistency_config: Optional[Dict] = None,
        kl_config: Optional[Dict] = None,
    ):
        super().__init__()
        
        # Initialize component losses
        self.sentiment_loss = SentimentLoss(**(sentiment_config or {}))
        self.consistency_loss = ConsistencyLoss(**(consistency_config or {}))
        self.kl_regularizer = KLRegularizer(**(kl_config or {}))
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        embeddings: Optional[Dict[str, torch.Tensor]] = None,
        probe_logits: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined multimodal loss.
        
        Args:
            outputs: Model predictions
            targets: Ground truth targets
            embeddings: Per-modality embeddings
            probe_logits: Sentiment probe outputs
            mask: Valid sequence mask
        
        Returns:
            Dictionary of all loss components
        """
        all_losses = {}
        total_loss = 0.0
        
        # Main sentiment prediction loss
        sentiment_losses = self.sentiment_loss(outputs, targets)
        for key, loss in sentiment_losses.items():
            all_losses[f'sentiment_{key}'] = loss
        total_loss += sentiment_losses['total']
        
        # Consistency regularization
        consistency_losses = self.consistency_loss(
            embeddings=embeddings,
            predictions=outputs.get('regression'),
            mask=mask
        )
        for key, loss in consistency_losses.items():
            all_losses[f'consistency_{key}'] = loss
        total_loss += consistency_losses['total']
        
        # KL regularization for probe
        if probe_logits is not None:
            kl_loss = self.kl_regularizer(probe_logits)
            all_losses['kl_regularization'] = kl_loss
            total_loss += kl_loss
        
        all_losses['total'] = total_loss
        return all_losses


def create_loss_function(config: Dict) -> MultimodalLoss:
    """Factory function to create loss from configuration."""
    return MultimodalLoss(
        sentiment_config=config.get('sentiment', {}),
        consistency_config=config.get('consistency', {}),
        kl_config=config.get('kl_regularization', {})
    )


@dataclass
class LossConfig:
    """Simple configuration class for losses."""
    sentiment_weight: float = 1.0
    consistency_weight: float = 0.1
    kl_weight: float = 0.01
    regression_weight: float = 1.0
    classification_weight: float = 1.0