# enhanced_msamaba/src/models/hierarchical_model.py
"""
Hierarchical Enhanced MSAmba Model with 3-level processing architecture.

This module orchestrates the complete hierarchical processing pipeline:
- Level 1: Token-level ISM per modality
- Level 2: Segment-level Cross-Modal ISM with gated exchange
- Level 3: Clip/dialog-level final fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math
from dataclasses import dataclass

from .ism import ISMBlock
from .chm import CHMBlock
from .sgsm import SGSM
from .saliency import SaliencyScorer as EmotionSaliencyScorer
from .memory import EMAMemory


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical processing."""
    d_model: int = 768
    d_state: int = 64
    n_layers: int = 6
    hierarchy_levels: int = 3
    segment_len: float = 0.75
    
    # Modality settings
    modalities: List[str] = None
    central_mod: str = "text"
    
    # Component flags
    use_sgsm: bool = True
    use_saliency: bool = True
    use_memory: bool = True
    use_self_attn: bool = True
    
    # SGSM settings
    sgsm_alpha: float = 0.3
    sgsm_drop: float = 0.2
    
    # Saliency settings
    saliency_floor: float = 0.25
    
    # Memory settings
    memory_tau: float = 0.9
    drop_memory: float = 0.2
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ["text", "audio", "vision"]


class ModalityEncoder(nn.Module):
    """Encodes raw features for each modality with normalization."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.projection = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim]
        Returns:
            Encoded tensor [batch, seq_len, d_model]
        """
        x = self.projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class GatedCrossModalExchange(nn.Module):
    """Gated exchange mechanism for segment-level cross-modal interaction."""
    
    def __init__(self, d_model: int, modalities: List[str]):
        super().__init__()
        self.d_model = d_model
        self.modalities = modalities
        self.n_modalities = len(modalities)
        
        # Gates for each modality pair
        self.gates = nn.ModuleDict()
        for i, mod_i in enumerate(modalities):
            for j, mod_j in enumerate(modalities):
                if i != j:
                    key = f"{mod_i}_to_{mod_j}"
                    self.gates[key] = nn.Sequential(
                        nn.Linear(d_model * 2, d_model),
                        nn.Tanh(),
                        nn.Linear(d_model, d_model),
                        nn.Sigmoid()
                    )
    
    def forward(
        self, 
        modality_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply gated cross-modal exchange.
        
        Args:
            modality_features: Dict of {modality: features [batch, seq_len, d_model]}
            
        Returns:
            Updated modality features with cross-modal information
        """
        batch_size, seq_len = next(iter(modality_features.values())).shape[:2]
        updated_features = {}
        
        for mod_i in self.modalities:
            feat_i = modality_features[mod_i]
            updates = []
            
            for mod_j in self.modalities:
                if mod_i != mod_j:
                    feat_j = modality_features[mod_j]
                    
                    # Compute gate
                    concat_feat = torch.cat([feat_i, feat_j], dim=-1)
                    gate = self.gates[f"{mod_j}_to_{mod_i}"](concat_feat)
                    
                    # Apply gated update
                    update = gate * feat_j
                    updates.append(update)
            
            # Combine updates
            if updates:
                total_update = torch.stack(updates, dim=0).sum(dim=0)
                updated_features[mod_i] = feat_i + total_update
            else:
                updated_features[mod_i] = feat_i
                
        return updated_features


class SegmentProcessor(nn.Module):
    """Processes temporal segments with cross-modal interaction."""
    
    def __init__(
        self,
        config: HierarchicalConfig,
        level: int = 2
    ):
        super().__init__()
        self.config = config
        self.level = level
        self.modalities = config.modalities
        
        # Segment-level ISMs for each modality
        self.segment_isms = nn.ModuleDict()
        for modality in self.modalities:
            self.segment_isms[modality] = ISMBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand_factor=config.expand,
                use_glce=config.use_glce
            )
        
        # Cross-modal exchange
        self.cross_modal_exchange = GatedCrossModalExchange(
            config.d_model, 
            config.modalities
        )
        
        # Optional CHM for final segment fusion
        if len(self.modalities) > 1:
            self.chm = CHMBlock(
                dim=config.d_model,
                heads=config.num_heads,
                use_self_attn=config.use_self_attn,
                central_modality=config.central_mod,
                use_memory=config.use_memory,
                memory_tau=config.memory_tau,
                drop_memory=config.drop_memory
            )
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        sentiment_context: Optional[Dict[str, torch.Tensor]] = None,
        saliency_weights: Optional[Dict[str, torch.Tensor]] = None,
        memory_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Process segment-level features.
        
        Args:
            modality_features: Dict of modality features
            sentiment_context: Optional sentiment conditioning
            saliency_weights: Optional saliency weights
            memory_state: Optional memory state from previous segments
            
        Returns:
            Tuple of (processed_features, segment_representations, updated_memory)
        """
        # Apply segment-level ISM to each modality
        segment_features = {}
        segment_representations = {}
        
        for modality in self.modalities:
            features = modality_features[modality]
            
            # Apply saliency reweighting if available
            if saliency_weights is not None and modality in saliency_weights:
                features = features * saliency_weights[modality].unsqueeze(-1)
            
            # Apply sentiment context to features if available
            if sentiment_context is not None and modality in sentiment_context:
                sentiment_params = sentiment_context[modality]
                if isinstance(sentiment_params, dict) and 'gamma' in sentiment_params and 'beta' in sentiment_params:
                    # Apply FiLM conditioning: γ ⊙ x + β
                    gamma = sentiment_params['gamma'].unsqueeze(1).expand_as(features)
                    beta = sentiment_params['beta'].unsqueeze(1).expand_as(features)
                    features = gamma * features + beta
            
            # Process with segment ISM
            ism_output = self.segment_isms[modality](features)
            
            if isinstance(ism_output, tuple):
                processed, attention_weights = ism_output
                # ISM returns (output, attention_weights), but we need both processed and representation
                # Use the main output for both since attention_weights can be None
                representation = processed
            else:
                # Handle case where ISM returns single output
                processed = ism_output
                representation = ism_output
            
            segment_features[modality] = processed
            segment_representations[modality] = representation
        
        # Apply cross-modal exchange
        segment_features = self.cross_modal_exchange(segment_features)
        
        # Final segment-level fusion if CHM is available
        updated_memory = memory_state
        if hasattr(self, 'chm') and len(self.modalities) > 1:
            # CHM returns (cross_modal_cls_token, cross_modal_features)
            cls_token, fused_features = self.chm(segment_features)
            
            # Update segment representations with fused information
            for modality in self.modalities:
                if modality in fused_features:
                    segment_representations[modality] = fused_features[modality]
        
        return segment_features, segment_representations, updated_memory


class HierarchicalMSAmba(nn.Module):
    """
    Complete hierarchical Enhanced MSAmba model with 3-level processing.
    
    Architecture:
    - Level 1: Token-level ISM per modality
    - Level 2: Segment-level cross-modal processing
    - Level 3: Clip/dialog-level final fusion
    """
    
    def __init__(
        self,
        config: HierarchicalConfig,
        input_dims: Dict[str, int],
        num_classes: int = 1,  # For regression; set > 1 for classification
        task_type: str = "regression"  # "regression" or "classification"
    ):
        super().__init__()
        self.config = config
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.task_type = task_type
        self.modalities = config.modalities
        
        # Input encoders for each modality
        self.encoders = nn.ModuleDict()
        for modality in self.modalities:
            self.encoders[modality] = ModalityEncoder(
                input_dims[modality],
                config.d_model
            )
        
        # Level 1: Token-level ISMs
        self.token_isms = nn.ModuleDict()
        for modality in self.modalities:
            self.token_isms[modality] = ISMBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand_factor=config.expand,
                use_glce=config.use_glce
            )
        
        # Level 2: Segment processor (if hierarchy_levels >= 2)
        if config.hierarchy_levels >= 2:
            self.segment_processor = SegmentProcessor(config, level=2)
        
        # Level 3: Final fusion (if hierarchy_levels >= 3)
        if config.hierarchy_levels >= 3:
            self.final_chm = CHMBlock(
                dim=config.d_model,
                heads=config.num_heads,
                use_self_attn=config.use_self_attn,
                central_modality=config.central_mod,
                use_memory=config.use_memory,
                memory_tau=config.memory_tau,
                drop_memory=config.drop_memory
            )
        
        # Optional components
        if config.use_sgsm:
            self.sgsm = SGSM(
                hidden_dim=config.d_model,
                alpha=config.sgsm_alpha,
                dropout=config.sgsm_drop
            )
        
        if config.use_saliency:
            self.saliency_scorer = EmotionSaliencyScorer(
                input_dim=config.d_model,
                saliency_floor=config.saliency_floor
            )
        
        if config.use_memory:
            self.memory = EMAMemory(
                memory_dim=config.d_model,
                tau=config.memory_tau,
                drop_memory_prob=config.drop_memory
            )
        
        # Final classifier/regressor
        self.classifier = self._build_classifier()
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _build_classifier(self) -> nn.Module:
        """Build final classification/regression head."""
        classifier_input_dim = len(self.modalities) * self.config.d_model
        
        if self.task_type == "regression":
            return nn.Sequential(
                nn.Linear(classifier_input_dim, self.config.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.config.d_model, self.num_classes)
            )
        else:  # classification
            return nn.Sequential(
                nn.Linear(classifier_input_dim, self.config.d_model),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.config.d_model, self.num_classes)
            )
    
    def _init_weights(self, module: nn.Module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _segment_sequence(
        self, 
        features: torch.Tensor, 
        segment_length: int
    ) -> List[torch.Tensor]:
        """Segment sequence into overlapping windows."""
        batch_size, seq_len, d_model = features.shape
        segments = []
        
        # Create overlapping segments with 50% overlap
        step_size = segment_length // 2
        for start in range(0, seq_len - segment_length + 1, step_size):
            end = start + segment_length
            segments.append(features[:, start:end, :])
        
        # Handle remaining sequence
        if seq_len > segment_length and (seq_len - 1) % step_size != 0:
            segments.append(features[:, -segment_length:, :])
        
        return segments
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        missing_mask: Optional[Dict[str, torch.Tensor]] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical model.
        
        Args:
            inputs: Dict of {modality: features [batch, seq_len, input_dim]}
            missing_mask: Optional mask for missing modalities
            return_intermediates: Whether to return intermediate representations
            
        Returns:
            Dict containing predictions and optionally intermediate results
        """
        batch_size = next(iter(inputs.values())).shape[0]
        
        # Handle missing modalities
        available_modalities = []
        
        for modality in self.modalities:
            if modality in inputs:
                # Check if modality is missing - handle both tensor and boolean values
                is_missing = False
                if missing_mask is not None and modality in missing_mask:
                    missing_value = missing_mask[modality]
                    if torch.is_tensor(missing_value):
                        # Handle tensor values safely
                        if missing_value.numel() == 0:
                            is_missing = False
                        elif missing_value.dim() == 0:  # Scalar tensor
                            is_missing = missing_value.item()
                        else:  # Multi-dimensional tensor
                            # For batch processing, we need to check if ALL samples are missing this modality
                            # If any sample has this modality available, we should include it
                            is_missing = missing_value.all().item()
                    else:
                        is_missing = bool(missing_value)
                
                if not is_missing:
                    available_modalities.append(modality)
        
        if not available_modalities:
            raise ValueError("At least one modality must be available")
        
        # Encode inputs
        encoded_features = {}
        for modality in available_modalities:
            encoded_features[modality] = self.encoders[modality](inputs[modality])
        
        # Initialize memory if using memory component
        memory_state = None
        if hasattr(self, 'memory'):
            memory_state = {mod: None for mod in available_modalities}
        
        # Level 1: Token-level processing
        token_features = {}
        token_representations = {}
        
        for modality in available_modalities:
            features = encoded_features[modality]
            
            # Process with token-level ISM
            ism_output = self.token_isms[modality](features)
            
            if isinstance(ism_output, tuple):
                processed, attention_weights = ism_output
                # ISM returns (output, attention_weights), but we need both processed and representation
                # Use the main output for both since attention_weights can be None
                representation = processed
            else:
                # Handle case where ISM returns single output
                processed = ism_output
                representation = ism_output
            
            # Ensure we have valid tensors
            if processed is None:
                raise ValueError(f"ISM for modality '{modality}' returned None for processed output")
            if representation is None:
                raise ValueError(f"ISM for modality '{modality}' returned None for representation output")
            
            token_features[modality] = processed
            token_representations[modality] = representation
        
        # Generate sentiment context if using SGSM
        sentiment_context = None
        if hasattr(self, 'sgsm'):
            sentiment_context = {}
            for modality in available_modalities:
                # Use pooled representation for sentiment conditioning
                pooled = token_representations[modality].mean(dim=1)  # [batch, d_model]
                context = self.sgsm.get_conditioning_params(pooled)
                sentiment_context[modality] = context
        
        # Generate saliency weights if using saliency
        saliency_weights = None
        if hasattr(self, 'saliency_scorer'):
            # Pass all modality features to saliency scorer at once
            _, saliency_scores = self.saliency_scorer(token_features, return_scores=True)
            saliency_weights = saliency_scores
        
        current_features = token_features
        current_representations = token_representations
        
        # Level 2: Segment-level processing (if enabled)
        if self.config.hierarchy_levels >= 2 and hasattr(self, 'segment_processor'):
            segment_features, segment_representations, memory_state = self.segment_processor(
                current_features,
                sentiment_context=sentiment_context,
                saliency_weights=saliency_weights,
                memory_state=memory_state
            )
            current_features = segment_features
            current_representations = segment_representations
        
        # Level 3: Final fusion (if enabled)
        if self.config.hierarchy_levels >= 3 and hasattr(self, 'final_chm'):
            # CHM returns (cross_modal_cls_token, cross_modal_features)
            cls_token, fused_features = self.final_chm(current_features)
            
            # Use fused features for final representation
            for modality in available_modalities:
                if modality in fused_features:
                    current_representations[modality] = fused_features[modality].mean(dim=1)
        else:
            # Pool current features for final representation
            for modality in available_modalities:
                current_representations[modality] = current_features[modality].mean(dim=1)
        
        # Combine modality representations
        combined_repr = torch.cat([
            current_representations[mod] for mod in available_modalities
        ], dim=-1)
        
        # Handle missing modalities by padding
        if len(available_modalities) < len(self.modalities):
            missing_dim = (len(self.modalities) - len(available_modalities)) * self.config.d_model
            padding = torch.zeros(
                batch_size, missing_dim,
                device=combined_repr.device,
                dtype=combined_repr.dtype
            )
            combined_repr = torch.cat([combined_repr, padding], dim=-1)
        
        # Final prediction
        predictions = self.classifier(combined_repr)
        
        # Pool predictions across sequence dimension to get single prediction per sample
        if predictions.dim() > 2:  # If predictions have sequence dimension
            predictions = predictions.mean(dim=1)  # Pool across sequence dimension
        
        # Prepare outputs - match loss function expectations
        if self.task_type == "regression":
            outputs = {"regression": predictions}
        else:  # classification
            outputs = {"classification": predictions}
        
        if return_intermediates:
            outputs.update({
                "token_features": token_features,
                "token_representations": token_representations,
                "current_features": current_features,
                "current_representations": current_representations,
                "sentiment_context": sentiment_context,
                "saliency_weights": saliency_weights,
                "combined_representation": combined_repr
            })
        
        return outputs
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }