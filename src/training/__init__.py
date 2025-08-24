"""Training utilities and components."""

from .trainer import MSAmbaTrainer as Trainer
from .losses import (
    MultimodalLoss,
    SentimentLoss,
    ConsistencyLoss,
    KLRegularizer,
)
from .metrics import (
    SentimentMetrics,
    compute_accuracy,
    compute_f1,
    compute_mae,
    compute_correlation,
    bootstrap_confidence,
)

__all__ = [
    # Trainer
    "Trainer",
    # Losses
    "MultimodalLoss",
    "SentimentLoss", 
    "ConsistencyLoss",
    "KLRegularizer",
    # Metrics
    "SentimentMetrics",
    "compute_accuracy",
    "compute_f1",
    "compute_mae", 
    "compute_correlation",
    "bootstrap_confidence",
]