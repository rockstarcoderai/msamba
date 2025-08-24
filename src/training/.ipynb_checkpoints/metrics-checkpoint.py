"""Evaluation metrics for multimodal sentiment analysis."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    mean_absolute_error,
    classification_report,
)
from scipy.stats import pearsonr, spearmanr
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def compute_accuracy(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    threshold: Optional[float] = None,
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: [B, C] logits or [B] continuous scores
        targets: [B] class indices or [B] continuous scores
        threshold: For binary classification from continuous scores
    
    Returns:
        Accuracy score
    """
    if predictions.dim() == 2 and predictions.size(-1) > 1:
        # Multi-class classification
        pred_classes = predictions.argmax(dim=-1)
        return accuracy_score(targets.cpu().numpy(), pred_classes.cpu().numpy())
    else:
        # Binary classification from continuous scores
        if threshold is None:
            threshold = 0.0
        pred_binary = (predictions.squeeze() > threshold).long()
        target_binary = (targets.squeeze() > threshold).long()
        return accuracy_score(target_binary.cpu().numpy(), pred_binary.cpu().numpy())


def compute_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor, 
    average: str = 'weighted',
    threshold: Optional[float] = None,
) -> float:
    """
    Compute F1 score.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        average: 'micro', 'macro', 'weighted'
        threshold: For binary classification
    
    Returns:
        F1 score
    """
    if predictions.dim() == 2 and predictions.size(-1) > 1:
        pred_classes = predictions.argmax(dim=-1)
    else:
        if threshold is None:
            threshold = 0.0
        pred_classes = (predictions.squeeze() > threshold).long()
        targets = (targets.squeeze() > threshold).long()
    
    return f1_score(
        targets.cpu().numpy(), 
        pred_classes.cpu().numpy(), 
        average=average,
        zero_division=0,
    )


def compute_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Mean Absolute Error."""
    return mean_absolute_error(
        targets.cpu().numpy().flatten(),
        predictions.cpu().numpy().flatten()
    )


def compute_correlation(
    predictions: torch.Tensor, 
    targets: torch.Tensor
) -> Tuple[float, float]:
    """
    Compute Pearson and Spearman correlations.
    
    Returns:
        (pearson_r, spearman_r)
    """
    pred_np = predictions.cpu().numpy().flatten()
    target_np = targets.cpu().numpy().flatten()
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(pred_np) | np.isnan(target_np))
    pred_np = pred_np[valid_mask]
    target_np = target_np[valid_mask]
    
    if len(pred_np) < 2:
        return 0.0, 0.0
    
    try:
        pearson_r, _ = pearsonr(pred_np, target_np)
        spearman_r, _ = spearmanr(pred_np, target_np)
        return float(pearson_r), float(spearman_r)
    except:
        return 0.0, 0.0


def bootstrap_confidence(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a metric.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        metric_fn: Function to compute metric
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        (metric_value, lower_ci, upper_ci)
    """
    pred_np = predictions.cpu().numpy().flatten()
    target_np = targets.cpu().numpy().flatten()
    
    n_samples = len(pred_np)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        sample_pred = pred_np[indices]
        sample_target = target_np[indices]
        
        # Compute metric
        try:
            metric_value = metric_fn(sample_pred, sample_target)
            bootstrap_metrics.append(metric_value)
        except:
            continue
    
    if not bootstrap_metrics:
        return 0.0, 0.0, 0.0
    
    bootstrap_metrics = np.array(bootstrap_metrics)
    metric_value = metric_fn(pred_np, target_np)
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(bootstrap_metrics, lower_percentile)
    upper_ci = np.percentile(bootstrap_metrics, upper_percentile)
    
    return metric_value, lower_ci, upper_ci


class SentimentMetrics:
    """Comprehensive sentiment analysis metrics."""
    
    def __init__(self, num_classes: int = 7):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.all_predictions = []
        self.all_targets = []
        self.all_regression_preds = []
        self.all_regression_targets = []
    
    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ):
        """
        Update metrics with new batch.
        
        Args:
            outputs: Model predictions
            targets: Ground truth targets
        """
        # Classification metrics
        if 'classification' in outputs and 'classification' in targets:
            pred_classes = outputs['classification'].argmax(dim=-1)
            self.all_predictions.extend(pred_classes.cpu().tolist())
            self.all_targets.extend(targets['classification'].cpu().tolist())
        
        # Regression metrics
        if 'regression' in outputs and 'regression' in targets:
            self.all_regression_preds.extend(outputs['regression'].cpu().tolist())
            self.all_regression_targets.extend(targets['regression'].cpu().tolist())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        metrics = {}
        
        # Classification metrics
        if self.all_predictions and self.all_targets:
            predictions = np.array(self.all_predictions)
            targets = np.array(self.all_targets)
            
            # Accuracy
            if self.num_classes == 7:
                metrics['Acc7'] = accuracy_score(targets, predictions)
            elif self.num_classes == 2:
                metrics['Acc2'] = accuracy_score(targets, predictions)
            
            # F1 scores
            metrics['F1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
            metrics['F1_weighted'] = f1_score(targets, predictions, average='weighted', zero_division=0)
            
            # Per-class metrics
            try:
                report = classification_report(targets, predictions, output_dict=True, zero_division=0)
                for class_idx in range(self.num_classes):
                    if str(class_idx) in report:
                        metrics[f'F1_class_{class_idx}'] = report[str(class_idx)]['f1-score']
            except:
                pass
        
        # Regression metrics
        if self.all_regression_preds and self.all_regression_targets:
            reg_preds = np.array(self.all_regression_preds).flatten()
            reg_targets = np.array(self.all_regression_targets).flatten()
            
            # MAE
            metrics['MAE'] = mean_absolute_error(reg_targets, reg_preds)
            
            # Correlations
            if len(reg_preds) > 1:
                try:
                    pearson_r, _ = pearsonr(reg_preds, reg_targets)
                    spearman_r, _ = spearmanr(reg_preds, reg_targets)
                    metrics['Corr_Pearson'] = float(pearson_r) if not np.isnan(pearson_r) else 0.0
                    metrics['Corr_Spearman'] = float(spearman_r) if not np.isnan(spearman_r) else 0.0
                except:
                    metrics['Corr_Pearson'] = 0.0
                    metrics['Corr_Spearman'] = 0.0
        
        return metrics
    
    def compute_bootstrap_ci(
        self, 
        metric_name: str, 
        n_bootstrap: int = 1000
    ) -> Tuple[float, float, float]:
        """Compute bootstrap confidence intervals for a specific metric."""
        if metric_name == 'MAE' and self.all_regression_preds:
            preds = torch.tensor(self.all_regression_preds)
            targets = torch.tensor(self.all_regression_targets)
            return bootstrap_confidence(preds, targets, compute_mae, n_bootstrap)
        
        elif 'Acc' in metric_name and self.all_predictions:
            preds = torch.tensor(self.all_predictions)
            targets = torch.tensor(self.all_targets) 
            return bootstrap_confidence(preds, targets, 
                                      lambda p, t: accuracy_score(t.numpy(), p.numpy()),
                                      n_bootstrap)
        
        elif 'F1' in metric_name and self.all_predictions:
            preds = torch.tensor(self.all_predictions)
            targets = torch.tensor(self.all_targets)
            avg_type = 'macro' if 'macro' in metric_name else 'weighted'
            return bootstrap_confidence(preds, targets,
                                      lambda p, t: f1_score(t.numpy(), p.numpy(), 
                                                           average=avg_type, zero_division=0),
                                      n_bootstrap)
        
        return 0.0, 0.0, 0.0


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 7,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataloader: Data to evaluate on
        device: Computing device
        num_classes: Number of sentiment classes
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    metrics_tracker = SentimentMetrics(num_classes)
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device - handle nested dictionaries
            inputs = {}
            for k, v in batch.items():
                if k not in ['targets', 'regression_targets', 'classification_targets', 'labels', 'ids']:
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)
                    elif isinstance(v, dict):
                        # Handle nested modality dictionaries
                        for mod, tensor in v.items():
                            if isinstance(tensor, torch.Tensor):
                                inputs[mod] = tensor.to(device)
                            else:
                                inputs[mod] = tensor
                    else:
                        inputs[k] = v
            
            # Ensure we only pass modality inputs to the model with correct data types
            model_inputs = {}
            for modality in ['text', 'audio', 'vision']:
                if modality in inputs:
                    tensor = inputs[modality]
                    if isinstance(tensor, torch.Tensor):
                        # Convert to float32 if it's boolean or other type
                        if tensor.dtype != torch.float32:
                            tensor = tensor.float()
                        model_inputs[modality] = tensor
                    else:
                        model_inputs[modality] = tensor
            
            # Forward pass
            outputs = model(model_inputs)
            
            # Prepare targets
            targets = {}
            if 'regression_targets' in batch:
                targets['regression'] = batch['regression_targets'].to(device)
            if 'classification_targets' in batch:
                targets['classification'] = batch['classification_targets'].to(device)
            
            # Update metrics
            metrics_tracker.update(outputs, targets)
    
    return metrics_tracker.compute()


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric: str = 'Acc7',
    confidence: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    Statistical comparison between models.
    
    Args:
        results: {model_name: {metric_name: score}}
        metric: Metric to compare
        confidence: Confidence level for significance tests
    
    Returns:
        Comparison statistics
    """
    model_names = list(results.keys())
    comparisons = {}
    
    for i, model_a in enumerate(model_names):
        comparisons[model_a] = {}
        for j, model_b in enumerate(model_names):
            if i != j and metric in results[model_a] and metric in results[model_b]:
                score_a = results[model_a][metric]
                score_b = results[model_b][metric]
                
                # Simple difference for now
                # Could be extended with statistical tests
                diff = score_a - score_b
                comparisons[model_a][f'vs_{model_b}'] = diff
    
    return comparisons


def create_metrics_summary(
    metrics: Dict[str, float],
    bootstrap_results: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> str:
    """Create formatted summary of metrics."""
    summary = ["=" * 50]
    summary.append("EVALUATION METRICS SUMMARY")
    summary.append("=" * 50)
    
    # Group metrics by type
    accuracy_metrics = {k: v for k, v in metrics.items() if 'Acc' in k}
    f1_metrics = {k: v for k, v in metrics.items() if 'F1' in k}
    correlation_metrics = {k: v for k, v in metrics.items() if 'Corr' in k}
    error_metrics = {k: v for k, v in metrics.items() if 'MAE' in k}
    
    # Format each group
    if accuracy_metrics:
        summary.append("\nACCURACY METRICS:")
        for metric, value in accuracy_metrics.items():
            if bootstrap_results and metric in bootstrap_results:
                mean, lower, upper = bootstrap_results[metric]
                summary.append(f"  {metric}: {value:.4f} [{lower:.4f}, {upper:.4f}]")
            else:
                summary.append(f"  {metric}: {value:.4f}")
    
    if f1_metrics:
        summary.append("\nF1 METRICS:")
        for metric, value in f1_metrics.items():
            if bootstrap_results and metric in bootstrap_results:
                mean, lower, upper = bootstrap_results[metric]
                summary.append(f"  {metric}: {value:.4f} [{lower:.4f}, {upper:.4f}]")
            else:
                summary.append(f"  {metric}: {value:.4f}")
    
    if correlation_metrics:
        summary.append("\nCORRELATION METRICS:")
        for metric, value in correlation_metrics.items():
            summary.append(f"  {metric}: {value:.4f}")
    
    if error_metrics:
        summary.append("\nERROR METRICS:")
        for metric, value in error_metrics.items():
            if bootstrap_results and metric in bootstrap_results:
                mean, lower, upper = bootstrap_results[metric]
                summary.append(f"  {metric}: {value:.4f} [{lower:.4f}, {upper:.4f}]")
            else:
                summary.append(f"  {metric}: {value:.4f}")
    
    summary.append("=" * 50)
    return "\n".join(summary)


@dataclass
class MetricConfig:
    """Simple configuration class for metrics."""
    compute_accuracy: bool = True
    compute_f1: bool = True
    compute_mae: bool = True
    compute_correlation: bool = True
    f1_average: str = 'weighted'
    threshold: Optional[float] = None


class MetricsCalculator:
    """Simple metrics calculator class."""
    
    def __init__(self, config: MetricConfig):
        self.config = config
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all configured metrics."""
        metrics = {}
        
        if self.config.compute_accuracy:
            metrics['accuracy'] = compute_accuracy(predictions, targets, self.config.threshold)
        
        if self.config.compute_f1:
            metrics['f1'] = compute_f1(predictions, targets, self.config.f1_average, self.config.threshold)
        
        if self.config.compute_mae:
            metrics['mae'] = compute_mae(predictions, targets)
        
        if self.config.compute_correlation:
            pearson_r, spearman_r = compute_correlation(predictions, targets)
            metrics['pearson_r'] = pearson_r
            metrics['spearman_r'] = spearman_r
        
        return metrics