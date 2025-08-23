"""Logging utilities for training and evaluation."""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.nn as nn
from datetime import datetime
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Setup logging configuration."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Add file handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Level: {log_level}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger for specific module."""
    return logging.getLogger(name)


class WandbLogger:
    """Weights & Biases logger wrapper."""
    
    def __init__(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ):
        self.enabled = WANDB_AVAILABLE
        
        if self.enabled:
            wandb.init(
                project=project,
                name=name,
                entity=entity,
                config=config,
                tags=tags,
                notes=notes,
            )
            self.run = wandb.run
        else:
            print("Warning: wandb not available. Logging disabled.")
            self.run = None
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if self.enabled and self.run:
            wandb.log(metrics, step=step)
    
    def log_model(
        self,
        model: nn.Module,
        name: str,
        aliases: Optional[List[str]] = None,
    ):
        """Log model artifact."""
        if self.enabled and self.run:
            model_artifact = wandb.Artifact(name, type="model")
            
            # Save model state dict
            model_path = f"{name}.pth"
            torch.save(model.state_dict(), model_path)
            model_artifact.add_file(model_path)
            
            wandb.log_artifact(model_artifact, aliases=aliases)
            
            # Clean up
            os.remove(model_path)
    
    def watch_model(self, model: nn.Module, log_freq: int = 100):
        """Watch model gradients and parameters."""
        if self.enabled and self.run:
            wandb.watch(model, log_freq=log_freq)
    
    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.run:
            wandb.finish()


class TrainingLogger:
    """Training progress logger."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        wandb_config: Optional[Dict] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"
        
        # Setup file logging
        self.logger = setup_logging(log_file=str(self.log_file))
        
        # Setup wandb if configured
        self.wandb_logger = None
        if wandb_config:
            self.wandb_logger = WandbLogger(
                project=wandb_config.get('project', 'enhanced-msamaba'),
                name=experiment_name,
                entity=wandb_config.get('entity'),
                config=wandb_config.get('config', {}),
                tags=wandb_config.get('tags'),
                notes=wandb_config.get('notes'),
            )
        
        # Metrics storage
        self.metrics_history = {
            'train': [],
            'val': [],
            'test': [],
        }
        
        self.step = 0
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        self.logger.info(f"Epoch {epoch+1}/{total_epochs} started")
        
        if self.wandb_logger:
            self.wandb_logger.log({'epoch': epoch}, step=self.step)
    
    def log_batch(
        self,
        loss: float,
        metrics: Dict[str, float],
        phase: str = 'train',
        batch_idx: Optional[int] = None,
    ):
        """Log batch metrics."""
        log_dict = {'loss': loss, **metrics}
        
        # Add phase prefix
        prefixed_dict = {f"{phase}_{k}": v for k, v in log_dict.items()}
        
        if self.wandb_logger:
            self.wandb_logger.log(prefixed_dict, step=self.step)
        
        # Log to file periodically
        if batch_idx is not None and batch_idx % 50 == 0:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
            self.logger.info(f"{phase.upper()} Batch {batch_idx} | {metric_str}")
        
        self.step += 1
    
    def log_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        test_metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None,
    ):
        """Log epoch end metrics."""
        # Store metrics
        self.metrics_history['train'].append(train_metrics)
        if val_metrics:
            self.metrics_history['val'].append(val_metrics)
        if test_metrics:
            self.metrics_history['test'].append(test_metrics)
        
        # Format log message
        log_parts = [f"Epoch {epoch+1} completed"]
        
        if learning_rate:
            log_parts.append(f"LR: {learning_rate:.6f}")
        
        for phase, metrics in [('TRAIN', train_metrics), ('VAL', val_metrics), ('TEST', test_metrics)]:
            if metrics:
                metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                log_parts.append(f"{phase}: {metric_str}")
        
        self.logger.info(" | ".join(log_parts))
        
        # Log to wandb
        if self.wandb_logger:
            wandb_dict = {}
            
            if learning_rate:
                wandb_dict['learning_rate'] = learning_rate
            
            for phase, metrics in [('train', train_metrics), ('val', val_metrics), ('test', test_metrics)]:
                if metrics:
                    wandb_dict.update({f"{phase}_{k}": v for k, v in metrics.items()})
            
            self.wandb_logger.log(wandb_dict, step=self.step)
    
    def save_metrics(self):
        """Save metrics history to file."""
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        
        # Convert numpy types for JSON serialization
        serializable_history = {}
        for phase, metrics_list in self.metrics_history.items():
            serializable_history[phase] = []
            for metrics in metrics_list:
                serialized_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (np.integer, np.floating)):
                        serialized_metrics[k] = float(v)
                    else:
                        serialized_metrics[k] = v
                serializable_history[phase].append(serialized_metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
    
    def log_model_checkpoint(self, epoch: int, model: nn.Module, is_best: bool = False):
        """Log model checkpoint."""
        checkpoint_info = f"Model checkpoint saved for epoch {epoch+1}"
        if is_best:
            checkpoint_info += " (BEST)"
        
        self.logger.info(checkpoint_info)
        
        if self.wandb_logger and is_best:
            self.wandb_logger.log_model(
                model, 
                f"{self.experiment_name}_best",
                aliases=["best"]
            )
    
    def close(self):
        """Close logger and save final metrics."""
        self.save_metrics()
        
        if self.wandb_logger:
            self.wandb_logger.finish()


def log_model_summary(model: nn.Module, input_shape: Optional[Dict[str, tuple]] = None):
    """Log model architecture summary."""
    logger = get_logger(__name__)
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=" * 70)
    logger.info("MODEL ARCHITECTURE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model structure
    logger.info("\nModel structure:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        logger.info(f"  {name}: {type(module).__name__} ({module_params:,} params)")
    
    logger.info("=" * 70)


def log_training_stats(
    epoch: int,
    train_loss: float,
    train_metrics: Dict[str, float],
    val_loss: Optional[float] = None,
    val_metrics: Optional[Dict[str, float]] = None,
    lr: Optional[float] = None,
):
    """Log training statistics."""
    logger = get_logger(__name__)
    
    stats = [f"Epoch {epoch+1}"]
    
    if lr:
        stats.append(f"LR: {lr:.2e}")
    
    # Training stats
    train_str = f"Train Loss: {train_loss:.4f}"
    if train_metrics:
        train_metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        train_str += f" | {train_metric_str}"
    stats.append(train_str)
    
    # Validation stats
    if val_loss is not None:
        val_str = f"Val Loss: {val_loss:.4f}"
        if val_metrics:
            val_metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            val_str += f" | {val_metric_str}"
        stats.append(val_str)
    
    logger.info(" | ".join(stats))


def create_experiment_summary(
    config: Dict[str, Any],
    results: Dict[str, float],
    save_path: str,
):
    """Create experiment summary file."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'results': results,
        'git_commit': _get_git_commit(),
    }
    
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)


def _get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd='.')
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


class MetricsAggregator:
    """Aggregate metrics across multiple runs."""
    
    def __init__(self):
        self.runs = []
    
    def add_run(self, metrics: Dict[str, float], run_name: str = None):
        """Add metrics from a single run."""
        run_data = {'metrics': metrics}
        if run_name:
            run_data['name'] = run_name
        self.runs.append(run_data)
    
    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute aggregate statistics across runs."""
        if not self.runs:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for run in self.runs:
            all_metrics.update(run['metrics'].keys())
        
        stats = {}
        for metric in all_metrics:
            values = [run['metrics'].get(metric) for run in self.runs if metric in run['metrics']]
            values = [v for v in values if v is not None]
            
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values),
                }
        
        return stats
    
    def save_summary(self, save_path: str):
        """Save aggregated summary."""
        summary = {
            'runs': self.runs,
            'statistics': self.compute_statistics(),
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)