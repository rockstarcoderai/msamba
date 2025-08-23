# enhanced_msamaba/src/training/trainer.py
"""
Main training loop for Enhanced MSAmba with comprehensive logging and validation.
Supports multi-GPU training, gradient accumulation, and advanced scheduling.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import time
from pathlib import Path
import json
import wandb
from dataclasses import dataclass, asdict
from tqdm import tqdm
import os

from ..models.hierarchical_model import HierarchicalMSAmba, HierarchicalConfig
from .losses import MultimodalLoss, LossConfig
from .metrics import MetricsCalculator, MetricConfig
from ..utils.logging import setup_logger
from ..utils.profiling import ProfilerManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    # Basic training settings
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Advanced training settings
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 1000
    scheduler: str = "cosine"  # "cosine", "linear", "constant"
    
    # Optimization
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Regularization
    dropout: float = 0.1
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    
    # Validation and saving
    val_frequency: int = 1  # Validate every N epochs
    save_frequency: int = 5  # Save checkpoint every N epochs
    save_best_only: bool = True
    max_checkpoints: int = 5
    
    # Logging
    log_frequency: int = 100  # Log every N steps
    use_wandb: bool = False
    wandb_project: str = "enhanced_msamaba"
    wandb_name: Optional[str] = None
    
    # Profiling
    profile_training: bool = False
    profile_steps: int = 100
    
    # Multi-GPU
    use_ddp: bool = False
    local_rank: int = -1


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self, 
        patience: int = 15, 
        min_delta: float = 1e-4, 
        mode: str = "min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.compare = lambda x, y: x < y if mode == "min" else x > y
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.compare(score, self.best_score - self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        scaler: Optional[GradScaler],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        prefix: str = "checkpoint"
    ) -> str:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'model_config': getattr(model, 'config', None)
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        # Save checkpoint
        if is_best:
            filename = f"{prefix}_best.pt"
        else:
            filename = f"{prefix}_epoch_{epoch:03d}.pt"
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if not is_best:
            self.checkpoints.append((filepath, metrics.get('val_loss', float('inf'))))
            self._cleanup_old_checkpoints()
        
        logger.info(f"Saved checkpoint: {filepath}")
        return str(filepath)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by validation loss and remove worst checkpoints
            self.checkpoints.sort(key=lambda x: x[1])
            
            while len(self.checkpoints) > self.max_checkpoints:
                filepath, _ = self.checkpoints.pop(-1)
                if filepath.exists():
                    filepath.unlink()
                    logger.info(f"Removed old checkpoint: {filepath}")
    
    def load_checkpoint(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
        device: torch.device = None
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {filepath}")
        return checkpoint


class MSAmbaTrainer:
    """Main trainer for Enhanced MSAmba."""
    
    def __init__(
        self,
        model: HierarchicalMSAmba,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        loss_config: LossConfig,
        metric_config: MetricConfig,
        device: torch.device,
        output_dir: str
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Setup loss and metrics
        self.loss_fn = MultimodalLoss(loss_config).to(device)
        self.metrics_calculator = MetricsCalculator(metric_config)
        
        # Setup training components
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Setup utilities
        self.checkpoint_manager = CheckpointManager(
            self.output_dir / "checkpoints",
            config.max_checkpoints
        )
        self.early_stopping = EarlyStopping(
            config.early_stopping_patience,
            config.early_stopping_min_delta
        )
        
        # Setup profiler
        self.profiler = ProfilerManager() if config.profile_training else None
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'learning_rates': []
        }
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        if self.config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps
            )
        elif self.config.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.config.scheduler.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_steps
            )
        elif self.config.scheduler.lower() == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
        
        return scheduler
    
    def _setup_logging(self):
        """Setup logging and wandb."""
        # Setup wandb if requested
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                config=asdict(self.config)
            )
            wandb.watch(self.model)
        
        # Save config
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Setup progress bar
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.config.mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = self.model(
                        {k: v for k, v in batch.items() 
                         if k in ['text', 'audio', 'vision']},
                        missing_mask=batch.get('missing_mask'),
                        return_intermediates=True
                    )
                    loss_dict = self.loss_fn(outputs, batch)
                    loss = loss_dict['total_loss']
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.gradient_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            
            else:
                outputs = self.model(
                    {k: v for k, v in batch.items() 
                     if k in ['text', 'audio', 'vision']},
                    missing_mask=batch.get('missing_mask'),
                    return_intermediates=True
                )
                loss_dict = self.loss_fn(outputs, batch)
                loss = loss_dict['total_loss']
                
                # Backward pass
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_norm
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
            
            # Log periodically
            if self.global_step % self.config.log_frequency == 0:
                self._log_training_step(loss_dict, batch_idx, len(self.train_loader))
            
            # Profile if enabled
            if self.profiler and self.global_step < self.config.profile_steps:
                self.profiler.step()
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    {k: v for k, v in batch.items() 
                     if k in ['text', 'audio', 'vision']},
                    missing_mask=batch.get('missing_mask'),
                    return_intermediates=True
                )
                
                # Compute loss
                loss_dict = self.loss_fn(outputs, batch)
                total_loss += loss_dict['total_loss'].item()
                
                # Collect predictions and targets
                predictions = outputs['predictions'].cpu()
                targets = batch['labels'].cpu()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
                all_outputs.append({
                    k: v.cpu() if torch.is_tensor(v) else v 
                    for k, v in outputs.items()
                })
        
        # Compute metrics
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics_calculator.compute_metrics(predictions, targets)
        metrics['val_loss'] = total_loss / len(self.val_loader)
        
        return metrics, all_outputs
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.config.epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self.train_epoch()
                self.training_history['train_loss'].append(train_metrics['train_loss'])
                
                # Validation phase
                if epoch % self.config.val_frequency == 0:
                    val_metrics, val_outputs = self.validate()
                    self.training_history['val_loss'].append(val_metrics['val_loss'])
                    self.training_history['val_metrics'].append(val_metrics)
                    
                    # Log validation results
                    self._log_validation_epoch(train_metrics, val_metrics)
                    
                    # Check for best model
                    is_best = val_metrics['val_loss'] < self.best_val_score
                    if is_best:
                        self.best_val_score = val_metrics['val_loss']
                        logger.info(f"New best validation score: {self.best_val_score:.6f}")
                    
                    # Save checkpoint
                    if epoch % self.config.save_frequency == 0 or is_best:
                        self.checkpoint_manager.save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            epoch=epoch,
                            step=self.global_step,
                            metrics=val_metrics,
                            is_best=is_best
                        )
                    
                    # Early stopping check
                    if self.early_stopping(val_metrics['val_loss']):
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.training_history['learning_rates'].append(current_lr)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Save final results
            training_time = time.time() - start_time
            self._save_training_results(training_time)
            
            # Cleanup
            if self.config.use_wandb:
                wandb.finish()
            
            if self.profiler:
                self.profiler.export_chrome_trace(
                    str(self.output_dir / "profiling_trace.json")
                )
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        return {
            'training_history': self.training_history,
            'best_val_score': self.best_val_score,
            'total_epochs': self.current_epoch + 1,
            'training_time': training_time
        }
    
    def _log_training_step(
        self, 
        loss_dict: Dict[str, torch.Tensor], 
        batch_idx: int, 
        total_batches: int
    ):
        """Log training step information."""
        step_info = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'batch': f"{batch_idx}/{total_batches}",
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        # Add loss components
        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                step_info[key] = value.item()
        
        # Log to wandb
        if self.config.use_wandb:
            wandb.log(step_info, step=self.global_step)
        
        # Log to console occasionally
        if self.global_step % (self.config.log_frequency * 5) == 0:
            logger.info(f"Step {self.global_step}: " + 
                       ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                 for k, v in step_info.items()]))
    
    def _log_validation_epoch(
        self, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float]
    ):
        """Log epoch validation results."""
        epoch_info = {
            'epoch': self.current_epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **val_metrics
        }
        
        # Log to wandb
        if self.config.use_wandb:
            wandb.log(epoch_info, step=self.global_step)
        
        # Log to console
        train_loss = train_metrics.get('train_loss', 0.0)
        val_loss = val_metrics.get('val_loss', 0.0)
        val_mae = val_metrics.get('mae', 0.0)
        val_corr = val_metrics.get('correlation', 0.0)
        
        logger.info(
            f"Epoch {self.current_epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_mae={val_mae:.4f}, "
            f"val_corr={val_corr:.4f}"
        )
    
    def _save_training_results(self, training_time: float):
        """Save comprehensive training results."""
        results = {
            'config': asdict(self.config),
            'model_config': asdict(self.model.config) if hasattr(self.model, 'config') else {},
            'training_history': self.training_history,
            'best_val_score': self.best_val_score,
            'total_epochs': self.current_epoch + 1,
            'training_time': training_time,
            'global_steps': self.global_step,
            'model_size': self.model.get_model_size() if hasattr(self.model, 'get_model_size') else {}
        }
        
        # Save results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save training history as numpy arrays for easy loading
        history_path = self.output_dir / "training_history.npz"
        np.savez(
            history_path,
            train_loss=np.array(self.training_history['train_loss']),
            val_loss=np.array(self.training_history['val_loss']),
            learning_rates=np.array(self.training_history['learning_rates'])
        )
        
        logger.info(f"Saved training results to {results_path}")
    
    def resume_training(
        self, 
        checkpoint_path: str, 
        additional_epochs: int = 0
    ) -> Dict[str, Any]:
        """Resume training from a checkpoint."""
        logger.info(f"Resuming training from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.device
        )
        
        # Update training state
        start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['step']
        self.best_val_score = checkpoint['metrics'].get('val_loss', float('inf'))
        
        # Update config for additional epochs
        if additional_epochs > 0:
            self.config.epochs = start_epoch + additional_epochs
        
        # Continue training from the loaded epoch
        original_epochs = self.config.epochs
        self.config.epochs = self.config.epochs - start_epoch
        
        # Run training
        results = self.train()
        
        # Restore original config
        self.config.epochs = original_epochs
        
        return results


def create_trainer(
    model_config: HierarchicalConfig,
    training_config: TrainingConfig,
    loss_config: LossConfig,
    metric_config: MetricConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dims: Dict[str, int],
    num_classes: int = 1,
    task_type: str = "regression",
    device: torch.device = None,
    output_dir: str = "outputs"
) -> MSAmbaTrainer:
    """
    Create and configure trainer.
    
    Args:
        model_config: Model configuration
        training_config: Training configuration
        loss_config: Loss configuration
        metric_config: Metrics configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dims: Input dimensions for each modality
        num_classes: Number of output classes
        task_type: Task type ("regression" or "classification")
        device: Training device
        output_dir: Output directory for checkpoints and logs
        
    Returns:
        Configured trainer
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = HierarchicalMSAmba(
        config=model_config,
        input_dims=input_dims,
        num_classes=num_classes,
        task_type=task_type
    )
    
    # Create trainer
    trainer = MSAmbaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        loss_config=loss_config,
        metric_config=metric_config,
        device=device,
        output_dir=output_dir
    )
    
    return trainer


# Utility functions for common training scenarios
def train_with_cross_validation(
    model_config: HierarchicalConfig,
    training_config: TrainingConfig,
    loss_config: LossConfig,
    metric_config: MetricConfig,
    data_loaders: List[Tuple[DataLoader, DataLoader]],  # List of (train, val) pairs
    input_dims: Dict[str, int],
    num_classes: int = 1,
    task_type: str = "regression",
    device: torch.device = None,
    output_base_dir: str = "outputs/cv"
) -> Dict[str, Any]:
    """Train model with k-fold cross-validation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cv_results = {
        'fold_results': [],
        'mean_metrics': {},
        'std_metrics': {}
    }
    
    for fold, (train_loader, val_loader) in enumerate(data_loaders):
        logger.info(f"Training fold {fold + 1}/{len(data_loaders)}")
        
        # Create output directory for this fold
        fold_output_dir = Path(output_base_dir) / f"fold_{fold}"
        
        # Create trainer
        trainer = create_trainer(
            model_config=model_config,
            training_config=training_config,
            loss_config=loss_config,
            metric_config=metric_config,
            train_loader=train_loader,
            val_loader=val_loader,
            input_dims=input_dims,
            num_classes=num_classes,
            task_type=task_type,
            device=device,
            output_dir=str(fold_output_dir)
        )
        
        # Train
        fold_results = trainer.train()
        cv_results['fold_results'].append(fold_results)
    
    # Compute mean and std metrics across folds
    all_metrics = []
    for fold_result in cv_results['fold_results']:
        if fold_result['training_history']['val_metrics']:
            best_metrics = min(
                fold_result['training_history']['val_metrics'],
                key=lambda x: x.get('val_loss', float('inf'))
            )
            all_metrics.append(best_metrics)
    
    if all_metrics:
        # Compute statistics
        metric_names = set()
        for metrics in all_metrics:
            metric_names.update(metrics.keys())
        
        for metric_name in metric_names:
            values = [m.get(metric_name, 0.0) for m in all_metrics]
            cv_results['mean_metrics'][metric_name] = np.mean(values)
            cv_results['std_metrics'][metric_name] = np.std(values)
    
    logger.info("Cross-validation completed")
    logger.info(f"Mean validation metrics: {cv_results['mean_metrics']}")
    
    return cv_results


# Example usage
if __name__ == "__main__":
    # Example configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model config
    model_config = HierarchicalConfig(
        d_model=768,
        d_state=64,
        n_layers=6,
        hierarchy_levels=3,
        modalities=["text", "audio", "vision"],
        use_sgsm=True,
        use_saliency=True,
        use_memory=True
    )
    
    # Training config
    training_config = TrainingConfig(
        epochs=50,
        batch_size=16,
        learning_rate=1e-4,
        mixed_precision=True,
        use_wandb=False,
        early_stopping_patience=10
    )
    
    # Loss and metric configs
    from .losses import LossConfig
    from .metrics import MetricConfig
    
    loss_config = LossConfig()
    metric_config = MetricConfig()
    
    # Input dimensions (example for MOSI)
    input_dims = {
        "text": 768,
        "audio": 74,
        "vision": 47
    }
    
    print("Trainer setup completed successfully!")
    print(f"Device: {device}")
    print(f"Model config: {model_config}")
    print(f"Training config: {training_config}")