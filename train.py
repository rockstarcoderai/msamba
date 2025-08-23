#!/usr/bin/env python3
"""Training script for Enhanced MSAmba."""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import (
    Config, 
    load_config, 
    save_config, 
    validate_config,
    get_device,
    setup_directories,
    print_config,
    create_ablation_configs,
)
from src.utils.logging import TrainingLogger, setup_logging, log_model_summary
from src.models import HierarchicalMSAmba
from src.data import create_dataloaders
from src.training import Trainer
from src.training.metrics import evaluate_model, SentimentMetrics


def set_random_seeds(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seeds set to {seed}. Deterministic: {deterministic}")


def create_model(config: Config, device: torch.device) -> nn.Module:
    """Create model from configuration."""
    print("Creating model...")
    
    model = HierarchicalMSAmba(config.model)
    model = model.to(device)
    
    # Log model summary
    log_model_summary(model)
    
    return model


def create_data_loaders(config: Config) -> tuple:
    """Create data loaders."""
    print(f"Loading {config.data.dataset.upper()} dataset...")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=config.data.dataset,
        data_path=config.data.data_path,
        batch_size=config.training.batch_size,
        num_workers=config.num_workers,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.data.random_seed,
        use_synthetic=config.data.use_synthetic,
    )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def train_single_experiment(
    config: Config,
    experiment_name: str,
    device: torch.device,
    output_dirs: Dict[str, Path],
) -> Dict[str, float]:
    """Train a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Starting experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Update config with experiment name
    config.experiment_name = experiment_name
    
    # Set random seeds
    set_random_seeds(config.seed, config.deterministic)
    
    # Create model
    model = create_model(config, device)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Setup logging
    wandb_config = None
    if config.logging.use_wandb:
        wandb_config = {
            'project': config.logging.project,
            'entity': config.logging.entity,
            'config': config.__dict__,
            'tags': getattr(config, 'tags', []),
            'notes': f"Enhanced MSAmba experiment: {experiment_name}",
        }
    
    logger = TrainingLogger(
        log_dir=str(output_dirs['logs']),
        experiment_name=experiment_name,
        wandb_config=wandb_config,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger,
        checkpoint_dir=str(output_dirs['checkpoints']),
    )
    
    # Train model
    print("Starting training...")
    best_metrics = trainer.train()
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_model(
        model=trainer.best_model if trainer.best_model else model,
        dataloader=test_loader,
        device=device,
        num_classes=config.model.num_classes,
    )
    
    print(f"Test results for {experiment_name}:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save final results
    results = {
        'best_val_metrics': best_metrics,
        'test_metrics': test_metrics,
        'config': config.__dict__,
    }
    
    results_file = output_dirs['results'] / f"{experiment_name}_results.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.close()
    
    return test_metrics


def run_ablation_study(
    base_config: Config,
    device: torch.device,
    output_dirs: Dict[str, Path],
) -> Dict[str, Dict[str, float]]:
    """Run full factorial ablation study."""
    print(f"\n{'='*80}")
    print("RUNNING ABLATION STUDY")
    print(f"{'='*80}")
    
    # Create ablation configurations
    ablation_configs = create_ablation_configs(base_config)
    
    all_results = {}
    
    for experiment_name, config in ablation_configs.items():
        try:
            results = train_single_experiment(
                config=config,
                experiment_name=experiment_name,
                device=device,
                output_dirs=output_dirs,
            )
            all_results[experiment_name] = results
            
        except Exception as e:
            print(f"Error in experiment {experiment_name}: {e}")
            continue
    
    # Generate comparison summary
    print(f"\n{'='*80}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*80}")
    
    # Find best configuration
    if all_results:
        primary_metric = 'Acc7' if base_config.model.num_classes == 7 else 'Acc2'
        if primary_metric not in next(iter(all_results.values())):
            primary_metric = 'F1_weighted'  # Fallback
        
        best_experiment = max(
            all_results.keys(),
            key=lambda x: all_results[x].get(primary_metric, 0)
        )
        
        print(f"\nBest configuration: {best_experiment}")
        print(f"Primary metric ({primary_metric}): {all_results[best_experiment][primary_metric]:.4f}")
        
        # Print all results
        print(f"\nAll results ({primary_metric}):")
        for exp_name in sorted(all_results.keys()):
            score = all_results[exp_name].get(primary_metric, 0)
            print(f"  {exp_name:<20}: {score:.4f}")
    
    # Save combined results
    summary_file = output_dirs['results'] / "ablation_summary.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    return all_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Enhanced MSAmba Training")
    
    # Configuration
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='Configuration file path'
    )
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, help='Experiment name override')
    parser.add_argument('--output_dir', type=str, help='Output directory override')
    
    # Model settings
    parser.add_argument('--central_mod', type=str, choices=['text', 'audio', 'vision'])
    parser.add_argument('--hierarchy_levels', type=int, choices=[1, 2, 3])
    parser.add_argument('--segment_len', type=float, help='Segment length in seconds')
    
    # Enhanced components
    parser.add_argument('--no-sgsm', action='store_true', help='Disable SGSM')
    parser.add_argument('--no-saliency', action='store_true', help='Disable saliency pruning')
    parser.add_argument('--no-memory', action='store_true', help='Disable EMA memory')
    parser.add_argument('--sgsm_alpha', type=float, help='SGSM conditioning strength')
    parser.add_argument('--saliency_floor', type=float, help='Saliency floor value')
    parser.add_argument('--memory_tau', type=float, help='EMA memory decay rate')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    
    # Data settings
    parser.add_argument('--dataset', type=str, choices=['mosi', 'mosei', 'chsims'])
    parser.add_argument('--data_path', type=str, help='Data path')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic data')
    
    # Device settings
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--num_workers', type=int, help='Number of data loading workers')
    
    # Ablation study
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    
    # Misc
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Load base configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Creating default configuration...")
        config = Config()
        save_config(config, args.config)
    
    # Apply command line overrides
    overrides = {}
    
    # Experiment settings
    if args.experiment_name:
        overrides['experiment_name'] = args.experiment_name
    if args.output_dir:
        overrides['output_dir'] = args.output_dir
    
    # Model overrides
    model_overrides = {}
    if args.central_mod:
        model_overrides['central_mod'] = args.central_mod
    if args.hierarchy_levels:
        model_overrides['hierarchy_levels'] = args.hierarchy_levels
    if args.segment_len:
        model_overrides['segment_len'] = args.segment_len
    if args.no_sgsm:
        model_overrides['use_sgsm'] = False
    if args.no_saliency:
        model_overrides['use_saliency'] = False
    if args.no_memory:
        model_overrides['use_memory'] = False
    if args.sgsm_alpha:
        model_overrides['sgsm_alpha'] = args.sgsm_alpha
    if args.saliency_floor:
        model_overrides['saliency_floor'] = args.saliency_floor
    if args.memory_tau:
        model_overrides['memory_tau'] = args.memory_tau
    
    if model_overrides:
        overrides['model'] = model_overrides
    
    # Training overrides
    training_overrides = {}
    if args.batch_size:
        training_overrides['batch_size'] = args.batch_size
    if args.learning_rate:
        training_overrides['learning_rate'] = args.learning_rate
    if args.num_epochs:
        training_overrides['num_epochs'] = args.num_epochs
    
    if training_overrides:
        overrides['training'] = training_overrides
    
    # Data overrides
    data_overrides = {}
    if args.dataset:
        data_overrides['dataset'] = args.dataset
    if args.data_path:
        data_overrides['data_path'] = args.data_path
    if args.use_synthetic:
        data_overrides['use_synthetic'] = True
    
    if data_overrides:
        overrides['data'] = data_overrides
    
    # System overrides
    if args.device:
        overrides['device'] = args.device
    if args.num_workers:
        overrides['num_workers'] = args.num_workers
    if args.seed:
        overrides['seed'] = args.seed
    
    # Logging overrides
    if args.no_wandb:
        overrides['logging'] = {'use_wandb': False}
    
    # Apply overrides
    for key, value in overrides.items():
        if isinstance(value, dict) and hasattr(config, key):
            for sub_key, sub_value in value.items():
                setattr(getattr(config, key), sub_key, sub_value)
        else:
            setattr(config, key, value)
    
    # Validate configuration
    validate_config(config)
    
    # Print configuration
    print_config(config)
    
    # Setup device
    device = torch.device(get_device(config))
    print(f"Using device: {device}")
    
    # Setup directories
    output_dirs = setup_directories(config)
    
    # Save configuration
    config_save_path = output_dirs['configs'] / f"{config.experiment_name}_config.yaml"
    save_config(config, config_save_path)
    
    # Setup logging
    setup_logging(
        log_level=config.logging.log_level,
        log_file=str(output_dirs['logs'] / f"{config.experiment_name}.log")
    )
    
    try:
        if args.ablation:
            # Run ablation study
            results = run_ablation_study(config, device, output_dirs)
        else:
            # Single experiment
            results = train_single_experiment(
                config=config,
                experiment_name=config.experiment_name,
                device=device,
                output_dirs=output_dirs,
            )
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"Results saved to: {output_dirs['results']}")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()