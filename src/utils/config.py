"""Configuration management utilities."""

import os
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Core parameters
    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    n_layers: int = 6
    
    # Modality dimensions
    text_dim: int = 768
    audio_dim: int = 5
    vision_dim: int = 20
    
    # Hierarchy settings
    hierarchy_levels: int = 3
    central_mod: str = "text"
    modalities: list = field(default_factory=lambda: ["text", "audio", "vision"])
    
    # ISM settings
    use_glce: bool = True
    glce_scales: list = field(default_factory=lambda: [1, 2, 4])
    
    # CHM settings
    use_self_attn: bool = True
    num_heads: int = 8
    
    # SGSM settings
    use_sgsm: bool = True
    sgsm_alpha: float = 0.3
    sgsm_drop: float = 0.2
    vad_dim: int = 3
    polarity_dim: int = 1
    
    # Saliency settings
    use_saliency: bool = True
    saliency_floor: float = 0.25
    emotion_classes: int = 7
    
    # Memory settings
    use_memory: bool = True
    memory_tau: float = 0.9
    memory_size: int = 64
    drop_memory: float = 0.2
    
    # Segmentation
    segment_len: float = 0.75
    segment_jitter: float = 0.1
    
    # Output settings
    num_classes: int = 7
    dropout: float = 0.1


@dataclass 
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 1000
    
    # Advanced training settings
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Regularization
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1
    
    # Loss weights
    regression_weight: float = 1.0
    classification_weight: float = 1.0
    consistency_weight: float = 0.1
    kl_weight: float = 0.01
    
    # Scheduling
    scheduler: str = "cosine"  # cosine, linear, step
    min_lr: float = 1e-6
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    save_every: int = 5
    keep_best: bool = True
    max_checkpoints: int = 5
    
    # Logging
    log_frequency: int = 100
    val_frequency: int = 1
    save_frequency: int = 5
    save_best_only: bool = True
    
    # Profiling
    profile_training: bool = False
    profile_steps: int = 100
    
    # Multi-GPU
    use_ddp: bool = False
    local_rank: int = -1


@dataclass
class DataConfig:
    """Data configuration."""
    # Dataset
    dataset: str = "mosi"  # mosi, mosei, chsims
    data_path: str = "data/"
    
    # Preprocessing
    normalize: bool = True
    use_synthetic: bool = False
    
    # Augmentation
    temporal_jitter: bool = True
    feature_noise: float = 0.01
    
    # Validation split
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    random_seed: int = 42


@dataclass
class LoggingConfig:
    """Logging configuration."""
    # Wandb
    use_wandb: bool = True
    project: str = "enhanced-msamaba"
    entity: Optional[str] = None
    
    # Local logging
    log_dir: str = "logs/"
    log_level: str = "INFO"
    
    # Metrics
    log_every: int = 10
    eval_every: int = 100


@dataclass
class Config:
    """Complete configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Experiment settings
    experiment_name: str = "baseline"
    output_dir: str = "outputs/"
    device: str = "auto"  # auto, cpu, cuda
    num_workers: int = 4
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert nested dict to Config object
    try:
        config = Config(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
        )
        
        # Update top-level fields
        for key, value in config_dict.items():
            if key not in ['model', 'training', 'data', 'logging']:
                setattr(config, key, value)
                
    except TypeError as e:
        raise ValueError(f"Invalid configuration: {e}")
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Config, save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = asdict(config)
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    logger.info(f"Saved configuration to {save_path}")


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """Merge override configuration into base configuration."""
    base_dict = asdict(base_config)
    
    # Recursively update nested dictionaries
    def deep_update(base: Dict, override: Dict) -> Dict:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    merged_dict = deep_update(base_dict, override_config)
    
    # Reconstruct Config object
    return Config(
        model=ModelConfig(**merged_dict.get('model', {})),
        training=TrainingConfig(**merged_dict.get('training', {})),
        data=DataConfig(**merged_dict.get('data', {})),
        logging=LoggingConfig(**merged_dict.get('logging', {})),
    )


def validate_config(config: Config) -> None:
    """Validate configuration parameters."""
    errors = []
    
    # Model validation
    if config.model.d_model <= 0:
        errors.append("d_model must be positive")
    
    if config.model.central_mod not in ["text", "audio", "vision"]:
        errors.append("central_mod must be one of: text, audio, vision")
    
    if config.model.hierarchy_levels not in [1, 2, 3]:
        errors.append("hierarchy_levels must be 1, 2, or 3")
    
    if not 0 <= config.model.sgsm_alpha <= 1:
        errors.append("sgsm_alpha must be in [0, 1]")
    
    if not 0 <= config.model.saliency_floor <= 1:
        errors.append("saliency_floor must be in [0, 1]")
    
    # Training validation
    if config.training.learning_rate <= 0:
        errors.append("learning_rate must be positive")
    
    if config.training.batch_size <= 0:
        errors.append("batch_size must be positive")
    
    if config.training.num_epochs <= 0:
        errors.append("num_epochs must be positive")
    
    # Data validation
    if config.data.dataset not in ["mosi", "mosei", "chsims"]:
        errors.append("dataset must be one of: mosi, mosei, chsims")
    
    if not 0 < config.data.val_ratio < 1:
        errors.append("val_ratio must be in (0, 1)")
    
    if not 0 < config.data.test_ratio < 1:
        errors.append("test_ratio must be in (0, 1)")
    
    if config.data.val_ratio + config.data.test_ratio >= 1:
        errors.append("val_ratio + test_ratio must be < 1")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    logger.info("Configuration validation passed")


def create_ablation_configs(base_config: Config) -> Dict[str, Config]:
    """Create configurations for ablation studies."""
    configs = {}
    
    # Base configuration
    configs['baseline'] = base_config
    
    # Single component ablations
    no_sgsm = merge_configs(base_config, {'model': {'use_sgsm': False}})
    configs['no_sgsm'] = no_sgsm
    
    no_saliency = merge_configs(base_config, {'model': {'use_saliency': False}})
    configs['no_saliency'] = no_saliency
    
    no_memory = merge_configs(base_config, {'model': {'use_memory': False}})
    configs['no_memory'] = no_memory
    
    # Pairwise ablations
    no_sgsm_saliency = merge_configs(base_config, {
        'model': {'use_sgsm': False, 'use_saliency': False}
    })
    configs['no_sgsm_saliency'] = no_sgsm_saliency
    
    no_sgsm_memory = merge_configs(base_config, {
        'model': {'use_sgsm': False, 'use_memory': False}
    })
    configs['no_sgsm_memory'] = no_sgsm_memory
    
    no_saliency_memory = merge_configs(base_config, {
        'model': {'use_saliency': False, 'use_memory': False}
    })
    configs['no_saliency_memory'] = no_saliency_memory
    
    # All ablated
    no_enhancements = merge_configs(base_config, {
        'model': {
            'use_sgsm': False,
            'use_saliency': False, 
            'use_memory': False
        }
    })
    configs['no_enhancements'] = no_enhancements
    
    return configs


def get_device(config: Config) -> str:
    """Get appropriate device based on configuration."""
    if config.device == "auto":
        import torch
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return config.device


def setup_directories(config: Config) -> Dict[str, Path]:
    """Setup output directories."""
    base_dir = Path(config.output_dir) / config.experiment_name
    
    dirs = {
        'base': base_dir,
        'checkpoints': base_dir / 'checkpoints',
        'logs': base_dir / 'logs', 
        'configs': base_dir / 'configs',
        'results': base_dir / 'results',
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def print_config(config: Config) -> None:
    """Print configuration in a readable format."""
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"\nEXPERIMENT: {config.experiment_name}")
    print(f"OUTPUT DIR: {config.output_dir}")
    print(f"DEVICE: {get_device(config)}")
    
    print(f"\nMODEL:")
    print(f"  Architecture: {config.model.hierarchy_levels}-level hierarchical")
    print(f"  Central modality: {config.model.central_mod}")
    print(f"  Model dimension: {config.model.d_model}")
    print(f"  State dimension: {config.model.d_state}")
    print(f"  SGSM: {'✓' if config.model.use_sgsm else '✗'}")
    print(f"  Saliency: {'✓' if config.model.use_saliency else '✗'}")
    print(f"  Memory: {'✓' if config.model.use_memory else '✗'}")
    
    print(f"\nTRAINING:")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Weight decay: {config.training.weight_decay}")
    
    print(f"\nDATA:")
    print(f"  Dataset: {config.data.dataset.upper()}")
    print(f"  Data path: {config.data.data_path}")
    print(f"  Validation ratio: {config.data.val_ratio}")
    
    print("=" * 60)