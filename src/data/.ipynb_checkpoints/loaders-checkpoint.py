



# enhanced_msamaba/src/data/loaders.py
"""
Data loaders for multimodal sentiment analysis datasets.
Supports CMU-MOSI, CMU-MOSEI, and CH-SIMS with proper preprocessing.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import h5py
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


def load_mosi(data_path: str, split: str = "train") -> Dict[str, Any]:
    """Load CMU-MOSI dataset."""
    data_path = Path(data_path)
    file_path = data_path / f"mosi_{split}.pkl"
    
    if not file_path.exists():
        # Return synthetic data if file doesn't exist
        logger.warning(f"MOSI data file not found: {file_path}. Using synthetic data.")
        return _generate_synthetic_mosi_data(split)
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def load_mosei(data_path: str, split: str = "train") -> Dict[str, Any]:
    """Load CMU-MOSEI dataset."""
    data_path = Path(data_path)
    file_path = data_path / f"mosei_{split}.pkl"
    
    if not file_path.exists():
        # Return synthetic data if file doesn't exist
        logger.warning(f"MOSEI data file not found: {file_path}. Using synthetic data.")
        return _generate_synthetic_mosei_data(split)
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def load_chsims(data_path: str, split: str = "train") -> Dict[str, Any]:
    """Load CH-SIMS dataset."""
    data_path = Path(data_path)
    file_path = data_path / f"chsims_{split}.pkl"
    
    if not file_path.exists():
        # Return synthetic data if file doesn't exist
        logger.warning(f"CH-SIMS data file not found: {file_path}. Using synthetic data.")
        return _generate_synthetic_chsims_data(split)
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def _generate_synthetic_mosi_data(split: str) -> Dict[str, Any]:
    """Generate synthetic MOSI-like data for testing."""
    n_samples = 100 if split == "train" else 20
    
    return {
        'text': np.random.randn(n_samples, 50, 768),
        'audio': np.random.randn(n_samples, 50, 74),
        'vision': np.random.randn(n_samples, 50, 47),
        'labels': np.random.uniform(-3, 3, n_samples),
        'metadata': {'split': split, 'dataset': 'mosi_synthetic'}
    }


def _generate_synthetic_mosei_data(split: str) -> Dict[str, Any]:
    """Generate synthetic MOSEI-like data for testing."""
    n_samples = 100 if split == "train" else 20
    
    return {
        'text': np.random.randn(n_samples, 50, 768),
        'audio': np.random.randn(n_samples, 50, 74),
        'vision': np.random.randn(n_samples, 50, 47),
        'labels': np.random.uniform(-3, 3, n_samples),
        'metadata': {'split': split, 'dataset': 'mosei_synthetic'}
    }


def _generate_synthetic_chsims_data(split: str) -> Dict[str, Any]:
    """Generate synthetic CH-SIMS-like data for testing."""
    n_samples = 100 if split == "train" else 20
    
    return {
        'text': np.random.randn(n_samples, 50, 768),
        'audio': np.random.randn(n_samples, 50, 74),
        'vision': np.random.randn(n_samples, 50, 47),
        'labels': np.random.randint(0, 7, n_samples),
        'metadata': {'split': split, 'dataset': 'chsims_synthetic'}
    }


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str  # "mosi", "mosei", "sims"
    data_path: str
    modalities: List[str] = None
    max_seq_len: int = 50
    segment_len: float = 0.75  # seconds
    overlap_ratio: float = 0.5
    normalize: bool = True
    augment_training: bool = True
    missing_data_prob: float = 0.1  # Probability of missing modality during training
    use_synthetic: bool = False  # Use synthetic data instead of loading from files
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ["text", "audio", "vision"]


class MultimodalDataset(Dataset):
    """
    Base dataset class for multimodal sentiment analysis.
    Handles loading, preprocessing, and augmentation of multimodal data.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        split: str = "train",
        transform: Optional[callable] = None
    ):
        self.config = config
        self.split = split
        self.transform = transform
        self.modalities = config.modalities
        
        # Load data
        self.data = self._load_data()
        
        # Preprocess
        self._preprocess_data()
        
        logger.info(f"Loaded {len(self.data['labels'])} samples for {split} split")
    
    def _load_data(self) -> Dict[str, Any]:
        """Load raw data from files."""
        # Check if synthetic data should be used
        if self.config.use_synthetic:
            return self._generate_synthetic_data()
        
        data_path = Path(self.config.data_path)
        
        if self.config.name.lower() in ["mosi", "mosei"]:
            # Look in dataset-specific subdirectory
            dataset_path = data_path / self.config.name.lower()
            if dataset_path.exists():
                return self._load_cmu_data(dataset_path)
            else:
                # Fallback to main data directory
                return self._load_cmu_data(data_path)
        elif self.config.name.lower() == "sims":
            return self._load_sims_data(data_path)
        else:
            raise ValueError(f"Unknown dataset: {self.config.name}")
    
    def _load_cmu_data(self, data_path: Path) -> Dict[str, Any]:
        """Load CMU-MOSI or CMU-MOSEI data."""
        # Try multiple file formats
        possible_files = [
            data_path / f"{self.split}.pkl",
            data_path / f"{self.config.name}_{self.split}.pkl",
            data_path / f"{self.split}.h5"
        ]
        
        for file_path in possible_files:
            if file_path.exists():
                if file_path.suffix == '.pkl':
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                    break
                elif file_path.suffix == '.h5':
                    data = self._load_h5_data(file_path)
                    break
        else:
            raise FileNotFoundError(f"No data file found for {self.split} split in {data_path}")
        
        return self._standardize_data_format(data)
    
    def _load_sims_data(self, data_path: Path) -> Dict[str, Any]:
        """Load CH-SIMS data."""
        file_path = data_path / f"sims_{self.split}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"SIMS data file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return self._standardize_data_format(data)
    
    def _load_h5_data(self, file_path: Path) -> Dict[str, Any]:
        """Load data from HDF5 format."""
        data = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                data[key] = f[key][:]
        return data
    
    def _generate_synthetic_data(self) -> Dict[str, Any]:
        """Generate synthetic data for testing."""
        n_samples = 100 if self.split == "train" else 20
        
        # Generate synthetic features based on dataset
        if self.config.name.lower() == "mosi":
            data = {
                'text': np.random.randn(n_samples, self.config.max_seq_len, 768),
                'audio': np.random.randn(n_samples, self.config.max_seq_len, 5),
                'vision': np.random.randn(n_samples, self.config.max_seq_len, 20),
                'labels': np.random.uniform(-3, 3, n_samples),
                'ids': list(range(n_samples))
            }
        elif self.config.name.lower() == "mosei":
            data = {
                'text': np.random.randn(n_samples, self.config.max_seq_len, 768),
                'audio': np.random.randn(n_samples, self.config.max_seq_len, 74),
                'vision': np.random.randn(n_samples, self.config.max_seq_len, 35),
                'labels': np.random.uniform(-3, 3, n_samples),
                'ids': list(range(n_samples))
            }
        elif self.config.name.lower() == "sims":
            data = {
                'text': np.random.randn(n_samples, self.config.max_seq_len, 768),
                'audio': np.random.randn(n_samples, self.config.max_seq_len, 33),
                'vision': np.random.randn(n_samples, self.config.max_seq_len, 709),
                'labels': np.random.uniform(-1, 1, n_samples),
                'ids': list(range(n_samples))
            }
        else:
            # Default synthetic data
            data = {
                'text': np.random.randn(n_samples, self.config.max_seq_len, 768),
                'audio': np.random.randn(n_samples, self.config.max_seq_len, 74),
                'vision': np.random.randn(n_samples, self.config.max_seq_len, 47),
                'labels': np.random.uniform(-3, 3, n_samples),
                'ids': list(range(n_samples))
            }
        
        logger.info(f"Generated synthetic {self.config.name.upper()} data: {n_samples} samples")
        return data
    
    def _standardize_data_format(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize data format across different datasets.
        Expected format: {
            'text': [N, seq_len, text_dim],
            'audio': [N, seq_len, audio_dim], 
            'vision': [N, seq_len, vision_dim],
            'labels': [N,] or [N, 1],
            'ids': [N,] (optional)
        }
        """
        standardized = {}
        
        # Handle different key naming conventions
        key_mappings = {
            'text': ['text', 'text_bert', 'bert', 'language'],
            'audio': ['audio', 'covarep', 'acoustic'],
            'vision': ['vision', 'facet', 'visual', 'openface'],
            'labels': ['labels', 'label', 'y', 'sentiment'],
            'ids': ['ids', 'id', 'video_ids', 'segment_ids']
        }
        
        for standard_key, possible_keys in key_mappings.items():
            found = False
            for key in possible_keys:
                if key in raw_data:
                    standardized[standard_key] = raw_data[key]
                    found = True
                    break
            
            if not found and standard_key in ['text', 'audio', 'vision']:
                if standard_key in self.modalities:
                    logger.warning(f"Required modality '{standard_key}' not found in data")
                    # Create dummy data for missing modality
                    n_samples = len(next(iter(raw_data.values())))
                    standardized[standard_key] = np.random.randn(n_samples, self.config.max_seq_len, 768)
        
        # Ensure labels exist
        if 'labels' not in standardized:
            raise ValueError("Labels not found in data")
        
        return standardized
    
    def _preprocess_data(self):
        """Preprocess loaded data."""
        # Convert to tensors and ensure proper shapes
        processed_data = {}
        
        for modality in self.modalities:
            if modality in self.data:
                features = self.data[modality]
                
                # Convert to tensor
                if not isinstance(features, torch.Tensor):
                    features = torch.FloatTensor(features)
                
                # Handle sequence length
                if features.shape[1] > self.config.max_seq_len:
                    features = features[:, :self.config.max_seq_len, :]
                elif features.shape[1] < self.config.max_seq_len:
                    # Pad sequences
                    pad_len = self.config.max_seq_len - features.shape[1]
                    features = F.pad(features, (0, 0, 0, pad_len))
                
                # Normalize if requested
                if self.config.normalize:
                    features = self._normalize_features(features)
                
                processed_data[modality] = features
        
        # Process labels
        labels = self.data['labels']
        if not isinstance(labels, torch.Tensor):
            labels = torch.FloatTensor(labels)
        
        # Ensure labels are proper shape
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        
        processed_data['labels'] = labels
        
        # Store IDs if available
        if 'ids' in self.data:
            processed_data['ids'] = self.data['ids']
        else:
            processed_data['ids'] = list(range(len(labels)))
        
        self.data = processed_data
        self.n_samples = len(labels)
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features to zero mean and unit variance."""
        # Compute statistics across batch and sequence dimensions
        mean = features.mean(dim=(0, 1), keepdim=True)
        std = features.std(dim=(0, 1), keepdim=True) + 1e-8
        return (features - mean) / std
    
    def _augment_sample(
        self, 
        sample: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply data augmentation during training."""
        if self.split != 'train' or not self.config.augment_training:
            return sample
        
        augmented = sample.copy()
        
        for modality in self.modalities:
            if modality in augmented:
                features = augmented[modality]
                
                # Add small amount of noise
                noise_level = 0.01
                noise = torch.randn_like(features) * noise_level
                features = features + noise
                
                # Random temporal jittering (shift sequences slightly)
                if random.random() < 0.3:
                    shift = random.randint(-2, 2)
                    if shift != 0:
                        features = torch.roll(features, shift, dims=0)
                
                # Random feature dropout
                if random.random() < 0.1:
                    dropout_mask = torch.rand(features.shape[-1]) > 0.1
                    features = features * dropout_mask.float()
                
                augmented[modality] = features
        
        return augmented
    
    def _simulate_missing_modality(
        self, 
        sample: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, bool]]:
        """Simulate missing modalities for robustness training."""
        missing_mask = {}
        processed_sample = sample.copy()
        
        if self.split == 'train':
            for modality in self.modalities:
                if random.random() < self.config.missing_data_prob:
                    missing_mask[modality] = True
                    # Zero out the modality
                    if modality in processed_sample:
                        processed_sample[modality] = torch.zeros_like(processed_sample[modality])
                else:
                    missing_mask[modality] = False
        else:
            # No missing modalities during evaluation
            for modality in self.modalities:
                missing_mask[modality] = False
        
        return processed_sample, missing_mask
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = {}
        
        # Get features for each modality
        for modality in self.modalities:
            if modality in self.data:
                sample[modality] = self.data[modality][idx].clone()
        
        # Get label
        sample['labels'] = self.data['labels'][idx].clone()
        
        # Get ID
        sample['ids'] = self.data['ids'][idx]
        
        # Apply augmentation
        sample = self._augment_sample(sample)
        
        # Simulate missing modalities
        sample, missing_mask = self._simulate_missing_modality(sample)
        sample['missing_mask'] = missing_mask
        
        # Apply custom transform if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_feature_dims(self) -> Dict[str, int]:
        """Get feature dimensions for each modality."""
        dims = {}
        for modality in self.modalities:
            if modality in self.data:
                dims[modality] = self.data[modality].shape[-1]
        return dims
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'n_samples': self.n_samples,
            'modalities': self.modalities,
            'feature_dims': self.get_feature_dims(),
            'max_seq_len': self.config.max_seq_len,
            'label_range': (
                self.data['labels'].min().item(),
                self.data['labels'].max().item()
            ),
            'label_mean': self.data['labels'].mean().item(),
            'label_std': self.data['labels'].std().item()
        }
        return stats


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching multimodal data.
    Handles variable length sequences and missing modalities.
    """
    collated = {}
    
    # Get all possible keys
    all_keys = set()
    for sample in batch:
        all_keys.update(sample.keys())
    
    for key in all_keys:
        if key == 'ids':
            collated[key] = [sample[key] for sample in batch]
        elif key == 'missing_mask':
            # Collate missing masks
            missing_masks = {}
            modalities = batch[0]['missing_mask'].keys()
            for modality in modalities:
                missing_masks[modality] = torch.tensor([
                    sample['missing_mask'][modality] for sample in batch
                ])
            collated[key] = missing_masks
        else:
            # Stack tensors
            values = [sample[key] for sample in batch if key in sample]
            if values:
                if isinstance(values[0], torch.Tensor):
                    collated[key] = torch.stack(values, dim=0)
                else:
                    collated[key] = values
    
    return collated


def create_dataloaders(
    config: DatasetConfig,
    batch_size: int = 32,
    num_workers: int = 4,
    splits: List[str] = ["train", "valid", "test"]
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for all specified splits.
    
    Args:
        config: Dataset configuration
        batch_size: Batch size for training
        num_workers: Number of worker processes
        splits: Which splits to create loaders for
        
    Returns:
        Dictionary of dataloaders
    """
    dataloaders = {}
    
    for split in splits:
        dataset = MultimodalDataset(config, split=split)
        
        # Use different batch size for validation/test
        current_batch_size = batch_size if split == 'train' else min(batch_size * 2, 64)
        
        dataloader = DataLoader(
            dataset,
            batch_size=current_batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=(split == 'train')
        )
        
        dataloaders[split] = dataloader
        
        # Log dataset info
        stats = dataset.get_dataset_stats()
        logger.info(f"{split.capitalize()} dataset: {stats}")
    
    return dataloaders


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get standard information for common datasets."""
    dataset_info = {
        "mosi": {
            "name": "CMU-MOSI",
            "modalities": ["text", "audio", "vision"],
            "feature_dims": {"text": 768, "audio": 74, "vision": 47},
            "num_classes": 1,  # Regression
            "task_type": "regression",
            "label_range": [-3, 3],
            "splits": ["train", "valid", "test"]
        },
        "mosei": {
            "name": "CMU-MOSEI", 
            "modalities": ["text", "audio", "vision"],
            "feature_dims": {"text": 768, "audio": 74, "vision": 35},
            "num_classes": 1,  # Regression
            "task_type": "regression", 
            "label_range": [-3, 3],
            "splits": ["train", "valid", "test"]
        },
        "sims": {
            "name": "CH-SIMS",
            "modalities": ["text", "audio", "vision"],
            "feature_dims": {"text": 768, "audio": 33, "vision": 709},
            "num_classes": 1,  # Regression
            "task_type": "regression",
            "label_range": [-1, 1],
            "splits": ["train", "valid", "test"]
        }
    }
    
    return dataset_info.get(dataset_name.lower(), {})


# Example usage and testing
if __name__ == "__main__":
    # Test dataset loading
    config = DatasetConfig(
        name="mosi",
        data_path="/path/to/mosi/data",
        modalities=["text", "audio", "vision"],
        max_seq_len=50,
        segment_len=0.75,
        normalize=True,
        augment_training=True,
        missing_data_prob=0.1
    )
    
    try:
        # Create dataloaders
        dataloaders = create_dataloaders(
            config,
            batch_size=16,
            num_workers=2,
            splits=["train", "valid"]
        )
        
        # Test a batch
        train_batch = next(iter(dataloaders['train']))
        print("Batch keys:", train_batch.keys())
        
        for modality in config.modalities:
            if modality in train_batch:
                print(f"{modality} shape: {train_batch[modality].shape}")
        
        print(f"Labels shape: {train_batch['labels'].shape}")
        print(f"Missing masks: {train_batch['missing_mask']}")
        
    except Exception as e:
        print(f"Error testing dataset loading: {e}")
        print("This is expected if data files are not available")