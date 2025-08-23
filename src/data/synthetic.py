# enhanced_msamaba/src/data/synthetic.py
"""
Synthetic data generation for testing Enhanced MSAmba without real datasets.
Creates realistic multimodal sequences with controllable sentiment patterns.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import random
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    # Dataset parameters
    n_samples_train: int = 2000
    n_samples_valid: int = 500
    n_samples_test: int = 500
    
    # Sequence parameters
    seq_len_range: Tuple[int, int] = (20, 80)
    target_seq_len: int = 50
    
    # Feature dimensions (matching real datasets)
    text_dim: int = 768  # BERT-like
    audio_dim: int = 74  # COVAREP-like features
    vision_dim: int = 47  # OpenFace-like features
    
    # Sentiment parameters
    sentiment_range: Tuple[float, float] = (-3.0, 3.0)  # MOSI/MOSEI range
    noise_level: float = 0.1
    correlation_strength: float = 0.7  # Cross-modal correlation
    
    # Temporal patterns
    add_temporal_patterns: bool = True
    sentiment_drift_prob: float = 0.3  # Probability of sentiment changing over time
    
    # Missing data simulation
    missing_prob: float = 0.05  # Probability of missing modality
    
    # Output format
    output_format: str = "mosi"  # "mosi", "mosei", or "sims"


class SentimentPatternGenerator:
    """Generate realistic sentiment patterns across modalities."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.random_state = np.random.RandomState(42)
    
    def generate_base_sentiment(self) -> float:
        """Generate base sentiment value."""
        return self.random_state.uniform(
            self.config.sentiment_range[0],
            self.config.sentiment_range[1]
        )
    
    def add_temporal_dynamics(
        self, 
        base_sentiment: float, 
        seq_len: int
    ) -> np.ndarray:
        """Add temporal dynamics to sentiment."""
        sentiment_sequence = np.full(seq_len, base_sentiment)
        
        if self.config.add_temporal_patterns:
            # Add sentiment drift
            if self.random_state.random() < self.config.sentiment_drift_prob:
                # Linear drift
                drift_strength = self.random_state.uniform(-1.0, 1.0)
                drift = np.linspace(0, drift_strength, seq_len)
                sentiment_sequence += drift
                
                # Ensure within bounds
                sentiment_sequence = np.clip(
                    sentiment_sequence,
                    self.config.sentiment_range[0],
                    self.config.sentiment_range[1]
                )
            
            # Add periodic patterns (emotional peaks/valleys)
            if self.random_state.random() < 0.2:
                period = self.random_state.uniform(5, 15)
                phase = self.random_state.uniform(0, 2 * np.pi)
                amplitude = self.random_state.uniform(0.2, 0.8)
                
                periodic = amplitude * np.sin(2 * np.pi * np.arange(seq_len) / period + phase)
                sentiment_sequence += periodic
                
                sentiment_sequence = np.clip(
                    sentiment_sequence,
                    self.config.sentiment_range[0],
                    self.config.sentiment_range[1]
                )
        
        return sentiment_sequence
    
    def generate_modality_variations(
        self, 
        sentiment_sequence: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate modality-specific sentiment variations."""
        variations = {}
        
        # Text: More direct sentiment representation
        text_noise = self.random_state.normal(0, 0.3, len(sentiment_sequence))
        variations['text'] = sentiment_sequence + text_noise
        
        # Audio: More volatile, emotional intensity
        audio_noise = self.random_state.normal(0, 0.5, len(sentiment_sequence))
        audio_intensity = 1.2 + 0.3 * self.random_state.random()
        variations['audio'] = sentiment_sequence * audio_intensity + audio_noise
        
        # Vision: Delayed and smoothed reactions
        vision_delay = self.random_state.randint(1, 4)
        vision_smooth = 0.7
        vision_base = np.roll(sentiment_sequence, vision_delay)
        vision_noise = self.random_state.normal(0, 0.4, len(sentiment_sequence))
        variations['vision'] = vision_base * vision_smooth + vision_noise
        
        # Clip to bounds
        for modality in variations:
            variations[modality] = np.clip(
                variations[modality],
                self.config.sentiment_range[0],
                self.config.sentiment_range[1]
            )
        
        return variations


class ModalityFeatureGenerator:
    """Generate modality-specific features based on sentiment."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.random_state = np.random.RandomState(42)
    
    def generate_text_features(
        self, 
        sentiment_sequence: np.ndarray, 
        seq_len: int
    ) -> np.ndarray:
        """Generate BERT-like text features."""
        features = np.zeros((seq_len, self.config.text_dim))
        
        for t in range(seq_len):
            sentiment = sentiment_sequence[t]
            
            # Base contextual embedding (BERT-like)
            base_embedding = self.random_state.normal(0, 1, self.config.text_dim)
            
            # Sentiment-dependent components
            # Positive/negative sentiment dimensions
            if sentiment > 0:
                # Positive sentiment features
                pos_indices = self.random_state.choice(
                    self.config.text_dim, 
                    size=self.config.text_dim // 4, 
                    replace=False
                )
                base_embedding[pos_indices] += sentiment * 0.5
            else:
                # Negative sentiment features
                neg_indices = self.random_state.choice(
                    self.config.text_dim, 
                    size=self.config.text_dim // 4, 
                    replace=False
                )
                base_embedding[neg_indices] += abs(sentiment) * 0.5
            
            # Add emotion-specific patterns
            emotion_strength = abs(sentiment)
            if emotion_strength > 1.5:  # Strong emotion
                strong_indices = self.random_state.choice(
                    self.config.text_dim,
                    size=self.config.text_dim // 8,
                    replace=False
                )
                base_embedding[strong_indices] *= (1 + emotion_strength * 0.3)
            
            # Normalize to reasonable range
            features[t] = base_embedding / np.linalg.norm(base_embedding) * 10
        
        return features.astype(np.float32)
    
    def generate_audio_features(
        self, 
        sentiment_sequence: np.ndarray, 
        seq_len: int
    ) -> np.ndarray:
        """Generate COVAREP-like audio features."""
        features = np.zeros((seq_len, self.config.audio_dim))
        
        # Feature categories (simplified COVAREP)
        pitch_dim = 12      # F0, voicing, etc.
        spectral_dim = 25   # Spectral features
        prosodic_dim = 15   # Prosodic features
        quality_dim = 22    # Voice quality features
        
        for t in range(seq_len):
            sentiment = sentiment_sequence[t]
            emotion_intensity = abs(sentiment)
            
            feature_idx = 0
            
            # Pitch features
            base_pitch = self.random_state.normal(100, 20)  # Base F0
            pitch_variation = sentiment * 10 + emotion_intensity * 5
            pitch_features = np.array([
                base_pitch + pitch_variation,  # F0
                emotion_intensity * 0.8,       # Voicing strength
                abs(sentiment) * 0.6          # Pitch stability
            ])
            # Add more pitch-related features
            pitch_features = np.concatenate([
                pitch_features,
                self.random_state.normal(0, 1, pitch_dim - 3)
            ])
            features[t, feature_idx:feature_idx + pitch_dim] = pitch_features
            feature_idx += pitch_dim
            
            # Spectral features
            spectral_shift = sentiment * 0.3
            spectral_spread = emotion_intensity * 0.4
            spectral_features = self.random_state.normal(
                spectral_shift, 1 + spectral_spread, spectral_dim
            )
            features[t, feature_idx:feature_idx + spectral_dim] = spectral_features
            feature_idx += spectral_dim
            
            # Prosodic features (rhythm, stress patterns)
            prosodic_base = self.random_state.normal(0, 1, prosodic_dim)
            if emotion_intensity > 1.0:
                prosodic_base *= (1 + emotion_intensity * 0.2)
            features[t, feature_idx:feature_idx + prosodic_dim] = prosodic_base
            feature_idx += prosodic_dim
            
            # Voice quality features
            quality_base = self.random_state.normal(0, 1, quality_dim)
            # Emotional speech affects voice quality
            if sentiment < -1.0:  # Sadness/depression
                quality_base[:5] *= 0.8  # Reduced energy
            elif sentiment > 1.5:  # Joy/excitement
                quality_base[:5] *= 1.3  # Increased energy
            
            features[t, feature_idx:feature_idx + quality_dim] = quality_base
        
        return features.astype(np.float32)
    
    def generate_vision_features(
        self, 
        sentiment_sequence: np.ndarray, 
        seq_len: int
    ) -> np.ndarray:
        """Generate OpenFace-like visual features."""
        features = np.zeros((seq_len, self.config.vision_dim))
        
        # Feature categories (simplified OpenFace)
        landmark_dim = 17   # Key facial landmarks
        au_dim = 17        # Action units
        gaze_dim = 8       # Gaze direction and eye features
        pose_dim = 5       # Head pose
        
        for t in range(seq_len):
            sentiment = sentiment_sequence[t]
            emotion_intensity = abs(sentiment)
            
            feature_idx = 0
            
            # Facial landmarks (relative positions)
            landmark_base = self.random_state.normal(0, 0.1, landmark_dim)
            
            # Sentiment affects facial expression
            if sentiment > 0:  # Positive emotions
                landmark_base[:5] += 0.2 * sentiment  # Mouth corners up
                landmark_base[5:8] += 0.15 * sentiment  # Eye crinkles
            else:  # Negative emotions
                landmark_base[:5] -= 0.2 * abs(sentiment)  # Mouth corners down
                landmark_base[8:12] += 0.1 * abs(sentiment)  # Eyebrow lowering
            
            features[t, feature_idx:feature_idx + landmark_dim] = landmark_base
            feature_idx += landmark_dim
            
            # Action Units (FACS)
            au_base = self.random_state.exponential(0.5, au_dim)
            
            # Map sentiment to specific AUs
            if sentiment > 1.0:  # Happy
                au_base[6] *= (1 + sentiment)    # AU12 (Lip Corner Puller)
                au_base[12] *= (1 + sentiment)   # AU06 (Cheek Raiser)
            elif sentiment < -1.0:  # Sad
                au_base[1] *= (1 + abs(sentiment))  # AU01 (Inner Brow Raiser)
                au_base[15] *= (1 + abs(sentiment)) # AU15 (Lip Corner Depressor)
            
            # High emotion intensity affects multiple AUs
            if emotion_intensity > 2.0:
                au_base *= (1 + emotion_intensity * 0.2)
            
            features[t, feature_idx:feature_idx + au_dim] = au_base
            feature_idx += au_dim
            
            # Gaze features
            gaze_base = self.random_state.normal(0, 0.3, gaze_dim)
            # Emotional state affects gaze patterns
            if emotion_intensity > 1.5:
                gaze_base[:2] *= (1 + emotion_intensity * 0.1)  # Gaze direction
            
            features[t, feature_idx:feature_idx + gaze_dim] = gaze_base
            feature_idx += gaze_dim
            
            # Head pose
            pose_base = self.random_state.normal(0, 0.2, pose_dim)
            # Sentiment affects posture
            if sentiment < -1.0:
                pose_base[0] -= 0.1 * abs(sentiment)  # Head down when sad
            
            features[t, feature_idx:feature_idx + pose_dim] = pose_base
        
        return features.astype(np.float32)


class SyntheticDataGenerator:
    """Main synthetic data generator."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.pattern_generator = SentimentPatternGenerator(config)
        self.feature_generator = ModalityFeatureGenerator(config)
        self.random_state = np.random.RandomState(42)
    
    def generate_sample(self, sample_id: int) -> Dict[str, Any]:
        """Generate a single synthetic multimodal sample."""
        # Generate sequence length
        seq_len = self.random_state.randint(
            self.config.seq_len_range[0],
            self.config.seq_len_range[1] + 1
        )
        
        # Generate base sentiment and temporal dynamics
        base_sentiment = self.pattern_generator.generate_base_sentiment()
        sentiment_sequence = self.pattern_generator.add_temporal_dynamics(
            base_sentiment, seq_len
        )
        
        # Generate modality-specific sentiment variations
        modality_sentiments = self.pattern_generator.generate_modality_variations(
            sentiment_sequence
        )
        
        # Generate features for each modality
        features = {}
        
        features['text'] = self.feature_generator.generate_text_features(
            modality_sentiments['text'], seq_len
        )
        
        features['audio'] = self.feature_generator.generate_audio_features(
            modality_sentiments['audio'], seq_len
        )
        
        features['vision'] = self.feature_generator.generate_vision_features(
            modality_sentiments['vision'], seq_len
        )
        
        # Pad/truncate to target length
        for modality in features:
            current_len = features[modality].shape[0]
            if current_len < self.config.target_seq_len:
                # Pad with last frame repeated
                pad_len = self.config.target_seq_len - current_len
                last_frame = features[modality][-1:].repeat(pad_len, axis=0)
                features[modality] = np.concatenate([features[modality], last_frame])
            elif current_len > self.config.target_seq_len:
                # Truncate
                features[modality] = features[modality][:self.config.target_seq_len]
        
        # Add cross-modal correlations
        features = self._add_cross_modal_correlations(features)
        
        # Add noise
        features = self._add_noise(features)
        
        # Simulate missing modalities
        missing_mask = self._generate_missing_mask()
        
        # Use final sentiment as label (average across modalities and time)
        final_sentiments = []
        for modality in modality_sentiments:
            if not missing_mask.get(modality, False):
                final_sentiments.append(modality_sentiments[modality][-5:].mean())
        
        label = np.mean(final_sentiments) if final_sentiments else base_sentiment
        
        # Add label noise
        label += self.random_state.normal(0, self.config.noise_level * 0.5)
        label = np.clip(label, self.config.sentiment_range[0], self.config.sentiment_range[1])
        
        sample = {
            'text': features['text'],
            'audio': features['audio'], 
            'vision': features['vision'],
            'label': label,
            'id': f"synthetic_{sample_id:06d}",
            'missing_mask': missing_mask,
            'base_sentiment': base_sentiment,
            'sentiment_sequence': sentiment_sequence
        }
        
        return sample
    
    def _add_cross_modal_correlations(
        self, 
        features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Add cross-modal correlations to make data more realistic."""
        if self.config.correlation_strength <= 0:
            return features
        
        modalities = list(features.keys())
        corr_strength = self.config.correlation_strength
        
        # Add pairwise correlations
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i < j:  # Avoid double correlation
                    # Create shared component
                    shared_component = self.random_state.normal(
                        0, 1, (features[mod1].shape[0], min(features[mod1].shape[1], features[mod2].shape[1]))
                    )
                    
                    # Add to both modalities
                    feat1_indices = self.random_state.choice(
                        features[mod1].shape[1], 
                        size=shared_component.shape[1], 
                        replace=False
                    )
                    feat2_indices = self.random_state.choice(
                        features[mod2].shape[1], 
                        size=shared_component.shape[1], 
                        replace=False
                    )
                    
                    features[mod1][:, feat1_indices] += corr_strength * shared_component
                    features[mod2][:, feat2_indices] += corr_strength * shared_component
        
        return features
    
    def _add_noise(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Add realistic noise to features."""
        for modality in features:
            noise_std = self.config.noise_level * features[modality].std()
            noise = self.random_state.normal(0, noise_std, features[modality].shape)
            features[modality] = features[modality] + noise
        
        return features
    
    def _generate_missing_mask(self) -> Dict[str, bool]:
        """Generate missing modality mask."""
        missing_mask = {}
        modalities = ['text', 'audio', 'vision']
        
        for modality in modalities:
            missing_mask[modality] = (
                self.random_state.random() < self.config.missing_prob
            )
        
        # Ensure at least one modality is present
        if all(missing_mask.values()):
            # Randomly select one modality to be present
            present_mod = self.random_state.choice(modalities)
            missing_mask[present_mod] = False
        
        return missing_mask
    
    def generate_dataset(self, split: str) -> List[Dict[str, Any]]:
        """Generate complete dataset for a split."""
        if split == "train":
            n_samples = self.config.n_samples_train
        elif split == "valid":
            n_samples = self.config.n_samples_valid
        elif split == "test":
            n_samples = self.config.n_samples_test
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Set random seed for reproducibility
        self.random_state = np.random.RandomState(42 + hash(split) % 1000)
        
        logger.info(f"Generating {n_samples} synthetic samples for {split} split...")
        
        samples = []
        for i in range(n_samples):
            sample = self.generate_sample(i)
            samples.append(sample)
            
            if (i + 1) % 500 == 0:
                logger.info(f"Generated {i + 1}/{n_samples} samples")
        
        logger.info(f"Completed generating {split} split with {len(samples)} samples")
        return samples
    
    def save_dataset(self, output_dir: str, splits: List[str] = ["train", "valid", "test"]):
        """Generate and save synthetic datasets."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset_info = {
            'config': self.config,
            'feature_dims': {
                'text': self.config.text_dim,
                'audio': self.config.audio_dim,
                'vision': self.config.vision_dim
            },
            'modalities': ['text', 'audio', 'vision'],
            'num_classes': 1,  # Regression
            'task_type': 'regression',
            'label_range': self.config.sentiment_range,
            'target_seq_len': self.config.target_seq_len,
            'splits': splits
        }
        
        # Save dataset info
        with open(output_path / 'dataset_info.pkl', 'wb') as f:
            pickle.dump(dataset_info, f)
        
        for split in splits:
            samples = self.generate_dataset(split)
            
            # Convert to the expected format
            formatted_data = self._format_for_dataloader(samples, split)
            
            # Save split data
            split_path = output_path / f'{split}.pkl'
            with open(split_path, 'wb') as f:
                pickle.dump(formatted_data, f)
            
            logger.info(f"Saved {split} split to {split_path}")
            
            # Save statistics
            stats = self._compute_split_statistics(samples)
            stats_path = output_path / f'{split}_stats.pkl'
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)
    
    def _format_for_dataloader(
        self, 
        samples: List[Dict[str, Any]], 
        split: str
    ) -> Dict[str, np.ndarray]:
        """Format samples for the dataloader."""
        n_samples = len(samples)
        
        # Initialize arrays
        text_features = np.zeros((n_samples, self.config.target_seq_len, self.config.text_dim))
        audio_features = np.zeros((n_samples, self.config.target_seq_len, self.config.audio_dim))
        vision_features = np.zeros((n_samples, self.config.target_seq_len, self.config.vision_dim))
        labels = np.zeros((n_samples, 1))
        ids = []
        
        for i, sample in enumerate(samples):
            text_features[i] = sample['text']
            audio_features[i] = sample['audio']
            vision_features[i] = sample['vision']
            labels[i, 0] = sample['label']
            ids.append(sample['id'])
        
        formatted_data = {
            'text': text_features.astype(np.float32),
            'audio': audio_features.astype(np.float32),
            'vision': vision_features.astype(np.float32),
            'labels': labels.astype(np.float32),
            'ids': ids
        }
        
        return formatted_data
    
    def _compute_split_statistics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute statistics for a split."""
        labels = [sample['label'] for sample in samples]
        base_sentiments = [sample['base_sentiment'] for sample in samples]
        
        stats = {
            'n_samples': len(samples),
            'label_mean': np.mean(labels),
            'label_std': np.std(labels),
            'label_min': np.min(labels),
            'label_max': np.max(labels),
            'base_sentiment_mean': np.mean(base_sentiments),
            'base_sentiment_std': np.std(base_sentiments),
            'missing_modality_freq': {}
        }
        
        # Compute missing modality frequencies
        for modality in ['text', 'audio', 'vision']:
            missing_count = sum(
                1 for sample in samples 
                if sample['missing_mask'].get(modality, False)
            )
            stats['missing_modality_freq'][modality] = missing_count / len(samples)
        
        # Compute feature statistics
        for modality in ['text', 'audio', 'vision']:
            features = np.stack([sample[modality] for sample in samples])
            stats[f'{modality}_feature_mean'] = np.mean(features)
            stats[f'{modality}_feature_std'] = np.std(features)
            stats[f'{modality}_feature_min'] = np.min(features)
            stats[f'{modality}_feature_max'] = np.max(features)
        
        return stats


def create_synthetic_datasets(
    output_dir: str,
    config: Optional[SyntheticConfig] = None,
    splits: List[str] = ["train", "valid", "test"]
) -> str:
    """
    Create synthetic multimodal datasets for testing.
    
    Args:
        output_dir: Directory to save datasets
        config: Synthetic data configuration
        splits: Which splits to create
        
    Returns:
        Path to the created dataset directory
    """
    if config is None:
        config = SyntheticConfig()
    
    generator = SyntheticDataGenerator(config)
    generator.save_dataset(output_dir, splits)
    
    logger.info(f"Created synthetic datasets in {output_dir}")
    return output_dir


def validate_synthetic_data(data_path: str) -> Dict[str, Any]:
    """Validate generated synthetic data."""
    data_path = Path(data_path)
    
    # Load dataset info
    with open(data_path / 'dataset_info.pkl', 'rb') as f:
        dataset_info = pickle.load(f)
    
    validation_results = {
        'dataset_info': dataset_info,
        'splits': {},
        'overall_valid': True
    }
    
    for split in dataset_info['splits']:
        split_path = data_path / f'{split}.pkl'
        if not split_path.exists():
            validation_results['overall_valid'] = False
            validation_results['splits'][split] = {'exists': False}
            continue
        
        # Load split data
        with open(split_path, 'rb') as f:
            split_data = pickle.load(f)
        
        # Validate split
        split_validation = {
            'exists': True,
            'n_samples': len(split_data['labels']),
            'expected_shapes': True,
            'no_nan_values': True,
            'label_range_valid': True
        }
        
        # Check shapes
        expected_shapes = {
            'text': (split_validation['n_samples'], dataset_info['target_seq_len'], dataset_info['feature_dims']['text']),
            'audio': (split_validation['n_samples'], dataset_info['target_seq_len'], dataset_info['feature_dims']['audio']),
            'vision': (split_validation['n_samples'], dataset_info['target_seq_len'], dataset_info['feature_dims']['vision']),
            'labels': (split_validation['n_samples'], 1)
        }
        
        for modality, expected_shape in expected_shapes.items():
            if modality in split_data:
                actual_shape = split_data[modality].shape
                if actual_shape != expected_shape:
                    split_validation['expected_shapes'] = False
                    logger.error(f"{split} {modality}: expected {expected_shape}, got {actual_shape}")
        
        # Check for NaN values
        for modality in ['text', 'audio', 'vision', 'labels']:
            if modality in split_data:
                if np.isnan(split_data[modality]).any():
                    split_validation['no_nan_values'] = False
                    logger.error(f"{split} {modality} contains NaN values")
        
        # Check label range
        labels = split_data['labels']
        label_min, label_max = dataset_info['label_range']
        if np.min(labels) < label_min - 0.1 or np.max(labels) > label_max + 0.1:
            split_validation['label_range_valid'] = False
            logger.error(f"{split} labels outside expected range: {np.min(labels):.2f} to {np.max(labels):.2f}")
        
        validation_results['splits'][split] = split_validation
        
        # Update overall validity
        if not all(split_validation[key] for key in ['expected_shapes', 'no_nan_values', 'label_range_valid']):
            validation_results['overall_valid'] = False
    
    return validation_results


def demo_synthetic_generation():
    """Demo synthetic data generation with visualization."""
    print("Demonstrating synthetic multimodal data generation...")
    
    # Create config for small demo
    config = SyntheticConfig(
        n_samples_train=10,
        n_samples_valid=5,
        n_samples_test=5,
        seq_len_range=(10, 30),
        target_seq_len=25,
        add_temporal_patterns=True,
        correlation_strength=0.7
    )
    
    generator = SyntheticDataGenerator(config)
    
    # Generate a few samples
    print("\nGenerating sample data...")
    samples = []
    for i in range(5):
        sample = generator.generate_sample(i)
        samples.append(sample)
        print(f"Sample {i}: label={sample['label']:.3f}, base_sentiment={sample['base_sentiment']:.3f}")
    
    # Show feature shapes
    sample = samples[0]
    print(f"\nFeature shapes:")
    print(f"Text: {sample['text'].shape}")
    print(f"Audio: {sample['audio'].shape}")
    print(f"Vision: {sample['vision'].shape}")
    
    # Show some statistics
    print(f"\nSample statistics:")
    for modality in ['text', 'audio', 'vision']:
        features = sample[modality]
        print(f"{modality}: mean={np.mean(features):.3f}, std={np.std(features):.3f}")
    
    # Show sentiment sequence for one sample
    print(f"\nSentiment sequence (first 10 timesteps):")
    sentiment_seq = sample['sentiment_sequence'][:10]
    print([f"{s:.2f}" for s in sentiment_seq])
    
    print("\nSynthetic data generation completed successfully!")


# Example usage and testing
if __name__ == "__main__":
    # Demo
    demo_synthetic_generation()
    
    # Create full synthetic datasets
    output_dir = "data/synthetic_mosi"
    
    # Configuration similar to MOSI dataset
    config = SyntheticConfig(
        n_samples_train=2000,
        n_samples_valid=500,
        n_samples_test=500,
        seq_len_range=(20, 80),
        target_seq_len=50,
        text_dim=768,
        audio_dim=74,
        vision_dim=47,
        sentiment_range=(-3.0, 3.0),
        correlation_strength=0.6,
        add_temporal_patterns=True,
        missing_prob=0.05,
        output_format="mosi"
    )
    
    try:
        # Generate datasets
        dataset_path = create_synthetic_datasets(output_dir, config)
        
        # Validate generated data
        validation_results = validate_synthetic_data(dataset_path)
        
        if validation_results['overall_valid']:
            print(f"✓ Successfully created valid synthetic datasets in {dataset_path}")
        else:
            print(f"✗ Validation failed for synthetic datasets")
            for split, results in validation_results['splits'].items():
                if not all(results.values()):
                    print(f"  - {split}: {results}")
        
        # Print dataset summary
        print(f"\nDataset Summary:")
        dataset_info = validation_results['dataset_info']
        print(f"  - Feature dims: {dataset_info['feature_dims']}")
        print(f"  - Sequence length: {dataset_info['target_seq_len']}")
        print(f"  - Label range: {dataset_info['label_range']}")
        print(f"  - Splits: {dataset_info['splits']}")
        
        for split in dataset_info['splits']:
            if split in validation_results['splits']:
                n_samples = validation_results['splits'][split]['n_samples']
                print(f"  - {split}: {n_samples} samples")
                
    except Exception as e:
        print(f"Error creating synthetic datasets: {e}")
        import traceback
        traceback.print_exc()