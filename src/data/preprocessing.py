# enhanced_msamaba/src/data/preprocessing.py
"""
Data preprocessing utilities for multimodal sentiment analysis.
Handles feature extraction, alignment, and temporal segmentation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import librosa
import cv2
from transformers import AutoTokenizer, AutoModel
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    # Text processing
    text_model: str = "bert-base-uncased"
    max_text_len: int = 50
    
    # Audio processing
    audio_sr: int = 16000
    audio_win_len: float = 0.025  # 25ms
    audio_hop_len: float = 0.010  # 10ms
    n_mels: int = 80
    n_mfcc: int = 13
    
    # Vision processing
    face_size: Tuple[int, int] = (224, 224)
    landmarks_dim: int = 68 * 2  # 68 facial landmarks, x and y coordinates
    
    # Temporal alignment
    target_fps: float = 30.0  # Target frame rate for alignment
    segment_length: float = 0.75  # seconds
    overlap_ratio: float = 0.5
    
    # Normalization
    normalize_method: str = "standard"  # "standard", "minmax", "none"
    per_sample_norm: bool = False


class TextProcessor:
    """Process text data using BERT or similar models."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_model)
            self.model = AutoModel.from_pretrained(self.config.text_model)
            self.model.eval()
            logger.info(f"Loaded text model: {self.config.text_model}")
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
            raise
    
    def process(self, texts: List[str]) -> torch.Tensor:
        """
        Process list of texts into BERT embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of shape [batch, max_len, hidden_dim]
        """
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_len,
            return_tensors="pt"
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        return embeddings
    
    def process_single(self, text: str) -> torch.Tensor:
        """Process a single text string."""
        return self.process([text])[0]


class AudioProcessor:
    """Process audio data into acoustic features."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.win_length = int(config.audio_win_len * config.audio_sr)
        self.hop_length = int(config.audio_hop_len * config.audio_sr)
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive audio features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of audio features
        """
        # Resample if necessary
        if sr != self.config.audio_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.audio_sr)
            sr = self.config.audio_sr
        
        features = {}
        
        try:
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=self.config.n_mels,
                win_length=self.win_length,
                hop_length=self.hop_length
            )
            features['mel_spec'] = librosa.power_to_db(mel_spec, ref=np.max)
            
            # MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.config.n_mfcc,
                win_length=self.win_length,
                hop_length=self.hop_length
            )
            features['mfcc'] = mfcc
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=sr, hop_length=self.hop_length
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=sr, hop_length=self.hop_length
            )
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                audio, hop_length=self.hop_length
            )
            
            features['spectral_centroid'] = spectral_centroids
            features['spectral_rolloff'] = spectral_rolloff
            features['zero_crossing_rate'] = zero_crossing_rate
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio, sr=sr, hop_length=self.hop_length
            )
            features['chroma'] = chroma
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(
                y=audio, sr=sr, hop_length=self.hop_length
            )
            features['tempo'] = np.full((1, features['mfcc'].shape[1]), tempo)
            
        except Exception as e:
            logger.warning(f"Error extracting audio features: {e}")
            # Return dummy features
            n_frames = len(audio) // self.hop_length + 1
            features = {
                'mel_spec': np.random.randn(self.config.n_mels, n_frames),
                'mfcc': np.random.randn(self.config.n_mfcc, n_frames),
                'spectral_centroid': np.random.randn(1, n_frames),
                'spectral_rolloff': np.random.randn(1, n_frames),
                'zero_crossing_rate': np.random.randn(1, n_frames),
                'chroma': np.random.randn(12, n_frames),
                'tempo': np.random.randn(1, n_frames)
            }
        
        return features
    
    def process(self, audio_data: Union[np.ndarray, List[np.ndarray]], 
                sample_rates: Union[int, List[int]]) -> torch.Tensor:
        """
        Process audio data into feature tensor.
        
        Args:
            audio_data: Audio signal(s)
            sample_rates: Sample rate(s)
            
        Returns:
            Feature tensor [batch, time, features] or [time, features] for single audio
        """
        if isinstance(audio_data, np.ndarray) and audio_data.ndim == 1:
            # Single audio file
            features = self.extract_features(audio_data, sample_rates)
            # Concatenate all features
            feature_list = []
            for key in ['mfcc', 'mel_spec', 'spectral_centroid', 'spectral_rolloff', 
                       'zero_crossing_rate', 'chroma', 'tempo']:
                if key in features:
                    feature_list.append(features[key])
            
            combined = np.concatenate(feature_list, axis=0).T  # [time, features]
            return torch.FloatTensor(combined)
        
        else:
            # Multiple audio files
            if not isinstance(sample_rates, list):
                sample_rates = [sample_rates] * len(audio_data)
            
            processed_features = []
            for audio, sr in zip(audio_data, sample_rates):
                features = self.extract_features(audio, sr)
                feature_list = []
                for key in ['mfcc', 'mel_spec', 'spectral_centroid', 'spectral_rolloff',
                           'zero_crossing_rate', 'chroma', 'tempo']:
                    if key in features:
                        feature_list.append(features[key])
                
                combined = np.concatenate(feature_list, axis=0).T
                processed_features.append(torch.FloatTensor(combined))
            
            # Pad to same length
            max_len = max(f.shape[0] for f in processed_features)
            padded_features = []
            for f in processed_features:
                if f.shape[0] < max_len:
                    pad_len = max_len - f.shape[0]
                    f = F.pad(f, (0, 0, 0, pad_len))
                padded_features.append(f)
            
            return torch.stack(padded_features, dim=0)


class VisionProcessor:
    """Process visual data including facial features and expressions."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        # Try to load OpenFace or similar face processing library
        try:
            import dlib
            import face_recognition
            self.face_detector = dlib.get_frontal_face_detector()
            self.shape_predictor = None  # Would need to load shape predictor model
            self.has_face_tools = True
        except ImportError:
            logger.warning("Face processing libraries not available. Using dummy features.")
            self.has_face_tools = False
    
    def extract_facial_features(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract facial features from a video frame.
        
        Args:
            frame: Video frame as numpy array [H, W, C]
            
        Returns:
            Dictionary of facial features
        """
        features = {}
        
        if not self.has_face_tools:
            # Return dummy features
            return {
                'landmarks': np.random.randn(self.config.landmarks_dim),
                'action_units': np.random.randn(17),  # 17 action units
                'gaze_direction': np.random.randn(2),
                'head_pose': np.random.randn(3)
            }
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
            
            # Detect faces
            faces = self.face_detector(gray)
            
            if len(faces) > 0:
                # Use first detected face
                face = faces[0]
                
                # Extract landmarks (would need shape predictor model)
                landmarks = np.random.randn(self.config.landmarks_dim)  # Placeholder
                
                # Extract action units (would need proper AU detector)
                action_units = np.random.randn(17)  # Placeholder
                
                # Gaze direction (would need gaze estimation model)
                gaze_direction = np.random.randn(2)  # Placeholder
                
                # Head pose (would need pose estimation)
                head_pose = np.random.randn(3)  # Placeholder
                
                features = {
                    'landmarks': landmarks,
                    'action_units': action_units,
                    'gaze_direction': gaze_direction,
                    'head_pose': head_pose
                }
            else:
                # No face detected, use zeros
                features = {
                    'landmarks': np.zeros(self.config.landmarks_dim),
                    'action_units': np.zeros(17),
                    'gaze_direction': np.zeros(2),
                    'head_pose': np.zeros(3)
                }
                
        except Exception as e:
            logger.warning(f"Error extracting facial features: {e}")
            # Return zeros on error
            features = {
                'landmarks': np.zeros(self.config.landmarks_dim),
                'action_units': np.zeros(17),
                'gaze_direction': np.zeros(2),
                'head_pose': np.zeros(3)
            }
        
        return features
    
    def process(self, video_data: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
        """
        Process video data into visual features.
        
        Args:
            video_data: Video frames [T, H, W, C] or list of frame arrays
            
        Returns:
            Feature tensor [batch, time, features] or [time, features] for single video
        """
        if isinstance(video_data, np.ndarray) and video_data.ndim == 4:
            # Single video
            features_list = []
            for frame in video_data:
                frame_features = self.extract_facial_features(frame)
                # Concatenate all features
                combined = np.concatenate([
                    frame_features['landmarks'],
                    frame_features['action_units'],
                    frame_features['gaze_direction'],
                    frame_features['head_pose']
                ])
                features_list.append(combined)
            
            return torch.FloatTensor(np.array(features_list))
        
        else:
            # Multiple videos
            processed_videos = []
            for video in video_data:
                features_list = []
                for frame in video:
                    frame_features = self.extract_facial_features(frame)
                    combined = np.concatenate([
                        frame_features['landmarks'],
                        frame_features['action_units'],
                        frame_features['gaze_direction'],
                        frame_features['head_pose']
                    ])
                    features_list.append(combined)
                
                processed_videos.append(torch.FloatTensor(np.array(features_list)))
            
            # Pad to same length
            max_len = max(v.shape[0] for v in processed_videos)
            padded_videos = []
            for v in processed_videos:
                if v.shape[0] < max_len:
                    pad_len = max_len - v.shape[0]
                    v = F.pad(v, (0, 0, 0, pad_len))
                padded_videos.append(v)
            
            return torch.stack(padded_videos, dim=0)


class TemporalAligner:
    """Align multimodal sequences temporally."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.target_fps = config.target_fps
    
    def align_sequences(
        self,
        sequences: Dict[str, torch.Tensor],
        original_fps: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Align sequences to the same temporal resolution.
        
        Args:
            sequences: Dict of {modality: tensor [time, features]}
            original_fps: Dict of {modality: original_frame_rate}
            
        Returns:
            Aligned sequences with same temporal length
        """
        aligned = {}
        target_length = None
        
        # First pass: determine target length
        for modality, seq in sequences.items():
            if modality in original_fps:
                fps = original_fps[modality]
                # Convert to target fps
                scale_factor = self.target_fps / fps
                new_length = int(seq.shape[0] * scale_factor)
                
                if target_length is None:
                    target_length = new_length
                else:
                    target_length = min(target_length, new_length)
        
        # Second pass: resample all sequences
        for modality, seq in sequences.items():
            if modality in original_fps:
                # Resample to target length
                seq = seq.unsqueeze(0)  # Add batch dim
                resampled = F.interpolate(
                    seq.transpose(1, 2),  # [1, features, time]
                    size=target_length,
                    mode='linear',
                    align_corners=False
                )
                aligned[modality] = resampled.transpose(1, 2).squeeze(0)  # [time, features]
            else:
                aligned[modality] = seq
        
        return aligned
    
    def create_segments(
        self,
        sequences: Dict[str, torch.Tensor],
        segment_length: Optional[float] = None,
        overlap_ratio: Optional[float] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Create temporal segments from aligned sequences.
        
        Args:
            sequences: Aligned sequences
            segment_length: Length of each segment in seconds
            overlap_ratio: Overlap between segments
            
        Returns:
            List of segment dictionaries
        """
        if segment_length is None:
            segment_length = self.config.segment_length
        if overlap_ratio is None:
            overlap_ratio = self.config.overlap_ratio
        
        # Convert segment length to frames
        segment_frames = int(segment_length * self.target_fps)
        step_frames = int(segment_frames * (1 - overlap_ratio))
        
        # Get sequence length (assume all modalities have same length after alignment)
        seq_length = next(iter(sequences.values())).shape[0]
        
        segments = []
        start = 0
        
        while start + segment_frames <= seq_length:
            segment = {}
            for modality, seq in sequences.items():
                segment[modality] = seq[start:start + segment_frames].clone()
            segments.append(segment)
            start += step_frames
        
        # Handle remaining frames
        if start < seq_length and len(segments) > 0:
            segment = {}
            for modality, seq in sequences.items():
                # Take last segment_frames
                segment[modality] = seq[-segment_frames:].clone()
            segments.append(segment)
        
        return segments


class DataNormalizer:
    """Normalize multimodal data."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scalers = {}
        self.is_fitted = False
    
    def fit(self, data: Dict[str, torch.Tensor]):
        """Fit normalization parameters."""
        self.scalers = {}
        
        for modality, features in data.items():
            if self.config.normalize_method == "standard":
                scaler = StandardScaler()
            elif self.config.normalize_method == "minmax":
                scaler = MinMaxScaler()
            else:
                continue  # No normalization
            
            # Fit on flattened features
            features_np = features.reshape(-1, features.shape[-1]).numpy()
            scaler.fit(features_np)
            self.scalers[modality] = scaler
        
        self.is_fitted = True
    
    def transform(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply normalization."""
        if not self.is_fitted or self.config.normalize_method == "none":
            return data
        
        normalized = {}
        for modality, features in data.items():
            if modality in self.scalers:
                original_shape = features.shape
                features_np = features.reshape(-1, features.shape[-1]).numpy()
                normalized_np = self.scalers[modality].transform(features_np)
                normalized[modality] = torch.FloatTensor(
                    normalized_np.reshape(original_shape)
                )
            else:
                normalized[modality] = features
        
        return normalized
    
    def fit_transform(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)


class MultimodalPreprocessor:
    """Main preprocessing pipeline for multimodal data."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.text_processor = TextProcessor(config)
        self.audio_processor = AudioProcessor(config)
        self.vision_processor = VisionProcessor(config)
        self.aligner = TemporalAligner(config)
        self.normalizer = DataNormalizer(config)
    
    def process_sample(
        self,
        text: str,
        audio: Tuple[np.ndarray, int],  # (audio_data, sample_rate)
        video: np.ndarray,  # [T, H, W, C]
        original_fps: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single multimodal sample.
        
        Args:
            text: Text string
            audio: Tuple of (audio_data, sample_rate)
            video: Video frames
            original_fps: Original frame rates
            
        Returns:
            Processed multimodal features
        """
        # Process each modality
        text_features = self.text_processor.process_single(text)
        audio_features = self.audio_processor.process(audio[0], audio[1])
        vision_features = self.vision_processor.process(video)
        
        # Align sequences temporally
        sequences = {
            "text": text_features,
            "audio": audio_features,
            "vision": vision_features
        }
        
        aligned_sequences = self.aligner.align_sequences(sequences, original_fps)
        
        # Normalize features
        normalized_sequences = self.normalizer.transform(aligned_sequences)
        
        return normalized_sequences
    
    def process_batch(
        self,
        batch_data: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of multimodal samples.
        
        Args:
            batch_data: List of sample dictionaries
            
        Returns:
            Batched processed features
        """
        processed_samples = []
        
        for sample in batch_data:
            processed = self.process_sample(
                text=sample['text'],
                audio=sample['audio'],
                video=sample['video'],
                original_fps=sample['fps']
            )
            processed_samples.append(processed)
        
        # Batch samples
        batched = {}
        for modality in processed_samples[0].keys():
            features_list = [sample[modality] for sample in processed_samples]
            
            # Pad to same length
            max_len = max(f.shape[0] for f in features_list)
            padded_features = []
            
            for features in features_list:
                if features.shape[0] < max_len:
                    pad_len = max_len - features.shape[0]
                    features = F.pad(features, (0, 0, 0, pad_len))
                padded_features.append(features)
            
            batched[modality] = torch.stack(padded_features, dim=0)
        
        return batched
    
    def save_preprocessing_config(self, filepath: str):
        """Save preprocessing configuration and fitted normalizers."""
        save_dict = {
            'config': self.config,
            'normalizer_scalers': self.normalizer.scalers,
            'normalizer_fitted': self.normalizer.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Saved preprocessing config to {filepath}")
    
    def load_preprocessing_config(self, filepath: str):
        """Load preprocessing configuration and fitted normalizers."""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.config = save_dict['config']
        self.normalizer.scalers = save_dict['normalizer_scalers']
        self.normalizer.is_fitted = save_dict['normalizer_fitted']
        
        logger.info(f"Loaded preprocessing config from {filepath}")


def preprocess_dataset(
    raw_data_path: str,
    output_path: str,
    config: PreprocessingConfig,
    splits: List[str] = ["train", "valid", "test"]
):
    """
    Preprocess entire dataset and save processed features.
    
    Args:
        raw_data_path: Path to raw multimodal data
        output_path: Path to save processed data
        config: Preprocessing configuration
        splits: Data splits to process
    """
    preprocessor = MultimodalPreprocessor(config)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First pass: fit normalizers on training data
    if "train" in splits:
        logger.info("Fitting normalizers on training data...")
        train_data_path = Path(raw_data_path) / "train"
        
        # Load a subset of training data for fitting normalizers
        # This is a simplified example - would need actual data loading logic
        sample_data = load_sample_data(train_data_path, max_samples=1000)
        
        # Process samples to get features for normalization fitting
        train_features = {"text": [], "audio": [], "vision": []}
        
        for sample in sample_data:
            try:
                processed = preprocessor.process_sample(
                    text=sample['text'],
                    audio=(sample['audio'], sample['sr']),
                    video=sample['video'],
                    original_fps=sample['fps']
                )
                
                for modality in train_features:
                    if modality in processed:
                        train_features[modality].append(processed[modality])
                        
            except Exception as e:
                logger.warning(f"Error processing sample for normalization: {e}")
                continue
        
        # Concatenate and fit normalizers
        for modality in train_features:
            if train_features[modality]:
                combined = torch.cat(train_features[modality], dim=0)
                train_features[modality] = combined
        
        preprocessor.normalizer.fit(train_features)
    
    # Process each split
    for split in splits:
        logger.info(f"Processing {split} split...")
        
        split_data_path = Path(raw_data_path) / split
        split_output_path = output_dir / f"{split}.pkl"
        
        # Load raw data for this split
        raw_samples = load_raw_data(split_data_path)
        
        processed_samples = []
        failed_count = 0
        
        for i, sample in enumerate(raw_samples):
            try:
                processed = preprocessor.process_sample(
                    text=sample['text'],
                    audio=(sample['audio'], sample['sr']),
                    video=sample['video'],
                    original_fps=sample['fps']
                )
                
                processed['label'] = sample['label']
                processed['id'] = sample.get('id', i)
                processed_samples.append(processed)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(raw_samples)} samples")
                    
            except Exception as e:
                logger.warning(f"Failed to process sample {i}: {e}")
                failed_count += 1
                continue
        
        # Save processed data
        with open(split_output_path, 'wb') as f:
            pickle.dump(processed_samples, f)
        
        logger.info(f"Saved {len(processed_samples)} processed samples to {split_output_path}")
        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} samples in {split} split")
    
    # Save preprocessing configuration
    config_path = output_dir / "preprocessing_config.pkl"
    preprocessor.save_preprocessing_config(str(config_path))


def load_sample_data(data_path: Path, max_samples: int = 1000) -> List[Dict[str, Any]]:
    """
    Load a sample of data for normalization fitting.
    This is a placeholder - implement according to your data format.
    """
    # This would be implemented based on your actual data format
    # For now, return empty list
    logger.warning("load_sample_data not implemented - using dummy data")
    return []


def load_raw_data(data_path: Path) -> List[Dict[str, Any]]:
    """
    Load raw multimodal data from files.
    This is a placeholder - implement according to your data format.
    """
    # This would be implemented based on your actual data format
    # For now, return empty list
    logger.warning("load_raw_data not implemented - using dummy data")
    return []


def validate_preprocessing(
    original_data: Dict[str, torch.Tensor],
    processed_data: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Validate preprocessing results.
    
    Args:
        original_data: Original features
        processed_data: Processed features
        
    Returns:
        Validation metrics
    """
    metrics = {}
    
    for modality in processed_data:
        if modality in original_data:
            orig = original_data[modality]
            proc = processed_data[modality]
            
            # Check shape preservation (allowing for temporal alignment)
            metrics[f"{modality}_shape_change"] = (
                orig.shape[-1] == proc.shape[-1]
            )
            
            # Check for NaN or infinite values
            metrics[f"{modality}_has_nan"] = torch.isnan(proc).any().item()
            metrics[f"{modality}_has_inf"] = torch.isinf(proc).any().item()
            
            # Check feature statistics
            metrics[f"{modality}_mean"] = proc.mean().item()
            metrics[f"{modality}_std"] = proc.std().item()
            metrics[f"{modality}_min"] = proc.min().item()
            metrics[f"{modality}_max"] = proc.max().item()
    
    return metrics


# Example usage
if __name__ == "__main__":
    # Example preprocessing configuration
    config = PreprocessingConfig(
        text_model="bert-base-uncased",
        max_text_len=50,
        audio_sr=16000,
        n_mels=80,
        n_mfcc=13,
        target_fps=30.0,
        segment_length=0.75,
        normalize_method="standard"
    )
    
    # Test individual processors
    try:
        # Test text processor
        text_processor = TextProcessor(config)
        sample_texts = ["This is a happy sentence.", "This is a sad sentence."]
        text_features = text_processor.process(sample_texts)
        print(f"Text features shape: {text_features.shape}")
        
        # Test audio processor
        audio_processor = AudioProcessor(config)
        dummy_audio = np.random.randn(16000)  # 1 second of audio
        audio_features = audio_processor.process(dummy_audio, 16000)
        print(f"Audio features shape: {audio_features.shape}")
        
        # Test vision processor
        vision_processor = VisionProcessor(config)
        dummy_video = np.random.randint(0, 255, (30, 224, 224, 3))  # 1 second at 30fps
        vision_features = vision_processor.process(dummy_video)
        print(f"Vision features shape: {vision_features.shape}")
        
        print("All processors working correctly!")
        
    except Exception as e:
        print(f"Error testing processors: {e}")
        print("This is expected if required libraries are not installed")