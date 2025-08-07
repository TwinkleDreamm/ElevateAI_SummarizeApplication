"""
Audio processing module for ElevateAI.
Handles audio extraction, enhancement, and vocal separation.
"""
import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

try:
    import librosa
    import soundfile as sf
    import noisereduce as nr
    AUDIO_DEPS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Audio processing dependencies not installed: {e}")
    AUDIO_DEPS_AVAILABLE = False

from .base_processor import BaseProcessor
from src.utils.exceptions import AudioProcessingError


class AudioProcessor(BaseProcessor):
    """Handles audio processing operations."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the audio processor.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.sample_rate = self.settings.audio_sample_rate
        self.noise_reduction_strength = self.settings.noise_reduction_strength
    
    def process(self, input_data: Union[str, Path], **kwargs) -> Path:
        """
        Process audio file with enhancement and vocal separation.
        
        Args:
            input_data: Path to audio file
            **kwargs: Additional processing parameters
            
        Returns:
            Path to processed audio file
            
        Raises:
            AudioProcessingError: If processing fails
        """
        if not self.validate_input(input_data):
            raise AudioProcessingError(f"Invalid input: {input_data}")
        
        audio_path = Path(input_data)
        self.log_processing_start("Audio processing", str(audio_path))
        
        try:
            # Load audio
            audio_data, sr = self._load_audio(audio_path)
            
            # Enhance audio quality
            enhanced_audio = self._enhance_audio(audio_data, sr)
            
            # Separate vocals if needed
            if kwargs.get('separate_vocals', True):
                enhanced_audio = self._separate_vocals(enhanced_audio, sr)
            
            # Save processed audio
            output_path = self.get_temp_file_path(".wav")
            self._save_audio(enhanced_audio, sr, output_path)
            
            self.log_processing_end("Audio processing")
            return output_path
            
        except Exception as e:
            self.handle_error(e, "Audio processing")
    
    def extract_audio_from_video(self, video_path: Union[str, Path]) -> Path:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
            
        Raises:
            AudioProcessingError: If extraction fails
        """
        try:
            import moviepy.editor as mp
        except ImportError:
            raise AudioProcessingError("moviepy not installed. Please install it to extract audio from video.")
        
        video_path = Path(video_path)
        self.log_processing_start("Audio extraction", str(video_path))
        
        try:
            # Load video and extract audio
            video = mp.VideoFileClip(str(video_path))
            audio = video.audio
            
            # Save audio to temporary file
            output_path = self.get_temp_file_path(".wav")
            audio.write_audiofile(
                str(output_path),
                verbose=False,
                logger=None
            )
            
            # Clean up
            audio.close()
            video.close()
            
            self.log_processing_end("Audio extraction")
            return output_path
            
        except Exception as e:
            self.handle_error(e, "Audio extraction from video")
    
    def _load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio_data, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                mono=True
            )
            return audio_data, sr
        except Exception as e:
            raise AudioProcessingError(f"Failed to load audio: {e}")
    
    def _enhance_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Enhance audio quality by reducing noise.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            
        Returns:
            Enhanced audio data
        """
        try:
            # Apply noise reduction
            enhanced_audio = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                prop_decrease=self.noise_reduction_strength
            )
            
            # Normalize audio
            enhanced_audio = librosa.util.normalize(enhanced_audio)
            
            return enhanced_audio
            
        except Exception as e:
            self.logger.warning(f"Audio enhancement failed: {e}")
            return audio_data  # Return original if enhancement fails
    
    def _separate_vocals(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Separate vocals from background music (simplified version).
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            
        Returns:
            Vocal-separated audio data
        """
        try:
            # Simple vocal separation using harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio_data)
            
            # Use harmonic component as it typically contains vocals
            return harmonic
            
        except Exception as e:
            self.logger.warning(f"Vocal separation failed: {e}")
            return audio_data  # Return original if separation fails
    
    def _save_audio(self, audio_data: np.ndarray, sample_rate: int, output_path: Path) -> None:
        """
        Save audio data to file.
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            output_path: Output file path
        """
        try:
            sf.write(str(output_path), audio_data, sample_rate)
        except Exception as e:
            raise AudioProcessingError(f"Failed to save audio: {e}")
    
    def get_audio_duration(self, audio_path: Union[str, Path]) -> float:
        """
        Get audio duration in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            duration = librosa.get_duration(filename=str(audio_path))
            return duration
        except Exception as e:
            self.logger.warning(f"Failed to get audio duration: {e}")
            return 0.0
    
    def validate_audio_format(self, file_path: Path) -> bool:
        """
        Validate if file format is supported for audio processing.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if format is supported
        """
        suffix = file_path.suffix.lower().lstrip('.')
        return suffix in self.settings.supported_audio_formats
