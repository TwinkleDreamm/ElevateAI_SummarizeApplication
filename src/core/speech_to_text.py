"""
Speech-to-text processing module for ElevateAI.
Handles audio transcription using Whisper and other STT models.
"""
from pathlib import Path
from typing import Optional, Union, Dict, Any

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_processor import BaseProcessor
from src.utils.exceptions import SpeechToTextError


class SpeechToTextProcessor(BaseProcessor):
    """Handles speech-to-text conversion operations."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the speech-to-text processor.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.whisper_model = None
        self.model_name = config.get('whisper_model', 'base') if config else 'base'
        self.language = config.get('language', self.settings.default_language) if config else self.settings.default_language
    
    def process(self, input_data: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Process audio file and convert to text.
        
        Args:
            input_data: Path to audio file
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary with transcription results
            
        Raises:
            SpeechToTextError: If processing fails
        """
        if not self.validate_input(input_data):
            raise SpeechToTextError(f"Invalid input: {input_data}")
        
        audio_path = Path(input_data)
        self.log_processing_start("Speech-to-text conversion", str(audio_path))
        
        try:
            # Choose STT method based on availability and configuration
            use_openai = kwargs.get('use_openai_api', False)
            
            if use_openai and OPENAI_AVAILABLE and self.settings.openai_api_key:
                result = self._transcribe_with_openai_api(audio_path, **kwargs)
            elif WHISPER_AVAILABLE:
                result = self._transcribe_with_whisper(audio_path, **kwargs)
            else:
                raise SpeechToTextError("No speech-to-text engine available")
            
            self.log_processing_end("Speech-to-text conversion")
            return result
            
        except Exception as e:
            self.handle_error(e, "Speech-to-text conversion")
    
    def _transcribe_with_whisper(self, audio_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using local Whisper model.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional parameters
            
        Returns:
            Transcription result dictionary
        """
        if not WHISPER_AVAILABLE:
            raise SpeechToTextError("Whisper not available. Please install openai-whisper.")
        
        try:
            # Load model if not already loaded
            if self.whisper_model is None:
                self.logger.info(f"Loading Whisper model: {self.model_name}")
                self.whisper_model = whisper.load_model(self.model_name)
            
            # Transcribe audio
            result = self.whisper_model.transcribe(
                str(audio_path),
                language=kwargs.get('language', self.language),
                task=kwargs.get('task', 'transcribe'),  # 'transcribe' or 'translate'
                verbose=False
            )
            
            return {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
                "confidence": self._calculate_average_confidence(result.get("segments", [])),
                "method": "whisper_local",
                "model": self.model_name
            }
            
        except Exception as e:
            raise SpeechToTextError(f"Whisper transcription failed: {e}")
    
    def _transcribe_with_openai_api(self, audio_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using OpenAI API.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional parameters
            
        Returns:
            Transcription result dictionary
        """
        if not OPENAI_AVAILABLE:
            raise SpeechToTextError("OpenAI client not available. Please install openai.")
        
        try:
            client = openai.OpenAI(api_key=self.settings.openai_api_key)
            
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=kwargs.get('language', self.language),
                    response_format="verbose_json"
                )
            
            return {
                "text": transcript.text.strip(),
                "language": transcript.language,
                "segments": getattr(transcript, 'segments', []),
                "confidence": None,  # OpenAI API doesn't provide confidence scores
                "method": "openai_api",
                "model": "whisper-1"
            }
            
        except Exception as e:
            raise SpeechToTextError(f"OpenAI API transcription failed: {e}")
    
    def _calculate_average_confidence(self, segments: list) -> Optional[float]:
        """
        Calculate average confidence score from segments.
        
        Args:
            segments: List of transcription segments
            
        Returns:
            Average confidence score or None
        """
        if not segments:
            return None
        
        try:
            confidences = []
            for segment in segments:
                if 'avg_logprob' in segment:
                    # Convert log probability to confidence (approximate)
                    confidence = min(1.0, max(0.0, (segment['avg_logprob'] + 1.0)))
                    confidences.append(confidence)
            
            return sum(confidences) / len(confidences) if confidences else None
            
        except Exception:
            return None
    
    def analyze_transcript_content(self, transcript: str) -> str:
        """
        Analyze transcript content to determine quality and length.
        
        Args:
            transcript: Transcribed text
            
        Returns:
            Content status: "no_content", "limited_content", or "sufficient_content"
        """
        if not transcript or not transcript.strip():
            return "no_content"
        
        # Clean and analyze text
        clean_text = transcript.strip()
        word_count = len(clean_text.split())
        
        # Define thresholds
        if word_count < 10:
            return "no_content"
        elif word_count < 50:
            return "limited_content"
        else:
            return "sufficient_content"
    
    def get_supported_languages(self) -> list:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        if WHISPER_AVAILABLE:
            # Whisper supports many languages
            return [
                "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
                "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",
                "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy",
                "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu",
                "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
                "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo",
                "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
                "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
            ]
        else:
            return self.settings.supported_languages
    
    def estimate_processing_time(self, audio_duration: float) -> float:
        """
        Estimate processing time based on audio duration.
        
        Args:
            audio_duration: Audio duration in seconds
            
        Returns:
            Estimated processing time in seconds
        """
        # Rough estimates based on model size and hardware
        model_multipliers = {
            "tiny": 0.1,
            "base": 0.2,
            "small": 0.4,
            "medium": 0.8,
            "large": 1.5
        }
        
        multiplier = model_multipliers.get(self.model_name, 0.5)
        return audio_duration * multiplier
