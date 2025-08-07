"""
Multi-modal AI features for ElevateAI.
Handles text-to-speech, speech-to-text, and image generation.
"""
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import io

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import LLMError


@dataclass
class TTSResult:
    """Result of text-to-speech conversion."""
    audio_data: bytes
    format: str
    duration: Optional[float]
    voice: str
    model: str


@dataclass
class ImageGenerationResult:
    """Result of image generation."""
    image_data: bytes
    format: str
    size: str
    prompt: str
    model: str
    revised_prompt: Optional[str] = None


class MultiModalAI:
    """Multi-modal AI capabilities for ElevateAI."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-modal AI system.
        
        Args:
            config: Optional configuration dictionary
        """
        if not OPENAI_AVAILABLE:
            raise LLMError("OpenAI library not available for multi-modal features")
        
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Initialize OpenAI clients
        self.azure_client = None
        self.openai_client = None
        
        self._initialize_clients()
        
        # Default parameters
        self.default_tts_voice = self.config.get('tts_voice', 'alloy')
        self.default_tts_model = self.config.get('tts_model', 'tts-1')
        self.default_image_size = self.config.get('image_size', '1024x1024')
        self.default_image_model = self.config.get('image_model', 'dall-e-3')
    
    def _initialize_clients(self) -> None:
        """Initialize OpenAI clients for multi-modal features."""
        try:
            # Azure OpenAI client
            if (settings.azure_openai_api_key and settings.azure_openai_endpoint):
                from openai import AzureOpenAI
                self.azure_client = AzureOpenAI(
                    api_key=settings.azure_openai_api_key,
                    api_version=settings.azure_openai_api_version,
                    azure_endpoint=settings.azure_openai_endpoint
                )
                self.logger.info("Azure OpenAI client initialized for multi-modal")
            
            # OpenAI client
            if settings.openai_api_key:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=settings.openai_api_key)
                self.logger.info("OpenAI client initialized for multi-modal")
            
            if not self.azure_client and not self.openai_client:
                self.logger.warning("No OpenAI API credentials configured for multi-modal features")

        except Exception as e:
            self.logger.error(f"Failed to initialize multi-modal clients: {e}")
            # Don't raise exception, just log the error
            pass
    
    def text_to_speech(self, text: str, **kwargs) -> TTSResult:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            **kwargs: Additional TTS parameters
            
        Returns:
            TTSResult object
            
        Raises:
            LLMError: If TTS conversion fails
        """
        if not text or not text.strip():
            raise LLMError("Empty text provided for TTS")
        
        try:
            self.logger.info(f"Converting text to speech: {len(text)} characters")
            
            # Use OpenAI client (Azure doesn't support TTS yet)
            client = self.openai_client
            if not client:
                raise LLMError("OpenAI client required for TTS")
            
            voice = kwargs.get('voice', self.default_tts_voice)
            model = kwargs.get('model', self.default_tts_model)
            
            # Generate speech
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=kwargs.get('format', 'mp3')
            )
            
            # Get audio data
            audio_data = response.content
            
            return TTSResult(
                audio_data=audio_data,
                format=kwargs.get('format', 'mp3'),
                duration=None,  # OpenAI doesn't provide duration
                voice=voice,
                model=model
            )
            
        except Exception as e:
            self.logger.error(f"TTS conversion failed: {e}")
            raise LLMError(f"TTS conversion failed: {e}")
    
    def save_tts_audio(self, tts_result: TTSResult, output_path: Union[str, Path]) -> Path:
        """
        Save TTS audio to file.
        
        Args:
            tts_result: TTS result to save
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        try:
            output_path = Path(output_path)
            
            with open(output_path, 'wb') as f:
                f.write(tts_result.audio_data)
            
            self.logger.info(f"TTS audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save TTS audio: {e}")
            raise LLMError(f"Failed to save TTS audio: {e}")
    
    def generate_image(self, prompt: str, **kwargs) -> ImageGenerationResult:
        """
        Generate image from text prompt.
        
        Args:
            prompt: Text prompt for image generation
            **kwargs: Additional image generation parameters
            
        Returns:
            ImageGenerationResult object
            
        Raises:
            LLMError: If image generation fails
        """
        if not prompt or not prompt.strip():
            raise LLMError("Empty prompt provided for image generation")
        
        try:
            self.logger.info(f"Generating image from prompt: {prompt[:100]}...")
            
            # Use OpenAI client (Azure may not support DALL-E)
            client = self.openai_client
            if not client:
                raise LLMError("OpenAI client required for image generation")
            
            model = kwargs.get('model', self.default_image_model)
            size = kwargs.get('size', self.default_image_size)
            quality = kwargs.get('quality', 'standard')
            
            # Generate image
            response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1,
                response_format='b64_json'
            )
            
            # Get image data
            image_data = base64.b64decode(response.data[0].b64_json)
            revised_prompt = getattr(response.data[0], 'revised_prompt', None)
            
            return ImageGenerationResult(
                image_data=image_data,
                format='png',
                size=size,
                prompt=prompt,
                model=model,
                revised_prompt=revised_prompt
            )
            
        except Exception as e:
            self.logger.error(f"Image generation failed: {e}")
            raise LLMError(f"Image generation failed: {e}")
    
    def save_generated_image(self, image_result: ImageGenerationResult, 
                           output_path: Union[str, Path]) -> Path:
        """
        Save generated image to file.
        
        Args:
            image_result: Image generation result
            output_path: Output file path
            
        Returns:
            Path to saved file
        """
        try:
            output_path = Path(output_path)
            
            with open(output_path, 'wb') as f:
                f.write(image_result.image_data)
            
            self.logger.info(f"Generated image saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save generated image: {e}")
            raise LLMError(f"Failed to save generated image: {e}")
    
    def analyze_image(self, image_path: Union[str, Path], prompt: str = "Describe this image") -> str:
        """
        Analyze image using vision model.
        
        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            
        Returns:
            Image analysis text
            
        Raises:
            LLMError: If image analysis fails
        """
        if not PIL_AVAILABLE:
            raise LLMError("PIL not available for image analysis")
        
        try:
            self.logger.info(f"Analyzing image: {image_path}")
            
            # Use OpenAI client with vision model
            client = self.openai_client
            if not client:
                raise LLMError("OpenAI client required for image analysis")
            
            # Load and encode image
            image_path = Path(image_path)
            if not image_path.exists():
                raise LLMError(f"Image file not found: {image_path}")
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Encode to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Analyze image
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")
            raise LLMError(f"Image analysis failed: {e}")
    
    def create_summary_visualization(self, summary: str, **kwargs) -> ImageGenerationResult:
        """
        Create a visual representation of a summary.
        
        Args:
            summary: Summary text to visualize
            **kwargs: Additional parameters
            
        Returns:
            ImageGenerationResult object
        """
        try:
            # Create visualization prompt
            viz_prompt = f"""Create a clean, professional infographic that visually represents this summary:

{summary[:500]}...

Style: Modern, minimalist infographic with clear sections, icons, and readable text. Use a professional color scheme."""
            
            return self.generate_image(viz_prompt, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Summary visualization failed: {e}")
            raise LLMError(f"Summary visualization failed: {e}")
    
    def create_audio_summary(self, summary: str, **kwargs) -> TTSResult:
        """
        Create audio version of summary.
        
        Args:
            summary: Summary text to convert
            **kwargs: Additional TTS parameters
            
        Returns:
            TTSResult object
        """
        try:
            # Optimize text for speech
            speech_text = self._optimize_text_for_speech(summary)
            
            return self.text_to_speech(speech_text, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Audio summary creation failed: {e}")
            raise LLMError(f"Audio summary creation failed: {e}")
    
    def _optimize_text_for_speech(self, text: str) -> str:
        """
        Optimize text for better speech synthesis.
        
        Args:
            text: Original text
            
        Returns:
            Optimized text for speech
        """
        # Add pauses for better speech flow
        optimized = text.replace('. ', '. ... ')
        optimized = optimized.replace(':', ': ... ')
        optimized = optimized.replace(';', '; ... ')
        
        # Replace abbreviations with full words
        replacements = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'and so on',
            'vs.': 'versus',
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Dr.': 'Doctor'
        }
        
        for abbrev, full in replacements.items():
            optimized = optimized.replace(abbrev, full)
        
        return optimized
    
    def get_available_voices(self) -> List[str]:
        """
        Get list of available TTS voices.
        
        Returns:
            List of voice names
        """
        # OpenAI TTS voices
        return ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
    
    def get_supported_image_sizes(self) -> List[str]:
        """
        Get list of supported image sizes.
        
        Returns:
            List of size strings
        """
        return ['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792']
    
    def get_multimodal_capabilities(self) -> Dict[str, Any]:
        """
        Get information about available multi-modal capabilities.
        
        Returns:
            Dictionary with capability information
        """
        return {
            'text_to_speech': {
                'available': bool(self.openai_client),
                'voices': self.get_available_voices(),
                'formats': ['mp3', 'opus', 'aac', 'flac']
            },
            'image_generation': {
                'available': bool(self.openai_client),
                'models': ['dall-e-2', 'dall-e-3'],
                'sizes': self.get_supported_image_sizes()
            },
            'image_analysis': {
                'available': bool(self.openai_client),
                'models': ['gpt-4-vision-preview']
            },
            'clients': {
                'azure_available': bool(self.azure_client),
                'openai_available': bool(self.openai_client)
            }
        }
