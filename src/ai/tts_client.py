"""
Text-to-Speech client using OpenAI's GPT-4o-mini TTS API.
Supports voice control through natural language instructions.
"""
import os
import tempfile
import base64
import json
from typing import Optional, Dict, Any
import requests
from pathlib import Path

from src.utils.logger import logger
from src.utils.exceptions import TTSProcessingError


class TTSClient:
    """OpenAI TTS client for text-to-speech conversion."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TTS client.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        # Get base URL from environment or use default
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
        self.base_url = base_url
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Available voices for GPT-4o-mini TTS
        self.available_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        
        # Supported models
        self.supported_models = {
            "gpt-4o-mini-tts": "standard_tts",
            "tts-1": "standard_tts",
            "gpt-4o-mini-audio-preview-2024-12-17": "audio_preview"
        }
    
    def _is_audio_preview_model(self, model: str) -> bool:
        """Check if the model is the new audio preview model."""
        return model in ["gpt-4o-mini-audio-preview-2024-12-17", "gpt-4o-audio-preview-2024-12-17"]
    
    def _call_standard_tts_api(self, text: str, voice: str, model: str, response_format: str, instructions: str) -> Optional[bytes]:
        """Call the standard TTS API endpoint."""
        url = f"{self.base_url}/audio/speech"
        
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "instructions": instructions
        }
        
        logger.info(f"Calling standard TTS API: {url}")
        
        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            audio_data = response.content
            logger.info(f"Standard TTS API success. Audio size: {len(audio_data)} bytes")
            return audio_data
        else:
            error_msg = f"Standard TTS API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise TTSProcessingError(error_msg)
    
    def _call_audio_preview_api(self, text: str, voice: str, model: str, response_format: str, instructions: str) -> Optional[bytes]:
        """Call the new audio preview API endpoint."""
        url = f"{self.base_url}/chat/completions"
        
        # Map response_format to audio format
        audio_format = response_format if response_format in ["mp3", "opus", "aac", "flac"] else "mp3"
        
        data = {
            "model": model,
            "modalities": ["text", "audio"],
            "audio": {"voice": voice, "format": audio_format},
            "echo": True,
            "messages": [
                {
                    "role": "system",
                    "content": f"output audio as HD quality. {instructions}"
                },
                {
                    "role": "user",
                    "content": "Act as an echo. Repeat the user's input word for word, without adding, removing, or changing anything."
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        }
        
        logger.info(f"Calling audio preview API: {url}")
        
        try:
            response = requests.post(url, headers=self.headers, json=data, timeout=60)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Check if the expected audio data is present
            if (
                response_data.get("choices") and
                len(response_data["choices"]) > 0 and
                response_data["choices"][0].get("message") and
                response_data["choices"][0]["message"].get("audio") and
                response_data["choices"][0]["message"]["audio"].get("data")
            ):
                audio_base64 = response_data["choices"][0]["message"]["audio"]["data"]
                # Decode base64 to binary
                audio_binary = base64.b64decode(audio_base64)
                logger.info(f"Audio preview API success. Audio size: {len(audio_binary)} bytes")
                return audio_binary
            else:
                error_msg = "Audio preview API: Could not find audio data in response"
                logger.error(error_msg)
                logger.error(f"Response: {json.dumps(response_data, indent=2)}")
                raise TTSProcessingError(error_msg)
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Audio preview API request error: {e}"
            logger.error(error_msg)
            raise TTSProcessingError(error_msg)
        except base64.binascii.Error as e:
            error_msg = f"Audio preview API: Error decoding base64 audio data: {e}"
            logger.error(error_msg)
            raise TTSProcessingError(error_msg)
        except Exception as e:
            error_msg = f"Audio preview API unexpected error: {e}"
            logger.error(error_msg)
            raise TTSProcessingError(error_msg)
    
    def text_to_speech(
        self, 
        text: str, 
        voice: str = "alloy", 
        model: str = "gpt-4o-mini-tts",
        response_format: str = "mp3",
        instructions: str = "Speak in a cheerful and positive tone."
    ) -> Optional[bytes]:
        """
        Convert text to speech using OpenAI TTS API.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model to use (gpt-4o-mini-tts or gpt-4o-mini-audio-preview-2024-12-17)
            response_format: Audio format (mp3, opus, aac, flac)
            instructions: Control voice aspects like accent, emotion, intonation, speed, tone, whispering
            
        Returns:
            Audio data as bytes, or None if failed
            
        Raises:
            TTSProcessingError: If TTS processing fails
        """
        if not text.strip():
            raise TTSProcessingError("Text cannot be empty")
        
        if voice not in self.available_voices:
            raise TTSProcessingError(f"Invalid voice. Available voices: {', '.join(self.available_voices)}")
        
        # Check if model is supported
        if model not in self.supported_models:
            raise TTSProcessingError(f"Unsupported model: {model}. Supported models: {', '.join(self.supported_models.keys())}")
        
        try:
            logger.info(f"Generating TTS for text (length: {len(text)}) with voice: {voice}, model: {model}")
            
            # Choose API based on model type
            if self._is_audio_preview_model(model):
                logger.info("Using audio preview API")
                return self._call_standard_tts_api(text, voice, model, response_format, instructions)
            else:
                logger.info("Using standard TTS API")
                return self._call_standard_tts_api(text, voice, model, response_format, instructions)
                
        except Exception as e:
            error_msg = f"TTS generation failed: {e}"
            logger.error(error_msg)
            raise TTSProcessingError(error_msg)
    
    def text_to_speech_file(
        self, 
        text: str, 
        output_path: str,
        voice: str = "alloy", 
        model: str = "gpt-4o-mini-tts",
        response_format: str = "mp3",
        instructions: str = "Speak in a cheerful and positive tone."
    ) -> Optional[str]:
        """
        Convert text to speech and save to file.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            voice: Voice to use
            model: TTS model to use
            response_format: Audio format
            instructions: Control voice aspects like accent, emotion, intonation, speed, tone, whispering
            
        Returns:
            Path to saved audio file, or None if failed
        """
        try:
            audio_data = self.text_to_speech(text, voice, model, response_format, instructions)
            if audio_data:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                
                logger.info(f"Audio saved to: {output_file}")
                return str(output_file)
            return None
            
        except Exception as e:
            logger.error(f"Failed to save TTS to file: {e}")
            return None
    
    def get_available_voices(self) -> list:
        """Get list of available voices."""
        return self.available_voices.copy()
    
    def validate_voice(self, voice: str) -> bool:
        """Check if voice is valid."""
        return voice in self.available_voices
    
    def get_supported_models(self) -> list:
        """Get list of supported models."""
        return list(self.supported_models.keys())
    
    def get_model_type(self, model: str) -> Optional[str]:
        """Get the type of a specific model."""
        return self.supported_models.get(model)
    
    def get_voice_info(self) -> Dict[str, Any]:
        """Get information about available voices and models."""
        return {
            "available_voices": self.available_voices,
            "supported_models": self.supported_models,
            "default_voice": "alloy",
            "default_model": "gpt-4o-mini-tts",
            "supported_formats": ["mp3", "opus", "aac", "flac"],
            "example_instructions": [
                "Speak in a cheerful and positive tone",
                "Speak slowly and clearly",
                "Use a professional business tone",
                "Speak with enthusiasm and energy",
                "Whisper softly and intimately",
                "Speak with a British accent",
                "Use emotional and expressive intonation",
                "Speak at a moderate pace"
            ]
        }
