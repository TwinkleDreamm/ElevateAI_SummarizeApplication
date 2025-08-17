"""
Settings Manager for ElevateAI application.
Handles loading, saving, and applying settings across the application.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st

from config.settings import settings as app_settings


class SettingsManager:
    """Manages application settings with persistence and validation."""
    
    def __init__(self):
        self.config_dir = Path("config")
        self.user_config_file = self.config_dir / "user_settings.json"
        self.default_settings = self._get_default_settings()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings for the application."""
        return {
            # Model settings
            'temperature': 0.7,
            'max_tokens': 2000,
            'top_p': 0.9,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'model_name': 'gpt-4',
            
            # Search settings
            'similarity_threshold': 0.25,
            'max_results': 10,
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'enable_web_search': False,
            'enable_function_calling': False,
            
            # Audio settings
            'enable_tts': True,
            'tts_voice': 'alloy',
            'audio_sample_rate': 16000,
            'noise_reduction': 0.8,
            'enable_vocal_separation': False,
            
            # Memory settings
            'enable_memory': True,
            'max_memory_context': 3,
            'memory_consolidation_threshold': 10,
            'store_conversations': True,
            'memory_retention_days': 30,
            'auto_cleanup': True,
            
            # Interface settings
            'theme': 'light',
            'language': 'vi',
            'auto_save': True,
            'show_processing_time': True,
            'show_confidence_score': True,
            'enable_animations': True,
            
            # Advanced settings
            'max_file_size_mb': 500,
            'enable_debug_mode': False,
            'cache_enabled': True,
            'log_level': 'INFO',
            'enable_metrics': True,
            'backup_enabled': True
        }
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file and session state."""
        # Start with default settings
        merged_settings = self.default_settings.copy()
        
        # Load from file
        file_settings = self.load_from_file()
        merged_settings.update(file_settings)
        
        # Load from session state (takes precedence)
        session_settings = self.load_from_session()
        merged_settings.update(session_settings)
        
        return merged_settings
    
    def load_from_file(self) -> Dict[str, Any]:
        """Load settings from JSON file."""
        try:
            if self.user_config_file.exists():
                with open(self.user_config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load settings from file: {e}")
        return {}
    
    def load_from_session(self) -> Dict[str, Any]:
        """Load settings from Streamlit session state."""
        if 'app_settings' in st.session_state:
            return st.session_state.app_settings
        return {}
    
    def save_to_file(self, settings_dict: Dict[str, Any]) -> bool:
        """Save settings to JSON file."""
        try:
            self.config_dir.mkdir(exist_ok=True)
            
            with open(self.user_config_file, 'w', encoding='utf-8') as f:
                json.dump(settings_dict, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving settings to file: {e}")
            return False
    
    def save_to_session(self, settings_dict: Dict[str, Any]):
        """Save settings to Streamlit session state."""
        if 'app_settings' not in st.session_state:
            st.session_state.app_settings = {}
        
        st.session_state.app_settings.update(settings_dict)
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        settings = self.load_settings()
        return settings.get(key, default)
    
    def set_setting(self, key: str, value: Any, save_to_file: bool = False):
        """Set a specific setting value."""
        # Update session state
        if 'app_settings' not in st.session_state:
            st.session_state.app_settings = {}
        
        st.session_state.app_settings[key] = value
        
        # Optionally save to file
        if save_to_file:
            current_settings = self.load_from_file()
            current_settings[key] = value
            self.save_to_file(current_settings)
    
    def apply_settings_to_app(self):
        """Apply current settings to the application."""
        current_settings = self.load_settings()
        
        # Apply theme
        if current_settings.get('theme') == 'dark':
            # Note: Streamlit doesn't support dynamic theme switching
            # This would need to be handled in the main app configuration
            pass
        
        # Apply language
        language = current_settings.get('language', 'vi')
        # Set language for the application
        
        # Apply debug mode
        if current_settings.get('enable_debug_mode'):
            # Enable debug logging
            pass
        
        # Apply log level
        log_level = current_settings.get('log_level', 'INFO')
        # Set logging level
        
        return current_settings
    
    def validate_settings(self, settings_dict: Dict[str, Any]) -> Dict[str, str]:
        """Validate settings and return any errors."""
        errors = {}
        
        # Validate temperature
        temp = settings_dict.get('temperature', 0.7)
        if not 0.0 <= temp <= 2.0:
            errors['temperature'] = "Temperature must be between 0.0 and 2.0"
        
        # Validate max_tokens
        max_tokens = settings_dict.get('max_tokens', 2000)
        if not 100 <= max_tokens <= 8000:
            errors['max_tokens'] = "Max tokens must be between 100 and 8000"
        
        # Validate similarity_threshold
        sim_threshold = settings_dict.get('similarity_threshold', 0.25)
        if not 0.0 <= sim_threshold <= 1.0:
            errors['similarity_threshold'] = "Similarity threshold must be between 0.0 and 1.0"
        
        # Validate max_results
        max_results = settings_dict.get('max_results', 10)
        if not 1 <= max_results <= 50:
            errors['max_results'] = "Max results must be between 1 and 50"
        
        # Validate chunk_size
        chunk_size = settings_dict.get('chunk_size', 1000)
        if not 100 <= chunk_size <= 2000:
            errors['chunk_size'] = "Chunk size must be between 100 and 2000"
        
        # Validate max_file_size_mb
        max_file_size = settings_dict.get('max_file_size_mb', 500)
        if not 10 <= max_file_size <= 1000:
            errors['max_file_size_mb'] = "Max file size must be between 10 and 1000 MB"
        
        return errors
    
    def reset_to_defaults(self):
        """Reset all settings to default values."""
        # Clear session state
        if 'app_settings' in st.session_state:
            del st.session_state.app_settings
        
        # Remove user config file
        if self.user_config_file.exists():
            self.user_config_file.unlink()
    
    def export_settings(self) -> str:
        """Export current settings as JSON string."""
        settings = self.load_settings()
        return json.dumps(settings, indent=2, ensure_ascii=False)
    
    def import_settings(self, settings_json: str) -> bool:
        """Import settings from JSON string."""
        try:
            settings_dict = json.loads(settings_json)
            
            # Validate imported settings
            errors = self.validate_settings(settings_dict)
            if errors:
                print(f"Validation errors: {errors}")
                return False
            
            # Save to file
            return self.save_to_file(settings_dict)
        except Exception as e:
            print(f"Error importing settings: {e}")
            return False
    
    def get_settings_summary(self) -> Dict[str, Any]:
        """Get a summary of current settings for display."""
        settings = self.load_settings()
        
        return {
            'model': {
                'temperature': settings.get('temperature', 0.7),
                'max_tokens': settings.get('max_tokens', 2000),
                'model_name': settings.get('model_name', 'gpt-4')
            },
            'search': {
                'similarity_threshold': settings.get('similarity_threshold', 0.25),
                'max_results': settings.get('max_results', 10),
                'enable_web_search': settings.get('enable_web_search', False),
                'enable_function_calling': settings.get('enable_function_calling', False)
            },
            'audio': {
                'enable_tts': settings.get('enable_tts', True),
                'tts_voice': settings.get('tts_voice', 'alloy')
            },
            'memory': {
                'enable_memory': settings.get('enable_memory', True),
                'max_memory_context': settings.get('max_memory_context', 3)
            },
            'interface': {
                'theme': settings.get('theme', 'light'),
                'language': settings.get('language', 'vi')
            }
        }


# Global settings manager instance
settings_manager = SettingsManager()


def get_settings() -> Dict[str, Any]:
    """Get current application settings."""
    return settings_manager.load_settings()


def get_setting(key: str, default: Any = None) -> Any:
    """Get a specific setting value."""
    return settings_manager.get_setting(key, default)


def set_setting(key: str, value: Any, save_to_file: bool = False):
    """Set a specific setting value."""
    settings_manager.set_setting(key, value, save_to_file)


def apply_settings():
    """Apply current settings to the application."""
    return settings_manager.apply_settings_to_app()
