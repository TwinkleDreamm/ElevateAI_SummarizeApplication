"""
Configuration settings for ElevateAI application.
"""
import os
from pathlib import Path
from typing import List, Optional


class Settings:
    """Application settings with environment variable support."""

    def __init__(self):
        # Load environment variables from .env file before reading them
        self._load_env_file()

        # Application Settings
        self.app_name = os.getenv("APP_NAME", "ElevateAI")
        self.app_version = os.getenv("APP_VERSION", "1.0.0")
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.use_openai = os.getenv("USE_OPENAI", "True").lower() == "true"

        # Azure OpenAI Configuration
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

        # OpenAI Configuration (and compatible providers)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
        self.openai_chat_model = os.getenv("OPENAI_CHAT_MODEL")
        
        # OpenAI Model Parameters
        self.openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        self.openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        self.openai_top_p = float(os.getenv("OPENAI_TOP_P", "0.9"))
        self.openai_frequency_penalty = float(os.getenv("OPENAI_FREQUENCY_PENALTY", "0.0"))
        self.openai_presence_penalty = float(os.getenv("OPENAI_PRESENCE_PENALTY", "0.0"))

        # Google Search API
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")

        # SerpAPI
        self.serpapi_api_key = os.getenv("SERPAPI_API_KEY")

        # File Processing Settings
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
        self.supported_video_formats = ["mp4", "avi", "mov", "mkv", "webm"]
        self.supported_audio_formats = ["mp3", "wav", "m4a", "flac", "ogg"]
        self.supported_document_formats = ["pdf", "docx", "txt", "xlsx"]

        # Document extraction performance/caching
        self.enable_extract_cache = os.getenv("ENABLE_EXTRACT_CACHE", "True").lower() == "true"
        # Consider PDFs larger than this size (MB) as large; prefer PyMuPDF when available
        self.pdf_large_file_size_mb = int(os.getenv("PDF_LARGE_FILE_SIZE_MB", "15"))

        # Vector Database Settings
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vectordb")
        # Local embedding model name (used only when not using OpenAI)
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.max_chunks_per_query = int(os.getenv("MAX_CHUNKS_PER_QUERY", "10"))

        # Search Settings
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        self.max_web_search_results = int(os.getenv("MAX_WEB_SEARCH_RESULTS", "5"))
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "5"))
        self.max_results = int(os.getenv("MAX_RESULTS", "10"))
        self.enable_web_search = os.getenv("ENABLE_WEB_SEARCH", "False").lower() == "true"
        self.enable_function_calling = os.getenv("ENABLE_FUNCTION_CALLING", "False").lower() == "true"

        # Audio Processing Settings
        self.audio_sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
        self.noise_reduction_strength = float(os.getenv("NOISE_REDUCTION_STRENGTH", "0.8"))
        self.vocal_separation_model = os.getenv("VOCAL_SEPARATION_MODEL", "spleeter:2stems-16kHz")
        self.enable_tts = os.getenv("ENABLE_TTS", "True").lower() == "true"
        self.tts_voice = os.getenv("TTS_VOICE", "alloy")
        self.enable_vocal_separation = os.getenv("ENABLE_VOCAL_SEPARATION", "False").lower() == "true"

        # Language Settings
        self.default_language = os.getenv("DEFAULT_LANGUAGE", "vi")
        self.supported_languages = ["vi", "en", "zh", "ja", "ko"]
        
        # Interface Settings
        self.theme = os.getenv("THEME", "light")
        self.auto_save = os.getenv("AUTO_SAVE", "True").lower() == "true"
        self.show_processing_time = os.getenv("SHOW_PROCESSING_TIME", "True").lower() == "true"
        self.show_confidence_score = os.getenv("SHOW_CONFIDENCE_SCORE", "True").lower() == "true"
        self.enable_animations = os.getenv("ENABLE_ANIMATIONS", "True").lower() == "true"
        
        # Performance Settings
        self.disable_nltk_downloads = os.getenv("DISABLE_NLTK_DOWNLOADS", "True").lower() == "true"
        self.cache_enabled = os.getenv("CACHE_ENABLED", "True").lower() == "true"
        self.enable_metrics = os.getenv("ENABLE_METRICS", "True").lower() == "true"
        self.backup_enabled = os.getenv("BACKUP_ENABLED", "True").lower() == "true"

    def _load_env_file(self):
        """Load environment variables from .env file."""
        try:
            from dotenv import load_dotenv
            env_file = Path(__file__).parent.parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
        except ImportError:
            pass  # dotenv not available, skip

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        """Get models directory."""
        return self.data_dir / "models"

    @property
    def temp_dir(self) -> Path:
        """Get temporary files directory."""
        return self.data_dir / "temp"

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.models_dir,
            self.temp_dir,
            Path(self.vector_db_path).parent
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
