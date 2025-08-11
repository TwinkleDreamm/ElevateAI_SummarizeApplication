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

        # OpenAI Configuration (fallback)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

        # Google Search API
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")

        # SerpAPI
        self.serpapi_api_key = os.getenv("SERPAPI_API_KEY")

        # File Processing Settings
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
        self.supported_video_formats = ["mp4", "avi", "mov", "mkv", "webm"]
        self.supported_audio_formats = ["mp3", "wav", "m4a", "flac", "ogg"]
        self.supported_document_formats = ["pdf", "docx", "txt"]

        # Vector Database Settings
        self.vector_db_path = os.getenv("VECTOR_DB_PATH", "./data/vectordb")
        # Local embedding model name (used only when not using OpenAI)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.max_chunks_per_query = int(os.getenv("MAX_CHUNKS_PER_QUERY", "10"))

        # Search Settings
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        self.max_web_search_results = int(os.getenv("MAX_WEB_SEARCH_RESULTS", "5"))
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "5"))

        # Audio Processing Settings
        self.audio_sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
        self.noise_reduction_strength = float(os.getenv("NOISE_REDUCTION_STRENGTH", "0.8"))
        self.vocal_separation_model = os.getenv("VOCAL_SEPARATION_MODEL", "spleeter:2stems-16kHz")

        # Language Settings
        self.default_language = os.getenv("DEFAULT_LANGUAGE", "vi")
        self.supported_languages = ["vi", "en", "zh", "ja", "ko"]
        
        # Performance Settings
        self.disable_nltk_downloads = os.getenv("DISABLE_NLTK_DOWNLOADS", "True").lower() == "true"

        # .env already loaded above

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
