"""
Embedding generation module for ElevateAI.
Handles text embedding generation using various models.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pickle

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Sentence Transformers not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
except Exception as e:
    print(f"Warning: Sentence Transformers import failed: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import EmbeddingError


class EmbeddingGenerator:
    """Handles text embedding generation using various models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedding generator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Model configuration
        self.model_name = self.config.get('model_name', settings.embedding_model)
        self.use_openai = self.config.get('use_openai', True)
        
        # Fix: Use correct base URL for third-party providers
        # For third-party providers, use azure_openai_endpoint
        # For OpenAI, use openai_api_base
        if (hasattr(self.settings, 'azure_openai_endpoint') and 
            self.settings.azure_openai_endpoint and
            "api.openai.com" not in self.settings.azure_openai_endpoint and
            "openai.azure.com" not in self.settings.azure_openai_endpoint):
            # Third-party provider detected
            self.openai_base_url = self.settings.azure_openai_endpoint
            self.logger.info(f"[EMBEDDING][CONFIG] Detected third-party provider: {self.openai_base_url}")
        else:
            # Use standard OpenAI
            self.openai_base_url = getattr(self.settings, 'openai_api_base', 'https://api.openai.com/v1')
            self.logger.info(f"[EMBEDDING][CONFIG] Using standard OpenAI: {self.openai_base_url}")

        # Initialize models
        self.sentence_transformer = None
        self.openai_client = None
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load embedding models with online-first, fallback to local."""
        self.online_available = False
        # Try OpenAI first
        if self.use_openai and OPENAI_AVAILABLE and self.settings.openai_api_key:
            try:
                key = self.settings.openai_api_key
                key_masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
                
                # Fix: Use correct model name for embeddings
                # For third-party providers, use the actual model name, not deployment name
                embedding_model = self.settings.azure_openai_embedding_deployment
                if not embedding_model:
                    embedding_model = 'text-embedding-ada-002'  # Default fallback
                
                self.logger.info(f"[EMBEDDING][ONLINE][LOAD] Initializing OpenAI embedding API model: {embedding_model}, base_url: {self.openai_base_url}, key: {key_masked}")
                self.openai_client = openai.OpenAI(api_key=self.settings.openai_api_key, base_url=self.openai_base_url)
                self.logger.info(f"[EMBEDDING][ONLINE][SUCCESS] Ready to use online model: {embedding_model}, base_url: {self.openai_base_url}, key: {key_masked}")
                self.online_available = True
            except Exception as e:
                self.logger.warning(f"[EMBEDDING][ONLINE][FAIL] Could not initialize OpenAI embedding API ({embedding_model}), base_url: {self.openai_base_url}. Falling back to local. Reason: {e}")
        
        # Fallback to local model if online not available
        if not self.online_available and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.logger.info(f"[EMBEDDING][LOCAL][LOAD] Loading local model: {self.model_name}")
                self.sentence_transformer = SentenceTransformer(self.model_name)
                self.logger.info(f"[EMBEDDING][LOCAL][SUCCESS] Ready to use local model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"[EMBEDDING][LOCAL][FAIL] Could not load local model ({self.model_name}): {e}")
                raise EmbeddingError(f"Failed to load embedding model: {e}")
        elif not self.online_available:
            self.logger.error("[EMBEDDING][FAIL] No embedding model available. Requires OPENAI_API_KEY or install sentence-transformers.")
            raise EmbeddingError("No embedding libraries available. Please install sentence-transformers or configure OpenAI.")
    
    def generate_embedding(self, text: str, **kwargs) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            **kwargs: Additional parameters
            
        Returns:
            Embedding vector as numpy array
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise EmbeddingError("Empty text provided for embedding")
        
        try:
            # Fix: Prioritize online models when available
            if self.online_available and self.openai_client:
                return self._generate_openai_embedding(text, **kwargs)
            elif self.sentence_transformer:
                return self._generate_sentence_transformer_embedding(text, **kwargs)
            else:
                raise EmbeddingError("No embedding model available")
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Embedding generation failed: {e}")
    
    def generate_embeddings_batch(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of input texts
            **kwargs: Additional parameters
            
        Returns:
            List of embedding vectors as numpy arrays
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            raise EmbeddingError("No texts provided for batch embedding")
        
        try:
            # Fix: Prioritize online models when available
            if self.online_available and self.openai_client:
                return self._generate_openai_embeddings_batch(texts, **kwargs)
            elif self.sentence_transformer:
                return self._generate_sentence_transformer_embeddings_batch(texts, **kwargs)
            else:
                raise EmbeddingError("No embedding model available")
                
        except Exception as e:
            self.logger.error(f"Batch embedding generation failed: {e}")
            raise EmbeddingError(f"Batch embedding generation failed: {e}")
    
    def _generate_sentence_transformer_embedding(self, text: str, **kwargs) -> np.ndarray:
        """Generate embedding using sentence transformer."""
        try:
            embedding = self.sentence_transformer.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=kwargs.get('normalize', True)
            )
            return embedding
        except Exception as e:
            raise EmbeddingError(f"Sentence transformer embedding failed: {e}")
    
    def _generate_sentence_transformer_embeddings_batch(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """Generate embeddings using sentence transformer in batch."""
        try:
            embeddings = self.sentence_transformer.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=kwargs.get('normalize', True),
                batch_size=kwargs.get('batch_size', 32),
                show_progress_bar=kwargs.get('show_progress', False)
            )
            return [embedding for embedding in embeddings]
        except Exception as e:
            raise EmbeddingError(f"Sentence transformer batch embedding failed: {e}")
    
    def _generate_openai_embedding(self, text: str, **kwargs) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        import time
        try:
            # Fix: Use correct model name from settings
            model_name = kwargs.get('model', self.settings.azure_openai_embedding_deployment or 'text-embedding-ada-002')
            
            self.logger.info(f"[EMBEDDING][ONLINE] Calling OpenAI embedding API (model: {model_name}, base_url: {self.openai_base_url})")
            start_time = time.time()
            response = self.openai_client.embeddings.create(
                model=model_name,
                input=text
            )
            elapsed = time.time() - start_time
            self.logger.info(f"[EMBEDDING][ONLINE] Received result (elapsed: {elapsed:.2f}s)")
            embedding = np.array(response.data[0].embedding)
            
            if kwargs.get('normalize', True):
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            self.logger.error(f"[EMBEDDING][ONLINE][ERROR] Error calling OpenAI embedding API: {e}")
            raise EmbeddingError(f"OpenAI embedding failed: {e}")
    
    def _generate_openai_embeddings_batch(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API in batch."""
        import time
        try:
            # Fix: Use correct model name from settings
            model_name = kwargs.get('model', self.settings.azure_openai_embedding_deployment or 'text-embedding-ada-002')
            
            self.logger.info(f"[EMBEDDING][ONLINE] Calling OpenAI embedding API batch (model: {model_name}, base_url: {self.openai_base_url}, batch_size: {len(texts)})")
            start_time = time.time()
            response = self.openai_client.embeddings.create(
                model=model_name,
                input=texts
            )
            elapsed = time.time() - start_time
            self.logger.info(f"[EMBEDDING][ONLINE] Received batch result (elapsed: {elapsed:.2f}s)")
            
            embeddings = []
            for data in response.data:
                embedding = np.array(data.embedding)
                if kwargs.get('normalize', True):
                    embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            self.logger.error(f"[EMBEDDING][ONLINE][ERROR] Error calling OpenAI embedding API batch: {e}")
            raise EmbeddingError(f"OpenAI batch embedding failed: {e}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the current model.
        
        Returns:
            Embedding dimension
        """
        if self.sentence_transformer:
            return self.sentence_transformer.get_sentence_embedding_dimension()
        elif self.use_openai:
            # Default dimensions for OpenAI models
            model_dims = {
                'text-embedding-ada-002': 1536,
                'text-embedding-3-small': 1536,
                'text-embedding-3-large': 3072
            }
            return model_dims.get(self.config.get('model', 'text-embedding-ada-002'), 1536)
        else:
            return 384  # Default dimension
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Ensure embeddings are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            self.logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    def save_model_cache(self, cache_path: Union[str, Path]) -> None:
        """
        Save model cache to disk.
        
        Args:
            cache_path: Path to save cache
        """
        try:
            cache_data = {
                'model_name': self.model_name,
                'use_openai': self.use_openai,
                'embedding_dimension': self.get_embedding_dimension()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.logger.info(f"Model cache saved to {cache_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save model cache: {e}")
    
    def load_model_cache(self, cache_path: Union[str, Path]) -> bool:
        """
        Load model cache from disk.
        
        Args:
            cache_path: Path to load cache from
            
        Returns:
            True if cache loaded successfully
        """
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache compatibility
            if cache_data.get('model_name') == self.model_name:
                self.logger.info(f"Model cache loaded from {cache_path}")
                return True
            else:
                self.logger.warning("Model cache incompatible with current configuration")
                return False
        except Exception as e:
            self.logger.warning(f"Failed to load model cache: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'use_openai': self.use_openai,
            'embedding_dimension': self.get_embedding_dimension(),
            'sentence_transformer_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'openai_available': OPENAI_AVAILABLE and bool(settings.openai_api_key)
        }
