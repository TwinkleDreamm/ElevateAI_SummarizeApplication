"""
Embedding generation module for ElevateAI.
Handles text embedding generation using various models.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pickle

try:
    # from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Sentence Transformers not available: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
except Exception as e:
    print(f"Warning: Sentence Transformers import failed: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
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
        self.embedding_model = self.config.get('model_name', settings.openai_embedding_model)
        self.use_openai = self.config.get('use_openai', True)
        
        # Fix: Use correct base URL for third-party providers
        # For third-party providers, use azure_openai_endpoint
        # For OpenAI, use openai_base_url
        if (hasattr(self.settings, 'azure_openai_endpoint') and 
            self.settings.azure_openai_endpoint and
            "api.openai.com" not in self.settings.azure_openai_endpoint and
            "openai.azure.com" not in self.settings.azure_openai_endpoint):
            # Third-party provider detected
            self.openai_base_url = self.settings.azure_openai_endpoint
            self.logger.info(f"[EMBEDDING][CONFIG] Detected third-party provider: {self.openai_base_url}")
        else:
            # Use standard OpenAI
            self.openai_base_url =self.settings.openai_base_url
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
                self.embedding_model = self.settings.openai_embedding_model
                if not self.embedding_model:
                    self.embedding_model = 'text-embedding-ada-002'  # Default fallback
                
                self.logger.info(f"[EMBEDDING][ONLINE][LOAD] Initializing OpenAI embedding API model: {self.embedding_model}, base_url: {self.openai_base_url}, key: {key_masked}")
                self.openai_client = OpenAI(api_key=self.settings.openai_api_key, base_url=self.openai_base_url)
                self.logger.info(f"[EMBEDDING][ONLINE][SUCCESS] Ready to use online model: {self.embedding_model}, base_url: {self.openai_base_url}, key: {key_masked}")
                self.online_available = True
            except Exception as e:
                self.logger.warning(f"[EMBEDDING][ONLINE][FAIL] Could not initialize OpenAI embedding API ({self.embedding_model}), base_url: {self.openai_base_url}. Falling back to local. Reason: {e}")
        
        # Fallback to local model if online not available
        if not self.online_available and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.logger.info(f"[EMBEDDING][LOCAL][LOAD] Loading local model: {self.embedding_model}")
                self.sentence_transformer = SentenceTransformer(self.embedding_model)
                self.logger.info(f"[EMBEDDING][LOCAL][SUCCESS] Ready to use local model: {self.embedding_model}")
            except Exception as e:
                self.logger.error(f"[EMBEDDING][LOCAL][FAIL] Could not load local model ({self.embedding_model}): {e}")
                raise EmbeddingError(f"Failed to load embedding model: {e}")
        elif not self.online_available:
            self.logger.error("[EMBEDDING][FAIL] No embedding model available. Requires OPENAI_API_KEY or install sentence-transformers.")
            raise EmbeddingError("No embedding libraries available. Please install sentence-transformers or configure OpenAI.")
    
    def generate_embedding(self, text: str, **kwargs) -> np.ndarray:
        """
        Generate embedding for a single text with fallback mechanism.
        
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
        
        # Try OpenAI first if available
        if self.online_available and self.openai_client:
            try:
                return self._generate_openai_embedding(text, **kwargs)
            except Exception as e:
                self.logger.warning(f"OpenAI embedding failed, falling back to local model: {e}")
                # Continue to fallback
        
        # Fallback to local model
        if self.sentence_transformer:
            try:
                return self._generate_sentence_transformer_embedding(text, **kwargs)
            except Exception as e:
                self.logger.error(f"Local embedding also failed: {e}")
                raise EmbeddingError(f"All embedding methods failed: {e}")
        else:
            raise EmbeddingError("No embedding model available")
    
    def generate_embeddings_batch(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batch with fallback mechanism.
        
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
        
        # Try OpenAI first if available
        if self.online_available and self.openai_client:
            try:
                return self._generate_openai_embeddings_batch(texts, **kwargs)
            except Exception as e:
                self.logger.warning(f"OpenAI batch embedding failed, falling back to local model: {e}")
                # Continue to fallback
        
        # Fallback to local model
        if self.sentence_transformer:
            try:
                return self._generate_sentence_transformer_embeddings_batch(texts, **kwargs)
            except Exception as e:
                self.logger.error(f"Local batch embedding also failed: {e}")
                raise EmbeddingError(f"All batch embedding methods failed: {e}")
        else:
            raise EmbeddingError("No embedding model available")
    
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
        """Generate embedding using OpenAI API with retry mechanism."""
        import time
        import random
        
        max_retries = kwargs.get('max_retries', 3)
        retry_delay = kwargs.get('retry_delay', 1.0)
        
        for attempt in range(max_retries):
            try:
                # Fix: Use correct model name from settings
                model_name = kwargs.get('model', self.embedding_model or 'text-embedding-ada-002')
                
                # Validate input text
                if not text or not text.strip():
                    raise EmbeddingError("Empty or whitespace-only text provided for embedding")
                
                self.logger.info(f"[EMBEDDING][ONLINE] Calling OpenAI embedding API (attempt {attempt + 1}/{max_retries}, model: {model_name}, base_url: {self.openai_base_url})")
                start_time = time.time()
                response = self.openai_client.embeddings.create(
                    model=model_name,
                    input=text
                )
                elapsed = time.time() - start_time
                self.logger.info(f"[EMBEDDING][ONLINE] Received result (elapsed: {elapsed:.2f}s)")
                
                # Validate response
                if not response or not response.data:
                    raise EmbeddingError("No embedding data received from OpenAI API")
                
                if len(response.data) == 0:
                    raise EmbeddingError("Empty embedding data received from OpenAI API")
                
                embedding_data = response.data[0]
                if not embedding_data or not embedding_data.embedding:
                    raise EmbeddingError("No embedding vector in response data")
                
                embedding = np.array(embedding_data.embedding)
                
                # Validate embedding vector
                if embedding.size == 0:
                    raise EmbeddingError("Empty embedding vector received")
                
                if kwargs.get('normalize', True):
                    embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
                
            except Exception as e:
                self.logger.warning(f"[EMBEDDING][ONLINE][RETRY] Attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Add jitter to avoid thundering herd
                    delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"[EMBEDDING][ONLINE][RETRY] Waiting {delay:.2f}s before retry...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"[EMBEDDING][ONLINE][ERROR] All {max_retries} attempts failed. Last error: {e}")
                    raise EmbeddingError(f"OpenAI embedding failed after {max_retries} attempts: {e}")
    
    def _generate_openai_embeddings_batch(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API in batch with retry mechanism."""
        import time
        import random
        
        max_retries = kwargs.get('max_retries', 3)
        retry_delay = kwargs.get('retry_delay', 1.0)
        
        for attempt in range(max_retries):
            try:
                # Fix: Use correct model name from settings
                model_name = kwargs.get('model', self.embedding_model or 'text-embedding-ada-002')
                
                # Validate input texts
                if not texts:
                    raise EmbeddingError("Empty text list provided for batch embedding")
                
                # Filter out empty texts
                valid_texts = [text for text in texts if text and text.strip()]
                if not valid_texts:
                    raise EmbeddingError("No valid texts provided for batch embedding")
                
                self.logger.info(f"[EMBEDDING][ONLINE] Calling OpenAI embedding API batch (attempt {attempt + 1}/{max_retries}, model: {model_name}, base_url: {self.openai_base_url}, batch_size: {len(valid_texts)})")
                start_time = time.time()
                response = self.openai_client.embeddings.create(
                    model=model_name,
                    input=valid_texts
                )
                elapsed = time.time() - start_time
                self.logger.info(f"[EMBEDDING][ONLINE] Received batch result (elapsed: {elapsed:.2f}s)")
                
                # Validate response
                if not response or not response.data:
                    raise EmbeddingError("No embedding data received from OpenAI API batch")
                
                if len(response.data) != len(valid_texts):
                    raise EmbeddingError(f"Mismatch in response data length: expected {len(valid_texts)}, got {len(response.data)}")
                
                embeddings = []
                for i, data in enumerate(response.data):
                    if not data or not data.embedding:
                        raise EmbeddingError(f"No embedding vector in response data at index {i}")
                    
                    embedding = np.array(data.embedding)
                    if embedding.size == 0:
                        raise EmbeddingError(f"Empty embedding vector received at index {i}")
                    
                    if kwargs.get('normalize', True):
                        embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
                
                return embeddings
                
            except Exception as e:
                self.logger.warning(f"[EMBEDDING][ONLINE][RETRY] Batch attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Add jitter to avoid thundering herd
                    delay = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"[EMBEDDING][ONLINE][RETRY] Waiting {delay:.2f}s before retry...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"[EMBEDDING][ONLINE][ERROR] All {max_retries} batch attempts failed. Last error: {e}")
                    raise EmbeddingError(f"OpenAI batch embedding failed after {max_retries} attempts: {e}")
    
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
            return model_dims.get(self.config.get('model', self.embedding_model), 1536)
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
                'model_name': self.embedding_model,
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
            if cache_data.get('model_name') == self.embedding_model:
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
            'use_openai': self.use_openai,
            'embedding_model': self.embedding_model,
            'embedding_dimension': self.get_embedding_dimension(),
            'sentence_transformer_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'openai_available': OPENAI_AVAILABLE and bool(settings.openai_api_key)
        }
