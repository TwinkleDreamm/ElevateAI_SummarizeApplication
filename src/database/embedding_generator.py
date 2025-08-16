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
        
        # Model configuration - Prioritize OpenAI API for better performance
        self.use_openai = self.config.get('use_openai', True)  # Default to True for online models
        self.openai_model_name = self.config.get('openai_model_name', 'text-embedding-ada-002')
        self.local_model_name = self.config.get('local_model_name', 'all-MiniLM-L6-v2')
        
        # Initialize models
        self.sentence_transformer = None
        self.openai_client = None
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load embedding models - prioritize online APIs."""
        # Prefer OpenAI if API key is configured
        if OPENAI_AVAILABLE and settings.openai_api_key:
            try:
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
                self.use_openai = True
                self.logger.info("OpenAI embedding client initialized (online)")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None

        # Fallback to local sentence-transformers if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.use_openai = False
                self.logger.info(f"Using local sentence transformer model: {self.local_model_name}")
                self.sentence_transformer = SentenceTransformer(self.local_model_name)
                return
            except Exception as e:
                self.logger.error(f"Failed to load sentence transformer: {e}")
                self.sentence_transformer = None

        # If we reach here, no embedding service is available
        self.logger.warning("No embedding service available (no OpenAI key and no local model)")
        raise EmbeddingError("No embedding service available. Configure OpenAI API key or install sentence-transformers.")
    
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
            if self.use_openai and self.openai_client:
                return self._generate_openai_embedding(text, **kwargs)
            elif self.sentence_transformer:
                return self._generate_sentence_transformer_embedding(text, **kwargs)
            else:
                raise EmbeddingError("No embedding model available")
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")
    
    def generate_embeddings_batch(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of input texts
            **kwargs: Additional parameters
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If batch embedding generation fails
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            raise EmbeddingError("No valid texts provided for embedding")
        
        try:
            if self.use_openai and self.openai_client:
                return self._generate_openai_embeddings_batch(valid_texts, **kwargs)
            elif self.sentence_transformer:
                return self._generate_sentence_transformer_embeddings_batch(valid_texts, **kwargs)
            else:
                raise EmbeddingError("No embedding model available")
                
        except Exception as e:
            self.logger.error(f"Batch embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}")
    
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
        try:
            response = self.openai_client.embeddings.create(
                model=kwargs.get('model', self.openai_model_name),
                input=text
            )
            embedding = np.array(response.data[0].embedding)
            
            if kwargs.get('normalize', True):
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}")
    
    def _generate_openai_embeddings_batch(self, texts: List[str], **kwargs) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API in batch."""
        try:
            # OpenAI API supports batch processing
            response = self.openai_client.embeddings.create(
                model=kwargs.get('model', self.openai_model_name),
                input=texts
            )
            
            embeddings = []
            for data in response.data:
                embedding = np.array(data.embedding)
                if kwargs.get('normalize', True):
                    embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
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
