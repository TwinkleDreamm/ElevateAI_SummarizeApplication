"""
Vector database module for ElevateAI using FAISS.
Handles vector storage, indexing, and similarity search.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import pickle
import json
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import VectorDatabaseError
from .embedding_generator import EmbeddingGenerator
from .metadata_manager import MetadataManager


class VectorDatabase:
    """FAISS-based vector database for semantic search."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the vector database.
        
        Args:
            config: Optional configuration dictionary
        """
        if not FAISS_AVAILABLE:
            raise VectorDatabaseError("FAISS not available. Please install faiss-cpu or faiss-gpu.")
        
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Database configuration
        self.db_path = Path(self.config.get('db_path', settings.vector_db_path))
        self.embedding_dim = self.config.get('embedding_dim', 384)
        self.index_type = self.config.get('index_type', 'flat')  # 'flat', 'ivf', 'hnsw'
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(config)
        self.metadata_manager = MetadataManager(config)
        
        # FAISS index
        self.index = None
        self.is_trained = False
        
        # Ensure database directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load database
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize or load existing database."""
        try:
            if self._database_exists():
                self.logger.info("Loading existing database...")
                try:
                    self.load_database()
                    self.logger.info("Database loaded successfully")
                except Exception as load_error:
                    self.logger.warning(f"Failed to load existing database: {load_error}")
                    self.logger.info("Creating new database as fallback")
                    self._create_new_database()
            else:
                self.logger.info("No existing database found, creating new one")
                self._create_new_database()
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise VectorDatabaseError(f"Failed to initialize database: {e}")
    
    def _database_exists(self) -> bool:
        """Check if database files exist."""
        index_file = self.db_path / "faiss_index.bin"
        metadata_file = self.db_path / "metadata.json"
        return index_file.exists() and metadata_file.exists()
    
    def _create_new_database(self) -> None:
        """Create a new FAISS database."""
        self.logger.info("Creating new vector database")
        
        # Get embedding dimension from the generator
        # If no embedding model is available, default to 1536 (OpenAI ada dims)
        try:
            if hasattr(self.embedding_generator, "has_model") and self.embedding_generator.has_model():
                self.embedding_dim = self.embedding_generator.get_embedding_dimension()
            else:
                # Default dimension matches OpenAI text-embedding-ada-002
                self.embedding_dim = 1536
        except Exception as e:
            self.logger.warning(f"Could not infer embedding dimension; defaulting to 1536. Reason: {e}")
            self.embedding_dim = 1536
        
        # Create FAISS index based on type
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        elif self.index_type == 'ivf':
            # IVF index for larger datasets
            nlist = self.config.get('nlist', 100)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif self.index_type == 'hnsw':
            # HNSW index for fast approximate search
            m = self.config.get('hnsw_m', 16)
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, m)
        else:
            raise VectorDatabaseError(f"Unsupported index type: {self.index_type}")
        
        self.is_trained = (self.index_type == 'flat' or self.index_type == 'hnsw')
        self.logger.info(f"Created {self.index_type} index with dimension {self.embedding_dim}")
    
    def add_to_vectordb(self, texts: List[str], metadata_list: List[Dict[str, Any]]) -> List[int]:
        """
        Add texts and their metadata to the vector database.
        
        Args:
            texts: List of texts to add
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of assigned IDs
            
        Raises:
            VectorDatabaseError: If addition fails
        """
        if not texts:
            return []
        
        if len(texts) != len(metadata_list):
            raise VectorDatabaseError("Number of texts and metadata entries must match")
        
        try:
            if hasattr(self.embedding_generator, "has_model") and not self.embedding_generator.has_model():
                raise VectorDatabaseError("No embedding model available. Configure OpenAI API key to enable embeddings.")
            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.embedding_generator.generate_embeddings_batch(texts)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings).astype('float32')

            # Check dimension mismatch between embeddings and index
            if embeddings_array.ndim != 2 or embeddings_array.shape[1] <= 0:
                raise VectorDatabaseError(f"Invalid embeddings shape: {embeddings_array.shape}")

            new_dim = embeddings_array.shape[1]
            if new_dim != self.embedding_dim:
                # If index empty, recreate with correct dim; else abort with clear error
                if self.index is None or getattr(self.index, 'ntotal', 0) == 0:
                    self.logger.warning(
                        f"Embedding dim {new_dim} != index dim {self.embedding_dim}. Recreating empty index with new dim.")
                    # Recreate index with the new dimension
                    self.embedding_dim = new_dim
                    if self.index_type == 'flat':
                        self.index = faiss.IndexFlatIP(self.embedding_dim)
                    elif self.index_type == 'ivf':
                        nlist = self.config.get('nlist', 100)
                        quantizer = faiss.IndexFlatIP(self.embedding_dim)
                        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                        self.is_trained = False
                    elif self.index_type == 'hnsw':
                        m = self.config.get('hnsw_m', 16)
                        self.index = faiss.IndexHNSWFlat(self.embedding_dim, m)
                    else:
                        raise VectorDatabaseError(f"Unsupported index type: {self.index_type}")
                else:
                    raise VectorDatabaseError(
                        f"Embedding dimension ({new_dim}) does not match existing index dimension ({self.embedding_dim}).")
            
            # Train index if needed
            if not self.is_trained and self.index_type == 'ivf':
                self.logger.info("Training IVF index")
                self.index.train(embeddings_array)
                self.is_trained = True
            
            # Get current size for ID assignment
            current_size = self.index.ntotal
            
            # Add embeddings to index
            self.index.add(embeddings_array)
            
            # Generate IDs
            ids = list(range(current_size, current_size + len(texts)))
            
            # Add metadata
            for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
                enhanced_metadata = {
                    **metadata,
                    'id': ids[i],
                    'text': text,
                    'added_at': datetime.now().isoformat(),
                    'embedding_model': self.embedding_generator.model_name
                }
                self.metadata_manager.add_metadata(ids[i], enhanced_metadata)
            
            self.logger.info(f"Added {len(texts)} vectors to database")
            
            # Fix: Auto-save database after adding data
            self.save_database()
            self.logger.info("Database auto-saved after adding new vectors")
            
            return ids
            
        except Exception as e:
            # Log rich error information
            try:
                err_type = type(e).__name__
                self.logger.error(f"Failed to add vectors to database: {err_type}: {e}")
            finally:
                pass
            raise VectorDatabaseError(f"Failed to add vectors: {type(e).__name__}: {e}")
    
    def search(self, query: str, k: int = 10, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the database.
        
        Args:
            query: Query text
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results with metadata
            
        Raises:
            VectorDatabaseError: If search fails
        """
        if self.index.ntotal == 0:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            
            # Search in index
            scores, indices = self.index.search(query_vector, k)
            
            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                if score >= threshold:
                    metadata = self.metadata_manager.get_metadata(int(idx))
                    if metadata:
                        result = {
                            'id': int(idx),
                            'score': float(score),
                            'metadata': metadata
                        }
                        results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise VectorDatabaseError(f"Search failed: {e}")
    
    def search_by_vector(self, query_vector: np.ndarray, k: int = 10, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search using a pre-computed vector.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of search results with metadata
        """
        if self.index.ntotal == 0:
            return []
        
        try:
            query_vector = query_vector.reshape(1, -1).astype('float32')
            scores, indices = self.index.search(query_vector, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                
                if score >= threshold:
                    metadata = self.metadata_manager.get_metadata(int(idx))
                    if metadata:
                        result = {
                            'id': int(idx),
                            'score': float(score),
                            'metadata': metadata
                        }
                        results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            raise VectorDatabaseError(f"Vector search failed: {e}")
    
    def delete_by_ids(self, ids: List[int]) -> None:
        """
        Delete vectors by IDs (FAISS doesn't support deletion, so we mark as deleted).
        
        Args:
            ids: List of IDs to delete
        """
        for id in ids:
            self.metadata_manager.delete_metadata(id)
        
        self.logger.info(f"Marked {len(ids)} vectors as deleted")
    
    def save_database(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save the database to disk.
        
        Args:
            path: Optional custom path to save to
        """
        save_path = Path(path) if path else self.db_path
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            index_file = save_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_file))
            
            # Save metadata
            self.metadata_manager.save_metadata(save_path / "metadata.json")
            
            # Save database info
            db_info = {
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'is_trained': self.is_trained,
                'total_vectors': self.index.ntotal,
                'created_at': datetime.now().isoformat(),
                'embedding_model': self.embedding_generator.model_name
            }
            
            with open(save_path / "db_info.json", 'w') as f:
                json.dump(db_info, f, indent=2)
            
            self.logger.info(f"Database saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save database: {e}")
            raise VectorDatabaseError(f"Failed to save database: {e}")
    
    def load_database(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Load database from disk.
        
        Args:
            path: Optional custom path to load from
        """
        load_path = Path(path) if path else self.db_path
        
        try:
            # Load FAISS index
            index_file = load_path / "faiss_index.bin"
            if not index_file.exists():
                raise VectorDatabaseError(f"Index file not found: {index_file}")
            
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            metadata_file = load_path / "metadata.json"
            if metadata_file.exists():
                self.metadata_manager.load_metadata(metadata_file)
            
            # Load database info
            db_info_file = load_path / "db_info.json"
            if db_info_file.exists():
                with open(db_info_file, 'r') as f:
                    db_info = json.load(f)
                
                self.embedding_dim = db_info.get('embedding_dim', self.embedding_dim)
                self.index_type = db_info.get('index_type', self.index_type)
                self.is_trained = db_info.get('is_trained', True)
            
            self.logger.info(f"Database loaded from {load_path}")
            self.logger.info(f"Total vectors: {self.index.ntotal}")
            
        except Exception as e:
            self.logger.error(f"Failed to load database: {e}")
            raise VectorDatabaseError(f"Failed to load database: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embedding_dim,
            'index_type': self.index_type,
            'is_trained': self.is_trained,
            'metadata_count': self.metadata_manager.get_metadata_count(),
            'database_path': str(self.db_path)
        }
    
    def clear_database(self) -> None:
        """Clear all data from the database."""
        self._create_new_database()
        self.metadata_manager.clear_metadata()
        self.logger.info("Database cleared")
