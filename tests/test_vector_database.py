"""
Tests for vector database functionality.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import tempfile
import shutil
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.embedding_generator import EmbeddingGenerator
from src.database.metadata_manager import MetadataManager
from src.utils.exceptions import EmbeddingError, VectorDatabaseError


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the sentence transformers to avoid loading actual models
        with patch('src.database.embedding_generator.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('src.database.embedding_generator.SentenceTransformer') as mock_st:
                mock_model = Mock()
                mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
                mock_model.get_sentence_embedding_dimension.return_value = 4
                mock_st.return_value = mock_model
                
                self.generator = EmbeddingGenerator({'model_name': 'test-model'})
    
    def test_generate_embedding_single(self):
        """Test generating embedding for single text."""
        text = "This is a test sentence."
        
        with patch.object(self.generator, '_generate_sentence_transformer_embedding') as mock_gen:
            mock_gen.return_value = np.array([0.1, 0.2, 0.3, 0.4])
            
            embedding = self.generator.generate_embedding(text)
            
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 4
            mock_gen.assert_called_once_with(text)
    
    def test_generate_embedding_empty_text(self):
        """Test generating embedding for empty text."""
        with pytest.raises(EmbeddingError):
            self.generator.generate_embedding("")
    
    def test_generate_embeddings_batch(self):
        """Test generating embeddings for multiple texts."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        
        with patch.object(self.generator, '_generate_sentence_transformer_embeddings_batch') as mock_gen:
            mock_gen.return_value = [
                np.array([0.1, 0.2, 0.3, 0.4]),
                np.array([0.2, 0.3, 0.4, 0.5]),
                np.array([0.3, 0.4, 0.5, 0.6])
            ]
            
            embeddings = self.generator.generate_embeddings_batch(texts)
            
            assert len(embeddings) == 3
            assert all(isinstance(emb, np.ndarray) for emb in embeddings)
            mock_gen.assert_called_once_with(texts)
    
    def test_calculate_similarity(self):
        """Test similarity calculation between embeddings."""
        emb1 = np.array([1.0, 0.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0, 0.0])
        emb3 = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Test orthogonal vectors (should be 0 similarity)
        sim1 = self.generator.calculate_similarity(emb1, emb2)
        assert abs(sim1) < 0.1
        
        # Test identical vectors (should be 1 similarity)
        sim2 = self.generator.calculate_similarity(emb1, emb3)
        assert abs(sim2 - 1.0) < 0.1
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        dim = self.generator.get_embedding_dimension()
        assert isinstance(dim, int)
        assert dim > 0


class TestMetadataManager:
    """Test cases for MetadataManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = MetadataManager()
    
    def test_add_and_get_metadata(self):
        """Test adding and retrieving metadata."""
        vector_id = 1
        metadata = {
            'source': 'test_document.pdf',
            'content_type': 'document',
            'text': 'This is test content.'
        }
        
        self.manager.add_metadata(vector_id, metadata)
        retrieved = self.manager.get_metadata(vector_id)
        
        assert retrieved is not None
        assert retrieved['source'] == 'test_document.pdf'
        assert retrieved['content_type'] == 'document'
        assert retrieved['vector_id'] == vector_id
        assert 'added_at' in retrieved
    
    def test_update_metadata(self):
        """Test updating metadata."""
        vector_id = 1
        metadata = {'source': 'original.pdf', 'content_type': 'document'}
        
        self.manager.add_metadata(vector_id, metadata)
        
        updates = {'source': 'updated.pdf', 'new_field': 'new_value'}
        success = self.manager.update_metadata(vector_id, updates)
        
        assert success
        
        retrieved = self.manager.get_metadata(vector_id)
        assert retrieved['source'] == 'updated.pdf'
        assert retrieved['new_field'] == 'new_value'
        assert 'updated_at' in retrieved
    
    def test_delete_metadata(self):
        """Test deleting metadata (soft delete)."""
        vector_id = 1
        metadata = {'source': 'test.pdf', 'content_type': 'document'}
        
        self.manager.add_metadata(vector_id, metadata)
        success = self.manager.delete_metadata(vector_id)
        
        assert success
        
        # Should return None for deleted metadata
        retrieved = self.manager.get_metadata(vector_id)
        assert retrieved is None
    
    def test_search_metadata(self):
        """Test searching metadata with filters."""
        # Add multiple metadata entries
        self.manager.add_metadata(1, {'source': 'doc1.pdf', 'content_type': 'document'})
        self.manager.add_metadata(2, {'source': 'video1.mp4', 'content_type': 'video'})
        self.manager.add_metadata(3, {'source': 'doc2.pdf', 'content_type': 'document'})
        
        # Search for documents
        results = self.manager.search_metadata({'content_type': 'document'})
        assert len(results) == 2
        
        # Search for specific source
        results = self.manager.search_metadata({'source': 'video1.mp4'})
        assert len(results) == 1
        assert results[0]['source'] == 'video1.mp4'
    
    def test_get_statistics(self):
        """Test getting metadata statistics."""
        # Add some metadata
        self.manager.add_metadata(1, {'content_type': 'document'})
        self.manager.add_metadata(2, {'content_type': 'video'})
        self.manager.add_metadata(3, {'content_type': 'document'})
        
        # Delete one
        self.manager.delete_metadata(2)
        
        stats = self.manager.get_statistics()
        
        assert stats['total_entries'] == 3
        assert stats['active_entries'] == 2
        assert stats['deleted_entries'] == 1
        assert stats['type_distribution']['document'] == 2
    
    def test_save_and_load_metadata(self):
        """Test saving and loading metadata to/from file."""
        # Add some metadata
        self.manager.add_metadata(1, {'source': 'test1.pdf'})
        self.manager.add_metadata(2, {'source': 'test2.pdf'})
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            self.manager.save_metadata(tmp_path)
            
            # Create new manager and load
            new_manager = MetadataManager()
            new_manager.load_metadata(tmp_path)
            
            # Verify data was loaded correctly
            assert new_manager.get_metadata_count() == 2
            assert new_manager.get_metadata(1)['source'] == 'test1.pdf'
            assert new_manager.get_metadata(2)['source'] == 'test2.pdf'
            
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)


class TestVectorDatabaseIntegration:
    """Integration tests for vector database components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Start persistent mocks
        self.faiss_available_patcher = patch('src.database.vector_database.FAISS_AVAILABLE', True)
        self.faiss_patcher = patch('src.database.vector_database.faiss')
        self.st_available_patcher = patch('src.database.embedding_generator.SENTENCE_TRANSFORMERS_AVAILABLE', True)
        self.st_patcher = patch('src.database.embedding_generator.SentenceTransformer')

        # Start all patches
        self.faiss_available_patcher.start()
        mock_faiss = self.faiss_patcher.start()
        self.st_available_patcher.start()
        mock_st = self.st_patcher.start()

        # Mock FAISS index
        self.mock_index = Mock()
        self.mock_index.ntotal = 0
        self.mock_index.add = Mock()
        self.mock_index.search = Mock(return_value=(np.array([[0.8, 0.7]]), np.array([[0, 1]])))

        mock_faiss.IndexFlatIP.return_value = self.mock_index
        mock_faiss.write_index = Mock()
        mock_faiss.read_index = Mock(return_value=self.mock_index)

        # Mock embedding generator
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_model.get_sentence_embedding_dimension.return_value = 4
        mock_st.return_value = mock_model

        # Store mocks for later use
        self.mock_faiss = mock_faiss
        self.mock_st = mock_st

        from src.database.vector_database import VectorDatabase

        self.vector_db = VectorDatabase({
            'db_path': str(self.temp_dir),
            'embedding_dim': 4
        })
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Stop all patches
        self.faiss_available_patcher.stop()
        self.faiss_patcher.stop()
        self.st_available_patcher.stop()
        self.st_patcher.stop()

        # Clean up temp directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_add_and_search_vectors(self):
        """Test adding vectors and searching."""
        texts = ["First document", "Second document", "Third document"]
        metadata_list = [
            {'source': 'doc1.txt', 'content_type': 'text'},
            {'source': 'doc2.txt', 'content_type': 'text'},
            {'source': 'doc3.txt', 'content_type': 'text'}
        ]
        
        # Add vectors
        ids = self.vector_db.add_to_vectordb(texts, metadata_list)
        assert len(ids) == 3
        
        # Search
        results = self.vector_db.search("First document", k=2)
        assert len(results) <= 2
    
    def test_database_persistence(self):
        """Test saving and loading database."""
        texts = ["Test document for persistence"]
        metadata_list = [{'source': 'test.txt', 'type': 'document'}]

        # Add data to database
        ids = self.vector_db.add_to_vectordb(texts, metadata_list)
        assert len(ids) == 1

        # Update mock index to reflect added data
        self.mock_index.ntotal = len(ids)

        # Test save_database method
        self.vector_db.save_database()

        # Verify that the mocked methods were called
        self.mock_faiss.write_index.assert_called_once()

        # Verify that metadata and db_info files were created
        assert (self.temp_dir / "metadata.json").exists()
        assert (self.temp_dir / "db_info.json").exists()

        # Verify content of db_info.json
        with open(self.temp_dir / "db_info.json", 'r') as f:
            db_info = json.load(f)
            assert db_info['embedding_dim'] == 4
            assert db_info['total_vectors'] == 1
            assert 'created_at' in db_info
    
    def test_database_stats(self):
        """Test getting database statistics."""
        stats = self.vector_db.get_database_stats()
        
        assert 'total_vectors' in stats
        assert 'embedding_dimension' in stats
        assert 'index_type' in stats
        assert 'database_path' in stats


if __name__ == "__main__":
    pytest.main([__file__])
