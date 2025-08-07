"""
Tests for text processing modules.
"""
import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.text_analyzer import TextAnalyzer
from src.analysis.text_cleaner import TextCleaner
from src.analysis.text_chunker import TextChunker


class TestTextAnalyzer:
    """Test cases for TextAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TextAnalyzer()
    
    def test_analyze_transcript_content_empty(self):
        """Test analysis of empty content."""
        result = self.analyzer.analyze_transcript_content("")
        assert result == "no_content"
    
    def test_analyze_transcript_content_short(self):
        """Test analysis of short content."""
        short_text = "This is a short text."
        result = self.analyzer.analyze_transcript_content(short_text)
        assert result == "limited_content"
    
    def test_analyze_transcript_content_sufficient(self):
        """Test analysis of sufficient content."""
        long_text = " ".join(["This is a longer text with more content."] * 20)
        result = self.analyzer.analyze_transcript_content(long_text)
        assert result == "sufficient_content"
    
    def test_analyze_text_comprehensive(self):
        """Test comprehensive text analysis."""
        test_text = """
        This is a sample text for testing the comprehensive analysis functionality.
        It contains multiple sentences and paragraphs to test various metrics.
        
        The text should be analyzed for word count, sentence count, and other features.
        This helps ensure the analyzer works correctly with different types of content.
        """
        
        result = self.analyzer.analyze_text(test_text)
        
        assert result.word_count > 0
        assert result.sentence_count > 0
        assert result.status in ["no_content", "limited_content", "sufficient_content"]
        assert 0 <= result.quality_score <= 1


class TestTextCleaner:
    """Test cases for TextCleaner."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = TextCleaner()
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        dirty_text = "  This   is    a   messy    text   with   extra   spaces.  "
        cleaned = self.cleaner.clean_text(dirty_text)
        
        assert "   " not in cleaned  # No triple spaces
        assert cleaned.strip() == cleaned  # No leading/trailing spaces
    
    def test_clean_text_empty(self):
        """Test cleaning empty text."""
        result = self.cleaner.clean_text("")
        assert result == ""
    
    def test_clean_text_with_options(self):
        """Test cleaning with various options."""
        test_text = "This is a test text with special characters! @#$%"
        
        # Test with special character removal
        cleaned = self.cleaner.clean_text(
            test_text,
            remove_special_chars=True,
            normalize_whitespace=True
        )
        
        assert len(cleaned) > 0
        assert "@#$%" not in cleaned
    
    def test_clean_transcript(self):
        """Test transcript-specific cleaning."""
        transcript = "Um, this is, uh, a transcript with, you know, filler words."
        cleaned = self.cleaner.clean_transcript(transcript)
        
        assert len(cleaned) > 0
        assert isinstance(cleaned, str)


class TestTextChunker:
    """Test cases for TextChunker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = TextChunker()
    
    def test_split_into_chunks_empty(self):
        """Test chunking empty text."""
        chunks = self.chunker.split_into_chunks("")
        assert len(chunks) == 0
    
    def test_split_into_chunks_short(self):
        """Test chunking short text."""
        short_text = "This is a short text that should fit in one chunk."
        chunks = self.chunker.split_into_chunks(short_text, chunk_size=1000)
        
        assert len(chunks) == 1
        assert chunks[0].content == short_text
        assert chunks[0].word_count > 0
    
    def test_split_into_chunks_long(self):
        """Test chunking long text."""
        long_text = " ".join(["This is a sentence."] * 100)
        chunks = self.chunker.split_into_chunks(long_text, chunk_size=200)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 250  # Allow some flexibility
            assert chunk.word_count > 0
    
    def test_split_strategies(self):
        """Test different chunking strategies."""
        test_text = """
        This is the first paragraph with multiple sentences. It contains important information.
        
        This is the second paragraph. It also has multiple sentences and important content.
        
        This is the third paragraph with even more content to test the chunking strategies.
        """
        
        # Test semantic chunking
        semantic_chunks = self.chunker.split_into_chunks(
            test_text, 
            strategy='semantic',
            chunk_size=200
        )
        
        # Test sentence-based chunking
        sentence_chunks = self.chunker.split_into_chunks(
            test_text,
            strategy='sentence',
            chunk_size=200
        )
        
        # Test paragraph-based chunking
        paragraph_chunks = self.chunker.split_into_chunks(
            test_text,
            strategy='paragraph',
            chunk_size=500
        )
        
        assert len(semantic_chunks) > 0
        assert len(sentence_chunks) > 0
        assert len(paragraph_chunks) > 0
    
    def test_chunk_statistics(self):
        """Test chunk statistics calculation."""
        test_text = "This is a test text for statistics calculation."
        chunks = self.chunker.split_into_chunks(test_text)
        
        stats = self.chunker.get_chunk_statistics(chunks)
        
        assert 'total_chunks' in stats
        assert 'avg_words_per_chunk' in stats
        assert 'total_words' in stats
        assert stats['total_chunks'] == len(chunks)


# Integration tests
class TestTextProcessingIntegration:
    """Integration tests for text processing pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TextAnalyzer()
        self.cleaner = TextCleaner()
        self.chunker = TextChunker()
    
    def test_full_processing_pipeline(self):
        """Test the complete text processing pipeline."""
        raw_text = """
        This is a sample document with multiple paragraphs and various content types.
        It contains some messy formatting   and   extra   spaces that need cleaning.
        
        The document also has different sections that should be properly chunked.
        Each section contains important information that needs to be preserved.
        
        Finally, the text should be analyzed for quality and content characteristics.
        This helps ensure the processing pipeline works correctly end-to-end.
        """
        
        # Step 1: Analyze raw text
        analysis = self.analyzer.analyze_text(raw_text)
        assert analysis.status != "no_content"
        
        # Step 2: Clean text
        cleaned_text = self.cleaner.clean_text(raw_text)
        assert len(cleaned_text) > 0
        assert "   " not in cleaned_text  # No excessive spaces
        
        # Step 3: Chunk cleaned text
        chunks = self.chunker.split_into_chunks(cleaned_text)
        assert len(chunks) > 0
        
        # Step 4: Verify chunk quality
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0
            assert chunk.word_count > 0
        
        # Step 5: Get statistics
        stats = self.chunker.get_chunk_statistics(chunks)
        assert stats['total_chunks'] == len(chunks)
        assert stats['total_words'] > 0


if __name__ == "__main__":
    pytest.main([__file__])
