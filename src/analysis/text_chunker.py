"""
Text chunking module for ElevateAI.
Handles intelligent text segmentation for vector database storage.
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    # import spacy
    # from spacy.lang.vi import Vietnamese
    # from spacy.lang.en import English
    SPACY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: spaCy not available: {e}")
    SPACY_AVAILABLE = False
except Exception as e:
    print(f"Warning: spaCy import failed: {e}")
    SPACY_AVAILABLE = False

from config.settings import settings
from src.utils.logger import logger


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    content: str
    start_index: int
    end_index: int
    chunk_id: str
    metadata: Dict[str, Any]
    word_count: int
    sentence_count: int


class TextChunker:
    """Handles intelligent text chunking for vector database storage."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the text chunker.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Chunking parameters
        self.chunk_size = self.config.get('chunk_size', settings.chunk_size)
        self.chunk_overlap = self.config.get('chunk_overlap', settings.chunk_overlap)
        self.min_chunk_size = self.config.get('min_chunk_size', 50)
        
        # Initialize NLP tools
        self.nlp_models = {}
        self._load_nlp_models()
    
    def _load_nlp_models(self) -> None:
        """Load NLP models for sentence segmentation."""
        # Only download NLTK data if not disabled
        if NLTK_AVAILABLE and not self.settings.disable_nltk_downloads:
            try:
                nltk.download('punkt', quiet=True)
                # Newer NLTK versions use 'punkt_tab' resources
                try:
                    nltk.download('punkt_tab', quiet=True)
                except Exception:
                    pass
                self.logger.info("✅ NLTK punkt data downloaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to download NLTK punkt: {e}")
        else:
            self.logger.info("⚡ NLTK downloads disabled for faster loading")

        # spaCy models are commented out in imports, so they're disabled
        # self.logger.info("⚡ spaCy models disabled for faster loading")
    
    def split_into_chunks(self, text: str, chunk_size: Optional[int] = None, 
                         overlap: Optional[int] = None, **kwargs) -> List[TextChunk]:
        """
        Split text into chunks with intelligent boundaries.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks in characters
            **kwargs: Additional chunking options
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        # Choose chunking strategy
        strategy = kwargs.get('strategy', 'semantic')
        language = kwargs.get('language', 'en')
        
        if strategy == 'semantic':
            return self._semantic_chunking(text, chunk_size, overlap, language)
        elif strategy == 'sentence':
            return self._sentence_based_chunking(text, chunk_size, overlap, language)
        elif strategy == 'paragraph':
            return self._paragraph_based_chunking(text, chunk_size, overlap)
        else:
            return self._fixed_size_chunking(text, chunk_size, overlap)
    
    def _semantic_chunking(self, text: str, chunk_size: int, overlap: int, language: str) -> List[TextChunk]:
        """
        Chunk text based on semantic boundaries (sentences and paragraphs).
        
        Args:
            text: Input text
            chunk_size: Maximum chunk size
            overlap: Overlap size
            language: Language code
            
        Returns:
            List of semantic chunks
        """
        # First, split into sentences
        sentences = self._split_into_sentences(text, language)
        
        if not sentences:
            return self._fixed_size_chunking(text, chunk_size, overlap)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(potential_chunk) <= chunk_size or not current_chunk:
                current_chunk = potential_chunk
                chunk_sentences.append(sentence)
            else:
                # Create chunk from current content
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk, 
                        current_start, 
                        current_start + len(current_chunk),
                        len(chunks),
                        {"sentences": chunk_sentences.copy(), "type": "semantic"}
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(chunk_sentences, overlap)
                current_chunk = " ".join(overlap_sentences + [sentence])
                current_start = current_start + len(current_chunk) - len(" ".join(overlap_sentences + [sentence]))
                chunk_sentences = overlap_sentences + [sentence]
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk,
                current_start,
                current_start + len(current_chunk),
                len(chunks),
                {"sentences": chunk_sentences, "type": "semantic"}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _sentence_based_chunking(self, text: str, chunk_size: int, overlap: int, language: str) -> List[TextChunk]:
        """
        Chunk text based on sentence boundaries.
        
        Args:
            text: Input text
            chunk_size: Maximum chunk size
            overlap: Overlap size
            language: Language code
            
        Returns:
            List of sentence-based chunks
        """
        sentences = self._split_into_sentences(text, language)
        
        if not sentences:
            return self._fixed_size_chunking(text, chunk_size, overlap)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(potential_chunk) <= chunk_size or not current_chunk:
                current_chunk = potential_chunk
                i += 1
            else:
                # Create chunk
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk,
                        current_start,
                        current_start + len(current_chunk),
                        len(chunks),
                        {"type": "sentence_based"}
                    )
                    chunks.append(chunk)
                
                # Calculate overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_start += len(current_chunk) - len(overlap_text) - len(sentence) - 1
                i += 1
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk,
                current_start,
                current_start + len(current_chunk),
                len(chunks),
                {"type": "sentence_based"}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _paragraph_based_chunking(self, text: str, chunk_size: int, overlap: int) -> List[TextChunk]:
        """
        Chunk text based on paragraph boundaries.
        
        Args:
            text: Input text
            chunk_size: Maximum chunk size
            overlap: Overlap size
            
        Returns:
            List of paragraph-based chunks
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return self._fixed_size_chunking(text, chunk_size, overlap)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if len(potential_chunk) <= chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # Create chunk
                if current_chunk:
                    chunk = self._create_chunk(
                        current_chunk,
                        current_start,
                        current_start + len(current_chunk),
                        len(chunks),
                        {"type": "paragraph_based"}
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + "\n\n" + paragraph
                current_start += len(current_chunk) - len(overlap_text) - len(paragraph) - 2
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk,
                current_start,
                current_start + len(current_chunk),
                len(chunks),
                {"type": "paragraph_based"}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fixed_size_chunking(self, text: str, chunk_size: int, overlap: int) -> List[TextChunk]:
        """
        Chunk text using fixed size with overlap.
        
        Args:
            text: Input text
            chunk_size: Chunk size
            overlap: Overlap size
            
        Returns:
            List of fixed-size chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            # Try to break at word boundary
            if end < len(text) and not text[end].isspace():
                last_space = chunk_text.rfind(' ')
                if last_space > chunk_size * 0.8:  # Only if we don't lose too much
                    end = start + last_space
                    chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunk = self._create_chunk(
                    chunk_text.strip(),
                    start,
                    end,
                    len(chunks),
                    {"type": "fixed_size"}
                )
                chunks.append(chunk)
            
            start = max(start + chunk_size - overlap, start + 1)
        
        return chunks
    
    def _split_into_sentences(self, text: str, language: str) -> List[str]:
        """Split text into sentences using available NLP tools."""
        if NLTK_AVAILABLE:
            try:
                return sent_tokenize(text)
            except Exception as e:
                self.logger.warning(f"NLTK sentence tokenization failed: {e}")
        
        if SPACY_AVAILABLE and language in self.nlp_models:
            try:
                nlp = self.nlp_models[language]
                doc = nlp(text)
                return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception as e:
                self.logger.warning(f"spaCy sentence segmentation failed: {e}")
        
        # Fallback: simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_chars: int) -> List[str]:
        """Get sentences for overlap based on character count."""
        if not sentences:
            return []
        
        overlap_sentences = []
        char_count = 0
        
        for sentence in reversed(sentences):
            if char_count + len(sentence) <= overlap_chars:
                overlap_sentences.insert(0, sentence)
                char_count += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk(self, content: str, start: int, end: int, chunk_id: int, metadata: Dict[str, Any]) -> TextChunk:
        """Create a TextChunk object with metadata."""
        return TextChunk(
            content=content,
            start_index=start,
            end_index=end,
            chunk_id=f"chunk_{chunk_id:04d}",
            metadata=metadata,
            word_count=len(content.split()),
            sentence_count=len(self._split_into_sentences(content, metadata.get('language', 'en')))
        )
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about the chunks."""
        if not chunks:
            return {}
        
        word_counts = [chunk.word_count for chunk in chunks]
        char_counts = [len(chunk.content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_words_per_chunk": sum(word_counts) / len(word_counts),
            "avg_chars_per_chunk": sum(char_counts) / len(char_counts),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "total_words": sum(word_counts),
            "total_chars": sum(char_counts)
        }
