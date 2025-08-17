"""
Text analysis module for ElevateAI.
Analyzes text content quality, language, and characteristics.
"""
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

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

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: NLTK not available: {e}")
    NLTK_AVAILABLE = False
except Exception as e:
    print(f"Warning: NLTK import failed: {e}")
    NLTK_AVAILABLE = False

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import TextProcessingError


@dataclass
class TextAnalysisResult:
    """Result of text analysis."""
    status: str  # "no_content", "limited_content", "sufficient_content"
    word_count: int
    sentence_count: int
    paragraph_count: int
    language: Optional[str]
    readability_score: Optional[float]
    key_topics: List[str]
    content_type: str  # "conversational", "formal", "technical", etc.
    quality_score: float  # 0-1 scale


class TextAnalyzer:
    """Analyzes text content for quality and characteristics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the text analyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Initialize NLP models
        self.nlp_models = {}
        self._load_nlp_models()
    
    def _load_nlp_models(self) -> None:
        """Load NLP models for text analysis."""
        # Only download NLTK data if not disabled
        if NLTK_AVAILABLE and not self.settings.disable_nltk_downloads:
            try:
                # Download required NLTK data
                nltk.download('punkt', quiet=True)
                try:
                    nltk.download('punkt_tab', quiet=True)
                except Exception:
                    pass
                nltk.download('stopwords', quiet=True)
                self.logger.info("✅ NLTK data downloaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to download NLTK data: {e}")
        else:
            self.logger.info("⚡ NLTK downloads disabled for faster loading")
        
        self.logger.info("⚡ NLP models disabled for faster loading")
    
    def analyze_transcript_content(self, transcript: str) -> str:
        """
        Analyze transcript content to determine processing approach.
        
        Args:
            transcript: Input transcript text
            
        Returns:
            Content status: "no_content", "limited_content", or "sufficient_content"
        """
        if not transcript or not transcript.strip():
            return "no_content"
        
        # Basic analysis
        clean_text = self._basic_clean(transcript)
        word_count = len(clean_text.split())
        
        # Define thresholds
        if word_count < 3:
            return "no_content"
        elif word_count < 100:
            return "limited_content"
        else:
            return "sufficient_content"
    
    def analyze_text(self, text: str) -> TextAnalysisResult:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            TextAnalysisResult with analysis details
        """
        if not text or not text.strip():
            return TextAnalysisResult(
                status="no_content",
                word_count=0,
                sentence_count=0,
                paragraph_count=0,
                language=None,
                readability_score=None,
                key_topics=[],
                content_type="unknown",
                quality_score=0.0
            )
        
        # Clean text for analysis
        clean_text = self._basic_clean(text)
        
        # Basic metrics
        word_count = len(clean_text.split())
        sentence_count = self._count_sentences(clean_text)
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # Language detection
        language = self._detect_language(clean_text)
        
        # Content analysis
        content_type = self._analyze_content_type(clean_text)
        key_topics = self._extract_key_topics(clean_text, language)
        
        # Quality assessment
        quality_score = self._calculate_quality_score(clean_text, word_count, sentence_count)
        readability_score = self._calculate_readability(clean_text)
        
        # Determine status
        status = self._determine_status(word_count, quality_score)
        
        return TextAnalysisResult(
            status=status,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            language=language,
            readability_score=readability_score,
            key_topics=key_topics,
            content_type=content_type,
            quality_score=quality_score
        )
    
    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning for analysis."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,!?;:\-\(\)]', '', text)
        return text.strip()
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        if NLTK_AVAILABLE:
            try:
                return len(sent_tokenize(text))
            except:
                pass
        
        # Fallback: simple sentence counting
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect text language."""
        # Simple heuristic-based language detection
        vietnamese_chars = re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text.lower())
        
        if len(vietnamese_chars) > len(text.split()) * 0.1:
            return 'vi'
        else:
            return 'en'  # Default to English
    
    def _analyze_content_type(self, text: str) -> str:
        """Analyze content type based on linguistic features."""
        # Check for conversational markers
        conversational_markers = ['uh', 'um', 'you know', 'like', 'so', 'well', 'actually']
        conversational_count = sum(1 for marker in conversational_markers if marker in text.lower())
        
        # Check for formal language
        formal_markers = ['therefore', 'furthermore', 'however', 'consequently', 'moreover']
        formal_count = sum(1 for marker in formal_markers if marker in text.lower())
        
        # Check for technical terms (simplified)
        technical_markers = ['algorithm', 'system', 'process', 'method', 'analysis', 'data']
        technical_count = sum(1 for marker in technical_markers if marker in text.lower())
        
        if conversational_count > formal_count and conversational_count > technical_count:
            return "conversational"
        elif technical_count > formal_count:
            return "technical"
        elif formal_count > 0:
            return "formal"
        else:
            return "general"
    
    def _extract_key_topics(self, text: str, language: Optional[str]) -> List[str]:
        """Extract key topics from text."""
        if not SPACY_AVAILABLE or not language or language not in self.nlp_models:
            return []
        
        try:
            nlp = self.nlp_models[language]
            doc = nlp(text[:1000])  # Limit text length for processing
            
            # Extract named entities and noun phrases
            topics = []
            
            # Named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']:
                    topics.append(ent.text.lower())
            
            # Noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit phrase length
                    topics.append(chunk.text.lower())
            
            # Remove duplicates and return top topics
            unique_topics = list(set(topics))
            return unique_topics[:10]
            
        except Exception as e:
            self.logger.warning(f"Topic extraction failed: {e}")
            return []
    
    def _calculate_quality_score(self, text: str, word_count: int, sentence_count: int) -> float:
        """Calculate text quality score (0-1)."""
        score = 0.0
        
        # Length factor
        if word_count > 50:
            score += 0.3
        elif word_count > 20:
            score += 0.2
        elif word_count > 10:
            score += 0.1
        
        # Sentence structure
        if sentence_count > 0:
            avg_words_per_sentence = word_count / sentence_count
            if 10 <= avg_words_per_sentence <= 25:
                score += 0.2
            elif 5 <= avg_words_per_sentence <= 35:
                score += 0.1
        
        # Vocabulary diversity (simplified)
        unique_words = len(set(text.lower().split()))
        if word_count > 0:
            diversity = unique_words / word_count
            score += min(0.3, diversity * 0.6)
        
        # Coherence (basic check for repeated patterns)
        if not re.search(r'(.{10,})\1{2,}', text):  # No excessive repetition
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_readability(self, text: str) -> Optional[float]:
        """Calculate readability score (simplified Flesch score)."""
        try:
            words = len(text.split())
            sentences = self._count_sentences(text)
            
            if sentences == 0:
                return None
            
            # Simplified readability calculation
            avg_sentence_length = words / sentences
            
            # Simple syllable estimation (very rough)
            syllables = sum(max(1, len(re.findall(r'[aeiouAEIOU]', word))) for word in text.split())
            avg_syllables_per_word = syllables / words if words > 0 else 0
            
            # Simplified Flesch formula
            score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            return max(0, min(100, score))
            
        except Exception:
            return None
    
    def _determine_status(self, word_count: int, quality_score: float) -> str:
        """Determine content status based on metrics."""
        if word_count < 10 or quality_score < 0.2:
            return "no_content"
        elif word_count < 100 or quality_score < 0.5:
            return "limited_content"
        else:
            return "sufficient_content"
