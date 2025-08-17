"""
Text cleaning module for ElevateAI.
Handles text normalization, cleaning, and preprocessing.
"""
import re
from typing import Optional, Dict, Any, List

try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from config.settings import settings
from src.utils.logger import logger


class TextCleaner:
    """Handles text cleaning and normalization operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the text cleaner.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Initialize language tools
        self.grammar_tools = {}
        self._load_language_tools()
    
    def _load_language_tools(self) -> None:
        """Load language processing tools."""
        self.logger.info("⚡ Grammar tools disabled for faster loading")
    
    def clean_text(self, text: str, **kwargs) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            **kwargs: Additional cleaning options
            
        Returns:
            Cleaned text
        """
        if not text or not text.strip():
            return ""
        
        cleaned_text = text
        
        # Apply cleaning steps
        if kwargs.get('normalize_whitespace', True):
            cleaned_text = self._normalize_whitespace(cleaned_text)
        
        if kwargs.get('remove_special_chars', False):
            cleaned_text = self._remove_special_characters(cleaned_text)
        
        if kwargs.get('fix_encoding', True):
            cleaned_text = self._fix_encoding_issues(cleaned_text)
        
        if kwargs.get('normalize_punctuation', True):
            cleaned_text = self._normalize_punctuation(cleaned_text)
        
        if kwargs.get('remove_repetitions', True):
            cleaned_text = self._remove_repetitions(cleaned_text)
        
        if kwargs.get('fix_grammar', False) and kwargs.get('language'):
            cleaned_text = self._fix_grammar(cleaned_text, kwargs['language'])
        
        if kwargs.get('remove_stopwords', False) and kwargs.get('language'):
            cleaned_text = self._remove_stopwords(cleaned_text, kwargs['language'])
        
        return cleaned_text.strip()
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        return text
    
    def _remove_special_characters(self, text: str) -> str:
        """Remove special characters while preserving important punctuation."""
        # Keep letters, numbers, basic punctuation, and Vietnamese characters
        pattern = r'[^\w\s\.,!?;:\-\(\)àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]'
        return re.sub(pattern, ' ', text)
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues."""
        # Common encoding fixes
        fixes = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€¦': '...',
            'â€"': '-',
            'â€"': '--',
            'Ã¡': 'á',
            'Ã©': 'é',
            'Ã­': 'í',
            'Ã³': 'ó',
            'Ãº': 'ú',
        }
        
        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # Fix spacing around punctuation
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        
        # Fix multiple punctuation marks
        text = re.sub(r'[.]{2,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Fix quotation marks
        text = re.sub(r'[\u201c\u201d]', '"', text)  # Smart double quotes
        text = re.sub(r'[\u2018\u2019]', "'", text)  # Smart single quotes
        
        return text
    
    def _remove_repetitions(self, text: str) -> str:
        """Remove excessive repetitions in text."""
        # Remove repeated words (more than 2 times)
        text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text)
        
        # Remove repeated phrases (basic)
        text = re.sub(r'(.{10,}?)\s+\1{2,}', r'\1', text)
        
        return text
    
    def _fix_grammar(self, text: str, language: str) -> str:
        """Fix grammar issues using language tools."""
        if not LANGUAGE_TOOL_AVAILABLE or language not in self.grammar_tools:
            return text
        
        try:
            tool = self.grammar_tools[language]
            matches = tool.check(text)
            
            # Apply corrections (limit to avoid over-correction)
            corrected_text = text
            for match in matches[:10]:  # Limit corrections
                if match.replacements:
                    corrected_text = corrected_text.replace(
                        text[match.offset:match.offset + match.errorLength],
                        match.replacements[0]
                    )
            
            return corrected_text
            
        except Exception as e:
            self.logger.warning(f"Grammar correction failed: {e}")
            return text
    
    def _remove_stopwords(self, text: str, language: str) -> str:
        """Remove stopwords from text."""
        if not NLTK_AVAILABLE:
            return text
        
        try:
            # Map language codes
            lang_map = {'vi': 'english', 'en': 'english'}  # NLTK doesn't have Vietnamese stopwords
            nltk_lang = lang_map.get(language, 'english')
            
            stop_words = set(stopwords.words(nltk_lang))
            
            # Add custom Vietnamese stopwords
            if language == 'vi':
                vietnamese_stopwords = {
                    'và', 'của', 'có', 'là', 'được', 'một', 'trong', 'cho', 'với', 'từ',
                    'này', 'đó', 'những', 'các', 'để', 'khi', 'như', 'về', 'sẽ', 'đã',
                    'rất', 'cũng', 'không', 'chỉ', 'hay', 'hoặc', 'mà', 'nếu', 'thì'
                }
                stop_words.update(vietnamese_stopwords)
            
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in stop_words]
            
            return ' '.join(filtered_words)
            
        except Exception as e:
            self.logger.warning(f"Stopword removal failed: {e}")
            return text
    
    def clean_transcript(self, transcript: str, language: Optional[str] = None) -> str:
        """
        Clean transcript text with specific settings for speech-to-text output.
        
        Args:
            transcript: Raw transcript text
            language: Language code for language-specific cleaning
            
        Returns:
            Cleaned transcript
        """
        return self.clean_text(
            transcript,
            normalize_whitespace=True,
            fix_encoding=True,
            normalize_punctuation=True,
            remove_repetitions=True,
            remove_special_chars=False,
            fix_grammar=False,  # Usually not needed for transcripts
            remove_stopwords=False,  # Keep all words for context
            language=language
        )
    
    def clean_document_text(self, text: str, language: Optional[str] = None) -> str:
        """
        Clean document text with comprehensive cleaning.
        
        Args:
            text: Raw document text
            language: Language code for language-specific cleaning
            
        Returns:
            Cleaned document text
        """
        return self.clean_text(
            text,
            normalize_whitespace=True,
            fix_encoding=True,
            normalize_punctuation=True,
            remove_repetitions=True,
            remove_special_chars=True,
            fix_grammar=True,
            remove_stopwords=False,
            language=language
        )
    
    def preprocess_for_embedding(self, text: str, language: Optional[str] = None) -> str:
        """
        Preprocess text specifically for embedding generation.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Preprocessed text optimized for embeddings
        """
        return self.clean_text(
            text,
            normalize_whitespace=True,
            fix_encoding=True,
            normalize_punctuation=True,
            remove_repetitions=True,
            remove_special_chars=False,
            fix_grammar=False,
            remove_stopwords=False,  # Keep stopwords for better embeddings
            language=language
        )
