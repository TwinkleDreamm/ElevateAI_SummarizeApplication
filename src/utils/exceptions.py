"""
Custom exceptions for ElevateAI application.
"""


class ElevateAIException(Exception):
    """Base exception for ElevateAI application."""
    pass


class FileProcessingError(ElevateAIException):
    """Exception raised when file processing fails."""
    pass


class AudioProcessingError(ElevateAIException):
    """Exception raised when audio processing fails."""
    pass


class VideoProcessingError(ElevateAIException):
    """Exception raised when video processing fails."""
    pass


class TextProcessingError(ElevateAIException):
    """Exception raised when text processing fails."""
    pass


class SpeechToTextError(ElevateAIException):
    """Exception raised when speech-to-text conversion fails."""
    pass


class EmbeddingError(ElevateAIException):
    """Exception raised when embedding generation fails."""
    pass


class VectorDatabaseError(ElevateAIException):
    """Exception raised when vector database operations fail."""
    pass


class SearchError(ElevateAIException):
    """Exception raised when search operations fail."""
    pass


class WebSearchError(ElevateAIException):
    """Exception raised when web search fails."""
    pass


class LLMError(ElevateAIException):
    """Exception raised when LLM operations fail."""
    pass


class ConfigurationError(ElevateAIException):
    """Exception raised when configuration is invalid."""
    pass


class UnsupportedFormatError(ElevateAIException):
    """Exception raised when file format is not supported."""
    pass


class FileSizeError(ElevateAIException):
    """Exception raised when file size exceeds limits."""
    pass


class TTSProcessingError(ElevateAIException):
    """Exception raised when text-to-speech processing fails."""
    pass
