"""
Base processor class for all data processing operations.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import ElevateAIException


class BaseProcessor(ABC):
    """Abstract base class for all processors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.settings = settings
        self.logger = logger
        
        # Ensure necessary directories exist
        self.settings.ensure_directories()
    
    @abstractmethod
    def process(self, input_data: Union[str, Path, bytes], **kwargs) -> Any:
        """
        Process the input data.
        
        Args:
            input_data: Input data to process
            **kwargs: Additional processing parameters
            
        Returns:
            Processed data
            
        Raises:
            ElevateAIException: If processing fails
        """
        pass
    
    def validate_input(self, input_data: Union[str, Path, bytes]) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if isinstance(input_data, (str, Path)):
            file_path = Path(input_data)
            if not file_path.exists():
                self.logger.error(f"File does not exist: {file_path}")
                return False
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.settings.max_file_size_mb:
                self.logger.error(
                    f"File size ({file_size_mb:.2f} MB) exceeds limit "
                    f"({self.settings.max_file_size_mb} MB)"
                )
                return False
        
        return True
    
    def get_temp_file_path(self, suffix: str = "") -> Path:
        """
        Get a temporary file path.
        
        Args:
            suffix: File suffix/extension
            
        Returns:
            Path to temporary file
        """
        import uuid
        temp_name = f"temp_{uuid.uuid4().hex}{suffix}"
        return self.settings.temp_dir / temp_name
    
    def cleanup_temp_files(self, file_paths: list[Path]) -> None:
        """
        Clean up temporary files.
        
        Args:
            file_paths: List of file paths to clean up
        """
        for file_path in file_paths:
            try:
                if file_path.exists():
                    file_path.unlink()
                    self.logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temp file {file_path}: {e}")
    
    def log_processing_start(self, operation: str, input_info: str) -> None:
        """Log the start of a processing operation."""
        self.logger.info(f"Starting {operation} for: {input_info}")
    
    def log_processing_end(self, operation: str, success: bool = True) -> None:
        """Log the end of a processing operation."""
        status = "completed" if success else "failed"
        self.logger.info(f"{operation} {status}")
    
    def handle_error(self, error: Exception, operation: str) -> None:
        """
        Handle and log errors during processing.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
        """
        error_msg = f"{operation} failed: {str(error)}"
        self.logger.error(error_msg)
        raise ElevateAIException(error_msg) from error
