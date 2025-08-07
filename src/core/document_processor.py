"""
Document processing module for ElevateAI.
Handles text extraction from various document formats.
"""
from pathlib import Path
from typing import Optional, Union

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    import requests
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

from .base_processor import BaseProcessor
from src.utils.exceptions import TextProcessingError, UnsupportedFormatError


class DocumentProcessor(BaseProcessor):
    """Handles document text extraction operations."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the document processor.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
    
    def process(self, input_data: Union[str, Path], **kwargs) -> str:
        """
        Process document and extract text.
        
        Args:
            input_data: Path to document file or URL
            **kwargs: Additional processing parameters
            
        Returns:
            Extracted text content
            
        Raises:
            TextProcessingError: If processing fails
        """
        # Check if input is URL
        if isinstance(input_data, str) and (input_data.startswith('http://') or input_data.startswith('https://')):
            return self.extract_text_from_url(input_data)
        
        # Process as file
        if not self.validate_input(input_data):
            raise TextProcessingError(f"Invalid input: {input_data}")
        
        file_path = Path(input_data)
        
        if not self.validate_document_format(file_path):
            raise UnsupportedFormatError(f"Unsupported document format: {file_path.suffix}")
        
        self.log_processing_start("Document processing", str(file_path))
        
        try:
            # Extract text based on file format
            suffix = file_path.suffix.lower()
            
            if suffix == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            elif suffix == '.docx':
                text = self.extract_text_from_docx(file_path)
            elif suffix == '.txt':
                text = self.extract_text_from_txt(file_path)
            else:
                raise UnsupportedFormatError(f"Unsupported format: {suffix}")
            
            self.log_processing_end("Document processing")
            return text
            
        except Exception as e:
            self.handle_error(e, "Document processing")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
            
        Raises:
            TextProcessingError: If extraction fails
        """
        if not PDF_AVAILABLE:
            raise TextProcessingError("PDF processing not available. Please install pdfplumber.")
        
        try:
            text_content = []
            
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
            
            return '\n\n'.join(text_content)
            
        except Exception as e:
            raise TextProcessingError(f"Failed to extract text from PDF: {e}")
    
    def extract_text_from_docx(self, docx_path: Path) -> str:
        """
        Extract text from DOCX file.
        
        Args:
            docx_path: Path to DOCX file
            
        Returns:
            Extracted text
            
        Raises:
            TextProcessingError: If extraction fails
        """
        if not DOCX_AVAILABLE:
            raise TextProcessingError("DOCX processing not available. Please install python-docx.")
        
        try:
            doc = Document(str(docx_path))
            text_content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            return '\n\n'.join(text_content)
            
        except Exception as e:
            raise TextProcessingError(f"Failed to extract text from DOCX: {e}")
    
    def extract_text_from_txt(self, txt_path: Path) -> str:
        """
        Extract text from TXT file.
        
        Args:
            txt_path: Path to TXT file
            
        Returns:
            Extracted text
            
        Raises:
            TextProcessingError: If extraction fails
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(txt_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            
            raise TextProcessingError("Could not decode text file with any supported encoding")
            
        except Exception as e:
            raise TextProcessingError(f"Failed to extract text from TXT: {e}")
    
    def extract_text_from_url(self, url: str) -> str:
        """
        Extract text from web URL.
        
        Args:
            url: Web URL to extract text from
            
        Returns:
            Extracted text
            
        Raises:
            TextProcessingError: If extraction fails
        """
        if not WEB_AVAILABLE:
            raise TextProcessingError("Web scraping not available. Please install beautifulsoup4 and requests.")
        
        try:
            # Fetch web page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse HTML and extract text
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            raise TextProcessingError(f"Failed to extract text from URL: {e}")
    
    def validate_document_format(self, file_path: Path) -> bool:
        """
        Validate if file format is supported for document processing.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if format is supported
        """
        suffix = file_path.suffix.lower().lstrip('.')
        return suffix in self.settings.supported_document_formats
    
    def get_document_info(self, file_path: Path) -> dict:
        """
        Get document information.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with document information
        """
        try:
            info = {
                "file_size": file_path.stat().st_size,
                "format": file_path.suffix.lower(),
                "name": file_path.name
            }
            
            # Add format-specific info
            if file_path.suffix.lower() == '.pdf' and PDF_AVAILABLE:
                try:
                    with pdfplumber.open(str(file_path)) as pdf:
                        info["pages"] = len(pdf.pages)
                except:
                    pass
            
            return info
            
        except Exception as e:
            self.logger.warning(f"Failed to get document info: {e}")
            return {"error": str(e)}
