"""
YouTube processing module for ElevateAI.
Handles YouTube video downloading and content extraction.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
from urllib.parse import urlparse, parse_qs

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("Warning: yt-dlp not installed. Install with: pip install yt-dlp")

# Simple base class to avoid import issues
class SimpleBaseProcessor:
    """Simple base processor to avoid import issues."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
    
    def log_processing_start(self, process_name: str, input_data: str):
        """Log processing start."""
        print(f"Starting {process_name} for: {input_data}")
    
    def log_processing_end(self, process_name: str, success: bool = True):
        """Log processing end."""
        print(f"Completed {process_name} - Success: {success}")
    
    def handle_error(self, error: Exception, process_name: str):
        """Handle processing error."""
        print(f"Error in {process_name}: {error}")
        raise error
    
    def cleanup_temp_files(self, file_paths: list):
        """Clean up temporary files."""
        for file_path in file_paths:
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
            except Exception:
                pass

class VideoProcessingError(Exception):
    """Custom exception for video processing errors."""
    pass

class UnsupportedFormatError(Exception):
    """Custom exception for unsupported format errors."""
    pass


class YouTubeProcessor(SimpleBaseProcessor):
    """Handles YouTube video processing operations."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the YouTube processor.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        
        # YouTube download configuration
        self.ydl_opts = {
            'format': 'best[height<=720]',  # Limit to 720p for faster processing
            'outtmpl': '%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
    
    def process(self, input_data: Union[str, Path, bytes], **kwargs) -> Dict[str, Any]:
        """
        Process YouTube URL and extract content.
        
        Args:
            input_data: YouTube video URL (string)
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary containing extracted content and metadata
            
        Raises:
            VideoProcessingError: If processing fails
        """
        if not YT_DLP_AVAILABLE:
            raise VideoProcessingError("YouTube processing dependencies not available")
        
        # Convert input_data to string if it's a Path
        url = str(input_data) if isinstance(input_data, Path) else input_data
        
        if not self._is_valid_youtube_url(url):
            raise VideoProcessingError(f"Invalid YouTube URL: {url}")
        
        self.log_processing_start("YouTube processing", url)
        
        try:
            # For now, just return basic info without downloading
            # This avoids complex dependencies during import
            video_info = self._get_video_info(url)
            
            self.log_processing_end("YouTube processing", True)
            
            return {
                "text": f"Video: {video_info.get('title', 'Unknown')} by {video_info.get('uploader', 'Unknown')}",
                "metadata": video_info,
                "source": url,
                "type": "youtube",
                "duration": video_info.get("duration"),
                "title": video_info.get("title"),
                "author": video_info.get("uploader"),
                "view_count": video_info.get("view_count"),
                "upload_date": video_info.get("upload_date"),
            }
            
        except Exception as e:
            self.log_processing_end("YouTube processing", False)
            self.handle_error(e, "YouTube processing")
            raise
    
    def _is_valid_youtube_url(self, url: str) -> bool:
        """Check if URL is a valid YouTube URL."""
        youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in youtube_domains)
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        parsed = urlparse(url)
        
        if 'youtube.com' in parsed.netloc:
            if parsed.path == '/watch':
                return parse_qs(parsed.query).get('v', [None])[0]
        elif 'youtu.be' in parsed.netloc:
            return parsed.path[1:]  # Remove leading slash
        
        return None
    
    def _get_video_info(self, url: str) -> Dict[str, Any]:
        """Get basic video info without downloading."""
        if not YT_DLP_AVAILABLE:
            raise VideoProcessingError("YouTube processing dependencies not available")
        
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    "title": info.get("title", "Unknown"),
                    "uploader": info.get("uploader", "Unknown"),
                    "duration": info.get("duration", 0),
                    "view_count": info.get("view_count", 0),
                    "upload_date": info.get("upload_date", ""),
                    "description": info.get("description", "")[:200],
                }
        except Exception as e:
            raise VideoProcessingError(f"Failed to get video info: {e}")
    
    def _download_video(self, url: str) -> Path:
        """Download YouTube video to temporary file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            temp_path = Path(tmp_file.name)
        
        try:
            ydl_opts = self.ydl_opts.copy()
            ydl_opts['outtmpl'] = str(temp_path)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # Check if file was downloaded successfully
            if temp_path.exists() and temp_path.stat().st_size > 0:
                return temp_path
            else:
                raise VideoProcessingError("Failed to download video")
                
        except Exception as e:
            # Clean up on failure
            if temp_path.exists():
                temp_path.unlink()
            raise VideoProcessingError(f"Failed to download video: {e}")
    
    def _extract_metadata(self, url: str, video_path: Path) -> Dict[str, Any]:
        """Extract metadata from YouTube video."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    "title": info.get("title", "Unknown"),
                    "uploader": info.get("uploader", "Unknown"),
                    "duration": info.get("duration", 0),
                    "view_count": info.get("view_count", 0),
                    "upload_date": info.get("upload_date", ""),
                    "description": info.get("description", "")[:500],  # Limit description length
                    "tags": info.get("tags", []),
                    "categories": info.get("categories", []),
                }
        except Exception as e:
            # Return basic metadata if extraction fails
            return {
                "title": "Unknown",
                "uploader": "Unknown",
                "duration": 0,
                "view_count": 0,
                "upload_date": "",
                "description": "",
                "tags": [],
                "categories": [],
            }
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video information without downloading."""
        if not YT_DLP_AVAILABLE:
            raise VideoProcessingError("YouTube processing dependencies not available")
        
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    "title": info.get("title", "Unknown"),
                    "uploader": info.get("uploader", "Unknown"),
                    "duration": info.get("duration", 0),
                    "view_count": info.get("view_count", 0),
                    "upload_date": info.get("upload_date", ""),
                    "thumbnail": info.get("thumbnail", ""),
                    "description": info.get("description", "")[:200],
                }
        except Exception as e:
            raise VideoProcessingError(f"Failed to get video info: {e}")
