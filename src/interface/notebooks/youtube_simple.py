"""
Simple YouTube processing module for ElevateAI.
Handles YouTube video info extraction without complex dependencies.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union
from urllib.parse import urlparse, parse_qs

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("Warning: yt-dlp not installed. Install with: pip install yt-dlp")


class SimpleYouTubeProcessor:
    """Simple YouTube processor that avoids complex dependencies."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the YouTube processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
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
        Process YouTube URL and extract basic info.
        
        Args:
            input_data: YouTube video URL (string)
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary containing extracted content and metadata
            
        Raises:
            Exception: If processing fails
        """
        if not YT_DLP_AVAILABLE:
            raise Exception("YouTube processing dependencies not available. Install yt-dlp: pip install yt-dlp")
        
        # Convert input_data to string if it's a Path
        url = str(input_data) if isinstance(input_data, Path) else input_data
        
        if not self._is_valid_youtube_url(url):
            raise Exception(f"Invalid YouTube URL: {url}")
        
        print(f"Starting YouTube processing for: {url}")
        
        try:
            # For now, just return basic info without downloading
            # This avoids complex dependencies during import
            video_info = self._get_video_info(url)
            
            print(f"Completed YouTube processing for: {url}")
            
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
            print(f"Error in YouTube processing: {e}")
            raise e
    
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
            raise Exception("YouTube processing dependencies not available")
        
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
            raise Exception(f"Failed to get video info: {e}")
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Get video information without downloading."""
        return self._get_video_info(url)
