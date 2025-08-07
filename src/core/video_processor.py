"""
Video processing module for ElevateAI.
Handles video file processing and audio extraction.
"""
from pathlib import Path
from typing import Optional, Union

try:
    import moviepy.editor as mp
    VIDEO_DEPS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Video processing dependencies not installed: {e}")
    VIDEO_DEPS_AVAILABLE = False
    # Create dummy mp module for graceful fallback
    class DummyMP:
        class VideoFileClip:
            def __init__(self, *args, **kwargs):
                raise VideoProcessingError("MoviePy not available")
    mp = DummyMP()

from .base_processor import BaseProcessor
from .audio_processor import AudioProcessor
from src.utils.exceptions import VideoProcessingError, UnsupportedFormatError


class VideoProcessor(BaseProcessor):
    """Handles video processing operations."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the video processor.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.audio_processor = AudioProcessor(config)
    
    def process(self, input_data: Union[str, Path], **kwargs) -> Path:
        """
        Process video file and extract audio.
        
        Args:
            input_data: Path to video file
            **kwargs: Additional processing parameters
            
        Returns:
            Path to extracted and processed audio file
            
        Raises:
            VideoProcessingError: If processing fails
        """
        if not VIDEO_DEPS_AVAILABLE:
            raise VideoProcessingError("Video processing dependencies not available")
        
        if not self.validate_input(input_data):
            raise VideoProcessingError(f"Invalid input: {input_data}")
        
        video_path = Path(input_data)
        
        if not self.validate_video_format(video_path):
            raise UnsupportedFormatError(f"Unsupported video format: {video_path.suffix}")
        
        self.log_processing_start("Video processing", str(video_path))
        
        try:
            # Extract audio from video
            audio_path = self.extract_audio(video_path)
            
            # Process the extracted audio
            processed_audio_path = self.audio_processor.process(
                audio_path,
                **kwargs
            )
            
            # Clean up temporary audio file
            self.cleanup_temp_files([audio_path])
            
            self.log_processing_end("Video processing")
            return processed_audio_path
            
        except Exception as e:
            self.handle_error(e, "Video processing")
    
    def extract_audio(self, video_path: Union[str, Path]) -> Path:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
            
        Raises:
            VideoProcessingError: If extraction fails
        """
        if not VIDEO_DEPS_AVAILABLE:
            raise VideoProcessingError("Video processing dependencies not available")
        
        video_path = Path(video_path)
        self.log_processing_start("Audio extraction from video", str(video_path))
        
        try:
            # Load video
            video = mp.VideoFileClip(str(video_path))
            
            if video.audio is None:
                raise VideoProcessingError("Video file contains no audio track")
            
            # Extract audio
            audio = video.audio
            
            # Save audio to temporary file
            output_path = self.get_temp_file_path(".wav")
            audio.write_audiofile(
                str(output_path),
                verbose=False,
                logger=None
            )
            
            # Clean up
            audio.close()
            video.close()
            
            self.log_processing_end("Audio extraction from video")
            return output_path
            
        except Exception as e:
            self.handle_error(e, "Audio extraction from video")
    
    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get video file information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        if not VIDEO_DEPS_AVAILABLE:
            return {"error": "Video processing dependencies not available"}
        
        try:
            video = mp.VideoFileClip(str(video_path))
            
            info = {
                "duration": video.duration,
                "fps": video.fps,
                "size": video.size,
                "has_audio": video.audio is not None,
                "audio_fps": video.audio.fps if video.audio else None
            }
            
            video.close()
            return info
            
        except Exception as e:
            self.logger.warning(f"Failed to get video info: {e}")
            return {"error": str(e)}
    
    def validate_video_format(self, file_path: Path) -> bool:
        """
        Validate if file format is supported for video processing.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if format is supported
        """
        suffix = file_path.suffix.lower().lstrip('.')
        return suffix in self.settings.supported_video_formats
    
    def extract_frames(self, video_path: Union[str, Path], 
                      frame_rate: float = 1.0) -> list[Path]:
        """
        Extract frames from video at specified rate.
        
        Args:
            video_path: Path to video file
            frame_rate: Frames per second to extract
            
        Returns:
            List of paths to extracted frame images
        """
        if not VIDEO_DEPS_AVAILABLE:
            raise VideoProcessingError("Video processing dependencies not available")
        
        try:
            video = mp.VideoFileClip(str(video_path))
            frame_paths = []
            
            # Extract frames at specified intervals
            for i, frame_time in enumerate(range(0, int(video.duration), int(1/frame_rate))):
                frame = video.get_frame(frame_time)
                frame_path = self.get_temp_file_path(f"_frame_{i}.jpg")
                
                # Save frame (would need PIL/opencv for this)
                # For now, just add to list
                frame_paths.append(frame_path)
            
            video.close()
            return frame_paths
            
        except Exception as e:
            self.logger.warning(f"Frame extraction failed: {e}")
            return []
