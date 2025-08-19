"""
Notebook ingest utilities: add sources into the vector DB with notebook-aware metadata.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import os

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None

from src.utils.logger import logger
from src.interface.app_context import get_context


def _process_file_to_text(tmp_path: Path, processors: Dict[str, Any]) -> Optional[str]:
    """Process a file to extract text using the provided processors.
    
    Args:
        tmp_path: Path to the temporary file
        processors: Dictionary of processors from the app context
        
    Returns:
        Extracted text or None if processing fails
    """
    file_extension = tmp_path.suffix.lower()

    try:
        if file_extension in [".mp4", ".avi", ".mov", ".mkv"]:
            audio_path = processors["video_processor"].extract_audio(tmp_path)
            transcript_result = processors["speech_processor"].process(audio_path)
            return transcript_result["text"]
        elif file_extension in [".mp3", ".wav", ".m4a"]:
            processed_audio = processors["audio_processor"].process(tmp_path)
            transcript_result = processors["speech_processor"].process(processed_audio)
            return transcript_result["text"]
        elif file_extension in [".pdf", ".docx", ".txt"]:
            return processors["document_processor"].process(tmp_path)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return None
    except Exception as e:
        logger.error(f"File processing failed for {tmp_path}: {str(e)}")
        return None


def ingest_uploaded_files(notebook_id: str, uploaded_files: List, *, clean_and_chunk: bool = True) -> int:
    """Ingest uploaded files into vector DB with metadata containing notebook_id and source info.

    Returns number of chunks added.
    """
    from src.utils.batch_processor import BatchProcessor
    
    # Get app context once and reuse it
    ctx = get_context()
    added_chunks = 0
    processed_texts: List[Dict[str, Any]] = []
    
    # Extract all needed processors from context to avoid thread issues
    document_processor = ctx["document_processor"]
    video_processor = ctx.get("video_processor")
    audio_processor = ctx.get("audio_processor")
    speech_processor = ctx.get("speech_processor")
    
    # Create a batch processor for parallel file processing
    batch_processor = BatchProcessor(max_workers=min(4, len(uploaded_files)))
    
    def process_single_file(file):
        # Create processors dictionary for thread-safe access
        processors = {
            "document_processor": document_processor,
            "video_processor": video_processor,
            "audio_processor": audio_processor,
            "speech_processor": speech_processor
        }
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = Path(tmp_file.name)
        try:
            text = _process_file_to_text(tmp_path, processors)
            if text:
                return {
                    "text": text,
                    "source": file.name,
                    "type": file.type,
                    "notebook_id": notebook_id,
                }
            return None
        finally:
            if tmp_path.exists():
                os.unlink(tmp_path)
    
    # Process files in parallel
    results = batch_processor.process_items(uploaded_files or [], process_single_file)
    
    # Collect successful results
    for result in results:
        if result.success and result.result:
            processed_texts.append(result.result)

    added_chunks += _commit_processed_texts(processed_texts)
    return added_chunks


def ingest_url(notebook_id: str, url: str) -> int:
    ctx = get_context()
    if not url:
        return 0
    
    # Check if it's a YouTube URL
    if "youtube.com" in url or "youtu.be" in url:
        try:
            # Process YouTube video
            youtube_result = ctx["youtube_processor"].process(url)
            
            if youtube_result and youtube_result.get("text"):
                # Add metadata for better context
                metadata = youtube_result.get("metadata", {})
                source_info = {
                    "text": youtube_result["text"],
                    "source": url,
                    "type": "youtube",
                    "notebook_id": notebook_id,
                    "youtube_title": metadata.get("title", ""),
                    "youtube_author": metadata.get("uploader", ""),
                    "youtube_duration": metadata.get("duration", 0),
                    "youtube_upload_date": metadata.get("upload_date", ""),
                }
                
                return _commit_processed_texts([source_info])
            else:
                # If no transcript available, try to get basic info
                video_info = ctx["youtube_processor"].get_video_info(url)
                basic_text = f"Video: {video_info.get('title', 'Unknown')} by {video_info.get('uploader', 'Unknown')}. Duration: {video_info.get('duration', 0)} seconds. Upload date: {video_info.get('upload_date', 'Unknown')}."
                
                return _commit_processed_texts([
                    {"text": basic_text, "source": url, "type": "youtube", "notebook_id": notebook_id}
                ])
                
        except Exception as e:
            # Fallback to basic URL processing if YouTube processing fails
            logger.warning(f"YouTube processing failed for {url}: {e}")
            try:
                text = ctx["document_processor"].extract_text_from_url(url)
                return _commit_processed_texts([
                    {"text": text, "source": url, "type": "url", "notebook_id": notebook_id}
                ])
            except Exception:
                return 0
    
    # Process regular URLs
    try:
        text = ctx["document_processor"].extract_text_from_url(url)
        return _commit_processed_texts([
            {"text": text, "source": url, "type": "url", "notebook_id": notebook_id}
        ])
    except Exception as e:
        logger.warning(f"URL processing failed for {url}: {e}")
        return 0


def _commit_processed_texts(processed_texts: List[Dict[str, Any]]) -> int:
    ctx = get_context()
    all_chunks: List[str] = []
    all_metadata: List[Dict[str, Any]] = []

    for item in processed_texts:
        cleaned_text = ctx["text_cleaner"].clean_text(item["text"])
        analysis = ctx["text_analyzer"].analyze_text(cleaned_text)
        if getattr(analysis, "status", None) == "no_content":
            continue
        chunks = ctx["text_chunker"].split_into_chunks(cleaned_text)
        for chunk in chunks:
            all_chunks.append(chunk.content)
            meta = {
                "source": item["source"],
                "content_type": item["type"],
                "chunk_id": chunk.chunk_id,
                "word_count": chunk.word_count,
                "analysis_status": getattr(analysis, "status", "processed"),
                "notebook_id": item.get("notebook_id"),
                # Store the raw text preview for better search experience in UI
                "text": chunk.content[:500],
            }
            all_metadata.append(meta)

    if not all_chunks:
        return 0

    # Ensure embeddings are available
    try:
        eg = ctx.get("vector_db").embedding_generator
        if hasattr(eg, "has_model") and not eg.has_model():  # type: ignore[attr-defined]
            return 0
    except Exception:
        pass

    ctx["vector_db"].add_to_vectordb(all_chunks, all_metadata)
    try:
        ctx["vector_db"].save_database()
    except Exception:
        logger.warning("Failed to save vector DB after ingest")
    return len(all_chunks)


