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


def _process_file_to_text(tmp_path: Path) -> Optional[str]:
    ctx = get_context()
    file_extension = tmp_path.suffix.lower()

    if file_extension in [".mp4", ".avi", ".mov", ".mkv"]:
        audio_path = ctx["video_processor"].extract_audio(tmp_path)
        transcript_result = ctx["speech_processor"].process(audio_path)
        return transcript_result["text"]
    elif file_extension in [".mp3", ".wav", ".m4a"]:
        processed_audio = ctx["audio_processor"].process(tmp_path)
        transcript_result = ctx["speech_processor"].process(processed_audio)
        return transcript_result["text"]
    elif file_extension in [".pdf", ".docx", ".txt"]:
        return ctx["document_processor"].process(tmp_path)
    else:
        logger.warning(f"Unsupported file type: {file_extension}")
        return None


def ingest_uploaded_files(notebook_id: str, uploaded_files: List, *, clean_and_chunk: bool = True) -> int:
    """Ingest uploaded files into vector DB with metadata containing notebook_id and source info.

    Returns number of chunks added.
    """
    ctx = get_context()
    added_chunks = 0
    processed_texts: List[Dict[str, Any]] = []

    for file in uploaded_files or []:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = Path(tmp_file.name)
        try:
            text = _process_file_to_text(tmp_path)
            if not text:
                continue
            processed_texts.append({
                "text": text,
                "source": file.name,
                "type": file.type,
                "notebook_id": notebook_id,
            })
        finally:
            if tmp_path.exists():
                os.unlink(tmp_path)

    added_chunks += _commit_processed_texts(processed_texts)
    return added_chunks


def ingest_url(notebook_id: str, url: str) -> int:
    ctx = get_context()
    if not url:
        return 0
    if "youtube.com" in url or "youtu.be" in url:
        # YouTube pipeline can be implemented later
        return 0
    text = ctx["document_processor"].extract_text_from_url(url)
    return _commit_processed_texts([
        {"text": text, "source": url, "type": "url", "notebook_id": notebook_id}
    ])


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


