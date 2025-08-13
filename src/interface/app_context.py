"""
Shared application context for Streamlit pages.

Provides lazy-initialized singletons for processors, vector DB, search, and AI components,
so all Streamlit pages can reuse the same instances.
"""
from __future__ import annotations

from typing import Any, Dict

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore

from src.utils.logger import logger


def _build_context() -> Dict[str, Any]:
    """Create and return an application context dictionary with initialized components."""
    # Local imports to avoid import cycles during Streamlit startup
    from src.core import (
        AudioProcessor,
        VideoProcessor,
        DocumentProcessor,
        SpeechToTextProcessor,
    )
    from src.analysis import TextAnalyzer, TextCleaner, TextChunker
    from src.database import VectorDatabase, EmbeddingGenerator
    from src.search import SemanticSearchEngine, RetrievalEngine, WebSearchEngine
    from src.ai import LLMClient, PromptEngineer, Summarizer, MultiModalAI

    context: Dict[str, Any] = {}

    # Core processors
    context["audio_processor"] = AudioProcessor()
    context["video_processor"] = VideoProcessor()
    context["document_processor"] = DocumentProcessor()
    context["speech_processor"] = SpeechToTextProcessor()

    # Analysis
    context["text_analyzer"] = TextAnalyzer()
    context["text_cleaner"] = TextCleaner()
    context["text_chunker"] = TextChunker()

    # Database and search
    vector_db = VectorDatabase()
    context["vector_db"] = vector_db
    context["embedding_generator"] = EmbeddingGenerator()
    search_engine = SemanticSearchEngine(vector_db=vector_db)
    context["search_engine"] = search_engine
    context["retrieval_engine"] = RetrievalEngine(search_engine=search_engine)
    context["web_search"] = WebSearchEngine()

    # AI components
    try:
        context["llm_client"] = LLMClient()
    except Exception as e:  # pragma: no cover
        logger.warning(f"LLM client initialization failed in context: {e}")
        context["llm_client"] = None

    try:
        context["prompt_engineer"] = PromptEngineer()
        context["summarizer"] = Summarizer()
    except Exception as e:  # pragma: no cover
        logger.warning(f"Text processing components failed: {e}")
        context["prompt_engineer"] = None
        context["summarizer"] = None

    try:
        context["multimodal_ai"] = MultiModalAI()
    except Exception as e:  # pragma: no cover
        logger.warning(f"Multimodal AI initialization failed: {e}")
        context["multimodal_ai"] = None

    return context


def get_context() -> Dict[str, Any]:
    """Get or create the shared app context stored in Streamlit session state."""
    if st is None:
        # Fallback for non-Streamlit environments
        return _build_context()

    if "_app_context" not in st.session_state:
        logger.info("Initializing shared app context")
        st.session_state._app_context = _build_context()
    return st.session_state._app_context


