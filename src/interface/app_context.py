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
    # Import YouTube processor separately to avoid dependency issues
    from src.interface.notebooks.youtube_simple import SimpleYouTubeProcessor
    from src.analysis import TextAnalyzer, TextCleaner, TextChunker
    from src.database import VectorDatabase, EmbeddingGenerator
    from src.search import SemanticSearchEngine, RetrievalEngine, WebSearchEngine
    from src.ai import LLMClient, PromptEngineer, Summarizer, MultiModalAI
    # Optional: LangChain/LangGraph (guarded via feature flags)
    from src.utils.feature_flags import feature_flags
    try:
        from src.ai.langchain import (
            LangchainPromptManager,
            LangchainLLMClient,
            LangchainMemoryManager,
        )
    except Exception:  # pragma: no cover
        LangchainPromptManager = None  # type: ignore
        LangchainLLMClient = None  # type: ignore
        LangchainMemoryManager = None  # type: ignore
    # LangGraph is disabled (we're using LangChain only)
    create_qa_workflow = None  # type: ignore
    create_summarization_workflow = None  # type: ignore

    context: Dict[str, Any] = {}

    # Core processors
    context["audio_processor"] = AudioProcessor()
    context["video_processor"] = VideoProcessor()
    context["document_processor"] = DocumentProcessor()
    context["speech_processor"] = SpeechToTextProcessor()
    context["youtube_processor"] = SimpleYouTubeProcessor()

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

    # Legacy AI components (kept for compatibility but not primary)
    try:
        context["legacy_llm_client"] = LLMClient()
    except Exception as e:  # pragma: no cover
        logger.warning(f"Legacy LLM client initialization failed in context: {e}")
        context["legacy_llm_client"] = None

    try:
        context["legacy_prompt_engineer"] = PromptEngineer()
        context["legacy_summarizer"] = Summarizer()
    except Exception as e:  # pragma: no cover
        logger.warning(f"Legacy text processing components failed: {e}")
        context["legacy_prompt_engineer"] = None
        context["legacy_summarizer"] = None

    # LangChain components (primary AI system)
    try:
        context["prompt_manager"] = LangchainPromptManager()
        context["llm_client"] = LangchainLLMClient()
        if feature_flags.is_enabled("enable_memory"):
            # Pass underlying LC LLM into memory manager for ConversationSummaryMemory
            lc_llm = getattr(context["llm_client"], "llm", None)
            context["memory_manager"] = LangchainMemoryManager(llm=lc_llm)
        # Add output parser for structured responses
        try:
            from src.ai.langchain import LangchainOutputParser
            context["output_parser"] = LangchainOutputParser()
        except Exception:
            context["output_parser"] = None
    except Exception as e:  # pragma: no cover
        logger.warning(f"LangChain components failed: {e}")

    # LangGraph workflows disabled
    context["qa_workflow"] = None
    context["summarization_workflow"] = None

    try:
        context["multimodal_ai"] = MultiModalAI()
    except Exception as e:  # pragma: no cover
        logger.warning(f"Multimodal AI initialization failed: {e}")
        context["multimodal_ai"] = None

    # TTS component
    try:
        from src.ai.tts_client import TTSClient
        logger.info("Initializing TTS client...")
        tts_client = TTSClient()
        logger.info(f"TTS client initialized successfully with voices: {tts_client.get_available_voices()}")
        context["tts_client"] = tts_client
    except Exception as e:  # pragma: no cover
        logger.error(f"TTS client initialization failed in context: {e}")
        import traceback
        logger.error(f"TTS client error details: {traceback.format_exc()}")
        context["tts_client"] = None

    return context


def get_context() -> Dict[str, Any]:
    """Get or create the shared app context stored in Streamlit session state."""
    if st is None:
        # Fallback for non-Streamlit environments
        return _build_context()

    if "_app_context" not in st.session_state:
        logger.info("Initializing shared app context")
        st.session_state._app_context = _build_context()
    
    # Ensure TTS client is always available
    context = st.session_state._app_context
    if context.get("tts_client") is None:
        logger.info("Re-initializing TTS client in existing context")
        try:
            from src.ai.tts_client import TTSClient
            context["tts_client"] = TTSClient()
            logger.info("TTS client re-initialized successfully")
        except Exception as e:
            logger.error(f"Failed to re-initialize TTS client: {e}")
    
    return context


