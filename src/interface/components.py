"""
UI components for ElevateAI Streamlit interface.
"""
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

from config.settings import settings
from src.utils.logger import logger


class UIComponents:
    """Reusable UI components for Streamlit interface."""
    
    @staticmethod
    def render_header():
        """Render application header."""
        st.set_page_config(
            page_title="ElevateAI - Intelligent Summarization",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🚀 ElevateAI")
        st.markdown("### Intelligent Video/Text Summarization Application")
        st.markdown("---")
    
    @staticmethod
    def get_current_settings() -> Dict[str, Any]:
        """Lấy settings hiện tại từ session state hoặc file cấu hình."""
        # Load from file first
        try:
            from pathlib import Path
            import json
            user_config_file = Path("config/user_settings.json")
            if user_config_file.exists():
                with open(user_config_file, 'r', encoding='utf-8') as f:
                    file_settings = json.load(f)
            else:
                file_settings = {}
        except Exception:
            file_settings = {}
        
        # Load from session state
        session_settings = st.session_state.get('app_settings', {})
        
        # Default settings
        default_settings = {
            'temperature': 0.7,
            'max_tokens': 2000,
            'top_p': 0.9,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'model_name': 'gpt-4',
            'similarity_threshold': 0.25,
            'max_results': 10,
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'enable_web_search': False,
            'enable_function_calling': False,
            'enable_tts': True,
            'tts_voice': 'alloy',
            'audio_sample_rate': 16000,
            'noise_reduction': 0.8,
            'enable_vocal_separation': False,
            'enable_memory': True,
            'max_memory_context': 3,
            'memory_consolidation_threshold': 10,
            'store_conversations': True,
            'memory_retention_days': 30,
            'auto_cleanup': True,
            'theme': 'light',
            'language': 'vi',
            'auto_save': True,
            'show_processing_time': True,
            'show_confidence_score': True,
            'enable_animations': True,
            'max_file_size_mb': 500,
            'enable_debug_mode': False,
            'cache_enabled': True,
            'log_level': 'INFO',
            'enable_metrics': True,
            'backup_enabled': True
        }
        
        # Merge settings (file settings > session settings > default settings)
        merged_settings = {**default_settings, **session_settings, **file_settings}
        return merged_settings

    @staticmethod
    def render_sidebar():
        """Render sidebar with configuration options."""
        # Get current settings
        current_settings = UIComponents.get_current_settings()
        
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Quick settings display
            st.markdown("**Current Settings:**")
            st.markdown(f"• Temperature: {current_settings.get('temperature', 0.7):.1f}")
            st.markdown(f"• Max Tokens: {current_settings.get('max_tokens', 2000)}")
            st.markdown(f"• Similarity: {current_settings.get('similarity_threshold', 0.25):.2f}")
            
            # Link to settings page
            if st.button("⚙️ Open Settings", use_container_width=True):
                st.query_params["page"] = "Settings"
                st.rerun()
            
            st.markdown("---")
            
            # Model settings (read-only display)
            st.subheader("Model Settings")
            st.markdown(f"**Temperature:** {current_settings.get('temperature', 0.7):.1f}")
            st.markdown(f"**Max Tokens:** {current_settings.get('max_tokens', 2000)}")
            st.markdown(f"**Model:** {current_settings.get('model_name', 'gpt-4')}")
            
            # Search settings (read-only display)
            st.subheader("Search Settings")
            st.markdown(f"**Similarity Threshold:** {current_settings.get('similarity_threshold', 0.25):.2f}")
            st.markdown(f"**Max Results:** {current_settings.get('max_results', 10)}")
            st.markdown(f"**Web Search:** {'✅' if current_settings.get('enable_web_search', False) else '❌'}")
            st.markdown(f"**Function Calling:** {'✅' if current_settings.get('enable_function_calling', False) else '❌'}")
            
            # Multi-modal settings (read-only display)
            st.subheader("Multi-modal Features")
            st.markdown(f"**TTS:** {'✅' if current_settings.get('enable_tts', True) else '❌'}")
            st.markdown(f"**TTS Voice:** {current_settings.get('tts_voice', 'alloy')}")
            
            # Memory settings (read-only display)
            st.subheader("Memory Settings")
            st.markdown(f"**Memory System:** {'✅' if current_settings.get('enable_memory', True) else '❌'}")
            st.markdown(f"**Max Context:** {current_settings.get('max_memory_context', 3)}")
            st.markdown(f"**Store Conversations:** {'✅' if current_settings.get('store_conversations', True) else '❌'}")

            return current_settings
    
    @staticmethod
    def render_file_uploader():
        """Render file upload interface."""
        st.subheader("📁 Upload Content")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to process",
            type=['mp4', 'avi', 'mov', 'mp3', 'wav', 'pdf', 'docx', 'txt', 'xlsx'],
            accept_multiple_files=True,
            help="Upload video, audio, or document files"
        )
        
        # URL input
        url_input = st.text_input(
            "Or enter a YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter a YouTube video URL to process"
        )
        
        return uploaded_files, url_input
    
    @staticmethod
    def render_query_interface():
        """Render query input interface."""
        st.subheader("💬 Ask Questions or Request Summary")
        
        # Query input
        query = st.text_area(
            "Enter your question or request:",
            placeholder="Summarize the main points of the video...\nWhat are the key insights about...?",
            height=100,
            help="Ask specific questions or request a general summary"
        )
        
        # Query type selection
        query_type = st.selectbox(
            "Query Type:",
            options=[
                "General Summary",
                "Specific Question",
                "Key Points",
                "Analysis",
                "Comparison"
            ],
            help="Select the type of response you want"
        )
        
        # Processing options
        col1, col2 = st.columns(2)
        
        with col1:
            use_chain_of_thought = st.checkbox(
                "Use Chain of Thought",
                value=True,
                help="Enable step-by-step reasoning"
            )
        
        with col2:
            include_sources = st.checkbox(
                "Include Source Attribution",
                value=True,
                help="Show sources for information"
            )
        
        return {
            'query': query,
            'query_type': query_type,
            'use_chain_of_thought': use_chain_of_thought,
            'include_sources': include_sources
        }
    
    @staticmethod
    def render_processing_status(status_text: str, progress: float = None):
        """Render processing status."""
        if progress is not None:
            st.progress(progress)
        
        with st.spinner(status_text):
            time.sleep(0.1)  # Small delay for visual effect
    
    @staticmethod
    def render_results(summary: str, sources: List[Dict[str, Any]], 
                      confidence_score: float, processing_time: float):
        """Render processing results."""
        st.subheader("📋 Results")
        
        # Summary
        st.markdown("### Summary")
        st.markdown(summary)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence Score", f"{confidence_score:.2f}")
        
        with col2:
            st.metric("Processing Time", f"{processing_time:.1f}s")
        
        with col3:
            st.metric("Sources Used", len(sources))
        
        # Sources
        if sources:
            st.markdown("### 📚 Sources")
            
            for i, source in enumerate(sources, 1):
                with st.expander(f"Source {i}: {source.get('title', 'Unknown')}"):
                    st.markdown(f"**Type:** {source.get('type', 'Unknown')}")
                    st.markdown(f"**Score:** {source.get('score', 0):.3f}")
                    
                    if 'content' in source:
                        st.markdown("**Content Preview:**")
                        st.text(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])
                    
                    if source.get('is_web_result'):
                        st.warning("⚠️ This is a web search result. Please verify the information.")
    
    @staticmethod
    def render_download_options(summary: str, audio_data: bytes = None, 
                               image_data: bytes = None):
        """Render download options."""
        st.subheader("💾 Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Text download
            st.download_button(
                label="📄 Download Summary (TXT)",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
        
        with col2:
            # Audio download
            if audio_data:
                st.download_button(
                    label="🔊 Download Audio (MP3)",
                    data=audio_data,
                    file_name="summary_audio.mp3",
                    mime="audio/mpeg"
                )
            else:
                st.button("🔊 Generate Audio", disabled=True, help="Audio generation not available")
        
        with col3:
            # Image download
            if image_data:
                st.download_button(
                    label="🖼️ Download Image (PNG)",
                    data=image_data,
                    file_name="summary_visual.png",
                    mime="image/png"
                )
            else:
                st.button("🖼️ Generate Visual", disabled=True, help="Image generation not available")
    
    @staticmethod
    def render_error(error_message: str):
        """Render error message."""
        st.error(f"❌ Error: {error_message}")
    
    @staticmethod
    def render_warning(warning_message: str):
        """Render warning message."""
        st.warning(f"⚠️ Warning: {warning_message}")
    
    @staticmethod
    def render_success(success_message: str):
        """Render success message."""
        st.success(f"✅ {success_message}")
    
    @staticmethod
    def render_info(info_message: str):
        """Render info message."""
        st.info(f"ℹ️ {info_message}")
    
    @staticmethod
    def render_database_stats(stats: Dict[str, Any]):
        """Render database statistics."""
        with st.expander("📊 Database Statistics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Documents", stats.get('total_vectors', 0))
            
            with col2:
                st.metric("Embedding Dimension", stats.get('embedding_dimension', 0))
            
            with col3:
                st.metric("Index Type", stats.get('index_type', 'Unknown'))
    
    @staticmethod
    def render_memory_stats():
        """Render memory system statistics."""
        try:
            from src.utils.memory import memory_manager

            with st.expander("🧠 Memory System"):
                stats = memory_manager.get_memory_stats()

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Short-term Memory")
                    short_stats = stats.get('short_term', {})
                    st.metric("Active Entries", short_stats.get('total_entries', 0))
                    st.metric("Conversation Turns", short_stats.get('conversation_turns', 0))
                    st.metric("Context Keys", short_stats.get('context_keys', 0))

                with col2:
                    st.subheader("Long-term Memory")
                    long_stats = stats.get('long_term', {})
                    st.metric("Stored Memories", long_stats.get('total_entries', 0))
                    if long_stats.get('avg_importance'):
                        st.metric("Avg Importance", f"{long_stats['avg_importance']:.2f}")
                    st.metric("Disk Files", long_stats.get('disk_files', 0))

                # Memory controls
                st.subheader("Memory Controls")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("🧹 Clear Short-term"):
                        memory_manager.short_term.clear_context()
                        st.success("Short-term memory cleared")

                with col2:
                    if st.button("🔄 Consolidate"):
                        memory_manager._consolidate_memories()
                        st.success("Memory consolidated")

                with col3:
                    if st.button("🗑️ Cleanup Old"):
                        cleaned = memory_manager.cleanup_memories()
                        st.success(f"Cleaned {cleaned['long_term_cleaned']} old memories")

        except Exception as e:
            st.error(f"Failed to load memory stats: {e}")

    @staticmethod
    def render_conversation_history():
        """Render conversation history from memory."""
        try:
            from src.utils.memory import memory_manager

            conversation_context = memory_manager.get_conversation_context(max_turns=10)

            if conversation_context:
                with st.expander("💬 Conversation History"):
                    for i, turn in enumerate(reversed(conversation_context), 1):
                        st.markdown(f"**Turn {i}** _{turn.timestamp.strftime('%H:%M:%S')}_")
                        st.markdown(f"**You:** {turn.user_input}")
                        st.markdown(f"**Assistant:** {turn.assistant_response[:200]}...")
                        st.markdown(f"*Confidence: {turn.confidence_score:.2f}, Time: {turn.processing_time:.1f}s*")
                        st.markdown("---")

        except Exception as e:
            st.error(f"Failed to load conversation history: {e}")

    @staticmethod
    def render_processing_history():
        """Render processing history."""
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []

        if st.session_state.processing_history:
            with st.expander("📜 Processing History"):
                for i, entry in enumerate(reversed(st.session_state.processing_history[-5:]), 1):
                    st.markdown(f"**{i}.** {entry.get('query', 'Unknown query')}")
                    st.markdown(f"   *Processed at: {entry.get('timestamp', 'Unknown time')}*")
                    st.markdown("---")
    
    @staticmethod
    def add_to_history(query: str, summary: str):
        """Add entry to processing history."""
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
        
        entry = {
            'query': query,
            'summary': summary[:100] + "..." if len(summary) > 100 else summary,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.session_state.processing_history.append(entry)
        
        # Keep only last 10 entries
        if len(st.session_state.processing_history) > 10:
            st.session_state.processing_history = st.session_state.processing_history[-10:]
