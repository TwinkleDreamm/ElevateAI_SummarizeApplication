"""
Streamlit web application for ElevateAI.
"""
import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

# Import core modules
from src.core.document_processor import DocumentProcessor
from src.core.speech_to_text import SpeechToTextProcessor
from src.ai.summarizer import Summarizer
from src.search.semantic_search import SemanticSearchEngine
from src.database.vector_database import VectorDatabase
from src.utils.logger import logger
from src.utils.memory import MemoryManager
from config.settings import settings


class StreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.setup_page_config()
        self.initialize_components()
        self.setup_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="ElevateAI",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_components(self):
        """Initialize core components."""
        try:
            self.document_processor = DocumentProcessor()
            self.stt_processor = SpeechToTextProcessor()
            self.summarizer = Summarizer()
            self.search_engine = SemanticSearchEngine()
            self.vector_db = VectorDatabase()
            self.memory_manager = MemoryManager()
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            logger.error(f"Component initialization failed: {e}")
    
    def setup_session_state(self):
        """Initialize session state variables."""
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
    def render_sidebar(self):
        """Render the sidebar with navigation and settings."""
        with st.sidebar:
            st.title("🚀 ElevateAI")
            st.markdown("---")
            
            # Navigation
            page = st.selectbox(
                "Choose a feature:",
                ["📄 Document Processing", "🔍 Search & Summarize", "🎤 Speech to Text", "💾 Database Management"]
            )
            
            st.markdown("---")
            
            # Settings
            st.subheader("⚙️ Settings")
            
            # Model settings
            use_openai = st.checkbox("Use OpenAI Models", value=True, help="Use online OpenAI models for better performance")
            
            if use_openai:
                st.info("🌐 Using online models (faster, no downloads)")
            else:
                st.warning("💾 Using local models (slower startup)")
            
            # Database stats
            try:
                stats = self.vector_db.get_database_stats()
                st.subheader("📊 Database Stats")
                st.metric("Documents", stats.get('total_vectors', 0))
                st.metric("Embedding Dim", stats.get('embedding_dimension', 0))
            except Exception as e:
                st.error(f"Failed to load stats: {e}")
            
            return page
    
    def render_document_processing(self):
        """Render document processing interface."""
        st.header("📄 Document Processing")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Choose files",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'txt', 'mp4', 'mp3', 'wav']
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if st.button(f"Process {uploaded_file.name}"):
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            try:
                                # Save uploaded file temporarily
                                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    tmp_path = tmp_file.name
                                
                                # Process the file
                                result = self.document_processor.process_file(tmp_path)
                                
                                # Add to vector database
                                if result.chunks:
                                    metadata_list = [{"source": uploaded_file.name, "chunk_id": i} for i in range(len(result.chunks))]
                                    self.vector_db.add_to_vectordb(result.chunks, metadata_list)
                                
                                st.success(f"✅ Processed {uploaded_file.name} - {len(result.chunks)} chunks added")
                                st.session_state.processed_files.append(uploaded_file.name)
                                
                                # Clean up
                                os.unlink(tmp_path)
                                
                            except Exception as e:
                                st.error(f"❌ Error processing {uploaded_file.name}: {e}")
        
        with col2:
            st.subheader("📝 Add Text Directly")
            text_input = st.text_area("Enter text to add to database:", height=200)
            
            if st.button("Add Text"):
                if text_input.strip():
                    try:
                        # Split into chunks if needed
                        chunks = [text_input.strip()]
                        metadata_list = [{"source": "manual_input", "type": "text"}]
                        
                        self.vector_db.add_to_vectordb(chunks, metadata_list)
                        st.success("✅ Text added to database")
                    except Exception as e:
                        st.error(f"❌ Error adding text: {e}")
                else:
                    st.warning("Please enter some text")
    
    def render_search_summarize(self):
        """Render search and summarization interface."""
        st.header("🔍 Search & Summarize")
        
        # Search section
        st.subheader("Search Documents")
        query = st.text_input("Enter your search query:")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_button = st.button("🔍 Search", type="primary")
        with col2:
            k = st.number_input("Results", min_value=1, max_value=50, value=10)
        with col3:
            threshold = st.number_input("Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        
        if search_button and query:
            with st.spinner("Searching..."):
                try:
                    results = self.search_engine.search(query, k=k, threshold=threshold)
                    st.session_state.search_results = results
                    
                    if results:
                        st.success(f"Found {len(results)} results")
                        
                        # Display results
                        for i, result in enumerate(results):
                            with st.expander(f"Result {i+1} (Score: {result['score']:.3f})"):
                                st.write(result['metadata'].get('text', 'No text available'))
                                st.caption(f"Source: {result['metadata'].get('source', 'Unknown')}")
                    else:
                        st.warning("No results found")
                        
                except Exception as e:
                    st.error(f"Search failed: {e}")
        
        # Summarization section
        st.markdown("---")
        st.subheader("Summarize Results")
        
        if st.session_state.search_results:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                focus_query = st.text_input("Focus topic (optional):", placeholder="What aspect to focus on?")
            
            with col2:
                summarize_button = st.button("📝 Summarize", type="primary")
            
            if summarize_button:
                with st.spinner("Generating summary..."):
                    try:
                        # Prepare chunks for summarization
                        chunks = []
                        for result in st.session_state.search_results:
                            chunks.append({
                                "text": result['metadata'].get('text', ''),
                                "source": result['metadata'].get('source', 'Unknown')
                            })
                        
                        # Generate summary
                        summary_result = self.summarizer.summarize_chunks(
                            chunks, 
                            query=focus_query if focus_query else query
                        )
                        
                        # Display summary
                        st.markdown("### 📋 Summary")
                        st.markdown(summary_result.summary)
                        
                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Word Count", getattr(summary_result, 'word_count', 'N/A'))
                        with col2:
                            st.metric("Compression", f"{getattr(summary_result, 'compression_ratio', 0):.1%}")
                        with col3:
                            st.metric("Sources", len(chunks))
                        
                    except Exception as e:
                        st.error(f"Summarization failed: {e}")
        else:
            st.info("Search for documents first to enable summarization")
    
    def render_speech_to_text(self):
        """Render speech-to-text interface."""
        st.header("🎤 Speech to Text")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Audio File")
            audio_file = st.file_uploader(
                "Choose an audio file",
                type=['mp3', 'wav', 'mp4', 'm4a', 'ogg']
            )
            
            if audio_file:
                st.audio(audio_file)
                
                if st.button("🎤 Transcribe", type="primary"):
                    with st.spinner("Transcribing audio..."):
                        try:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
                                tmp_file.write(audio_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            # Process audio
                            result = self.stt_processor.process(tmp_path)
                            
                            # Display results
                            st.subheader("📝 Transcription")
                            st.text_area("Transcribed text:", value=result.get('text', ''), height=200)
                            
                            # Add to database option
                            if st.button("Add transcription to database"):
                                chunks = [result.get('text', '')]
                                metadata_list = [{"source": audio_file.name, "type": "transcription"}]
                                self.vector_db.add_to_vectordb(chunks, metadata_list)
                                st.success("✅ Transcription added to database")
                            
                            # Clean up
                            os.unlink(tmp_path)
                            
                        except Exception as e:
                            st.error(f"Transcription failed: {e}")
        
        with col2:
            st.subheader("ℹ️ Info")
            st.info("""
            **Supported formats:**
            - MP3, WAV, MP4, M4A, OGG
            
            **Features:**
            - Automatic language detection
            - High-quality transcription
            - Direct database integration
            """)
    
    def render_database_management(self):
        """Render database management interface."""
        st.header("💾 Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Database Statistics")
            try:
                stats = self.vector_db.get_database_stats()
                
                st.metric("Total Vectors", stats.get('total_vectors', 0))
                st.metric("Embedding Dimension", stats.get('embedding_dimension', 0))
                st.metric("Index Type", stats.get('index_type', 'Unknown'))
                st.metric("Database Path", stats.get('database_path', 'Unknown'))
                
            except Exception as e:
                st.error(f"Failed to load database stats: {e}")
        
        with col2:
            st.subheader("🔧 Database Actions")
            
            if st.button("💾 Save Database", type="secondary"):
                try:
                    self.vector_db.save_database()
                    st.success("✅ Database saved successfully")
                except Exception as e:
                    st.error(f"Failed to save database: {e}")
            
            if st.button("🗑️ Clear Database", type="secondary"):
                if st.checkbox("I understand this will delete all data"):
                    try:
                        self.vector_db.clear_database()
                        st.success("✅ Database cleared successfully")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to clear database: {e}")
    
    def run(self):
        """Run the Streamlit application."""
        try:
            # Render sidebar and get selected page
            page = self.render_sidebar()
            
            # Render main content based on selected page
            if page == "📄 Document Processing":
                self.render_document_processing()
            elif page == "🔍 Search & Summarize":
                self.render_search_summarize()
            elif page == "🎤 Speech to Text":
                self.render_speech_to_text()
            elif page == "💾 Database Management":
                self.render_database_management()
            
        except Exception as e:
            st.error(f"Application error: {e}")
            logger.error(f"Streamlit app error: {e}")


def main():
    """Main entry point for the Streamlit application."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
