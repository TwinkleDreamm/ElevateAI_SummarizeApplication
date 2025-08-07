"""
Main Streamlit application for ElevateAI.
"""
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import time
import tempfile
import os

from config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import ElevateAIException
from src.interface.components import UIComponents

# Import core modules
from src.core import AudioProcessor, VideoProcessor, DocumentProcessor, SpeechToTextProcessor
from src.analysis import TextAnalyzer, TextCleaner, TextChunker
from src.database import VectorDatabase, EmbeddingGenerator
from src.search import SemanticSearchEngine, RetrievalEngine, WebSearchEngine
from src.ai import LLMClient, PromptEngineer, Summarizer, MultiModalAI


class StreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        """Initialize the Streamlit application."""
        if not STREAMLIT_AVAILABLE:
            raise ImportError("Streamlit not available. Please install streamlit package.")
        
        self.logger = logger
        self.settings = settings
        
        # Initialize components
        self._initialize_components()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_components(self):
        """Initialize all application components."""
        try:
            # Core processors
            self.audio_processor = AudioProcessor()
            self.video_processor = VideoProcessor()
            self.document_processor = DocumentProcessor()
            self.speech_processor = SpeechToTextProcessor()
            
            # Analysis components
            self.text_analyzer = TextAnalyzer()
            self.text_cleaner = TextCleaner()
            self.text_chunker = TextChunker()
            
            # Database and search
            self.vector_db = VectorDatabase()
            self.embedding_generator = EmbeddingGenerator()
            self.search_engine = SemanticSearchEngine()
            self.retrieval_engine = RetrievalEngine()
            self.web_search = WebSearchEngine()
            
            # AI components (with graceful fallback)
            try:
                self.llm_client = LLMClient()
                self.logger.info("LLM client initialized")
            except Exception as e:
                self.logger.warning(f"LLM client initialization failed: {e}")
                self.llm_client = None

            try:
                self.prompt_engineer = PromptEngineer()
                self.summarizer = Summarizer()
                self.logger.info("Text processing components initialized")
            except Exception as e:
                self.logger.warning(f"Text processing components failed: {e}")
                self.prompt_engineer = None
                self.summarizer = None

            try:
                self.multimodal_ai = MultiModalAI()
                self.logger.info("Multimodal AI initialized")
            except Exception as e:
                self.logger.warning(f"Multimodal AI initialization failed: {e}")
                self.multimodal_ai = None
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            st.error(f"Failed to initialize application components: {e}")
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state."""
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
        
        if 'current_summary' not in st.session_state:
            st.session_state.current_summary = None
        
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
        
        if 'vector_db_stats' not in st.session_state:
            st.session_state.vector_db_stats = {}
    
    def run(self):
        """Run the Streamlit application."""
        try:
            # Render header
            UIComponents.render_header()
            
            # Render sidebar and get configuration
            config = UIComponents.render_sidebar()
            
            # Main interface
            self._render_main_interface(config)
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            UIComponents.render_error(f"Application error: {e}")
    
    def _render_main_interface(self, config: Dict[str, Any]):
        """Render the main application interface."""
        # File upload section
        uploaded_files, url_input = UIComponents.render_file_uploader()
        
        # Process files if uploaded
        if uploaded_files or url_input:
            if st.button("🚀 Process Content", type="primary"):
                self._process_content(uploaded_files, url_input, config)
        
        # Query interface
        query_config = UIComponents.render_query_interface()
        
        # Process query if provided
        if query_config['query'] and st.button("💡 Get Answer", type="primary"):
            self._process_query(query_config, config)
        
        # Display current results
        if st.session_state.current_summary:
            self._display_results(config)
        
        # Display database stats
        if st.session_state.vector_db_stats:
            UIComponents.render_database_stats(st.session_state.vector_db_stats)
        
        # Display memory and history
        UIComponents.render_memory_stats()
        UIComponents.render_conversation_history()
        UIComponents.render_processing_history()
    
    def _process_content(self, uploaded_files: List, url_input: str, config: Dict[str, Any]):
        """Process uploaded content."""
        try:
            UIComponents.render_processing_status("Processing content...", 0.1)
            
            processed_texts = []
            
            # Process uploaded files
            if uploaded_files:
                for i, file in enumerate(uploaded_files):
                    progress = 0.1 + (0.4 * i / len(uploaded_files))
                    UIComponents.render_processing_status(f"Processing {file.name}...", progress)
                    
                    text = self._process_single_file(file)
                    if text:
                        processed_texts.append({
                            'text': text,
                            'source': file.name,
                            'type': file.type
                        })
            
            # Process URL
            if url_input:
                UIComponents.render_processing_status("Processing URL...", 0.6)
                text = self._process_url(url_input)
                if text:
                    processed_texts.append({
                        'text': text,
                        'source': url_input,
                        'type': 'url'
                    })
            
            if processed_texts:
                # Add to vector database
                UIComponents.render_processing_status("Adding to database...", 0.8)
                self._add_to_database(processed_texts)
                
                UIComponents.render_processing_status("Complete!", 1.0)
                UIComponents.render_success(f"Successfully processed {len(processed_texts)} items")
                
                # Update database stats
                st.session_state.vector_db_stats = self.vector_db.get_database_stats()
            else:
                UIComponents.render_warning("No content could be processed")
                
        except Exception as e:
            self.logger.error(f"Content processing failed: {e}")
            UIComponents.render_error(f"Content processing failed: {e}")
    
    def _process_single_file(self, file) -> Optional[str]:
        """Process a single uploaded file."""
        try:
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                tmp_file.write(file.read())
                tmp_path = Path(tmp_file.name)
            
            try:
                # Determine file type and process accordingly
                file_extension = tmp_path.suffix.lower()
                
                if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                    # Video processing
                    audio_path = self.video_processor.extract_audio(tmp_path)
                    transcript_result = self.speech_processor.process(audio_path)
                    return transcript_result['text']
                
                elif file_extension in ['.mp3', '.wav', '.m4a']:
                    # Audio processing
                    processed_audio = self.audio_processor.process(tmp_path)
                    transcript_result = self.speech_processor.process(processed_audio)
                    return transcript_result['text']
                
                elif file_extension in ['.pdf', '.docx', '.txt']:
                    # Document processing
                    return self.document_processor.process(tmp_path)
                
                else:
                    self.logger.warning(f"Unsupported file type: {file_extension}")
                    return None
                    
            finally:
                # Clean up temporary file
                if tmp_path.exists():
                    os.unlink(tmp_path)
                    
        except Exception as e:
            self.logger.error(f"Failed to process file {file.name}: {e}")
            return None
    
    def _process_url(self, url: str) -> Optional[str]:
        """Process a URL (YouTube or web page)."""
        try:
            if 'youtube.com' in url or 'youtu.be' in url:
                # YouTube processing would require additional libraries
                UIComponents.render_warning("YouTube processing not yet implemented")
                return None
            else:
                # Web page processing
                return self.document_processor.extract_text_from_url(url)
                
        except Exception as e:
            self.logger.error(f"Failed to process URL {url}: {e}")
            return None
    
    def _add_to_database(self, processed_texts: List[Dict[str, Any]]):
        """Add processed texts to vector database."""
        try:
            all_chunks = []
            all_metadata = []
            
            for item in processed_texts:
                # Clean text
                cleaned_text = self.text_cleaner.clean_text(item['text'])
                
                # Analyze content
                analysis = self.text_analyzer.analyze_text(cleaned_text)
                
                if analysis.status == 'no_content':
                    continue
                
                # Chunk text
                chunks = self.text_chunker.split_into_chunks(cleaned_text)
                
                for chunk in chunks:
                    all_chunks.append(chunk.content)
                    metadata = {
                        'source': item['source'],
                        'content_type': item['type'],
                        'chunk_id': chunk.chunk_id,
                        'word_count': chunk.word_count,
                        'analysis_status': analysis.status
                    }
                    all_metadata.append(metadata)
            
            if all_chunks:
                # Add to vector database
                self.vector_db.add_to_vectordb(all_chunks, all_metadata)
                self.logger.info(f"Added {len(all_chunks)} chunks to database")
            
        except Exception as e:
            self.logger.error(f"Failed to add to database: {e}")
            raise
    
    def _process_query(self, query_config: Dict[str, Any], config: Dict[str, Any]):
        """Process user query."""
        try:
            query = query_config['query']
            UIComponents.render_processing_status("Searching for relevant content...", 0.2)
            
            start_time = time.time()
            
            # Perform search
            if config.get('enable_web_search'):
                # Use web search fallback
                local_results = self.search_engine.search(query, k=config['max_results'])
                results, used_web = self.web_search.fallback_search(query, local_results)
            else:
                # Local search only
                search_results = self.search_engine.search(
                    query,
                    k=config['max_results'],
                    threshold=config['similarity_threshold']
                )
                results = [
                    {
                        'text': result.text,
                        'score': result.score,
                        'metadata': result.metadata
                    }
                    for result in search_results
                ]
                used_web = False
            
            if not results:
                UIComponents.render_warning("No relevant content found for your query")
                return
            
            UIComponents.render_processing_status("Generating summary...", 0.6)
            
            # Generate summary with memory context
            summary_result = self.summarizer.summarize_chunks(
                results,
                query=query,
                use_chain_of_thought=query_config['use_chain_of_thought']
            )

            # Generate LLM response with memory integration
            from src.ai.prompt_engineer import PromptEngineer
            prompt_engineer = PromptEngineer()

            # Build context-aware prompt
            if config.get('enable_memory', True):
                messages = prompt_engineer.build_qa_prompt(
                    context=summary_result.summary,
                    question=query,
                    use_chain_of_thought=query_config['use_chain_of_thought']
                )

                # Generate enhanced response with memory
                enhanced_response = self.llm_client.generate_response(
                    messages,
                    use_memory=True,
                    store_in_memory=config.get('store_conversations', True),
                    max_memory_context=config.get('max_memory_context', 3),
                    context_sources=[r.get('metadata', {}).get('source', 'unknown') for r in results]
                )

                # Use enhanced response as final summary
                final_summary = enhanced_response.content
            else:
                final_summary = summary_result.summary
            
            processing_time = time.time() - start_time
            
            # Store results in session state
            st.session_state.current_summary = {
                'summary': final_summary,
                'sources': results,
                'confidence_score': summary_result.confidence_score,
                'processing_time': processing_time,
                'used_web_search': used_web,
                'query': query,
                'used_memory': config.get('enable_memory', True)
            }
            
            # Add to history
            UIComponents.add_to_history(query, summary_result.summary)
            
            UIComponents.render_processing_status("Complete!", 1.0)
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            UIComponents.render_error(f"Query processing failed: {e}")
    
    def _display_results(self, config: Dict[str, Any]):
        """Display processing results."""
        try:
            result = st.session_state.current_summary
            
            # Display main results
            UIComponents.render_results(
                summary=result['summary'],
                sources=result['sources'],
                confidence_score=result['confidence_score'],
                processing_time=result['processing_time']
            )
            
            # Web search warning
            if result.get('used_web_search'):
                UIComponents.render_warning(
                    "Some results include web search content. Please verify information from external sources."
                )
            
            # Generate multi-modal content
            audio_data = None
            image_data = None
            
            if config.get('enable_tts'):
                try:
                    tts_result = self.multimodal_ai.create_audio_summary(
                        result['summary'],
                        voice=config.get('tts_voice', 'alloy')
                    )
                    audio_data = tts_result.audio_data
                except Exception as e:
                    self.logger.warning(f"TTS generation failed: {e}")
            
            if config.get('enable_image_gen'):
                try:
                    image_result = self.multimodal_ai.create_summary_visualization(
                        result['summary']
                    )
                    image_data = image_result.image_data
                except Exception as e:
                    self.logger.warning(f"Image generation failed: {e}")
            
            # Download options
            UIComponents.render_download_options(
                summary=result['summary'],
                audio_data=audio_data,
                image_data=image_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to display results: {e}")
            UIComponents.render_error(f"Failed to display results: {e}")


def main():
    """Main entry point for Streamlit app."""
    try:
        app = StreamlitApp()
        app.run()
    except Exception as e:
        st.error(f"Failed to start application: {e}")


if __name__ == "__main__":
    main()
