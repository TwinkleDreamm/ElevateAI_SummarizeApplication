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
            
            # Database and search - Fix: Load existing database first
            self.vector_db = VectorDatabase()
            self.embedding_generator = EmbeddingGenerator()
            self.search_engine = SemanticSearchEngine(vector_db=self.vector_db)
            self.retrieval_engine = RetrievalEngine(search_engine=self.search_engine)
            self.web_search = WebSearchEngine()
            
            # Fix: Initialize search engine with existing database
            if hasattr(self.search_engine, 'set_vector_database'):
                self.search_engine.set_vector_database(self.vector_db)
            
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
            
            # Fix: Display database status after initialization
            self._display_database_status()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            st.error(f"Failed to initialize application components: {e}")
    
    def _display_database_status(self):
        """Display current database status."""
        try:
            stats = self.vector_db.get_database_stats()
            self.logger.info(f"Database status: {stats}")
            
            # Store in session state for UI display
            st.session_state.vector_db_stats = stats
            
        except Exception as e:
            self.logger.warning(f"Failed to get database status: {e}")
            st.session_state.vector_db_stats = {}
    
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
            # Check if function calling is enabled from sidebar config
            if config.get('enable_function_calling', False):
                self._process_query_with_function_calling(query_config, config)
            else:
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
                
                # Fix: Ensure database is saved after processing
                try:
                    self.vector_db.save_database()
                    self.logger.info("Database saved after content processing")
                except Exception as save_error:
                    self.logger.warning(f"Failed to save database: {save_error}")
                
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
                # If no embedding model, skip DB insert with user-friendly warning
                try:
                    if hasattr(self.vector_db, 'embedding_generator') and hasattr(self.vector_db.embedding_generator, 'has_model'):
                        if not self.vector_db.embedding_generator.has_model():
                            self.logger.warning("No embedding model available. Skipping database insertion.")
                            UIComponents.render_warning("Embeddings are disabled. Configure OPENAI_API_KEY to enable the vector database.")
                            return
                except Exception:
                    pass

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
            
            # Fix: Ensure database is saved after processing
            try:
                self.vector_db.save_database()
                self.logger.info("Database saved after query processing")
            except Exception as save_error:
                self.logger.warning(f"Failed to save database: {save_error}")
            
            UIComponents.render_processing_status("Complete!", 1.0)
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            UIComponents.render_error(f"Query processing failed: {e}")
    
    def _process_query_with_function_calling(self, query_config: Dict[str, Any], config: Dict[str, Any]):
        """Process user query with function calling visualization."""
        try:
            query = query_config['query']
            
            # Step 1: Show function calling initialization
            st.markdown("## 🚀 Function Calling Mode Activated")
            st.markdown("*AI will now use specialized functions to process your request...*")
            
            # Step 2: Search for relevant content
            with st.expander("### 🔍 Step 1: Content Search", expanded=True):
                st.markdown("**Searching for relevant content in the database...**")
                
                # Create a progress bar with better styling
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(3):
                    progress_bar.progress((i + 1) * 0.33)
                    if i == 0:
                        status_text.text("🔍 Searching database...")
                    elif i == 1:
                        status_text.text("📊 Calculating similarity scores...")
                    else:
                        status_text.text("✅ Found relevant content!")
                    time.sleep(0.5)
                
                start_time = time.time()
                
                # Perform search
                if config.get('enable_web_search'):
                    local_results = self.search_engine.search(query, k=config['max_results'])
                    results, used_web = self.web_search.fallback_search(query, local_results)
                else:
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
                
                # Display search results in a nice format
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📄 Documents Found", len(results))
                with col2:
                    st.metric("📊 Top Score", f"{results[0]['score']:.3f}")
                with col3:
                    st.metric("⏱️ Search Time", f"{time.time() - start_time:.1f}s")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
            
            # Step 3: Function calling analysis
            with st.expander("### 🤖 Step 2: AI Function Analysis", expanded=True):
                st.markdown("**AI is analyzing retrieved content and deciding which functions to call...**")
                
                # Show available functions in a nice format
                if self.llm_client:
                    available_functions = self.llm_client.get_available_functions()
                    
                    # Display available functions in a clean format
                    st.markdown("**🔧 Available AI Functions:**")
                    function_cols = st.columns(4)
                    for i, func in enumerate(available_functions[:8]):  # Show first 8
                        with function_cols[i % 4]:
                            st.info(f"• {func}")
                    
                    # Register custom summary functions if not already done
                    self._register_summary_functions()
                    
                    # Simulate function calling process on RAG context
                    st.markdown("**🤖 AI Decision Process (RAG Context Analysis):**")
                    
                    # Get the most relevant context from RAG results
                    if len(results) > 0:
                        rag_context = results[0]['text']
                        
                        # Display context info
                        context_col1, context_col2 = st.columns(2)
                        with context_col1:
                            st.metric("📄 Context Length", f"{len(rag_context)} chars")
                        with context_col2:
                            st.metric("📊 Similarity Score", f"{results[0]['score']:.3f}")
                        
                        # Function 1: Analyze document structure
                        st.markdown("**1️⃣ Analyzing document structure...**")
                        structure_result = self.llm_client.call_function(
                            "analyze_document_structure", 
                            {"text": rag_context}
                        )
                        if structure_result.get('success'):
                            structure_data = structure_result['result']
                            
                            # Display structure results in a nice format
                            struct_col1, struct_col2, struct_col3 = st.columns(3)
                            with struct_col1:
                                st.metric("📊 Total Sections", structure_data.get('total_sections', 0))
                            with struct_col2:
                                st.metric("📋 Structure Type", structure_data.get('structure_type', 'unknown'))
                            with struct_col3:
                                st.metric("📝 Total Words", structure_data.get('total_words', 0))
                            
                            # Show sections in an expandable
                            with st.expander("📋 View Document Sections", expanded=False):
                                for i, section in enumerate(structure_data.get('sections', [])[:5]):  # Show first 5
                                    st.markdown(f"**{i+1}. {section.get('title', 'Untitled')}** ({section.get('word_count', 0)} words)")
                        else:
                            st.error("❌ Failed to analyze document structure")
                        
                        # Function 2: Extract key topics
                        st.markdown("**2️⃣ Extracting key topics...**")
                        topics_result = self.llm_client.call_function(
                            "extract_key_topics", 
                            {"text": rag_context, "max_topics": 5}
                        )
                        if topics_result.get('success'):
                            topics_data = topics_result['result']
                            
                            # Display topics in a nice format
                            st.markdown(f"**🎯 Found {len(topics_data.get('key_topics', []))} key topics:**")
                            
                            # Create a nice topic display
                            topic_cols = st.columns(3)
                            for i, topic in enumerate(topics_data.get('key_topics', [])[:6]):  # Show first 6
                                with topic_cols[i % 3]:
                                    st.metric(
                                        f"#{i+1} {topic['topic'].title()}", 
                                        f"{topic['frequency']} times",
                                        delta=f"Frequency: {topic['frequency']}"
                                    )
                        else:
                            st.error("❌ Failed to extract key topics")
                        
                        # Function 3: Suggest summary format
                        st.markdown("**3️⃣ Suggesting summary format...**")
                        format_result = self.llm_client.call_function(
                            "suggest_summary_format", 
                            {"text": rag_context, "target_length": "medium"}
                        )
                        if format_result.get('success'):
                            format_data = format_result['result']
                            
                            # Display format suggestion in a nice format
                            format_col1, format_col2, format_col3 = st.columns(3)
                            with format_col1:
                                st.metric("📄 Content Type", format_data.get('content_type', 'unknown'))
                            with format_col2:
                                st.metric("📋 Suggested Format", format_data.get('suggested_format', 'unknown'))
                            with format_col3:
                                st.metric("📏 Target Length", format_data.get('target_length', 'unknown'))
                        else:
                            st.error("❌ Failed to suggest summary format")
                        
                        # Store function results for enhanced summary
                        function_results = {
                            'structure': structure_result,
                            'topics': topics_result,
                            'format': format_result,
                            'rag_context': rag_context,
                            'rag_score': results[0]['score']
                        }
                    else:
                        st.warning("⚠️ No RAG results to analyze")
                        function_results = {}
                else:
                    st.warning("⚠️ LLM client not available - using basic summarization")
                    function_results = {}
            
            # Step 4: Generate enhanced summary
            with st.expander("### 📝 Step 3: Enhanced Summary Generation", expanded=True):
                st.markdown("**Generating summary using RAG context and function analysis results...**")
                
                # Create a better progress indicator
                summary_progress = st.progress(0)
                summary_status = st.empty()
                
                # Generate summary with function calling
                if self.llm_client and function_results:
                    # Create enhanced prompt with RAG context and function results
                    rag_context = function_results.get('rag_context', '')
                    structure_data = function_results.get('structure', {}).get('result', {})
                    topics_data = function_results.get('topics', {}).get('result', {})
                    format_data = function_results.get('format', {}).get('result', {})
                    rag_score = function_results.get('rag_score', 0)
                    
                    # Update progress
                    summary_progress.progress(0.3)
                    summary_status.text("🤖 Creating enhanced prompt with function insights...")
                    
                    enhanced_prompt = f"""
                    Based on the RAG (Retrieval-Augmented Generation) analysis, please provide a comprehensive summary.
                    
                    User Query: {query}
                    RAG Context (Top Result): {rag_context[:500]}...
                    RAG Similarity Score: {rag_score:.3f}
                    
                    Function Analysis Results:
                    - Document Structure: {structure_data.get('total_sections', 0)} sections, {structure_data.get('structure_type', 'unknown')} type
                    - Key Topics: {[topic['topic'] for topic in topics_data.get('key_topics', [])]}
                    - Recommended Format: {format_data.get('suggested_format', 'general')}
                    - Content Type: {format_data.get('content_type', 'unknown')}
                    
                    Please create a well-structured summary that incorporates these insights.
                    """
                    
                    messages = [{"role": "user", "content": enhanced_prompt}]
                    
                    # Update progress
                    summary_progress.progress(0.6)
                    summary_status.text("🚀 Calling AI with function definitions...")
                    
                    try:
                        # Use function calling enhanced response
                        st.info("🤖 Calling AI model with function definitions...")
                        
                        enhanced_response = self.llm_client.generate_response_with_functions(
                            messages,
                            use_functions=True
                        )
                        
                        # Debug: Check if response is valid
                        if enhanced_response and hasattr(enhanced_response, 'content'):
                            final_summary = enhanced_response.content
                            
                            # Check if content is meaningful
                            if final_summary and final_summary.strip() and len(final_summary.strip()) > 50:
                                st.success(f"✅ AI Response received! Length: {len(final_summary)} characters")
                            else:
                                st.warning("⚠️ AI response is too short or empty, falling back to basic summarization...")
                                raise Exception("Empty or invalid response from AI model")
                        else:
                            st.error("❌ Invalid response from AI model")
                            raise Exception("No valid response from AI model")
                        
                        # Update progress
                        summary_progress.progress(1.0)
                        summary_status.text("✅ Enhanced summary generated successfully!")
                        
                        # Display success message with metrics
                        success_col1, success_col2, success_col3 = st.columns(3)
                        with success_col1:
                            st.success("🎯 Function Calling")
                        with success_col2:
                            st.success("📊 Enhanced Analysis")
                        with success_col3:
                            st.success("📝 Quality Summary")
                        
                        # Show summary metrics
                        if function_results.get('structure') and function_results.get('topics'):
                            metrics_col1, metrics_col2 = st.columns(2)
                            with metrics_col1:
                                st.metric("📊 Analysis Sections", structure_data.get('total_sections', 0))
                            with metrics_col2:
                                st.metric("🎯 Key Topics Found", len(topics_data.get('key_topics', [])))
                        
                        # Display the generated summary prominently
                        st.markdown("---")
                        st.markdown("## 📋 Summary Generated Successfully!")
                        st.success("✅ AI has generated an enhanced summary using function calling analysis.")
                        st.info("📊 View detailed results in the tabs below.")
                        
                        # Don't display summary here - it will be shown in the final tabs
                        # st.markdown(final_summary)
                        
                    except Exception as e:
                        st.error(f"❌ Function calling failed: {str(e)}")
                        st.info("🔄 Falling back to basic RAG summary...")
                        
                        # Fallback to basic RAG summarization
                        try:
                            summary_result = self.summarizer.summarize_chunks(
                                results,
                                query=query,
                                use_chain_of_thought=query_config['use_chain_of_thought']
                            )
                            final_summary = summary_result.summary
                            
                            # Display fallback summary
                            st.markdown("---")
                            st.markdown("## 📋 Basic Summary (Fallback)")
                            st.markdown(final_summary)
                            st.info("✅ Basic summary generated successfully")
                            
                        except Exception as fallback_error:
                            st.error(f"❌ Fallback also failed: {str(fallback_error)}")
                            final_summary = "Error: Unable to generate summary"
                            st.markdown("## 📋 Error Summary")
                            st.markdown("*Summary generation failed. Please check your configuration and try again.*")
                else:
                    # Basic RAG summarization
                    summary_progress.progress(0.5)
                    summary_status.text("📝 Generating basic summary...")
                    
                    summary_result = self.summarizer.summarize_chunks(
                        results,
                        query=query,
                        use_chain_of_thought=query_config['use_chain_of_thought']
                    )
                    final_summary = summary_result.summary
                    # Initialize empty function_results for basic mode
                    function_results = {}
                    
                    summary_progress.progress(1.0)
                    summary_status.text("✅ Basic summary generated!")
                    
                    # Display basic summary
                    st.markdown("---")
                    st.markdown("## 📋 Basic Summary")
                    st.markdown(final_summary)
                
                # Clear progress indicators
                summary_progress.empty()
                summary_status.empty()
                
                processing_time = time.time() - start_time
                
                # Store results in session state
                st.session_state.current_summary = {
                    'summary': final_summary,
                    'sources': results,
                    'confidence_score': 0.9,  # High confidence with function calling
                    'processing_time': processing_time,
                    'used_web_search': used_web,
                    'query': query,
                    'used_memory': config.get('enable_memory', True),
                    'used_function_calling': True,
                    'function_results': function_results
                }
                
                # Add to history
                UIComponents.add_to_history(query, final_summary)
                
                # Save database
                try:
                    self.vector_db.save_database()
                    self.logger.info("Database saved after function calling query processing")
                except Exception as save_error:
                    self.logger.warning(f"Failed to save database: {save_error}")
                
                UIComponents.render_processing_status("Complete!", 1.0)
                
                # st.success("🎉 **Function Calling Enhanced Summary Complete!**")
                
        except Exception as e:
            self.logger.error(f"Function calling query processing failed: {e}")
            UIComponents.render_error(f"Function calling query processing failed: {e}")
    
    def _register_summary_functions(self):
        """Register custom summary functions for function calling."""
        if not self.llm_client:
            return
            
        try:
            # Check if functions are already registered
            available_functions = self.llm_client.get_available_functions()
            if 'analyze_document_structure' in available_functions:
                return  # Already registered
            
            # Import and register custom functions
            from demo_function_calling_summary import create_summary_functions
            summary_functions = create_summary_functions()
            
            for name, func in summary_functions.items():
                if name == "analyze_document_structure":
                    self.llm_client.register_function(
                        name=name,
                        description="Analyze document structure and organization",
                        parameters={
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Text to analyze"}
                            },
                            "required": ["text"]
                        },
                        function=func
                    )
                elif name == "extract_key_topics":
                    self.llm_client.register_function(
                        name=name,
                        description="Extract key topics and themes from text",
                        parameters={
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Text to analyze"},
                                "max_topics": {"type": "integer", "description": "Maximum number of topics", "default": 5}
                            },
                            "required": ["text"]
                        },
                        function=func
                    )
                elif name == "suggest_summary_format":
                    self.llm_client.register_function(
                        name=name,
                        description="Suggest best format for summarizing text",
                        parameters={
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Text to summarize"},
                                "target_length": {"type": "string", "enum": ["short", "medium", "long"], "description": "Target summary length", "default": "medium"}
                            },
                            "required": ["text"]
                        },
                        function=func
                    )
            
            self.logger.info("Custom summary functions registered for function calling")
            
        except Exception as e:
            self.logger.warning(f"Failed to register summary functions: {e}")
    
    def _display_results(self, config: Dict[str, Any]):
        """Display processing results."""
        try:
            result = st.session_state.current_summary
            
            # Check if function calling was used
            if result.get('used_function_calling'):
                # Create a nice header
                st.markdown("---")
                st.markdown("# 🎉 Function Calling Enhanced Summary Complete!")
                
                # Display function calling results if available
                function_results = result.get('function_results', {})
                if function_results:
                    # Create tabs for better organization
                    tab1, tab2, tab3 = st.tabs(["📋 Summary", "📊 Analysis Results", "⚙️ Processing Info"])
                    
                    with tab1:
                        st.markdown("## 📋 Enhanced Summary")
                        st.markdown("### Generated Summary")
                        st.markdown(result['summary'])
                        
                        # Display additional info
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Confidence Score", f"{result['confidence_score']:.2f}")
                        with col2:
                            st.metric("⏱️ Processing Time", f"{result['processing_time']:.1f}s")
                        with col3:
                            st.metric("📄 Sources Used", len(result.get('sources', [])))
                    
                    with tab2:
                        st.markdown("## 🤖 Function Calling Analysis Results")
                        
                        # Display structure analysis
                        if function_results.get('structure'):
                            structure_data = function_results['structure'].get('result', {})
                            if structure_data:
                                st.markdown("### 📋 Document Structure Analysis")
                                struct_col1, struct_col2, struct_col3 = st.columns(3)
                                with struct_col1:
                                    st.metric("📊 Total Sections", structure_data.get('total_sections', 0))
                                with struct_col2:
                                    st.metric("📋 Structure Type", structure_data.get('structure_type', 'unknown'))
                                with struct_col3:
                                    st.metric("📝 Total Words", structure_data.get('total_words', 0))
                        
                        # Display topics analysis
                        if function_results.get('topics'):
                            topics_data = function_results['topics'].get('result', {})
                            if topics_data and topics_data.get('key_topics'):
                                st.markdown("### 🎯 Key Topics Extracted")
                                # Display topics in a nice grid
                                topic_cols = st.columns(3)
                                for i, topic in enumerate(topics_data['key_topics'][:6]):  # Show top 6
                                    with topic_cols[i % 3]:
                                        st.metric(
                                            f"#{i+1} {topic['topic'].title()}", 
                                            f"{topic['frequency']} times"
                                        )
                        
                        # Display format suggestion
                        if function_results.get('format'):
                            format_data = function_results['format'].get('result', {})
                            if format_data:
                                st.markdown("### 📝 Summary Format Recommendation")
                                format_col1, format_col2, format_col3 = st.columns(3)
                                with format_col1:
                                    st.metric("📄 Content Type", format_data.get('content_type', 'unknown'))
                                with format_col2:
                                    st.metric("📋 Suggested Format", format_data.get('suggested_format', 'unknown'))
                                with format_col3:
                                    st.metric("📏 Target Length", format_data.get('target_length', 'unknown'))
                        
                        # Display RAG quality metrics
                        if function_results.get('rag_score'):
                            st.markdown("### 🔍 RAG Search Quality")
                            rag_col1, rag_col2 = st.columns(2)
                            with rag_col1:
                                st.metric("📊 Similarity Score", f"{function_results['rag_score']:.3f}")
                            with rag_col2:
                                st.metric("📄 Context Length", f"{len(function_results.get('rag_context', ''))} chars")
                    
                    with tab3:
                        st.markdown("## ⚙️ Processing Information")
                        # Display processing details
                        proc_col1, proc_col2, proc_col3 = st.columns(3)
                        with proc_col1:
                            st.metric("⏱️ Processing Time", f"{result['processing_time']:.1f}s")
                        with proc_col2:
                            st.metric("🎯 Function Calling", "Enabled")
                        with proc_col3:
                            st.metric("📊 Confidence Score", f"{result['confidence_score']:.2f}")
                        
                        # Additional info
                        st.markdown("### 📋 Additional Details")
                        st.markdown(f"• **Query:** {result.get('query', 'N/A')}")
                        st.markdown(f"• **Memory Used:** {'Yes' if result.get('used_memory') else 'No'}")
                        st.markdown(f"• **Web Search:** {'Yes' if result.get('used_web_search') else 'No'}")
                        st.markdown(f"• **Sources Found:** {len(result.get('sources', []))}")
                        
                        # Show sources if available
                        if result.get('sources'):
                            with st.expander("📚 View Sources", expanded=False):
                                for i, source in enumerate(result['sources'][:3]):  # Show first 3
                                    st.markdown(f"**Source {i+1}:** {source.get('metadata', {}).get('source', 'Unknown')}")
                                    st.markdown(f"**Similarity:** {source.get('score', 0):.3f}")
                                    st.markdown("---")
                
                else:
                    # Fallback if no function results
                    st.markdown("## 📋 Summary Results")
                    st.markdown("### Generated Summary")
                    st.markdown(result['summary'])
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Confidence Score", f"{result['confidence_score']:.2f}")
                    with col2:
                        st.metric("⏱️ Processing Time", f"{result['processing_time']:.1f}s")
                    with col3:
                        st.metric("📄 Sources Used", len(result.get('sources', [])))
                    
                    # Show sources if available
                    if result.get('sources'):
                        with st.expander("📚 View Sources", expanded=False):
                            for i, source in enumerate(result['sources'][:3]):
                                st.markdown(f"**Source {i+1}:** {source.get('metadata', {}).get('source', 'Unknown')}")
                                st.markdown(f"**Similarity:** {source.get('score', 0):.3f}")
                                st.markdown("---")
            else:
                # Basic results display
                st.markdown("## 📋 Summary Results")
                st.markdown("### Generated Summary")
                st.markdown(result['summary'])
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Confidence Score", f"{result['confidence_score']:.2f}")
                with col2:
                    st.metric("⏱️ Processing Time", f"{result['processing_time']:.1f}s")
                with col3:
                    st.metric("📄 Sources Used", len(result.get('sources', [])))
                
                # Show sources if available
                if result.get('sources'):
                    with st.expander("📚 View Sources", expanded=False):
                        for i, source in enumerate(result['sources'][:3]):
                            st.markdown(f"**Source {i+1}:** {source.get('metadata', {}).get('source', 'Unknown')}")
                            st.markdown(f"**Similarity:** {source.get('score', 0):.3f}")
                            st.markdown("---")
            
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
