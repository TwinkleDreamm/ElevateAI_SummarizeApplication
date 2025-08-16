# ElevateAI - Intelligent Content Analysis & Summarization Application

## 🎯 Project Overview
ElevateAI is an advanced AI-powered application that provides intelligent analysis and summarization of various content types including videos, audio files, and documents. The application features a sophisticated memory system and uses cutting-edge AI technologies including speech-to-text, natural language processing, vector databases, and large language models.

## 🚀 Features
- **Multi-format Support**: Process video, audio, PDF, DOCX, TXT files, and YouTube links
- **Advanced Memory System**: Short-term and long-term memory with conversation tracking
- **Advanced Audio Processing**: Vocal separation, noise reduction, speech enhancement
- **Intelligent Text Processing**: Content analysis, cleaning, chunking, and embedding
- **Semantic Search**: Multi-hop retrieval with reranking capabilities
- **Web Fallback**: Automatic web search when local content is insufficient
- **Multi-modal AI**: Text-to-speech, speech-to-text, image generation (when API configured)
- **Interactive Interface**: User-friendly Streamlit multi-page web application
- **Graceful Fallbacks**: Works even when some dependencies are missing
- **Notebook System**: Organize and manage multiple knowledge collections

## 🔧 Recent Fixes & Improvements (v1.1.0)
- **✅ Issue #1: Embedding Model Configuration**
  - Fixed online embedding model detection for third-party providers
  - Corrected base URL and API key selection logic
  - Enhanced model name handling for embeddings

- **✅ Issue #2: Vector Database Persistence**
  - Resolved database data loss after application restart
  - Implemented auto-save functionality
  - Fixed shared database instances between components

- **✅ Issue #3: LLM Client Provider Detection**
  - Fixed Azure OpenAI parameter handling
  - Implemented automatic third-party provider detection
  - Enhanced provider compatibility and error handling

- **✅ Issue #4: UI Bug Fixes**
  - Fixed column reference errors in notebook interface
  - Improved error handling in Streamlit components
  - Enhanced user experience and stability

## 🏗️ Architecture
The application follows an Object-Oriented Programming (OOP) design with the following main components:

### Core Modules
1. **Data Processing** (`src/core/`)
   - Audio/Video processing
   - Document text extraction
   - Speech-to-text conversion

2. **Text Analysis** (`src/analysis/`)
   - Content analysis and validation
   - Text cleaning and preprocessing
   - Intelligent chunking strategies

3. **Vector Database** (`src/database/`)
   - FAISS vector storage
   - Embedding generation (OpenAI + Sentence Transformers)
   - Metadata management

4. **Search Engine** (`src/search/`)
   - Semantic search
   - Multi-hop retrieval
   - Reranking algorithms

5. **AI Integration** (`src/ai/`)
   - Azure OpenAI integration
   - OpenAI API support
   - Function calling capabilities
   - Prompt engineering
   - Multi-modal AI features

6. **Web Interface** (`src/interface/`)
   - Streamlit multi-page application
   - Notebook management system
   - File upload handling
   - Result visualization

7. **Memory System** (`src/utils/memory/`)
   - Short-term and long-term memory
   - Conversation tracking
   - Context management

## 🛠️ Technology Stack
- **Audio/Video Processing**: moviepy, ffmpeg-python, whisper, librosa
- **Text Processing**: SpaCy, NLTK, pdfplumber, python-docx, pyvi (Vietnamese)
- **Vector Database**: FAISS, sentence-transformers, ChromaDB
- **AI/ML**: Azure OpenAI, OpenAI API, LangChain, Transformers
- **Web Framework**: Streamlit 1.48.0
- **Package Management**: pip, conda
- **Development Tools**: pytest, black, flake8, mypy

## 📁 Project Structure
```
ElevateAI_SummarizeApplication/
├── src/                          # Core source code
│   ├── ai/                      # AI integration layer
│   │   ├── llm_client.py       # LLM client (Azure OpenAI, OpenAI)
│   │   ├── multimodal.py       # Multi-modal AI capabilities
│   │   ├── tts_client.py       # Text-to-speech client
│   │   ├── function_calling.py # Function calling system
│   │   ├── prompt_engineer.py  # Advanced prompt management
│   │   └── summarizer.py       # Content summarization
│   ├── analysis/                # Text analysis and processing
│   │   ├── text_analyzer.py    # Content analysis
│   │   ├── text_cleaner.py     # Text preprocessing
│   │   └── text_chunker.py     # Intelligent text chunking
│   ├── core/                    # Data processing layer
│   │   ├── audio_processor.py  # Audio processing
│   │   ├── video_processor.py  # Video processing
│   │   ├── document_processor.py # Document processing
│   │   └── speech_to_text.py   # Speech recognition
│   ├── database/                # Vector database and storage
│   │   ├── vector_database.py  # FAISS vector storage
│   │   ├── embedding_generator.py # Text embeddings
│   │   └── metadata_manager.py # Metadata management
│   ├── interface/               # User interface components
│   │   ├── app.py              # Main Streamlit app
│   │   ├── pages/              # Multi-page navigation
│   │   │   ├── 01_Notebooks.py # Notebook management
│   │   │   └── 04_Settings.py  # Application settings
│   │   ├── components.py       # Reusable UI components
│   │   └── utils/              # Interface utilities
│   ├── search/                  # Search and retrieval engine
│   │   ├── semantic_search.py  # Semantic search
│   │   ├── retrieval_engine.py # Multi-hop retrieval
│   │   └── web_search.py       # Web search fallback
│   └── utils/                   # Utility functions and helpers
│       ├── logger.py           # Logging system
│       ├── memory.py           # Memory management
│       ├── cache.py            # Caching system
│       └── exceptions.py       # Custom exceptions
├── data/                        # Data storage directory
│   ├── vectordb/               # Vector database files
│   ├── memory/                 # Memory system storage
│   ├── cache/                  # General cache storage
│   ├── embedding_cache/        # Embedding result cache
│   ├── llm_cache/              # LLM response cache
│   ├── models/                 # Local AI models
│   ├── temp/                   # Temporary files
│   └── notes/                  # User notes and documents
├── config/                      # Configuration files
│   └── settings.py             # Application settings
├── tests/                       # Test suite
├── docs/                        # Documentation
├── logs/                        # Application logs
├── requirements.txt             # Python dependencies (239 packages)
├── start_app.py                 # Application startup script
├── main.py                      # Alternative entry point
├── pyproject.toml              # Project configuration
└── setup.py                     # Package setup
```

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.9 or higher (3.11 recommended)
- **Conda**: Environment management
- **Java**: 17+ (Java 21 recommended) for LanguageTool grammar checking
  - Download from: https://adoptium.net/temurin/releases/
- **FFmpeg**: For video/audio processing (optional but recommended)
- **API Credentials**: Azure OpenAI or OpenAI API (optional - app works without them)

### Quick Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ElevateAI_SummarizeApplication
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n Project_1 python=3.11
   conda activate Project_1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   **Note**: This installs 239 packages including:
   - Core AI: openai, langchain, transformers, sentence-transformers
   - ML/Data: numpy, pandas, scikit-learn, torch
   - Web: streamlit, requests, beautifulsoup4
   - Audio/Video: moviepy, librosa, soundfile
   - Text Processing: spacy, nltk, pdfplumber, python-docx
   - Vector DB: faiss-cpu, chromadb

4. **Configure environment variables (optional)**
   - Copy `.env.example` to `.env`
   - Add your API keys and configuration:
   ```env
   # Azure OpenAI
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
   
   # OpenAI (fallback)
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_BASE_URL=https://api.openai.com/v1
   
   # Google Search (optional)
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CSE_ID=your_custom_search_engine_id
   
   # SerpAPI (optional)
   SERPAPI_API_KEY=your_serpapi_key
   ```
   - **Note**: App works without API keys but with limited functionality

5. **Setup Java for LanguageTool (recommended)**
   - Install Java 21 from https://adoptium.net/temurin/releases/
   - Set JAVA_HOME environment variable
   - Add Java to PATH

6. **Run the application**
   ```bash
   conda activate Project_1
   python start_app.py
   ```
   - Opens Streamlit app at http://localhost:8501
   - Use sidebar navigation to access different pages

### Manual Setup
If you prefer manual setup:

1. **Activate conda environment**
   ```bash
   conda activate Project_1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download language models**
   ```bash
   conda activate Project_1
   python -m spacy download en_core_web_sm
   # Vietnamese model is included in requirements.txt
   ```

4. **Set up configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run tests**
   ```bash
   conda activate Project_1
   pytest tests/ -v
   ```

## 🎯 Usage

### Web Interface
1. **Start the application**:
   ```bash
   python start_app.py
   ```
2. **Open browser** to `http://localhost:8501`
3. **Navigate using sidebar**:
   - **Home**: View and manage your notebooks
   - **Create Notebook**: Create a new knowledge collection
   - **Notebook**: View and chat with a specific notebook
   - **Settings**: Configure application parameters

### Supported File Types
- **Video**: MP4, AVI, MOV, MKV, WebM
- **Audio**: MP3, WAV, M4A, FLAC, OGG
- **Documents**: PDF, DOCX, TXT
- **URLs**: YouTube videos, web pages, RSS feeds

### Key Features
- **Multi-format Processing**: Handles video, audio, and text files
- **Intelligent Summarization**: Context-aware summaries with various styles
- **Semantic Search**: Find relevant information across all processed content
- **Multi-modal Output**: Text, audio, and visual summaries
- **Web Search Fallback**: Searches the web when local content is insufficient
- **Source Attribution**: Tracks and cites information sources
- **Memory System**: Short-term and long-term memory for conversation continuity
- **Context Awareness**: Remembers previous interactions and learned facts
- **Notebook Management**: Organize content into themed collections

## 🔧 Configuration

### Environment Variables
See `.env.example` for all available configuration options:

- **AI Models**: Azure OpenAI, OpenAI API settings
- **Processing**: File size limits, supported formats
- **Search**: Similarity thresholds, result limits
- **Performance**: Caching, batch processing settings
- **Audio**: Sample rates, noise reduction, vocal separation

### Advanced Configuration
Edit `config/settings.py` for advanced customization:
- Model parameters and deployment names
- Processing strategies and chunking settings
- Database settings and vector dimensions
- Performance tuning and resource limits

## 🧪 Testing
Run the test suite:
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_text_processing.py -v
pytest tests/test_vector_database.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📊 Performance Optimization

### Caching
The application includes intelligent caching:
- **Embedding Cache**: Reuses embeddings for identical text
- **LLM Response Cache**: Caches AI responses for repeated queries
- **File Processing Cache**: Avoids reprocessing unchanged files
- **Metadata Cache**: Caches document metadata and analysis results

### Memory Management
- Automatic memory optimization and cleanup
- Batch processing for large datasets
- Configurable memory thresholds and retention policies
- LRU eviction for short-term memory

### API Rate Limiting
- Built-in rate limiting for API calls
- Configurable limits per service
- Automatic retry with exponential backoff
- Graceful degradation when APIs are unavailable

## 🔍 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure conda environment Project_1 is activated
2. **Model Download Failures**: Check internet connection and try manual download
3. **API Errors**: Verify API keys and endpoints in `.env`
4. **Memory Issues**: Reduce batch sizes in configuration
5. **File Processing Errors**: Check file formats and sizes

### Common Issues & Solutions

#### Java/LanguageTool Issues
- **Error**: "Detected java 11.0. LanguageTool requires Java >= 17"
- **Solution**: Install Java 21 and set JAVA_HOME:
  ```bash
  # Windows (PowerShell as Admin)
  [Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Java\jdk-21", "Machine")
  
  # Linux/macOS
  export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
  export PATH=$JAVA_HOME/bin:$PATH
  ```

#### NumPy Compatibility Issues
- **Error**: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"
- **Solution**: Ensure NumPy 1.x is installed:
  ```bash
  conda activate Project_1
  pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
  ```

#### Missing Dependencies
- **Error**: "No module named 'langchain'" or similar
- **Solution**: Ensure you're in the correct conda environment:
  ```bash
  conda activate Project_1
  pip install -r requirements.txt
  ```

#### Streamlit Issues
- **Error**: "Port 8501 is already in use"
- **Solution**: Kill existing process or change port:
  ```bash
  # Kill process on port 8501
  lsof -ti:8501 | xargs kill -9
  
  # Or change port in start_app.py
  ```

### Debug Mode
Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python start_app.py
```

### Performance Monitoring
View performance metrics in the web interface or check logs for detailed timing information.

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📝 License
This project is developed for educational and research purposes.

## 🙏 Acknowledgments
- OpenAI for GPT models and APIs
- Hugging Face for transformer models and sentence-transformers
- Streamlit for the web interface framework
- FAISS for efficient vector similarity search
- The open-source community for various libraries and tools

## 📚 Additional Documentation
- **ARCHITECTURE.md**: Detailed system architecture and design patterns
- **MEMORY_SYSTEM.md**: Memory system implementation details
- **CHANGELOG.md**: Version history and changes
- **docs/**: Additional technical documentation
