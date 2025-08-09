# ElevateAI - Intelligent Content Analysis & Summarization Application

## 🎯 Project Overview
ElevateAI is an advanced AI-powered application that provides intelligent analysis and summarization of various content types including videos, audio files, and documents. The application features a sophisticated memory system and uses cutting-edge AI technologies including speech-to-text, natural language processing, vector databases, and large language models.

## 🚀 Features
- **Multi-format Support**: Process video, audio, PDF, DOCX, TXT files, and YouTube links
- **Advanced Memory System**: Short-term and long-term memory with conversation tracking
- **Online AI Models**: Optimized for cloud-based AI services (OpenAI, Azure) for better performance
- **Advanced Audio Processing**: Vocal separation, noise reduction, speech enhancement
- **Intelligent Text Processing**: Content analysis, cleaning, chunking, and embedding
- **Semantic Search**: Multi-hop retrieval with reranking capabilities
- **Web Fallback**: Automatic web search when local content is insufficient
- **Multi-modal AI**: Text-to-speech, speech-to-text, image generation (when API configured)
- **Interactive Interface**: User-friendly Streamlit web application
- **Lightweight & Fast**: Minimal local storage, fast startup, online model prioritization

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
   - Intelligent chunking

3. **Vector Database** (`src/database/`)
   - FAISS vector storage
   - Embedding generation
   - Metadata management

4. **Search Engine** (`src/search/`)
   - Semantic search
   - Multi-hop retrieval
   - Reranking algorithms

5. **AI Integration** (`src/ai/`)
   - Azure OpenAI integration
   - Function calling
   - Prompt engineering

6. **Web Interface** (`src/interface/`)
   - Streamlit application
   - File upload handling
   - Result visualization

## 🛠️ Technology Stack
- **Audio/Video Processing**: moviepy, ffmpeg-python, whisper
- **Text Processing**: SpaCy, NLTK, pdfplumber, python-docx
- **Vector Database**: FAISS, sentence-transformers
- **AI/ML**: Azure OpenAI, Langchain
- **Web Framework**: Streamlit
- **Package Management**: pip, conda

## 📁 Project Structure
```
Project_1/
├── src/
│   ├── core/           # Core data processing
│   ├── analysis/       # Text analysis and preprocessing
│   ├── database/       # Vector database management
│   ├── search/         # Semantic search engine
│   ├── ai/            # AI integration and LLM
│   ├── interface/     # Web interface
│   └── utils/         # Utility functions
├── tests/             # Unit tests
├── data/              # Sample data and models
├── config/            # Configuration files
├── requirements.txt   # Dependencies
└── main.py           # Application entry point
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Conda environment named `Project_1`
- **OpenAI API key (recommended)** for optimal performance with online models
- **Java 17+ (Java 21 recommended)** for LanguageTool grammar checking
  - Download from: https://adoptium.net/temurin/releases/
- Azure OpenAI or OpenAI API credentials (optional - app works without them)
- FFmpeg (for video processing, optional)

### 🚀 Performance Optimization
ElevateAI is optimized for **online AI models** to provide:
- ✅ **Fast startup** (no large model downloads)
- ✅ **Minimal storage** (~5MB vs ~500MB for local models)
- ✅ **Latest model versions** automatically
- ✅ **Better accuracy** with cloud-based models
- ⚠️ **Requires internet** and API keys for full functionality

### Quick Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Project_1
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

3. **Configure environment variables (optional)**
   - Copy `.env.example` to `.env`
   - Add your API keys and configuration:
   ```env
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   OPENAI_API_KEY=your_openai_api_key_here  # Fallback
   ```
   - **Note**: App works without API keys but with limited functionality

4. **Setup Java for LanguageTool (recommended)**
   - Install Java 21 from https://adoptium.net/temurin/releases/
   - Set JAVA_HOME environment variable
   - Add Java to PATH

5. **Run the application**
   ```bash
   conda activate Project_1
   python start_app.py
   ```

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
   python -m spacy download vi_core_news_sm  # Optional for Vietnamese
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
1. Start the Streamlit application:
   ```bash
   python run_app.py
   ```
2. Open your browser to `http://localhost:8501`
3. Upload video, audio, or document files
4. Ask questions or request summaries
5. Download results in various formats

### Supported File Types
- **Video**: MP4, AVI, MOV, MKV, WebM
- **Audio**: MP3, WAV, M4A, FLAC, OGG
- **Documents**: PDF, DOCX, TXT
- **URLs**: YouTube videos, web pages

### Key Features
- **Multi-format Processing**: Handles video, audio, and text files
- **Intelligent Summarization**: Context-aware summaries with various styles
- **Semantic Search**: Find relevant information across all processed content
- **Multi-modal Output**: Text, audio, and visual summaries
- **Web Search Fallback**: Searches the web when local content is insufficient
- **Source Attribution**: Tracks and cites information sources
- **Memory System**: Short-term and long-term memory for conversation continuity
- **Context Awareness**: Remembers previous interactions and learned facts

## 🔧 Configuration

### Environment Variables
See `.env.example` for all available configuration options:

- **AI Models**: Azure OpenAI, OpenAI API settings
- **Processing**: File size limits, supported formats
- **Search**: Similarity thresholds, result limits
- **Performance**: Caching, batch processing settings

### Advanced Configuration
Edit `config/settings.py` for advanced customization:
- Model parameters
- Processing strategies
- Database settings
- Performance tuning

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

### Memory Management
- Automatic memory optimization
- Batch processing for large datasets
- Configurable memory thresholds

### API Rate Limiting
- Built-in rate limiting for API calls
- Configurable limits per service
- Automatic retry with backoff

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
- Hugging Face for transformer models
- Streamlit for the web interface framework
- The open-source community for various libraries and tools
