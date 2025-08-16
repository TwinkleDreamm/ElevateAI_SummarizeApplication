# 🚀 ElevateAI Installation Guide

## Quick Setup

### **Install in Current Environment**
```bash
# Clone repository
git clone https://github.com/TwinkleDreamm/ElevateAI_SummarizeApplication.git
cd ElevateAI_SummarizeApplication

# Install package and dependencies
pip install -e .
```

### **Alternative: Manual Dependencies**
```bash
# Install dependencies only
pip install -r requirements.txt

# Create directories
mkdir -p data/cache data/vectordb logs
```

## Prerequisites

### Required
- **Python 3.9+** (3.12 recommended)
- **pip** (usually comes with Python)

### Optional (for full functionality)
- **OpenAI API Key** - For online models (recommended)
- **Java 17+** - For LanguageTool grammar checking
- **FFmpeg** - For video processing

## Configuration

### 1. **Environment Variables**
Edit `.env` file (created automatically):
```bash
# Required for optimal performance
OPENAI_API_KEY=your_openai_api_key_here

# Optional
AZURE_OPENAI_API_KEY=your_azure_key
LOG_LEVEL=INFO
```

### 2. **API Keys Setup**
- **OpenAI**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Azure OpenAI**: Get from Azure Portal

## Running the Application

### Streamlit UI (Recommended)
```bash
# If installed as package
elevateai

# Or run directly
python main.py
# Opens automatically in browser
```

### FastAPI Backend (Optional)
```bash
uvicorn src.server.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

## Verification

### Test Installation
```bash
python -c "from src.interface.streamlit_app import StreamlitApp; print('✅ Installation OK')"
```

### Run Tests
```bash
pytest
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'fastapi'**
```bash
pip install -r requirements.txt
# Or reinstall package
pip install -e .
```

**2. Java not found (LanguageTool)**
- Install Java 17+: https://adoptium.net/temurin/releases/
- Or disable grammar checking in settings

**3. OpenAI API errors**
- Check API key in `.env` file
- Verify account has credits
- App works without API key (uses local models)

**4. Memory/Performance issues**
- Use online models (set OPENAI_API_KEY)
- Reduce chunk size in settings
- Close other applications

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/TwinkleDreamm/ElevateAI_SummarizeApplication/issues)
- **Documentation**: Check README.md
- **Logs**: Check `logs/elevateai.log`

## Development Setup

### Additional Dev Dependencies
```bash
pip install -e ".[dev]"
```

### Code Quality Tools
```bash
# Format code
black .
isort .

# Type checking
mypy src/

# Linting
flake8 src/

# Tests with coverage
pytest --cov=src/
```

## Uninstall

```bash
# Uninstall package
pip uninstall elevateai

# Remove project directory (optional)
rm -rf ElevateAI_SummarizeApplication/
```
