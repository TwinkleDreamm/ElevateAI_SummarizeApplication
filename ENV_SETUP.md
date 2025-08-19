# Environment Setup Guide for ElevateAI with LangChain/LangGraph

## Quick Start

1. **Create `.env` file with your configuration**
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Configure your API keys in `.env`**

## Environment Variables

### Required Configuration

```bash
# OpenAI Configuration (choose one)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2000
OPENAI_TOP_P=0.9
OPENAI_FREQUENCY_PENALTY=0.0
OPENAI_PRESENCE_PENALTY=0.0

# OR Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

### LangChain/LangGraph Features (Primary System)

```bash
# LangChain and LangGraph are the primary AI system
USE_STRUCTURED_OUTPUT=true
ENABLE_MEMORY=true

# Optional: LangSmith for debugging and monitoring
LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

### Optional Features

```bash
# Search APIs
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_google_cse_id_here
SERPAPI_API_KEY=your_serpapi_key_here

# Performance tuning
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE_MB=500
MAX_RESULTS=10
ENABLE_WEB_SEARCH=False
ENABLE_FUNCTION_CALLING=False
```

## Core Features with LangChain/LangGraph

### 1. Structured Output Parsing
- **Primary system** with `USE_STRUCTURED_OUTPUT=true`
- Extracts structured data from AI responses
- Provides confidence scores and citations
- Maps sources to specific parts of answers

### 2. Advanced Document Processing
- **Primary system** using LangChain document loaders for PDF, DOCX, TXT
- Smart text splitting with context preservation
- Automatic format detection and processing

### 3. Multi-Agent Workflows
- **QA Workflow**: Search → Analyze → Synthesize → Validate
- **Summarization Workflow**: Chunk → Analyze → Generate → Refine
- Automatic quality checking and iterative improvement

### 4. Enhanced Memory Management
- **Conversation memory** with automatic summarization
- **Entity memory** for tracking important information
- **Context window** management

### 5. Dynamic Settings Management
- **Real-time settings updates** without restart
- **Automatic LangchainLLMClient reconfiguration** when settings change
- **Settings validation and application** through UI
- **Configuration consistency checking** between UI and backend

## Troubleshooting

### LangChain/LangGraph Not Working
1. Check if packages are installed: `pip list | grep langchain`
2. Verify API keys are set correctly
3. Check logs for import errors

### Fallback Behavior
- If LangChain/LangGraph fails: falls back to legacy components
- If structured parsing fails: returns plain text
- Legacy components are kept for emergency fallback only

### Settings Update Issues
- If settings don't apply: check if LangchainLLMClient is available
- Use "Check AI Settings Status" button in Settings page to verify
- Settings are applied immediately when "Save & Apply" is clicked
- Some settings may require a page refresh to take full effect

### Performance Issues
- Reduce `CHUNK_SIZE` for faster processing
- Set `ENABLE_MEMORY=false` to disable memory features
- Use `fast_mode` in UI for quicker responses

## API Key Setup

### OpenAI
1. Visit https://platform.openai.com/api-keys
2. Create new API key
3. Add to `.env`: `OPENAI_API_KEY=sk-...`

### OpenAI Model Parameters
```bash
# Model configuration
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=2000
OPENAI_TOP_P=0.9
OPENAI_FREQUENCY_PENALTY=0.0
OPENAI_PRESENCE_PENALTY=0.0
```

### Custom OpenAI-Compatible Providers (Ollama, LM Studio, etc.)
1. Set your custom provider's base URL
2. Add to `.env`:
   ```
   OPENAI_API_KEY=your_api_key_here
   OPENAI_BASE_URL=http://localhost:11434/v1  # Example: Ollama
   # or
   OPENAI_BASE_URL=http://localhost:1234/v1   # Example: LM Studio
   OPENAI_CHAT_MODEL=llama3.2:3b             # Use model name your provider supports
   ```
3. The system will automatically detect and use the custom base URL

### Azure OpenAI
1. Create Azure OpenAI resource
2. Get endpoint and API key from Azure portal
3. Add to `.env`:
   ```
   AZURE_OPENAI_API_KEY=your_key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
   ```

### Audio & Interface Settings
```bash
# Audio processing
ENABLE_TTS=True
TTS_VOICE=alloy
AUDIO_SAMPLE_RATE=16000
NOISE_REDUCTION_STRENGTH=0.8
ENABLE_VOCAL_SEPARATION=False

# Interface
THEME=light
AUTO_SAVE=True
SHOW_PROCESSING_TIME=True
SHOW_CONFIDENCE_SCORE=True
ENABLE_ANIMATIONS=True

# Performance
CACHE_ENABLED=True
ENABLE_METRICS=True
BACKUP_ENABLED=True
```

### LangSmith (Optional)
1. Visit https://smith.langchain.com/
2. Create account and get API key
3. Add to `.env`:
   ```
   LANGCHAIN_API_KEY=ls_...
   ```

## Development

### Testing Core System
```bash
# Test LangChain components
python -c "from src.ai.langchain import LangchainPromptManager; print('OK')"

# Test LangGraph workflows
python -c "from src.ai.langgraph import create_qa_workflow; print('OK')"
```

### Disabling Features
```bash
# Disable structured output
USE_STRUCTURED_OUTPUT=false

# Disable memory features
ENABLE_MEMORY=false
```
