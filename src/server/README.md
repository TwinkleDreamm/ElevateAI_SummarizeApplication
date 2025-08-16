# ElevateAI FastAPI Backend

Backend API server for ElevateAI Streamlit frontend.

## Run API Server

```bash
conda activate Project_1
uvicorn src.server.main:app --reload --port 8000
```

## API Endpoints

- **GET /healthz** - Health check
- **POST /api/ingest/text** - Add documents to vector database
- **POST /api/search** - Semantic search
- **POST /api/summarize** - Summarize text chunks
- **POST /api/summarize-from-search** - Search + summarize in one step
- **POST /api/stt** - Speech-to-text (multipart/form-data)

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## CORS Configuration

Configured for Streamlit frontend (localhost:8501). Adjust for production.

