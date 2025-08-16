from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import health, ingest, search, summarize, stt

app = FastAPI(
    title="ElevateAI API",
    version="1.0.0",
    description="Backend API for ElevateAI - AI-powered document processing and summarization"
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers only (no HTML UI)
app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingest"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(summarize.router, prefix="/api", tags=["summarize"])
app.include_router(stt.router, prefix="/api", tags=["stt"])
