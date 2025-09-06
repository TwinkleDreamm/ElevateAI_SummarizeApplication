# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Avoid interactive prompts during apt operations
ARG DEBIAN_FRONTEND=noninteractive

# Base envs for Python and Streamlit in containerized envs
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8080

# Workdir
WORKDIR /app

# System deps first (audio/video, graphviz, magic, build tools for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    graphviz \
    git \
    build-essential \
    libsndfile1 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./

# Patch a known typo in requirements to ensure successful install
# blinker version '1.9.0G' is invalid; fix to '1.9.0'
RUN sed -i 's/^blinker==1\.9\.0G$/blinker==1.9.0/' requirements.txt

# Upgrade pip and install Python deps
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code
COPY . .

## Create a non-root user and give permissions to app dirs
RUN useradd -ms /bin/bash appuser \
    && mkdir -p /app/data /app/logs /app/config /app/tmp \
    && chown -R appuser:appuser /app

# Switch to non-root for better security on Cloud Run
USER appuser

# Expose the local default Streamlit port (Cloud Run ignores EXPOSE and uses $PORT)
EXPOSE 8501

# Cloud Run will inject $PORT; start_app.py defaults to 8501 locally and binds 0.0.0.0
CMD ["bash", "-lc", "python -u start_app.py"]
