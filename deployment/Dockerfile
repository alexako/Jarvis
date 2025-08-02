# Multi-stage Dockerfile for Jarvis API production deployment
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libportaudio2 \
    libportaudio-dev \
    libasound2-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Piper TTS
RUN curl -L -o piper.tar.gz https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz \
    && tar -xzf piper.tar.gz \
    && mv piper/piper /usr/local/bin/ \
    && chmod +x /usr/local/bin/piper \
    && rm -rf piper.tar.gz piper

# Create non-root user
RUN groupadd -r jarvis && useradd -r -g jarvis jarvis

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install production-specific dependencies
RUN pip install --no-cache-dir \
    uvicorn[standard]==0.24.0 \
    gunicorn==21.2.0 \
    slowapi==0.1.9 \
    pyjwt==2.8.0 \
    prometheus-client==0.19.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/audio \
    && chown -R jarvis:jarvis /app

# Download Piper model
RUN mkdir -p /app/models/piper \
    && curl -L -o /app/models/piper/en_GB-alan-medium.onnx \
       https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx \
    && curl -L -o /app/models/piper/en_GB-alan-medium.onnx.json \
       https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json \
    && chown -R jarvis:jarvis /app/models

# Switch to non-root user
USER jarvis

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "jarvis_api_production.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]