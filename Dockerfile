FROM python:3.10-slim

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
 
# Install system dependencies required for OCR, image processing and common binaries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-fra \
        tesseract-ocr-ara \
        libsm6 \
        libxext6 \
        libxrender1 \
        libglib2.0-0 \
        ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY . /app

# Create persistent data directories expected by the app
RUN mkdir -p /app/data/models /app/data/samples /app/data/temp /app/downloaddoc

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    TESSERACT_PATH=/usr/bin/tesseract \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LANG=C.UTF-8

# Expose API port
EXPOSE 8000

# Run the FastAPI app via Uvicorn. The project exposes the FastAPI app object at `api.app:app`.
# Using Uvicorn directly avoids relying on the CLI wrapper and gives better process management.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
