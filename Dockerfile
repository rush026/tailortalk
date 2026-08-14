FROM python:3.11-slim

# System dependencies for OpenCV headless + image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models so they're baked into the image
RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
    AutoImageProcessor.from_pretrained('facebook/dinov2-base'); \
    AutoModel.from_pretrained('facebook/dinov2-base')"
RUN python -c "from rembg import new_session; new_session('u2net')"

# Copy application code
COPY . .

# Streamlit config
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
