# ── DetectFarm Dockerfile for Hugging Face Spaces ─────────────────────────────
#
# Hugging Face Spaces (Docker SDK) requires:
#   - Port 7860 exposed
#   - App listening on 0.0.0.0:7860
#   - No root-only file writes (use /tmp or app directory)
#
# Build locally: docker build -t detectfarm .
# Run locally:   docker run -p 7860:7860 detectfarm
# On HF Spaces:  push this file + all project files to the Space's git repo

FROM python:3.11-slim

# Install system dependencies required by OpenCV and scikit-image
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (Docker layer caching — faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create the output directory (must exist before app starts)
RUN mkdir -p static/output

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Environment variables
ENV FLASK_DEBUG=false
ENV PORT=7860
# Set GEMINI_API_KEY as a Secret in HF Spaces settings (not here)
# ENV GEMINI_API_KEY=your_key_here  ← NEVER put your actual key in Dockerfile

# Use gunicorn for production (not Flask dev server)
# - 1 worker (HF free tier has limited RAM)
# - timeout 120s (image processing can take 20-30s)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
