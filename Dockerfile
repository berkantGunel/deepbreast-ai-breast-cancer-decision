# ================================
# DeepBreast AI - Backend Dockerfile
# ================================
# Optimized single-stage build for broad compatibility (Linux/Mac/M1/Windows)

FROM python:3.11-slim

# System dependencies
# Includes fix for libgl1 (OpenCV) and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install
# Installing uvicorn/gunicorn explicitly to ensure availability in path
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn gunicorn

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Environment settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
# Ensure /app is in PYTHONPATH to find src module
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start application
# Using python -m uvicorn is safest for path resolution
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
