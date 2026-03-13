FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for scikit-learn
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p models

# Expose ports
# 8001 for HTTP API, 50052 for gRPC
EXPOSE 8001 50052

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')"

# Run the application
CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8001"]
