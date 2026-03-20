FROM python:3.11-slim

WORKDIR /app

# Install build tools that some Python packages may need
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8004/health')"

CMD ["uvicorn", "llm_service:app", "--host", "0.0.0.0", "--port", "8004"]
