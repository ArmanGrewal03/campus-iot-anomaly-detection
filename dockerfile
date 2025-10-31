# Start from a lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (for PostgreSQL and other packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency file first (for Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Run the Flask app (assuming app.py is your entrypoint)
# You can also use 'flask run' if you’ve set FLASK_APP env var
CMD ["python", "app.py"]
