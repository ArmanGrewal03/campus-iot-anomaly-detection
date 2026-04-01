FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_ops.py .

EXPOSE 8004

CMD ["python", "ml_ops.py"]
