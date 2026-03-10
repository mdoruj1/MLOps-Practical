# ─────────────────────────────────────────────────────────────────
# Phase 3: Dockerfile — Iris Classifier API
# Build : docker build -t iris-classifier:v1 .
# Run   : docker run -d -p 80:80 --name iris-api iris-classifier:v1
# Test  : curl http://localhost/health
# Docs  : http://localhost/docs
# ─────────────────────────────────────────────────────────────────

# 1. Lightweight base image (slim = no build tools → smaller image)
FROM python:3.9-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy requirements FIRST (Docker layer caching: only re-run pip
#    install when requirements.txt changes, not on every code change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application source code
COPY main.py .
COPY models/ ./models/

# 5. Expose port 80 (documentation only; -p flag does the actual mapping)
EXPOSE 80

# 6. Non-root user for security (best practice in production)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# 7. Start Uvicorn ASGI server
#    --host 0.0.0.0  : Listen on all interfaces (required inside container)
#    --port 80       : Map to the exposed port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
