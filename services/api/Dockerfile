# ==============================================================================
# Dota 2 Match Analysis System — ML Inference API (online)
# ==============================================================================
# Build context: project root.
# ==============================================================================

FROM python:3.12-slim

WORKDIR /app

# Install system deps (psycopg2 needs libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API package
COPY services/api/api/ ./api/

# Mount point for model files — shared volume with trainer
VOLUME /models

EXPOSE 8080

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
