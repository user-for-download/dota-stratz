# ==============================================================================
# Dota 2 Match Analysis System — ML Trainer (batch)
# ==============================================================================
# Build context: project root (one level above this file).
# This is an *offline* batch service — runs train.py, writes model files to
# the shared ml_models volume, then exits.
# ==============================================================================

FROM python:3.12-slim

WORKDIR /app

# Install system deps (psycopg2 needs libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY services/trainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trainer package
COPY services/trainer/trainer/ ./trainer/

# Mount point for model output — shared volume with inference API
VOLUME /models

ENTRYPOINT ["python", "-m", "trainer.main"]
