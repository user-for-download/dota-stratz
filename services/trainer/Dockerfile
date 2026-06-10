# ==============================================================================
# Dota 2 Match Analysis System — ML Trainer (batch)
# ==============================================================================
# Build context: project root (one level above this file).
# This is an *offline* batch service — runs train.py, writes model files to
# the shared ml_models volume, then exits.
# ==============================================================================

# Stage 1: Builder — compile dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system deps needed for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY services/trainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime — minimal production image
FROM python:3.12-slim

# libgomp1 needed by lightgbm at runtime (OpenMP threading support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with pinned UID/GID
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --gid 1001 appuser

WORKDIR /app

# Copy only installed packages from builder (no gcc/build tools)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the trainer package
COPY services/trainer/trainer/ ./trainer/

# Mount point for model output — shared volume with inference API
VOLUME /models

# Switch to non-root user
USER appuser

ENTRYPOINT ["python", "-m", "trainer.main"]
