# ==============================================================================
# Dota 2 Match Analysis System — ML Trainer (batch)
# ==============================================================================
# Build context: project root (one level above this file).
# This is an *offline* batch service — runs PyTorch DraftBERT training via
# trainer.main, writes model files to the shared ml_models volume, then exits.
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
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Stage 2: Runtime — minimal production image
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
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

# Copy entrypoint wrapper for volume permission handling
COPY services/trainer/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Mount point for model output — shared volume with inference API
VOLUME /models

# Switch to non-root user
USER appuser

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "trainer.main"]
