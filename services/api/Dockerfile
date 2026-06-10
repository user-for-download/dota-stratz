# ==============================================================================
# Dota 2 Match Analysis System — ML Inference API (online)
# ==============================================================================
# Build context: project root.
# ==============================================================================

# Stage 1: Builder — compile dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system deps needed for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime — minimal production image
FROM python:3.12-slim

# Runtime libs: libgomp1 (lightgbm OpenMP), libpq5 (psycopg2)
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

# Copy the API package
COPY services/api/api/ ./api/

# Mount point for model files — shared volume with trainer
VOLUME /models

# Switch to non-root user
USER appuser

EXPOSE 8080

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8080"]
