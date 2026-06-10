# ==============================================================================
# Dota 2 Match Analysis System — id-fetcher
# ==============================================================================
# Build context: project root (one level above this file).
#   docker build -f services/id-fetcher/Dockerfile -t dota2-id-fetcher .
# ==============================================================================

FROM golang:1.26-alpine AS builder

WORKDIR /app

# Workspace definition — must be first so Go knows which local modules to use.
COPY go.work ./

# Copy the entire source so go work sync operates on complete module state.
COPY . .

# Download all workspace dependencies.
RUN go work sync

# Build a fully static binary.
RUN CGO_ENABLED=0 go build \
    -trimpath \
    -ldflags="-s -w" \
    -o /id-fetcher \
    ./services/id-fetcher

# ------------------------------------------------------------------------------
# Stage 2 — Minimal runtime image
# ------------------------------------------------------------------------------
FROM alpine:3.21
RUN apk add --no-cache ca-certificates tzdata
RUN adduser -D appuser

COPY --from=builder /id-fetcher /id-fetcher
COPY services/id-fetcher/config/config.yaml /app/config/config.yaml

WORKDIR /app
USER appuser
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD wget -qO- http://localhost:9094/healthz || exit 1
ENTRYPOINT ["/id-fetcher"]
