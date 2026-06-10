# ==============================================================================
# Dota 2 Match Analysis System — proxy-manager
# ==============================================================================
# Build context: project root (one level above this file).
#   docker build -f services/proxy-manager/Dockerfile -t dota2-proxy-manager .
#
# Bake:           make bake or make bake-proxy-manager
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1 — Build the binary
# ------------------------------------------------------------------------------
FROM golang:1.26-alpine AS builder

WORKDIR /app

# Workspace definition — must be first so Go knows which local modules to use.
COPY go.work ./

# Copy the entire source first so that go work sync operates on the
# complete module state (go.mod files may be updated by the sync).
COPY . .

# Download all workspace dependencies (cached until a manifest changes).
RUN go work sync

# Build a fully static binary.
RUN CGO_ENABLED=0 go build \
    -trimpath \
    -ldflags="-s -w" \
    -o /proxy-manager \
    ./services/proxy-manager

# ------------------------------------------------------------------------------
# Stage 2 — Minimal runtime image
# ------------------------------------------------------------------------------
FROM alpine:3.21

# File descriptor notes:
# PROXY_VALIDATION_CONCURRENCY (default 150) workers each open a fresh TCP
# connection per proxy check. PAM's limits.conf is a no-op inside containers;
# set ulimits at the container runtime level instead:
#   docker run --ulimit nofile=8192:8192 ...
#   docker-compose: proxy-manager: ulimits: nofile: soft: 8192 hard: 8192
RUN apk add --no-cache ca-certificates tzdata
RUN adduser -D appuser

COPY --from=builder /proxy-manager /proxy-manager

USER appuser
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD wget -qO- http://localhost:${PROXY_METRICS_PORT:-9090}/healthz || exit 1
ENTRYPOINT ["/proxy-manager"]
