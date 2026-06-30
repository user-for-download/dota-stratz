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

COPY go.work go.work.sum ./

COPY services/id-fetcher/go.mod services/id-fetcher/go.sum ./services/id-fetcher/
COPY services/detail-fetcher/go.mod services/detail-fetcher/go.sum ./services/detail-fetcher/
COPY services/parser/go.mod services/parser/go.sum ./services/parser/
COPY services/proxy-manager/go.mod services/proxy-manager/go.sum ./services/proxy-manager/
COPY shared/go-common/go.mod shared/go-common/go.sum ./shared/go-common/

RUN go work sync

COPY . .

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
ENTRYPOINT ["/proxy-manager"]
