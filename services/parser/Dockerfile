FROM golang:1.26-alpine AS builder
WORKDIR /app
# Workspace definition — must be first so Go knows which local modules to use.
COPY go.work ./
# Copy the entire source so go work sync operates on complete module state.
COPY . .
RUN go work sync
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /parser ./services/parser

FROM alpine:3.21
RUN apk add --no-cache ca-certificates tzdata
RUN adduser -D appuser
COPY --from=builder /parser /parser
COPY services/parser/config/config.yaml /app/config/config.yaml
WORKDIR /app
USER appuser
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD wget -qO- http://localhost:9093/healthz || exit 1
ENTRYPOINT ["/parser"]
