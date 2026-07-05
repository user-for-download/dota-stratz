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
