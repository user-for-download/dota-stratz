# Service Quick Reference

| Service | Package | Lang | Port | Deps | Config |
|---------|---------|------|------|------|--------|
| ID Fetcher | `services/id-fetcher/` | Go | 9094 | Redis, RabbitMQ | `config/config.yaml` |
| Detail Fetcher | `services/detail-fetcher/` | Go | 9091 | Redis, RabbitMQ | `config/config.yaml` |
| Parser | `services/parser/` | Go | 9093 | PostgreSQL, RabbitMQ | `config/config.yaml` |
| Proxy Manager | `services/proxy-manager/` | Go | 9090 | Redis | Env vars only |
| Trainer | `services/trainer/` | Python | — | PostgreSQL | Env vars |
| API | `services/api/` | Python | 8080 | PostgreSQL | Env vars |

## Shared Library
**Module**: `github.com/dota-stratz/shared/go-common` at `shared/go-common/`

| Package | Description |
|---------|-------------|
| `cache` | Redis connection with 3-retry ping |
| `db` | pgxpool connection with ping |
| `mq` | AMQP connection + channel |
| `logger` | Global zap.Logger from `LOG_LEVEL` |
| `proxypool` | Redis-backed proxy pool (~710 lines). `MakeTransport` supports HTTP/HTTPS/SOCKS4/SOCKS5 |
