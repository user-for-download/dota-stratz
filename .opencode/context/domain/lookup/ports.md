# Port Allocation Map

| Port | Service | Notes |
|------|---------|-------|
| 5432 | PostgreSQL | Data port |
| 5672 | RabbitMQ | AMQP |
| 15672 | RabbitMQ | Management UI |
| 15692 | RabbitMQ | Prometheus metrics |
| 6379 | Redis | Data port |
| 9090 | Proxy Manager | Metrics + health |
| 9091 | Detail Fetcher | Metrics |
| 9092 | Prometheus | Self (host network) |
| 9093 | Parser | Metrics + health |
| 9094 | ID Fetcher | Metrics + health |
| 8080 | ML Inference API | Prediction endpoint |

Prometheus + Grafana use `network_mode: host`. All other services connect via `dota2-net` bridge network.
