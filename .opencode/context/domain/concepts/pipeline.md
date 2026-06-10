# Pipeline Data Flow

**Core concept**: Event-driven microservice pipeline that ingests Dota 2 matches from OpenDota API through a multi-stage queue architecture into PostgreSQL for analytics and ML.

```
[OpenDota API] ──► [ID Fetcher] ──► [queue.match_ids] ──► [Detail Fetcher]
                                                                │
                                                                ▼
                                                       [queue.raw_matches]
                                                                │
                                                                ▼
                                                           [Parser]
                                                                │
                                                                ▼
                                                         [PostgreSQL]
```

## Key Points
- **No coordinator**: ID Fetcher owns its cron schedule (`FETCH_SCHEDULE`), rest of pipeline is reactive
- **Queue isolation**: `queue.match_ids` (IDs), `queue.raw_matches` (full JSON). Each has a DLQ
- **Idempotent inserts**: All DB writes use `ON CONFLICT DO NOTHING` — safe retry
- **FK violation fallback**: Parser detects 23503, routes offending match to DLQ, commits healthy matches
- **Graceful shutdown**: SIGINT/SIGTERM drains in-flight work via bounded wait groups with timeouts

## Services
| Stage | Service | Reads From | Writes To |
|-------|---------|-----------|-----------|
| 1 | ID Fetcher | OpenDota Explorer API | `queue.match_ids` |
| 2 | Detail Fetcher | `queue.match_ids`, OpenDota API | `queue.raw_matches` |
| 3 | Parser | `queue.raw_matches` | PostgreSQL (20+ tables) |
| — | Proxy Manager | — | Redis (proxy pool) |
