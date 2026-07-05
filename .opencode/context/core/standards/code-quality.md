# Code Quality Standards

**Core principle**: Production Go code that is readable, testable, and maintainable.

## Key Points
- **Language**: Go 1.26, standard library + minimal dependencies
- **Error handling**: Return errors, not panic. Use `fmt.Errorf("context: %w", err)` for wrapping
- **Concurrency**: Use `errgroup` for goroutine groups, `context.Context` for cancellation
- **Naming**: `camelCase` for unexported, `PascalCase` for exported. Avoid stutter (`pool.Pool`)
- **Imports**: stdlib first, third-party second, internal last, grouped with blank lines

## Patterns
- **Config**: YAML + env-var substitution via `os.Getenv` defaults in `config/config.go`
- **Metrics**: Prometheus counters/histograms in `internal/metrics/` per service
- **Logging**: Global `zap.Logger` via `shared/go-common/logger`
- **Health checks**: `GET /healthz` returning `"ok"` on every service

## Review Gates
- All `go vet` + `gofmt -s` clean
- No `context.Background()` in request paths (use request context)
- No naked `go func()` without wait group or errgroup
- All env vars documented in `deploy/.env.example`
