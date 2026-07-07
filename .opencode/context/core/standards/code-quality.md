# Code Quality Standards

**Core principle**: Production code that is readable, testable, and maintainable.

## Go Code Standards
- **Language**: Go 1.26, standard library + minimal dependencies
- **Error handling**: Return errors, not panic. Use `fmt.Errorf("context: %w", err)` for wrapping
- **Concurrency**: Use `errgroup` for goroutine groups, `context.Context` for cancellation
- **Naming**: `camelCase` for unexported, `PascalCase` for exported. Avoid stutter (`pool.Pool`)
- **Imports**: stdlib first, third-party second, internal last, grouped with blank lines

## Python Code Standards
- **Language**: Python 3.12+, type hints required
- **ML Framework**: PyTorch 2.2+ with functional patterns
- **Performance**: Vectorized operations preferred over Python loops
- **Memory**: Use appropriate dtypes, batch operations when possible
- **Labels**: Use `make_target()` for correct team-relative labels

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
- Python code uses vectorized operations where possible
- ML training includes early stopping and proper label handling
