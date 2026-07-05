# Test Coverage Standards

**Core principle**: Tests protect the pipeline. Cover integration paths, edge cases, and failure modes.

## Key Points
- **Framework**: Go standard `testing` package + `testify/assert`
- **Location**: `*_test.go` alongside implementation, integration tests in `internal/test/`
- **Pattern**: Table-driven tests for multiple cases, `t.Run()` for sub-tests
- **Coverage**: Target >70% for service packages, >40% for shared library
- **CI gate**: `make test` runs all module tests; `make check` = fmt + vet + test

## What to Test
- **Integration paths**: Happy path through each service (fetch → queue → parse → DB)
- **Edge cases**: Nil/empty data, network timeouts, duplicate messages, partial batches
- **Failure modes**: Proxy pool depletion, FK violations, broker restart, shutdown mid-batch
- **No mocks**: Use testcontainers or Docker Compose for Postgres/RabbitMQ/Redis

## Test Patterns
- Use `t.Parallel()` for independent sub-tests
- Use `testdata/` for fixture files (sample match JSON, etc.)
- Clean up resources with `t.Cleanup()` — never leak containers or goroutines
