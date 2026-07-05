package db

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

// Connect creates a pgxpool.Pool from a DSN string.
// If maxConns > 0 the pool will be capped at that value; otherwise the pgx
// default (4 × GOMAXPROCS) is used. Callers processing batch workloads
// (e.g. the parser) should tune this per their peak concurrency.
// Sets MaxConnIdleTime (5m) and MaxConnLifetime (1h) to prevent broken
// connections from accumulating after network changes or DB restarts.
func Connect(ctx context.Context, dsn string, maxConns int) (*pgxpool.Pool, error) {
	config, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("unable to parse database DSN: %w", err)
	}

	if maxConns > 0 {
		config.MaxConns = int32(maxConns)
	}

	// Prevent silent connection rot from firewalls/NAT dropping idle TCP connections.
	config.MaxConnIdleTime = 5 * time.Minute
	config.MaxConnLifetime = 1 * time.Hour
	config.HealthCheckPeriod = 1 * time.Minute

	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("unable to create connection pool: %w", err)
	}

	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("unable to ping database: %w", err)
	}

	return pool, nil
}
