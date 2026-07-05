package cache

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// Connect creates a new Redis client and verifies the connection with retries.
// Retries up to 3 times with exponential backoff (1s, 2s, 4s) to handle the
// startup race where Redis is still being initialised. The caller can cancel
// the retry loop via ctx.
func Connect(ctx context.Context, addr string, password string, db int) (*redis.Client, error) {
	rdb := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})

	const maxRetries = 3
	var lastErr error
	for attempt := range maxRetries {
		if err := rdb.Ping(ctx).Err(); err == nil {
			return rdb, nil
		} else {
			lastErr = err
		}
		if attempt < maxRetries-1 {
			select {
			case <-ctx.Done():
				rdb.Close()
				return nil, fmt.Errorf("redis connect cancelled after %d retries: %w", attempt+1, ctx.Err())
			case <-time.After(time.Duration(1<<attempt) * time.Second):
			}
		}
	}
	rdb.Close()
	return nil, fmt.Errorf("failed to connect to Redis after %d retries: %w", maxRetries, lastErr)
}
