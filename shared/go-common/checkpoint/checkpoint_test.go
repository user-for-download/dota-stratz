package checkpoint

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestReadWatermark_CancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled

	watermark, ok, err := ReadWatermark(ctx, nil)
	assert.ErrorIs(t, err, context.Canceled)
	assert.False(t, ok)
	assert.Equal(t, int64(0), watermark)
}

func TestReadWatermark_NilPool(t *testing.T) {
	watermark, ok, err := ReadWatermark(context.Background(), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "pool is nil")
	assert.False(t, ok)
	assert.Equal(t, int64(0), watermark)
}

func TestReadWatermark_DeadlineExceeded(t *testing.T) {
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-1*time.Hour))
	defer cancel()

	watermark, ok, err := ReadWatermark(ctx, nil)
	assert.Error(t, err)
	assert.False(t, ok)
	assert.Equal(t, int64(0), watermark)
}

func TestConstants(t *testing.T) {
	assert.Equal(t, "parser", CheckpointPipelineParser)
	assert.Equal(t, "id-fetcher", CheckpointPipelineIDFetcher)
}

// Note: ReadWatermark paths that require a real pgxpool (successful read,
// pgx.ErrNoRows, query error) are not covered by these unit tests. They
// require a PostgreSQL integration test environment.
