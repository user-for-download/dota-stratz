package repository

import (
	"context"
	"testing"

	"github.com/dota-stratz/services/parser/internal/models"
	"github.com/stretchr/testify/assert"
)

func TestMaxMatchID_EmptySlice(t *testing.T) {
	result := maxMatchID([]models.OpenDotaMatch{})
	assert.Equal(t, int64(0), result)
}

func TestMaxMatchID_SingleMatch(t *testing.T) {
	matches := []models.OpenDotaMatch{
		{MatchID: 42},
	}
	result := maxMatchID(matches)
	assert.Equal(t, int64(42), result)
}

func TestMaxMatchID_MultipleMatches(t *testing.T) {
	matches := []models.OpenDotaMatch{
		{MatchID: 100},
		{MatchID: 999},
		{MatchID: 50},
		{MatchID: 500},
	}
	result := maxMatchID(matches)
	assert.Equal(t, int64(999), result)
}

func TestMaxMatchID_NegativeIDs(t *testing.T) {
	// Match IDs should always be positive, but maxMatchID should handle
	// edge cases gracefully.
	matches := []models.OpenDotaMatch{
		{MatchID: -1},
		{MatchID: 0},
		{MatchID: 42},
	}
	result := maxMatchID(matches)
	assert.Equal(t, int64(42), result)
}

func TestMaxMatchID_AllZero(t *testing.T) {
	matches := []models.OpenDotaMatch{
		{MatchID: 0},
		{MatchID: 0},
	}
	result := maxMatchID(matches)
	assert.Equal(t, int64(0), result)
}

func TestNewRepository_NonNil(t *testing.T) {
	// NewRepository requires a pool, so we can't test it without a real DB.
	// This is a placeholder to document the interface.
	// Integration tests should validate:
	// - Ping() returns nil when DB is reachable
	// - ReadCheckpoint() returns watermark when row exists
	// - WriteBatch() commits all statements atomically
	// - WriteBatch() rolls back on error
	t.Skip("Requires PostgreSQL integration — see deploy/migration/ for schema")
}

func TestWriteBatch_CancelledContext(t *testing.T) {
	t.Skip("Requires PostgreSQL integration — see deploy/migration/ for schema")

	// WriteBatch should return early when context is cancelled.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_ = ctx
}

// Note: Full integration tests for WriteBatch require:
// 1. A running PostgreSQL instance with the migration schema applied
// 2. A pgxpool connected to it
// 3. Real or synthetic OpenDotaMatch data
//
// Key scenarios to cover in integration tests:
// - Successful batch insert of 1+ matches
// - ON CONFLICT DO NOTHING idempotency (re-insert same data)
// - FK violation triggers constraint error (unseeded hero_id)
// - Checkpoint watermark upsert advances correctly
// - Checkpoint watermark GREATEST() prevents rewinding
// - Transaction rollback on batch exec failure
// - context.WithoutCancel keeps I/O alive during graceful shutdown
// - Large batch (100 matches) completes within 30s deadline
