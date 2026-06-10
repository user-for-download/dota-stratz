package repository

import (
	"strings"
	"testing"

	"github.com/dota-stratz/services/parser/internal/models"
)

// TestMaxMatchID verifies the pure helper used to compute the
// parser's watermark for a batch.
//
//   - empty input        → 0 (caller guards with `if watermark > 0`)
//   - single match       → its match_id
//   - out-of-order       → the maximum, not the last
//   - negative IDs       → still finds the max (defensive — OpenDota
//     should never return negatives, but the upsert uses GREATEST
//     anyway so a bug here would be caught at the DB layer)
func TestMaxMatchID(t *testing.T) {
	tests := []struct {
		name    string
		matches []models.OpenDotaMatch
		want    int64
	}{
		{
			name:    "empty slice",
			matches: nil,
			want:    0,
		},
		{
			name:    "single match",
			matches: []models.OpenDotaMatch{{MatchID: 42}},
			want:    42,
		},
		{
			name: "out of order",
			matches: []models.OpenDotaMatch{
				{MatchID: 100},
				{MatchID: 50},
				{MatchID: 200},
				{MatchID: 75},
			},
			want: 200,
		},
		{
			name: "first element is max",
			matches: []models.OpenDotaMatch{
				{MatchID: 999},
				{MatchID: 1},
				{MatchID: 2},
			},
			want: 999,
		},
		{
			name: "last element is max",
			matches: []models.OpenDotaMatch{
				{MatchID: 1},
				{MatchID: 2},
				{MatchID: 999},
			},
			want: 999,
		},
		{
			name: "all zeros",
			matches: []models.OpenDotaMatch{
				{MatchID: 0},
				{MatchID: 0},
			},
			want: 0,
		},
		{
			name: "mix with zero (zero should be ignored as max)",
			matches: []models.OpenDotaMatch{
				{MatchID: 0},
				{MatchID: 10},
				{MatchID: 0},
			},
			want: 10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := maxMatchID(tt.matches)
			if got != tt.want {
				t.Errorf("maxMatchID = %d, want %d", got, tt.want)
			}
		})
	}
}

// TestCheckpointUpsertSQLStructure verifies that the checkpoint
// upsert SQL is shaped the way the spec requires. We do NOT execute
// the SQL here (no real DB available in this unit test) — the goal
// is to catch refactors that accidentally drop the GREATEST(...)
// monotonicity guarantee or change the conflict target.
//
// Specifically the SQL must:
//   - INSERT into ingestion_checkpoints (id, last_parsed_match_id)
//   - target the single-row (id=1) primary key on conflict
//   - use GREATEST(current, EXCLUDED) so a late batch with a smaller
//     match_id can never rewind the watermark
//   - bump updated_at on every upsert
//
// If any of these break, the id-fetcher can re-emit already-parsed
// match IDs and the pipeline regresses to the pre-P1-1 state.
func TestCheckpointUpsertSQLStructure(t *testing.T) {
	sql := checkpointUpsertSQL

	// Required fragments (case-insensitive, collapsed whitespace).
	requiredFragments := []string{
		"INSERT INTO ingestion_checkpoints",
		"ON CONFLICT (id) DO UPDATE",
		"GREATEST",
		"EXCLUDED.last_parsed_match_id",
		"updated_at = NOW",
		"$1", // parameterised watermark
	}

	normalised := strings.ToLower(strings.Join(strings.Fields(sql), " "))
	for _, frag := range requiredFragments {
		fragLower := strings.ToLower(strings.Join(strings.Fields(frag), " "))
		if !strings.Contains(normalised, fragLower) {
			t.Errorf("checkpointUpsertSQL missing required fragment %q.\nGot: %s", frag, sql)
		}
	}

	// Defensive: the upsert must NOT update the id column. Updating
	// id would either violate the CHECK (id = 1) constraint or — if
	// the new value happened to be 1 — silently break the
	// "single-row table" invariant. Either way it's a logic bug.
	if strings.Contains(normalised, "set id =") {
		t.Errorf("checkpointUpsertSQL must not update the id column.\nGot: %s", sql)
	}
}
