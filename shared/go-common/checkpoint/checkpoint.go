// Package checkpoint holds constants and small helpers shared by the
// services that read/write the ingestion_checkpoints table.
//
// The table itself (deploy/migration/001_init.sql) is a single-row
// configuration table keyed by id=1; it does NOT have a pipeline_name
// column. The constants below are used in structured log fields and
// (eventually) in cross-service metrics so we can correlate writes
// from different pipeline components without depending on a string
// stored in the database.
package checkpoint

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// CheckpointPipelineParser is the logical pipeline name used by the
// parser service when it writes to ingestion_checkpoints. It is not
// persisted as a column value — it is a Go-side identifier for logs
// and (future) metrics.
const CheckpointPipelineParser = "parser"

// CheckpointPipelineIDFetcher is the logical pipeline name used by the
// id-fetcher when it reads from ingestion_checkpoints.
const CheckpointPipelineIDFetcher = "id-fetcher"

// readCheckpointSQL selects the parser's last_parsed_match_id watermark
// from the single-row ingestion_checkpoints table. Centralised here so
// the parser (which writes the value) and the id-fetcher (which reads
// it) cannot drift out of sync on the column name.
const readCheckpointSQL = `SELECT last_parsed_match_id FROM ingestion_checkpoints WHERE id = 1`

// ReadWatermark returns the parser's high-water mark from
// ingestion_checkpoints.
//
// Return contract:
//   - (watermark, true,  nil)  → row exists; use watermark as the
//     "match_id > X" lower bound for the next fetch.
//   - (0,         false, nil)  → row is missing entirely (e.g. the
//     migration that seeds the row has not been applied yet, or a
//     fresh DB). Callers should treat this as "no watermark yet" and
//     fall back to the bootstrap (rolling window) path.
//   - (0,         false, err)  → query error; callers should log and
//     fall back to the rolling window.
//
// The bool distinguishes "watermark=0 row exists" from "row missing".
// A row with watermark=0 is a legitimate state (e.g. the parser has
// committed a batch whose max match_id is 0, which is unlikely but
// possible if OpenDota ever reuses IDs). The id-fetcher should use
// the watermark only when it is > 0.
func ReadWatermark(ctx context.Context, pool *pgxpool.Pool) (int64, bool, error) {
	if ctx.Err() != nil {
		return 0, false, ctx.Err()
	}
	if pool == nil {
		return 0, false, fmt.Errorf("checkpoint.ReadWatermark: pool is nil")
	}
	var watermark int64
	err := pool.QueryRow(ctx, readCheckpointSQL).Scan(&watermark)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return 0, false, nil
		}
		return 0, false, fmt.Errorf("failed to read checkpoint: %w", err)
	}
	return watermark, true, nil
}
