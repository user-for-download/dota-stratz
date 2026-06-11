package repository

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/dota-stratz/services/parser/internal/models"
	"github.com/dota-stratz/shared/go-common/checkpoint"
	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"go.uber.org/zap"
)

// checkpointUpsertSQL is the SQL used to advance the parser watermark in
// ingestion_checkpoints after a successful batch commit.
//
// Schema note: the ingestion_checkpoints table in deploy/migration/001_init.sql
// is a single-row configuration table (PRIMARY KEY id with a CHECK constraint
// that id=1), not a multi-pipeline registry keyed by pipeline_name. We
// therefore upsert on (id=1) and use GREATEST(..., EXCLUDED.last_parsed_match_id)
// to guarantee the watermark is monotonic — a late-arriving batch with a
// smaller match_id can never rewind the watermark and cause the id-fetcher
// to re-fetch already-parsed matches.
//
// The constant pipeline identifier used by the parser lives in
// shared/go-common/checkpoint as CheckpointPipelineParser; the table itself
// does not store that string — it is used only in structured logs.
const checkpointUpsertSQL = `
INSERT INTO ingestion_checkpoints (id, last_parsed_match_id)
VALUES (1, $1)
ON CONFLICT (id) DO UPDATE SET
    last_parsed_match_id = GREATEST(
        ingestion_checkpoints.last_parsed_match_id,
        EXCLUDED.last_parsed_match_id
    ),
    updated_at = NOW()`

// maxMatchID returns the largest match_id in the slice, or 0 when the slice
// is empty. Used to compute the watermark we write to ingestion_checkpoints.
func maxMatchID(matches []models.OpenDotaMatch) int64 {
	var maxID int64
	for _, m := range matches {
		if m.MatchID > maxID {
			maxID = m.MatchID
		}
	}
	return maxID
}

// Repository handles batch writes of parsed match data into Postgres.
// Each WriteBatch call is wrapped in a single transaction for atomicity.
type Repository struct {
	pool *pgxpool.Pool
}

func NewRepository(pool *pgxpool.Pool) *Repository {
	return &Repository{pool: pool}
}

// Ping checks the database connection health.
func (r *Repository) Ping(ctx context.Context) error {
	return r.pool.Ping(ctx)
}

// ReadCheckpoint returns the parser's high-water mark from
// ingestion_checkpoints. Thin wrapper over the shared
// checkpoint.ReadWatermark so the column name and (id=1) key live in
// exactly one place — both the parser (writer) and the id-fetcher
// (reader) use the same SQL.
//
// The ok bool is false when the row is missing entirely (e.g. the
// migration that seeds the row has not been applied yet). Callers should
// treat that as "fresh DB, use the bootstrap path".
func (r *Repository) ReadCheckpoint(ctx context.Context) (int64, bool, error) {
	return checkpoint.ReadWatermark(ctx, r.pool)
}

// WriteBatch inserts an entire batch of matches in a single transaction.
// All inserts use ON CONFLICT DO NOTHING for idempotency.
func (r *Repository) WriteBatch(ctx context.Context, matches []models.OpenDotaMatch) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}

	tx, err := r.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	// Use background context for Rollback — the caller's ctx may already be
	// cancelled during shutdown, and a cancelled Rollback leaks a server-side
	// transaction that only gets cleaned up when the connection is returned.
	cleanupCtx := context.Background()
	defer func() {
		// Rollback is a no-op if the transaction was already committed.
		// pgx returns ErrTxClosed in that case, which is the expected path
		// on success. We only log on unexpected errors.
		if rerr := tx.Rollback(cleanupCtx); rerr != nil && !errors.Is(rerr, pgx.ErrTxClosed) {
			// If the rollback itself fails, the connection is in an unknown
			// state. The pool will eventually recover via health checks.
			logger.Log.Warn("Transaction rollback failed (connection may be corrupted)",
				zap.Error(rerr))
		}
	}()

	batch := &pgx.Batch{}

	for _, m := range matches {
		// 1. Matches table
		writeMatch(batch, m)

		// 2. Players (110 columns matching the schema)
		for _, p := range m.Players {
			writePlayer(batch, m, p)

			// --- Player child tables ---
			// Each helper returns an error on JSON schema mismatch so the
			// entire match is sent to the DLQ instead of silently dropping data.

			if err := writeKillsLog(batch, m.MatchID, p.PlayerSlot, p.KillsLog); err != nil {
				return fmt.Errorf("match %d player %d kills_log: %w", m.MatchID, p.PlayerSlot, err)
			}
			if err := writeBuybackLog(batch, m.MatchID, p.PlayerSlot, p.BuybackLog); err != nil {
				return fmt.Errorf("match %d player %d buyback_log: %w", m.MatchID, p.PlayerSlot, err)
			}
			if err := writeRunesLog(batch, m.MatchID, p.PlayerSlot, p.RunesLog); err != nil {
				return fmt.Errorf("match %d player %d runes_log: %w", m.MatchID, p.PlayerSlot, err)
			}
			if err := writePurchaseLog(batch, m.MatchID, p.PlayerSlot, p.PurchaseLog); err != nil {
				return fmt.Errorf("match %d player %d purchase_log: %w", m.MatchID, p.PlayerSlot, err)
			}
			if err := writeObsLog(batch, m.MatchID, p.PlayerSlot, p.ObsLog, "player_obs_log"); err != nil {
				return fmt.Errorf("match %d player %d obs_log: %w", m.MatchID, p.PlayerSlot, err)
			}
			if err := writeObsLog(batch, m.MatchID, p.PlayerSlot, p.SenLog, "player_sen_log"); err != nil {
				return fmt.Errorf("match %d player %d sen_log: %w", m.MatchID, p.PlayerSlot, err)
			}
			if err := writeObsLeftLog(batch, m.MatchID, p.PlayerSlot, p.ObsLeftLog, "player_obs_left_log"); err != nil {
				return fmt.Errorf("match %d player %d obs_left_log: %w", m.MatchID, p.PlayerSlot, err)
			}
			if err := writeObsLeftLog(batch, m.MatchID, p.PlayerSlot, p.SenLeftLog, "player_sen_left_log"); err != nil {
				return fmt.Errorf("match %d player %d sen_left_log: %w", m.MatchID, p.PlayerSlot, err)
			}
			if err := writeAbilityUpgrades(batch, m.MatchID, p.PlayerSlot, p.AbilityUpgradesArr); err != nil {
				return fmt.Errorf("match %d player %d ability_upgrades: %w", m.MatchID, p.PlayerSlot, err)
			}
			if err := writeBenchmarks(batch, m.MatchID, p.PlayerSlot, p.Benchmarks); err != nil {
				return fmt.Errorf("match %d player %d benchmarks: %w", m.MatchID, p.PlayerSlot, err)
			}
			if err := writePermanentBuffs(batch, m.MatchID, p.PlayerSlot, p.PermanentBuffs); err != nil {
				return fmt.Errorf("match %d player %d permanent_buffs: %w", m.MatchID, p.PlayerSlot, err)
			}
			if err := writeNeutralItemHistory(batch, m.MatchID, p.PlayerSlot, p.NeutralItemHistory); err != nil {
				return fmt.Errorf("match %d player %d neutral_item_history: %w", m.MatchID, p.PlayerSlot, err)
			}
			// 2f. Minute-by-minute gold/XP arrays (JSONB, minute=0 sentinel)
			writeMinuteStats(batch, m.MatchID, p.PlayerSlot, p.GoldT, p.XPT)
		}

		// 3. Picks/Bans
		if err := writePicksBans(batch, m.MatchID, m.PicksBans); err != nil {
			return fmt.Errorf("match %d picks_bans: %w", m.MatchID, err)
		}

		// 4. Objectives
		if err := writeObjectives(batch, m.MatchID, m.Objectives); err != nil {
			return fmt.Errorf("match %d objectives: %w", m.MatchID, err)
		}

		// 5. Chat
		if err := writeChat(batch, m.MatchID, m.Chat); err != nil {
			return fmt.Errorf("match %d chat: %w", m.MatchID, err)
		}

		// 6. Gold Advantage (minute-by-minute array)
		if err := writeGoldAdv(batch, m.MatchID, m.GoldAdv); err != nil {
			return fmt.Errorf("match %d gold_adv: %w", m.MatchID, err)
		}

		// 7. XP Advantage
		if err := writeXPAdv(batch, m.MatchID, m.XPAdv); err != nil {
			return fmt.Errorf("match %d xp_adv: %w", m.MatchID, err)
		}

		// 8. Teamfights
		if err := writeTeamfights(batch, m.MatchID, m.Teamfights); err != nil {
			return fmt.Errorf("match %d teamfights: %w", m.MatchID, err)
		}
	}

	// Queue the watermark (checkpoint) upsert in the SAME pgx.Batch so it
	// runs inside the same transaction as the match inserts above. This
	// guarantees the watermark only advances when the entire batch commits;
	// a rolled-back batch leaves the watermark untouched, so the id-fetcher
	// will re-publish the same match_ids on the next cron tick and the
	// parser will retry them (idempotent via ON CONFLICT DO NOTHING).
	//
	// GREATEST(...) in the SQL makes the upsert monotonic: a late or
	// out-of-order batch with a smaller match_id can never rewind the
	// watermark, which would otherwise cause the id-fetcher to re-emit
	// already-parsed match_ids.
	if watermark := maxMatchID(matches); watermark > 0 {
		batch.Queue(checkpointUpsertSQL, watermark)
		// Tag the log with the pipeline name constant from the shared
		// checkpoint package — makes it easy to grep parser-vs-id-fetcher
		// checkpoint activity in a shared log aggregator.
		logger.Log.Debug("Parser checkpoint queued in batch",
			zap.String("pipeline", checkpoint.CheckpointPipelineParser),
			zap.Int64("watermark", watermark),
			zap.Int("batch_size", len(matches)))
	}

	// Execute the entire batch in one round-trip, within the transaction.
	// IMPORTANT: We use context.WithoutCancel(ctx) for all batch I/O so that
	// cancellation during graceful shutdown does NOT abort queries mid-flight.
	// If the cancelable ctx were passed here and a query was interrupted, pgx
	// would leave the connection in an undefined state. When that connection is
	// returned to the pool, the next worker would crash with a protocol sync
	// error (see Issue #14 in the audit).
	//
	// We layer a generous deadline on the orphaned context so a stuck DB or
	// network split cannot block shutdown indefinitely. 30s is far longer than
	// any expected batch run, but bounded so SIGTERM can eventually succeed.
	batchCtx, batchCancel := context.WithTimeout(context.WithoutCancel(ctx), 30*time.Second)
	defer batchCancel()
	br := tx.SendBatch(batchCtx, batch)

	var batchErr error
	for i := 0; i < batch.Len(); i++ {
		if _, err := br.Exec(); err != nil {
			batchErr = fmt.Errorf("batch exec failed at query %d: %w", i, err)
			break
		}
	}

	// Close MUST be called even on partial failure to drain any remaining
	// results and release the connection. If Close fails, the connection is
	// corrupted — return the error so the caller can shut down and restart.
	if closeErr := br.Close(); closeErr != nil {
		return fmt.Errorf("batch close failed (connection may be corrupted): %w", closeErr)
	}
	if batchErr != nil {
		return batchErr
	}

	if err := tx.Commit(batchCtx); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}
