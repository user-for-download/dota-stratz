package api

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/dota-stratz/services/id-fetcher/internal/metrics"
	"github.com/dota-stratz/services/id-fetcher/internal/queue"
	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

// existenceChecker queries which match IDs already exist in the database
// so the fetcher can skip publishing them. Implementations must be safe
// for concurrent use (the fetcher calls ExistingMatchIDs from Run).
type existenceChecker interface {
	// ExistingMatchIDs returns the set of match_id values in the DB whose
	// match_id falls within [minID, maxID] (inclusive). An empty map or
	// nil are both valid results meaning "no matches exist in this range".
	ExistingMatchIDs(ctx context.Context, minID, maxID int64) (map[int64]struct{}, error)
}

// postgresExistenceChecker is the production implementation backed by
// the matches table in Postgres.
type postgresExistenceChecker struct {
	pool *pgxpool.Pool
}

func NewPostgresExistenceChecker(pool *pgxpool.Pool) *postgresExistenceChecker {
	return &postgresExistenceChecker{pool: pool}
}

func (c *postgresExistenceChecker) ExistingMatchIDs(ctx context.Context, minID, maxID int64) (map[int64]struct{}, error) {
	rows, err := c.pool.Query(ctx,
		`SELECT match_id FROM matches WHERE match_id >= $1 AND match_id <= $2`,
		minID, maxID)
	if err != nil {
		return nil, fmt.Errorf("query existing matches: %w", err)
	}
	defer rows.Close()

	result := make(map[int64]struct{})
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("scan match_id: %w", err)
		}
		result[id] = struct{}{}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("rows iteration: %w", err)
	}
	return result, nil
}

// openDotaSource is the subset of *OpenDotaClient used by Fetcher. It
// is defined as an interface so tests can inject deterministic fakes
// without standing up an HTTP server or a proxy pool. Both methods
// must honour ctx cancellation exactly like the real client.
type openDotaSource interface {
	FetchMatches(ctx context.Context) ([]MatchNode, error)
	FetchMatchesSince(ctx context.Context, watermark int64, lookbackDays int, maxResults int) ([]MatchNode, error)
}

// matchIDPublisher is the subset of *queue.Publisher used by Fetcher.
// Same rationale as openDotaSource: an interface seam for testing.
type matchIDPublisher interface {
	PublishBatch(ctx context.Context, queueName string, matchIDs []int64) error
}

// Compile-time interface satisfaction checks.
var _ openDotaSource = (*OpenDotaClient)(nil)
var _ matchIDPublisher = (*queue.Publisher)(nil)

// lastMaxMatchIDKey is the Redis key where the highest match ID ever seen
// by the id-fetcher is stored. Used to avoid re-publishing matches that
// have already been queued.
const lastMaxMatchIDKey = "dota2:fetcher:last_max_match_id"

// Fetcher fetches match IDs from OpenDota and publishes them to RabbitMQ.
// Duplicate suppression is three-layered:
//
//  1. Watermark (last_parsed_match_id) — matches fully parsed and stored
//     in the local DB are skipped.
//  2. Redis lastMaxMatchID — the highest match ID previously returned by
//     an OpenDota fetch is stored in Redis. Any match at or below this
//     value on subsequent runs is skipped, preventing re-publication of
//     already-queued match IDs.
//  3. Local DB lookup — if the Fetcher has a DB connection (optional),
//     fetched match IDs are checked against the matches table before
//     publishing (not yet implemented — the first two layers are
//     sufficient for production).
type Fetcher struct {
	client    openDotaSource
	publisher matchIDPublisher
	queueName string
	batchSize int

	// watermark is the parser's last_parsed_match_id. When > 0, matches
	// at or below this value are skipped in the Run loop.
	watermark int64

	// rdb is the shared Redis client used by the proxy pool. It is also
	// used to persist the highest match ID seen across fetcher runs so
	// that already-queued match IDs are not re-published on restart or
	// cron tick. Nil is allowed (tests / disabled) — the Redis-based
	// filter is skipped when rdb is nil.
	rdb redis.Cmdable

	// lastMaxMatchID is the highest match ID seen in the previous fetch.
	// Loaded from Redis at the start of each Run call.
	lastMaxMatchID int64

	// existsChecker queries the Postgres matches table for match IDs
	// that have already been committed. When non-nil and forceDownload
	// is false, matches found in the DB are skipped (Layer 3 filter).
	existsChecker existenceChecker

	// forceDownload bypasses the DB existence check so ALL match IDs
	// returned by OpenDota are published to the queue, regardless of
	// whether they already exist in the database. Controlled by the
	// FORCE_DOWNLOAD_REWRITE env var.
	forceDownload bool
}

func NewFetcher(client openDotaSource, pub matchIDPublisher, qName string, bSize int, rdb redis.Cmdable) *Fetcher {
	return &Fetcher{
		client:    client,
		publisher: pub,
		queueName: qName,
		batchSize: bSize,
		rdb:       rdb,
	}
}

// SetExistenceChecker configures an optional DB-backed filter that
// skips match IDs already committed to the matches table. Pass
// forceDownload=true to bypass the filter and publish everything.
func (f *Fetcher) SetExistenceChecker(ec existenceChecker, forceDownload bool) {
	f.existsChecker = ec
	f.forceDownload = forceDownload
}

// SetWatermark configures the parser high-water mark this Fetcher
// should use to filter OpenDota responses. Safe to call once before the
// first Run; the field is read without locking in Run, so callers must
// not call SetWatermark concurrently with Run.
func (f *Fetcher) SetWatermark(w int64, lookbackDays int) {
	if w < 0 {
		w = 0
	}
	f.watermark = w
	_ = lookbackDays // retained for API compatibility with existing callers
}

// Watermark returns the current watermark value (0 if unset). Useful
// for tests and for logging during boot.
func (f *Fetcher) Watermark() int64 {
	return f.watermark
}

// loadLastMaxMatchID reads the highest previously-fetched match ID from
// Redis. On first run (key absent) the value stays 0 so no additional
// filtering is applied.
func (f *Fetcher) loadLastMaxMatchID(ctx context.Context) {
	if f.rdb == nil {
		return
	}
	val, err := f.rdb.Get(ctx, lastMaxMatchIDKey).Result()
	if err != nil {
		if err == redis.Nil {
			logger.Log.Info("No lastMaxMatchID in Redis, fetching full window")
		} else {
			logger.Log.Warn("Failed to read lastMaxMatchID from Redis, fetching full window",
				zap.Error(err))
		}
		f.lastMaxMatchID = 0
		return
	}
	id, err := strconv.ParseInt(val, 10, 64)
	if err != nil {
		logger.Log.Warn("Invalid lastMaxMatchID in Redis, fetching full window",
			zap.String("raw", val), zap.Error(err))
		f.lastMaxMatchID = 0
		return
	}
	f.lastMaxMatchID = id
	logger.Log.Info("Loaded lastMaxMatchID from Redis",
		zap.Int64("last_max_match_id", id))
}

// saveLastMaxMatchID persists the highest match ID from this fetch run
// into Redis so subsequent runs can skip already-queued matches.
// Errors are logged but not returned — the fetcher should never fail
// a publish run over a Redis SET failure.
func (f *Fetcher) saveLastMaxMatchID(ctx context.Context, maxID int64) {
	if f.rdb == nil || maxID <= f.lastMaxMatchID {
		return
	}
	if err := f.rdb.Set(ctx, lastMaxMatchIDKey, strconv.FormatInt(maxID, 10), 0).Err(); err != nil {
		logger.Log.Error("Failed to persist lastMaxMatchID to Redis",
			zap.Int64("max_id", maxID),
			zap.Error(err))
		return
	}
	f.lastMaxMatchID = maxID
	logger.Log.Info("Persisted lastMaxMatchID to Redis",
		zap.Int64("max_id", maxID))
}

// Run fetches matches via the rolling-window query (matches.sql) and
// publishes their IDs to RabbitMQ in batchSize-sized messages.
//
// Deduplication is applied in this order:
//  1. Watermark filter — skips matches at or below last_parsed_match_id.
//  2. Redis filter — skips matches at or below the highest match ID seen
//     in any previous fetch run, preventing re-publication of
//     already-queued IDs.
//
// After a successful publish cycle the highest match ID from this batch
// is persisted to Redis.
//
// On context cancellation (graceful shutdown) Run stops after the
// current in-flight batch finishes — any partial batch is flushed
// with a fresh context so the IDs aren't lost.
func (f *Fetcher) Run(ctx context.Context) error {
	matches, err := f.fetch(ctx)
	if err != nil {
		metrics.PaginationRunsTotal.WithLabelValues("error").Inc()
		return err
	}

	logger.Log.Info("Fetched matches from OpenDota",
		zap.Int("count", len(matches)),
		zap.Int64("watermark", f.watermark))

	// Load the highest match ID previously seen so we can skip
	// already-queued matches.
	f.loadLastMaxMatchID(ctx)

	// Layer 3: pre-load the set of match IDs already committed to the
	// Postgres matches table. Only needed when there is a DB connection
	// and forceDownload is false.
	var dbExisting map[int64]struct{}
	if f.existsChecker != nil && !f.forceDownload && len(matches) > 0 {
		minID, maxID := matches[0].MatchID, matches[0].MatchID
		for _, m := range matches[1:] {
			if m.MatchID < minID {
				minID = m.MatchID
			}
			if m.MatchID > maxID {
				maxID = m.MatchID
			}
		}
		existing, err := f.existsChecker.ExistingMatchIDs(ctx, minID, maxID)
		if err != nil {
			logger.Log.Warn("Failed to query existing matches from DB, "+
				"proceeding without DB filter",
				zap.Error(err),
				zap.Int64("range_min", minID),
				zap.Int64("range_max", maxID))
		} else {
			dbExisting = existing
			logger.Log.Info("DB existence filter active",
				zap.Int("skip_count", len(dbExisting)),
				zap.Int64("range_min", minID),
				zap.Int64("range_max", maxID))
		}
	}

	var (
		totalPublished int
		thisRunMaxID   int64
		batch          []int64
	)

	for _, m := range matches {
		// Layer 1: skip already-parsed matches (watermark).
		if f.watermark > 0 && m.MatchID <= f.watermark {
			continue
		}
		// Layer 2: skip already-queued matches (Redis).
		if f.lastMaxMatchID > 0 && m.MatchID <= f.lastMaxMatchID {
			continue
		}
		// Layer 3: skip matches already committed to the database.
		if dbExisting != nil {
			if _, exists := dbExisting[m.MatchID]; exists {
				continue
			}
		}

		if ctx.Err() != nil {
			// Flush any accumulated IDs before returning so they aren't lost.
			// Use a fresh context since the caller's ctx is already cancelled.
			if len(batch) > 0 {
				flushCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
				if flushErr := f.publisher.PublishBatch(flushCtx, f.queueName, batch); flushErr != nil {
					logger.Log.Error("Failed to flush partial match batch on shutdown",
						zap.Int("batch_size", len(batch)),
						zap.Error(flushErr))
				} else {
					// Save watermark after successful flush to prevent
					// re-publishing these IDs on next run.
					for _, id := range batch {
						if id > thisRunMaxID {
							thisRunMaxID = id
						}
					}
					if thisRunMaxID > f.lastMaxMatchID {
						f.saveLastMaxMatchID(flushCtx, thisRunMaxID)
					}
				}
				cancel()
			}
			metrics.PaginationRunsTotal.WithLabelValues("cancelled").Inc()
			return ctx.Err()
		}

		batch = append(batch, m.MatchID)

		if len(batch) >= f.batchSize {
			if err := f.publisher.PublishBatch(ctx, f.queueName, batch); err != nil {
				metrics.PaginationRunsTotal.WithLabelValues("error").Inc()
				return err
			}
			totalPublished += len(batch)
			metrics.MatchIDsPublishedTotal.Add(float64(len(batch)))
			batch = batch[:0]
		}
	}

	// Flush remainder
	if len(batch) > 0 {
		if err := f.publisher.PublishBatch(ctx, f.queueName, batch); err != nil {
			metrics.PaginationRunsTotal.WithLabelValues("error").Inc()
			return err
		}
		totalPublished += len(batch)
		metrics.MatchIDsPublishedTotal.Add(float64(len(batch)))
	}

	// Compute the max match ID of this run for Redis persistence.
	for _, m := range matches {
		if m.MatchID > thisRunMaxID {
			thisRunMaxID = m.MatchID
		}
	}
	if thisRunMaxID > f.lastMaxMatchID {
		f.saveLastMaxMatchID(ctx, thisRunMaxID)
	}

	logger.Log.Info("Fetch run complete",
		zap.Int("total_published", totalPublished),
		zap.Int64("watermark", f.watermark),
		zap.Int64("last_max_match_id", f.lastMaxMatchID))

	metrics.PaginationRunsTotal.WithLabelValues("success").Inc()
	return nil
}

// fetch always uses the rolling-window query (matches.sql) which returns
// all match IDs in the configured time window with a generous LIMIT 50000.
//
// The watermark and Redis filters are applied in Go inside Run() so a
// single round-trip covers the entire window.
func (f *Fetcher) fetch(ctx context.Context) ([]MatchNode, error) {
	return f.client.FetchMatches(ctx)
}
