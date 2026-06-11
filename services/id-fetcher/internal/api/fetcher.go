package api

import (
	"context"
	"time"

	"github.com/dota-stratz/services/id-fetcher/internal/metrics"
	"github.com/dota-stratz/services/id-fetcher/internal/queue"
	"github.com/dota-stratz/shared/go-common/logger"
	"go.uber.org/zap"
)

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

// Fetcher fetches match IDs from OpenDota and publishes them to RabbitMQ.
//
// On boot the caller invokes SetWatermark(w, lookbackDays). When
// Watermark > 0 the fetcher uses the watermark-based query
// (matches_watermark.sql + in-Go `match_id > watermark` filter) so
// the pipeline advances past the parser's high-water mark and does
// not re-emit already-parsed match IDs. When Watermark == 0 the
// fetcher falls back to the rolling N-day window so a fresh DB still
// gets bootstrapped.
type Fetcher struct {
	client    openDotaSource
	publisher matchIDPublisher
	queueName string
	batchSize int

	// watermark is the parser's last_parsed_match_id. When > 0, the
	// fetcher uses the watermark-based path instead of the rolling
	// window. Read-only after construction — use SetWatermark.
	watermark int64

	// watermarkLookbackDays is the rolling window (in days) used by
	// the watermark query. Must be >= the rolling-window lookback
	// used by the bootstrap path; enforced in config.Load.
	watermarkLookbackDays int
}

func NewFetcher(client openDotaSource, pub matchIDPublisher, qName string, bSize int) *Fetcher {
	return &Fetcher{
		client:    client,
		publisher: pub,
		queueName: qName,
		batchSize: bSize,
	}
}

// SetWatermark configures the parser high-water mark this Fetcher
// should use to filter OpenDota responses. A watermark of 0 (the
// default) selects the rolling-window path; any positive value
// selects the watermark-based path. Safe to call once before the
// first Run; the field is read without locking in Run, so callers
// must not call SetWatermark concurrently with Run.
func (f *Fetcher) SetWatermark(w int64, lookbackDays int) {
	if w < 0 {
		w = 0
	}
	if lookbackDays < 0 {
		lookbackDays = 0
	}
	f.watermark = w
	f.watermarkLookbackDays = lookbackDays
}

// Watermark returns the current watermark value (0 if unset). Useful
// for tests and for logging during boot.
func (f *Fetcher) Watermark() int64 {
	return f.watermark
}

// Run fetches matches via the rolling-window query (matches.sql) and
// publishes their IDs to RabbitMQ in batchSize-sized messages. When a
// watermark is configured, already-parsed match IDs (<= watermark) are
// skipped in Go code so the pipeline does not re-queue committed work.
//
// Previously this method toggled between matches.sql (rolling window)
// and matches_watermark.sql (watermark filter pushed into SQL), but the
// watermark SQL's small LIMIT (batchSize × 5) caused a 2+ day catch-up
// lag behind the 360-day window. Using matches.sql (LIMIT 50000) in a
// single round-trip and filtering in Go is both simpler and faster.
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

	totalPublished := 0
	var batch []int64

	for _, m := range matches {
		// Skip already-parsed matches when a watermark is set.
		// The rolling-window query returns everything in the time
		// window; we filter in Go so we don't re-queue IDs the
		// parser has already committed.
		if f.watermark > 0 && m.MatchID <= f.watermark {
			continue
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

	logger.Log.Info("Fetch run complete",
		zap.Int("total_published", totalPublished),
		zap.Int64("watermark", f.watermark))

	metrics.PaginationRunsTotal.WithLabelValues("success").Inc()
	return nil
}

// fetch always uses the rolling-window query (matches.sql) which returns
// all match IDs in the configured time window with a generous LIMIT 50000.
//
// The watermark filter is applied in Go inside Run() so a single
// round-trip covers the entire window — no more 2500-row limit that
// takes multiple cron ticks to drain.
//
// Previously this method dispatched to FetchMatchesSince when a watermark
// was set, but the small overscan window (batchSize × 5) caused a
// multi-day catch-up lag. The rolling query is a single call regardless
// of watermark state.
func (f *Fetcher) fetch(ctx context.Context) ([]MatchNode, error) {
	return f.client.FetchMatches(ctx)
}
