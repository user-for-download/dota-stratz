package api

import (
	"context"
	"fmt"
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

// watermarkOverscanMultiplier is the factor by which the watermark
// path over-fetches from OpenDota to leave slack for the in-Go filter
// (`match_id > watermark`). 5× the batch size means a single OpenDota
// round-trip yields enough rows to fill one batch even if 80% of the
// response is at or below the watermark (e.g. a long quiet period
// followed by a burst of new matches). Tuned empirically — 5× is
// generous for typical Dota 2 traffic (1k matches/day × 30-day
// lookback = 30k rows, well under the 1M Explorer limit).
const watermarkOverscanMultiplier = 5

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

func NewFetcher(client *OpenDotaClient, pub *queue.Publisher, qName string, bSize int) *Fetcher {
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

// Run fetches matches (rolling window or watermark, depending on the
// configured mode) and publishes their IDs to RabbitMQ in
// batchSize-sized messages.
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
		zap.Int64("watermark", f.watermark),
		zap.Bool("watermark_path", f.watermark > 0))

	totalPublished := 0
	var batch []int64

	for _, m := range matches {
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
		zap.Int64("watermark", f.watermark),
		zap.Bool("watermark_path", f.watermark > 0))

	metrics.PaginationRunsTotal.WithLabelValues("success").Inc()
	return nil
}

// fetch dispatches to the rolling-window or watermark path based on
// the configured mode. Extracted so the branching is testable in
// isolation from the publish loop.
func (f *Fetcher) fetch(ctx context.Context) ([]MatchNode, error) {
	if f.watermark > 0 {
		if f.watermarkLookbackDays <= 0 {
			// Programmer error: SetWatermark called with a positive
			// watermark but no lookback days. Config.Load already
			// enforces the inverse (lookback >= rolling window), so
			// reaching this branch indicates a wiring bug in main.go.
			return nil, fmt.Errorf("fetcher: watermark path requires watermark_lookback_days > 0 (got %d)", f.watermarkLookbackDays)
		}
		maxResults := f.batchSize * watermarkOverscanMultiplier
		return f.client.FetchMatchesSince(ctx, f.watermark, f.watermarkLookbackDays, maxResults)
	}
	return f.client.FetchMatches(ctx)
}
