package worker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/dota-stratz/services/detail-fetcher/internal/api"
	"github.com/dota-stratz/services/detail-fetcher/internal/metrics"
	"github.com/dota-stratz/services/detail-fetcher/internal/publisher"
	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/jackc/pgx/v5/pgxpool"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

// matchExistenceChecker checks whether a match_id already exists in the
// Postgres matches table. When available, the Worker skips the download
// entirely and Acks the message immediately, avoiding 5+ minutes of
// futile proxy + direct retries for matches that are already committed.
type matchExistenceChecker interface {
	// MatchExists returns true when the given match_id exists in the
	// matches table. Errors are logged but treated as "don't know" —
	// the worker proceeds with the normal fetch flow when the DB is
	// unreachable, so a transient DB outage cannot drop messages.
	MatchExists(ctx context.Context, matchID int64) (bool, error)
}

// postgresMatchChecker is the production implementation backed by the
// dota-stratz Postgres matches table.
type postgresMatchChecker struct {
	pool *pgxpool.Pool
}

func NewPostgresMatchChecker(pool *pgxpool.Pool) *postgresMatchChecker {
	return &postgresMatchChecker{pool: pool}
}

func (c *postgresMatchChecker) MatchExists(ctx context.Context, matchID int64) (bool, error) {
	var exists bool
	err := c.pool.QueryRow(ctx,
		`SELECT EXISTS(SELECT 1 FROM matches WHERE match_id = $1)`, matchID).Scan(&exists)
	if err != nil {
		return false, fmt.Errorf("check match existence: %w", err)
	}
	return exists, nil
}

// Compile-time check that *postgresMatchChecker satisfies the interface.
var _ matchExistenceChecker = (*postgresMatchChecker)(nil)

// MatchIDMessage is the payload consumed from the match_ids queue (produced
// by the id-fetcher service).
type MatchIDMessage struct {
	MatchID int64 `json:"match_id"`
}

// matchFetcher is the subset of *api.Client used by Worker. Defined as an
// interface so tests can inject deterministic fakes without an HTTP server
// or proxy pool.
type matchFetcher interface {
	FetchRaw(ctx context.Context, matchID int64) ([]byte, error)
	// FetchRawDirect fetches from OpenDota without a proxy. Used as a
	// last-resort fallback when all proxy-based retries are exhausted.
	FetchRawDirect(ctx context.Context, matchID int64) ([]byte, error)
}

// matchPublisher is the subset of *publisher.Publisher used by Worker.
// Same rationale as matchFetcher: an interface seam for testing.
type matchPublisher interface {
	Publish(ctx context.Context, queueName string, msg publisher.RawMatchMessage) error
}

// Compile-time interface satisfaction checks.
var _ matchFetcher = (*api.Client)(nil)
var _ matchPublisher = (*publisher.Publisher)(nil)

// Worker orchestrates the fetch→publish loop with retries and DLQ escalation.
type Worker struct {
	client     matchFetcher
	publisher  matchPublisher
	dbChecker  matchExistenceChecker
	queueName  string
	maxRetries int
	retryDelay time.Duration
}

// SetDBExistenceChecker attaches a Postgres-backed existence checker so the
// worker can skip matches that have already been committed to the database.
// This avoids 5+ minutes of futile proxy retries + direct fallback for
// matches that are already parsed. The checker is nil-safe — when nil
// (default) the worker always fetches. Safe to call at any time before
// Process; not guarded by a mutex (set once at startup).
func (w *Worker) SetDBExistenceChecker(c matchExistenceChecker) {
	w.dbChecker = c
}

func NewWorker(client matchFetcher, pub matchPublisher, queueName string, maxRetries, retryDelaySec int) *Worker {
	return &Worker{
		client:     client,
		publisher:  pub,
		queueName:  queueName,
		maxRetries: maxRetries,
		retryDelay: time.Duration(retryDelaySec) * time.Second,
	}
}

// ProcessResult describes what the caller should do with the delivery.
type ProcessResult int

const (
	// Ack signals the delivery should be acknowledged (success).
	Ack ProcessResult = iota
	// NackDLQ signals Nack without requeue — route to dead-letter queue.
	NackDLQ
	// NackRequeue signals Nack with requeue — the message should be retried.
	NackRequeue
)

// Process handles one match ID delivery.
// The ctx is the application-level shutdown context and is passed through to
// the HTTP client so in-flight requests can be cancelled during shutdown.
func (w *Worker) Process(ctx context.Context, d amqp.Delivery) (result ProcessResult) {
	// Recover from any panic in the fetch+publish pipeline. Without this
	// recover, a single misbehaving message (e.g. an HTTP transport bug
	// that panics on a malformed header) would crash the whole service.
	// We log the panic, increment a counter for observability, and Nack
	// the offending message to DLQ so the bad payload can be inspected
	// without blocking the rest of the queue.

	defer func() {
		if r := recover(); r != nil {
			metrics.DLQRoutedTotal.Inc()
			logger.Log.Error("Worker panic recovered, sending message to DLQ",
				zap.Any("panic", r),
				zap.Int("delivery_bytes", len(d.Body)))
			result = NackDLQ
		}
	}()

	metrics.MessagesReceivedTotal.Inc()

	var msg MatchIDMessage
	if err := json.Unmarshal(d.Body, &msg); err != nil {
		logger.Log.Error("Invalid message payload, sending to DLQ",
			zap.Error(err))
		metrics.DLQRoutedTotal.Inc()
		result = NackDLQ
		return
	}

	// Skip fetch if the match already exists in the database. This avoids
	// 5+ minutes of futile proxy retries + direct fallback for matches
	// that are already committed. The checker is nil-safe (default nil
	// for backward compatibility and tests).
	if w.dbChecker != nil {
		exists, dbErr := w.dbChecker.MatchExists(ctx, msg.MatchID)
		if dbErr != nil {
			logger.Log.Warn("DB existence check failed, proceeding with fetch",
				zap.Int64("match_id", msg.MatchID),
				zap.Error(dbErr))
		} else if exists {
			logger.Log.Debug("Match already exists in DB, skipping",
				zap.Int64("match_id", msg.MatchID))
			metrics.SkippedTotal.Inc()
			result = Ack
			return
		}
	}

	var lastErr error
	var rawJSON []byte
	fetchAttempts := 0
	for attempt := range w.maxRetries {
		if ctx.Err() != nil {
			// Shutdown — requeue so the next instance can pick it up.
			result = NackRequeue
			return
		}

		// Fetch from OpenDota only if we don't already have the data
		// from a previous attempt. Once fetched successfully, subsequent
		// iterations only retry the publish step (fixes audit finding #9:
		// redundant OpenDota calls on transient RabbitMQ failures).
		if rawJSON == nil {
			fetchAttempts++
			var fetchErr error
			rawJSON, fetchErr = w.client.FetchRaw(ctx, msg.MatchID)
			if fetchErr != nil {
				if errors.Is(fetchErr, api.ErrMatchNotFound) {
					logger.Log.Warn("Match not found or private, discarding",
						zap.Int64("match_id", msg.MatchID))
					metrics.FetchesTotal.WithLabelValues("not_found").Inc()
					result = Ack
					return
				}

				lastErr = fetchErr
				metrics.FetchesTotal.WithLabelValues("error").Inc()
				logger.Log.Warn("Fetch failed, retrying",
					zap.Int64("match_id", msg.MatchID),
					zap.Int("attempt", attempt+1),
					zap.Int("max", w.maxRetries),
					zap.Error(fetchErr))

				// Cap the shift at 6 (×64) so a misconfigured maxRetries
				// cannot overflow time.Duration via (1<<attempt).
				// At retryDelay=1s this is a 64s cap.
				shift := attempt
				if shift > 6 {
					shift = 6
				}
				select {
				case <-time.After(w.retryDelay * (1 << shift)):
				case <-ctx.Done():
					result = NackRequeue
					return
				}
				continue
			}
			metrics.FetchesTotal.WithLabelValues("success").Inc()
		}

		// Publish raw JSON to the parser queue.
		pubMsg := publisher.RawMatchMessage{
			MatchID:   msg.MatchID,
			RawJSON:   json.RawMessage(rawJSON),
			FetchedAt: time.Now(),
		}
		if err := w.publisher.Publish(ctx, w.queueName, pubMsg); err != nil {
			logger.Log.Error("Failed to publish to raw_matches, retrying",
				zap.Int64("match_id", msg.MatchID),
				zap.Int("attempt", attempt+1),
				zap.Int("max", w.maxRetries),
				zap.Error(err))
			metrics.PublishesTotal.WithLabelValues("error").Inc()
			lastErr = err

			shift := attempt
			if shift > 6 {
				shift = 6
			}
			select {
			case <-time.After(w.retryDelay * (1 << shift)):
			case <-ctx.Done():
				result = NackRequeue
				return
			}
			continue
		}

		metrics.PublishesTotal.WithLabelValues("success").Inc()
		result = Ack
		return
	}

	// Proxy retries exhausted — try direct connection as last resort.
	// Free proxies are often too slow for the full match download JSON
	// even though they pass the lightweight validation ping. A direct
	// connection bypasses the proxy bottleneck entirely.
	if rawJSON == nil && ctx.Err() == nil {
		logger.Log.Warn("Proxies exhausted, trying direct connection",
			zap.Int64("match_id", msg.MatchID),
			zap.Int("attempts_fetch", fetchAttempts),
			zap.Error(lastErr))

		fetchAttempts++
		var directErr error
		rawJSON, directErr = w.client.FetchRawDirect(ctx, msg.MatchID)
		if directErr != nil {
			if errors.Is(directErr, api.ErrMatchNotFound) {
				logger.Log.Warn("Match not found via direct connection, discarding",
					zap.Int64("match_id", msg.MatchID))
				metrics.FetchesTotal.WithLabelValues("not_found").Inc()
				result = Ack
				return
			}
			lastErr = directErr
			metrics.FetchesTotal.WithLabelValues("error").Inc()

			logger.Log.Error("All attempts (proxies + direct) failed, sending to DLQ",
				zap.Int64("match_id", msg.MatchID),
				zap.Int("attempts_fetch", fetchAttempts),
				zap.Error(lastErr))
			metrics.DLQRoutedTotal.Inc()
			result = NackDLQ
			return
		}
		metrics.FetchesTotal.WithLabelValues("success").Inc()
		logger.Log.Info("Direct connection succeeded after proxy retries exhausted",
			zap.Int64("match_id", msg.MatchID),
			zap.Int("attempts_fetch", fetchAttempts))

		// Publish the directly-fetched match.
		pubMsg := publisher.RawMatchMessage{
			MatchID:   msg.MatchID,
			RawJSON:   json.RawMessage(rawJSON),
			FetchedAt: time.Now(),
		}
		if err := w.publisher.Publish(ctx, w.queueName, pubMsg); err != nil {
			logger.Log.Error("Failed to publish after direct fallback",
				zap.Int64("match_id", msg.MatchID),
				zap.Error(err))
			metrics.PublishesTotal.WithLabelValues("error").Inc()
			lastErr = err

			logger.Log.Error("Publish failed after direct fallback, sending to DLQ",
				zap.Int64("match_id", msg.MatchID),
				zap.Int("attempts_fetch", fetchAttempts),
				zap.Error(lastErr))
			metrics.DLQRoutedTotal.Inc()
			result = NackDLQ
			return
		}
		metrics.PublishesTotal.WithLabelValues("success").Inc()
		result = Ack
		return
	}

	logger.Log.Error("Exhausted retries, sending to DLQ",
		zap.Int64("match_id", msg.MatchID),
		zap.Int("attempts_fetch", fetchAttempts),
		zap.Error(lastErr))
	metrics.DLQRoutedTotal.Inc()
	result = NackDLQ
	return
}
