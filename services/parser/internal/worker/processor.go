package worker

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/dota-stratz/services/parser/internal/metrics"
	"github.com/dota-stratz/services/parser/internal/models"
	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/jackc/pgx/v5/pgconn"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

// ErrFatalPanic is returned by Run() when a panic is recovered during batch
// processing. The caller (main.go) should terminate the process so the
// container supervisor restarts it fresh — this prevents a zombie service
// that passes healthchecks but never consumes messages.
var ErrFatalPanic = errors.New("processor: recovered panic, terminating")

// maxConsecutiveBatchFailures is the number of sequential batch write
// failures before the processor escalates to DLQ (poison-pill guard).
// Prevents a persistent non-FK error (unique violation, truncation, etc.)
// from creating an infinite requeue loop that blocks the entire pipeline.
// See CRITICAL BUG C-4.
const maxConsecutiveBatchFailures = 3

// matchWriter is the subset of *repository.Repository used by Processor.
// Defined as an interface so tests can inject fakes without a real database.
type matchWriter interface {
	WriteBatch(ctx context.Context, matches []models.OpenDotaMatch) error
}

// Processor batches messages from RabbitMQ and writes parsed matches to Postgres.
type Processor struct {
	repo         matchWriter
	batchSize    int
	fetchTimeout time.Duration
	msgs         <-chan amqp.Delivery

	// consecutiveBatchFailures tracks how many sequential batch writes
	// have failed with a non-FK error. Reset to 0 after a successful
	// write or after escalating to DLQ.
	consecutiveBatchFailures int
}

func NewProcessor(
	repo matchWriter,
	msgs <-chan amqp.Delivery,
	batchSize int,
	fetchTimeout time.Duration,
) *Processor {
	return &Processor{
		repo:         repo,
		msgs:         msgs,
		batchSize:    batchSize,
		fetchTimeout: fetchTimeout,
	}
}

// Run starts the processing loop. It accumulates messages into batches,
// unmarshals and validates them, writes to Postgres, and ACKs on success.
// Returns ErrFatalPanic if a panic is recovered, indicating the caller should
// terminate the process to avoid a zombie service.
func (p *Processor) Run(ctx context.Context) (err error) {
	// currentBatch tracks the deliveries that are still in-flight (not yet
	// Ack'd or Nack'd) during the current batch iteration. Used by the
	// panic recovery below to Nack them before process exit.
	var currentBatch []amqp.Delivery

	// Recover from any panic during the batch loop. When a panic occurs,
	// Nack any in-flight deliveries so they are requeued by RabbitMQ
	// immediately rather than stuck unacked until the heartbeat timeout.
	// Then return ErrFatalPanic so the caller (main.go) can terminate the
	// process. The container supervisor (docker-compose restart:
	// unless-stopped) restarts it fresh. Without process exit, the
	// panic-recovered goroutine returns silently, the processor never
	// consumes another message, but the healthcheck (Postgres ping) still
	// passes — creating a zombie service. See CRITICAL BUG C-3.
	defer func() {
		if r := recover(); r != nil {
			err = ErrFatalPanic
			logger.Log.Error("Processor panic, terminating process",
				zap.Any("panic", r))
			// Nack all currently in-flight deliveries so they requeue
			// immediately via RabbitMQ, rather than waiting for the TCP
			// heartbeat timeout to detect the dropped connection.
			for _, d := range currentBatch {
				if err := d.Nack(false, true); err != nil {
					logger.Log.Warn("Failed to Nack delivery during panic recovery "+
						"(may be already Ack'd/Nack'd)",
						zap.Uint64("delivery_tag", d.DeliveryTag),
						zap.Error(err))
				}
			}
		}
	}()

	logger.Log.Info("Parser processor started",
		zap.Int("batch_size", p.batchSize),
		zap.Duration("fetch_timeout", p.fetchTimeout))

	for {
		if ctx.Err() != nil {
			logger.Log.Info("Context cancelled, stopping processor")
			return nil
		}

		batch := p.fetchBatch(ctx)
		if len(batch) == 0 {
			// Brief sleep to prevent CPU spin when channel is closed during
			// shutdown but ctx is not yet cancelled.
			select {
			case <-ctx.Done():
				return nil
			case <-time.After(10 * time.Millisecond):
			}
			continue
		}
		currentBatch = batch // track raw batch during unmarshal phase

		var validMatches []models.OpenDotaMatch
		var validDeliveries []amqp.Delivery

		// 1. Unmarshal and filter poison pills.
		//
		// NOTE: Invalid/malformed messages are Nack'd without requeue, which
		// routes them to the DLQ via the queue's x-dead-letter-exchange binding
		// (declared in consumer.go). The sendToDLQ manual-publish pattern was
		// removed in favor of native DLX routing because the Ack+manual-publish
		// approach silently lost messages when the DLQ publish failed (the Ack
		// already consumed the original and the Nack in sendToDLQ was a no-op).
		for _, d := range batch {
			var envelope models.RawMatchMessage
			if err := json.Unmarshal(d.Body, &envelope); err != nil {
				logger.Log.Warn("Invalid envelope, routing to DLQ via DLX",
					zap.Error(err),
				)
				if err := d.Nack(false, false); err != nil { // → DLQ via dead-letter exchange
					logger.Log.Error("Failed to Nack invalid envelope to DLQ",
						zap.Uint64("delivery_tag", d.DeliveryTag),
						zap.Error(err))
				}
				metrics.DLQMessages.Inc()
				continue
			}

			var match models.OpenDotaMatch
			if err := json.Unmarshal(envelope.RawJSON, &match); err != nil {
				logger.Log.Warn("Invalid match JSON, routing to DLQ via DLX",
					zap.Error(err),
					zap.Int64("match_id", envelope.MatchID),
				)
				if err := d.Nack(false, false); err != nil { // → DLQ via dead-letter exchange
					logger.Log.Error("Failed to Nack invalid match to DLQ",
						zap.Uint64("delivery_tag", d.DeliveryTag),
						zap.Error(err))
				}
				metrics.DLQMessages.Inc()
				continue
			}

			// Basic validation.
			if match.MatchID == 0 || match.Duration <= 0 {
				logger.Log.Warn("Match failed validation, routing to DLQ via DLX",
					zap.Int64("match_id", match.MatchID),
					zap.Int("duration", match.Duration),
				)
				if err := d.Nack(false, false); err != nil { // → DLQ via dead-letter exchange
					logger.Log.Error("Failed to Nack invalid match to DLQ",
						zap.Uint64("delivery_tag", d.DeliveryTag),
						zap.Error(err))
				}
				metrics.DLQMessages.Inc()
				continue
			}

			validMatches = append(validMatches, match)
			validDeliveries = append(validDeliveries, d)
		}

		// Switch tracking to only the un-Nack'd (still in-flight) deliveries.
		// Nil currentBatch first to avoid the deferred panic handler Nacking
		// deliveries that were already Nack'd to DLQ (Bug #11).
		currentBatch = nil
		currentBatch = validDeliveries

		if len(validMatches) == 0 {
			logger.Log.Debug("No valid matches in batch, skipping")
			currentBatch = nil
			continue
		}

		// 2. Execute batch insert.
		startTime := time.Now()
		if err := p.repo.WriteBatch(ctx, validMatches); err != nil {
			if isConstraintViolation(err) {
				// FK violation (e.g. unseeded hero_id) — Do NOT requeue
				// the entire batch. Fall back to individual record processing
				// so healthy matches are committed and the offending match
				// is sent to the DLQ (Issue #28).
				logger.Log.Warn("Batch FK violation, falling back to individual inserts",
					zap.Error(err),
					zap.Int("batch_size", len(validMatches)))
				p.processIndividualMatches(ctx, validDeliveries, validMatches)
				// BUG-010: reset the consecutive failure counter so the
				// next non-FK error starts fresh instead of inheriting
				// the previous streak and prematurely escalating to DLQ.
				p.consecutiveBatchFailures = 0
				currentBatch = nil
				continue
			}

			p.consecutiveBatchFailures++
			logger.Log.Error("Batch DB write failed, requeuing batch",
				zap.Error(err),
				zap.Int("batch_size", len(validMatches)),
				zap.Int("consecutive_failures", p.consecutiveBatchFailures),
			)
			metrics.MatchesFailed.Add(float64(len(validMatches)))

			if p.consecutiveBatchFailures >= maxConsecutiveBatchFailures {
				// Poison pill guard: after N consecutive failures, route
				// the batch to DLQ instead of looping forever. This prevents
				// a persistent non-FK error (unique violation, data truncation,
				// NOT NULL violation) from blocking the entire pipeline.
				// See CRITICAL BUG C-4.
				logger.Log.Error("Batch failed consecutively, routing to DLQ",
					zap.Int("failures", p.consecutiveBatchFailures),
					zap.Int("batch_size", len(validMatches)))
				for _, d := range validDeliveries {
					if err := d.Nack(false, false); err != nil {
						logger.Log.Error("Failed to Nack poison batch to DLQ",
							zap.Uint64("delivery_tag", d.DeliveryTag),
							zap.Error(err))
					}
				}
				p.consecutiveBatchFailures = 0
				currentBatch = nil
				continue
			}

			for _, d := range validDeliveries {
				if err := d.Nack(false, true); err != nil {
					logger.Log.Error("Failed to Nack batch for requeue",
						zap.Uint64("delivery_tag", d.DeliveryTag),
						zap.Error(err))
				}
			}
			currentBatch = nil
			// Bounded backoff that honors shutdown — without the select on
			// ctx.Done() this would block graceful shutdown for 2s per failure
			// burst and could swallow SIGTERM during a DB outage.
			select {
			case <-time.After(2 * time.Second):
			case <-ctx.Done():
				return nil
			}
			continue
		}

		duration := time.Since(startTime)

		// 3. Success! ACK all valid deliveries and reset failure counter.
		p.consecutiveBatchFailures = 0
		for _, d := range validDeliveries {
			if err := d.Ack(false); err != nil {
				logger.Log.Error("Failed to Ack successful batch",
					zap.Uint64("delivery_tag", d.DeliveryTag),
					zap.Error(err))
			}
		}
		currentBatch = nil

		metrics.MatchesParsed.Add(float64(len(validMatches)))
		metrics.BatchSize.Observe(float64(len(validMatches)))
		metrics.BatchProcessingDuration.Observe(duration.Seconds())

		// Guard against division by zero (Issue #15) — if the batch was so
		// fast that duration rounds to 0s, clamp to 1ms so mps is sensible.
		sec := duration.Seconds()
		if sec <= 0 {
			sec = 0.001
		}
		logger.Log.Info("Successfully parsed and committed batch",
			zap.Int("matches", len(validMatches)),
			zap.Duration("duration", duration),
			zap.Float64("matches_per_sec", float64(len(validMatches))/sec),
		)
	}
}

// fetchBatch accumulates up to batchSize messages or until the timeout elapses.
// When the channel closes unexpectedly (e.g. reconnect in progress), it waits
// briefly and re-checks context to avoid an infinite CPU spin.
func (p *Processor) fetchBatch(ctx context.Context) []amqp.Delivery {
	var batch []amqp.Delivery
	timer := time.NewTimer(p.fetchTimeout)
	defer func() {
		if !timer.Stop() {
			// Drain the timer channel to prevent a stale value from
			// leaking — Stop() returns false when the timer has already
			// fired but the value was never received (e.g. the !ok path
			// returned before the <-timer.C case was selected).
			select {
			case <-timer.C:
			default:
			}
		}
	}()

	for len(batch) < p.batchSize {
		select {
		case d, ok := <-p.msgs:
			if !ok {
				// Channel closed — this only happens during shutdown
				// (ConsumeWithReconnect never closes the output channel
				// on reconnect). Brief pause before returning to avoid
				// any accidental tight loop. Honour ctx cancellation
				// during this pause so shutdown is not delayed.
				logger.Log.Warn("Message channel closed, yielding")
				select {
				case <-time.After(100 * time.Millisecond):
				case <-ctx.Done():
				}
				return batch
			}
			batch = append(batch, d)
		case <-timer.C:
			return batch
		case <-ctx.Done():
			return batch
		}
	}
	return batch
}

// isConstraintViolation returns true if the error is (or wraps) a PostgreSQL
// foreign key violation (SQLSTATE 23503) or a CHECK constraint violation
// (SQLSTATE 23514). Both are permanent failures — retrying will not fix them.
//
// FK violations include unseeded hero_id when Valve adds a new hero (Issue #28).
// CHECK violations include schema-level constraints like chk_duration_positive
// (removed in migration 009 but defensively checked here for any future
// constraints). Routing both to DLQ prevents an infinite batch-requeue loop
// (audit findings #1 and #2).
func isConstraintViolation(err error) bool {
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		return pgErr.Code == "23503" || pgErr.Code == "23514"
	}
	return false
}

// processIndividualMatches is a safe-mode fallback when a batch Write fails
// due to a foreign key violation.  Matches are written one-by-one so the
// offending match can be identified, sent to DLQ, and removed from the failed
// batch without affecting healthy matches (Issue #28).
func (p *Processor) processIndividualMatches(
	ctx context.Context,
	deliveries []amqp.Delivery,
	matches []models.OpenDotaMatch,
) {
	for i, m := range matches {
		if ctx.Err() != nil {
			// Shutdown — Nack remaining deliveries for requeue so they
			// are not lost, then stop.
			for j := i; j < len(matches); j++ {
				if err := deliveries[j].Nack(false, true); err != nil {
					logger.Log.Error("Failed to Nack remaining delivery during shutdown",
						zap.Uint64("delivery_tag", deliveries[j].DeliveryTag),
						zap.Error(err))
				}
			}
			return
		}

		if err := p.repo.WriteBatch(ctx, []models.OpenDotaMatch{m}); err != nil {
			if isConstraintViolation(err) {
				logger.Log.Warn("FK violation on individual match, routing to DLQ via DLX",
					zap.Int64("match_id", m.MatchID),
					zap.Error(err))
				// Nack without requeue → DLQ via dead-letter exchange binding.
				if err := deliveries[i].Nack(false, false); err != nil {
					logger.Log.Error("Failed to Nack FK-violating match to DLQ",
						zap.Uint64("delivery_tag", deliveries[i].DeliveryTag),
						zap.Error(err))
				}
				metrics.DLQMessages.Inc()
			} else {
				logger.Log.Error("Non-FK error on individual match, requeueing",
					zap.Int64("match_id", m.MatchID),
					zap.Error(err))
				if err := deliveries[i].Nack(false, true); err != nil {
					logger.Log.Error("Failed to Nack individual match for requeue",
						zap.Uint64("delivery_tag", deliveries[i].DeliveryTag),
						zap.Error(err))
				}
			}
		} else {
			// Success — ACK the individual message.
			if err := deliveries[i].Ack(false); err != nil {
				logger.Log.Error("Failed to Ack individual match",
					zap.Uint64("delivery_tag", deliveries[i].DeliveryTag),
					zap.Error(err))
			}
			metrics.MatchesParsed.Inc()
		}
	}
}
