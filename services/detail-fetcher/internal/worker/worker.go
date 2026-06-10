package worker

import (
	"context"
	"encoding/json"
	"errors"
	"time"

	"github.com/dota-stratz/services/detail-fetcher/internal/api"
	"github.com/dota-stratz/services/detail-fetcher/internal/metrics"
	"github.com/dota-stratz/services/detail-fetcher/internal/publisher"
	"github.com/dota-stratz/shared/go-common/logger"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

// MatchIDMessage is the payload consumed from the match_ids queue (produced
// by the id-fetcher service).
type MatchIDMessage struct {
	MatchID int64 `json:"match_id"`
}

// Worker orchestrates the fetch→publish loop with retries and DLQ escalation.
type Worker struct {
	client     *api.Client
	publisher  *publisher.Publisher
	queueName  string
	maxRetries int
	retryDelay time.Duration
}

func NewWorker(client *api.Client, pub *publisher.Publisher, queueName string, maxRetries, retryDelaySec int) *Worker {
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

	logger.Log.Error("Exhausted retries, sending to DLQ",
		zap.Int64("match_id", msg.MatchID),
		zap.Int("attempts_fetch", fetchAttempts),
		zap.Error(lastErr))
	metrics.DLQRoutedTotal.Inc()
	result = NackDLQ
	return
}
