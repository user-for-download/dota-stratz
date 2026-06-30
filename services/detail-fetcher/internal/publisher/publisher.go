package publisher

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/dota-stratz/shared/go-common/mq"
	"go.uber.org/zap"
)

// RawMatchMessage is the payload published to the raw_matches queue for the
// parser service to consume.
//
// RawJSON is typed as json.RawMessage (which is also []byte under the hood)
// so it is marshaled as the inner JSON, not as a base64-encoded string.
// Using plain []byte here would cause the parser service to receive a
// string-typed field ("raw_json":"eyJ...") that fails to unmarshal
// directly into models.OpenDotaMatch — see Issue #32.
type RawMatchMessage struct {
	MatchID   int64           `json:"match_id"`
	RawJSON   json.RawMessage `json:"raw_json"`
	FetchedAt time.Time       `json:"fetched_at"`
}

// Publisher sends match JSON to the raw_matches queue with publisher confirms
// for at-least-once delivery. Wraps the shared mq.Publisher for connection
// lifecycle and reconnection.
type Publisher struct {
	inner *mq.Publisher
}

// NewPublisher creates a Publisher that publishes to the given queue with
// automatic reconnection and publisher confirms.
func NewPublisher(url string, rawMatchesQueue, dlqQueue string) (*Publisher, error) {
	inner, err := mq.NewPublisher(url, &mq.QueueConfig{
		Name:       rawMatchesQueue,
		DLQName:    dlqQueue,
		MessageTTL: mq.DefaultMessageTTL,
	})
	if err != nil {
		return nil, err
	}
	return &Publisher{inner: inner}, nil
}

// Publish sends a single RawMatchMessage to the given queue and waits for
// the broker to confirm delivery. Thread-safe.
func (p *Publisher) Publish(ctx context.Context, queueName string, msg RawMatchMessage) error {
	body, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	logger.Log.Debug("Publishing raw match",
		zap.Int64("match_id", msg.MatchID),
		zap.Int("json_bytes", len(msg.RawJSON)))

	return p.inner.Publish(ctx, queueName, body)
}

// Close cleanly shuts down the publisher, aborting any in-progress
// reconnect loops before closing the connection.
func (p *Publisher) Close() {
	p.inner.Close()
}
