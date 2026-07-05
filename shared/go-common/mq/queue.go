package mq

import (
	"fmt"

	amqp "github.com/rabbitmq/amqp091-go"
)

// QueueConfig holds the names and TTL for a queue and its dead-letter queue.
// All fields are required.
type QueueConfig struct {
	// Name is the primary queue name (e.g. "queue.raw_matches").
	Name string

	// DLQName is the dead-letter queue name (e.g. "queue.raw_matches.dlq").
	DLQName string

	// MessageTTL is the TTL in milliseconds for messages in the DLQ.
	// After this period, unconsumed DLQ messages are discarded.
	// Default: 86400000 (24 hours).
	MessageTTL int32
}

// DefaultMessageTTL is the default TTL for messages in the dead-letter queue.
const DefaultMessageTTL int32 = 86400000 // 24 hours

// maxQueueBytes prevents OOM when downstream services are down.
// 2GB is enough for ~125 match JSONs (16MB each) before backpressure kicks in.
const maxQueueBytes int64 = 2147483648

// DeclareQueueWithDLQ declares the DLQ (with TTL) and the main queue (with
// dead-letter exchange binding and max length). Idempotent — safe to call on
// every reconnect.
func DeclareQueueWithDLQ(ch *amqp.Channel, cfg QueueConfig) error {
	ttl := cfg.MessageTTL
	if ttl <= 0 {
		ttl = DefaultMessageTTL
	}

	// Declare DLQ with TTL.
	_, err := ch.QueueDeclare(cfg.DLQName, true, false, false, false, amqp.Table{
		"x-message-ttl": ttl,
	})
	if err != nil {
		return fmt.Errorf("failed to declare DLQ %s: %w", cfg.DLQName, err)
	}

	// Declare main queue with DLX binding and max length backpressure.
	_, err = ch.QueueDeclare(cfg.Name, true, false, false, false, amqp.Table{
		"x-dead-letter-exchange":    "",
		"x-dead-letter-routing-key": cfg.DLQName,
		"x-max-length-bytes":        maxQueueBytes,
		"x-overflow":                "reject-publish",
	})
	if err != nil {
		return fmt.Errorf("failed to declare queue %s: %w", cfg.Name, err)
	}

	return nil
}
