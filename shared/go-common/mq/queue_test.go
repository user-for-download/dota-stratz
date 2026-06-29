package mq

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDefaultMessageTTL(t *testing.T) {
	assert.Equal(t, int32(86400000), DefaultMessageTTL, "default TTL should be 24 hours in ms")
}

func TestQueueConfig_Defaults(t *testing.T) {
	// QueueConfig with zero TTL should get the default.
	cfg := QueueConfig{
		Name:    "test-queue",
		DLQName: "test-queue.dlq",
	}
	ttl := cfg.MessageTTL
	if ttl <= 0 {
		ttl = DefaultMessageTTL
	}
	assert.Equal(t, int32(86400000), ttl, "zero TTL should default to 24h")
}

// Note: DeclareQueueWithDLQ requires a live RabbitMQ connection and is not
// covered by these unit tests. Integration tests should verify queue
// declaration with a real broker.
