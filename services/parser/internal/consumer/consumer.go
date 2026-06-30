package consumer

import (
	"context"

	"github.com/dota-stratz/shared/go-common/mq"
	amqp "github.com/rabbitmq/amqp091-go"
)

// Consumer wraps the shared mq.Consumer with parser-specific queue naming.
type Consumer struct {
	inner *mq.Consumer
}

// NewConsumer connects to RabbitMQ, declares the raw_matches queue + DLQ,
// sets QoS, and returns a Consumer ready for use.
func NewConsumer(url, queue, dlqName string, prefetch int) (*Consumer, error) {
	inner, err := mq.NewConsumer(url, mq.QueueConfig{
		Name:       queue,
		DLQName:    dlqName,
		MessageTTL: mq.DefaultMessageTTL,
	}, prefetch)
	if err != nil {
		return nil, err
	}
	return &Consumer{inner: inner}, nil
}

// ConsumeWithReconnect starts consuming with automatic reconnection.
func (c *Consumer) ConsumeWithReconnect(ctx context.Context) <-chan amqp.Delivery {
	return c.inner.ConsumeWithReconnect(ctx, "parser")
}

// Close cleanly shuts down the consumer.
func (c *Consumer) Close() {
	c.inner.Close()
}
