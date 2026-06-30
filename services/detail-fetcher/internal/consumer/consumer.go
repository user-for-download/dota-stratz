package consumer

import (
	"github.com/dota-stratz/shared/go-common/mq"
	amqp "github.com/rabbitmq/amqp091-go"
)

// Consumer wraps the shared mq.Consumer with detail-fetcher-specific naming.
type Consumer struct {
	inner *mq.Consumer
}

func NewConsumer(url string, queueName, dlqName string, prefetch int) (*Consumer, error) {
	inner, err := mq.NewConsumer(url, mq.QueueConfig{
		Name:       queueName,
		DLQName:    dlqName,
		MessageTTL: mq.DefaultMessageTTL,
	}, prefetch)
	if err != nil {
		return nil, err
	}
	return &Consumer{inner: inner}, nil
}

func (c *Consumer) Close() {
	c.inner.Close()
}

func (c *Consumer) Consume(queueName string) (<-chan amqp.Delivery, error) {
	return c.inner.Consume("detail-fetcher")
}

// ConsumeTag starts consuming with a custom consumer tag. Used by
// consumeWithReconnect in main.go (reconnection loop).
func (c *Consumer) ConsumeTag(tag string) (<-chan amqp.Delivery, error) {
	return c.inner.Consume(tag)
}
