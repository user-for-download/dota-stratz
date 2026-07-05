package mq

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/dota-stratz/shared/go-common/logger"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

// Consumer wraps a RabbitMQ connection and channel with automatic reconnection.
// Messages are delivered through a channel that survives broker restarts.
type Consumer struct {
	url      string
	cfg      QueueConfig
	prefetch int

	conn *amqp.Connection
	ch   *amqp.Channel

	closeOnce sync.Once
}

// NewConsumer connects to RabbitMQ, declares the queue + DLQ, sets QoS,
// and returns a Consumer ready to consume messages.
func NewConsumer(url string, cfg QueueConfig, prefetch int) (*Consumer, error) {
	conn, ch, err := Connect(url)
	if err != nil {
		return nil, err
	}

	if err := DeclareQueueWithDLQ(ch, cfg); err != nil {
		ch.Close()
		conn.Close()
		return nil, fmt.Errorf("failed to declare queues: %w", err)
	}

	if err := ch.Qos(prefetch, 0, false); err != nil {
		ch.Close()
		conn.Close()
		return nil, fmt.Errorf("failed to set QoS: %w", err)
	}

	return &Consumer{
		url:      url,
		cfg:      cfg,
		prefetch: prefetch,
		conn:     conn,
		ch:       ch,
	}, nil
}

// Consume starts consuming from the configured queue with the given
// consumer tag. Messages must be Ack'd or Nack'd manually.
func (c *Consumer) Consume(tag string) (<-chan amqp.Delivery, error) {
	return c.ch.Consume(
		c.cfg.Name,
		tag,
		false, // manual ack
		false, // exclusive
		false, // no-local
		false, // no-wait
		nil,   // args
	)
}

// ConsumeWithReconnect wraps consumption with automatic reconnection.
// If the broker restarts or the channel dies, it reconnects with exponential
// backoff (1s → 30s max) and resumes delivery on the returned channel.
// The channel is never closed on reconnect — only on permanent shutdown via
// context cancellation.
//
// The returned channel is closed when ctx is cancelled, allowing callers to
// range over it. The caller should exit via context cancellation.
func (c *Consumer) ConsumeWithReconnect(ctx context.Context, tag string) <-chan amqp.Delivery {
	outCh := make(chan amqp.Delivery, c.prefetch)

	go func() {
		defer close(outCh)
		backoff := 1 * time.Second
		const maxBackoff = 30 * time.Second
		var cons *Consumer

		for {
			select {
			case <-ctx.Done():
				if cons != nil {
					cons.Close()
				}
				c.Close()
				return
			default:
			}

			// Close the previous consumer before creating a new one to
			// prevent connection/channel leaks on reconnect.
			if cons != nil {
				cons.Close()
				cons = nil
			}

			var err error
			cons, err = NewConsumer(c.url, c.cfg, c.prefetch)
			if err != nil {
				logger.Log.Warn("Consumer reconnect init failed, retrying",
					zap.Error(err),
					zap.Duration("backoff", backoff))
				select {
				case <-time.After(backoff):
				case <-ctx.Done():
					c.Close()
					return
				}
				backoff = min(backoff*2, maxBackoff)
				continue
			}

			msgs, err := cons.Consume(tag)
			if err != nil {
				cons.Close()
				cons = nil
				logger.Log.Warn("Consumer reconnect consume failed, retrying",
					zap.Error(err),
					zap.Duration("backoff", backoff))
				select {
				case <-time.After(backoff):
				case <-ctx.Done():
					c.Close()
					return
				}
				backoff = min(backoff*2, maxBackoff)
				continue
			}

			backoff = 1 * time.Second // reset on success

			// Forward messages until channel closes or ctx is cancelled.
			for d := range msgs {
				select {
				case outCh <- d:
				case <-ctx.Done():
					cons.Close()
					c.Close()
					return
				}
			}

			// Message channel closed (connection lost). Sleep with backoff
			// before reconnecting.
			cons.Close()
			cons = nil
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				c.Close()
				return
			}
			backoff = min(backoff*2, maxBackoff)
		}
	}()

	return outCh
}

// Close cleanly shuts down the consumer's channel and connection.
// Safe to call multiple times from different goroutines.
func (c *Consumer) Close() {
	c.closeOnce.Do(func() {
		if c.ch != nil {
			c.ch.Close()
			c.ch = nil
		}
		if c.conn != nil {
			c.conn.Close()
			c.conn = nil
		}
	})
}
