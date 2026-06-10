package consumer

import (
	"fmt"
	"time"

	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/dota-stratz/shared/go-common/mq"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

// Consumer wraps a RabbitMQ channel for consuming from the raw_matches queue.
type Consumer struct {
	conn     *amqp.Connection
	ch       *amqp.Channel
	url      string
	queue    string
	dlqName  string
	prefetch int
}

func NewConsumer(url, queue, dlqName string, prefetch int) (*Consumer, error) {
	conn, ch, err := mq.Connect(url)
	if err != nil {
		return nil, err
	}

	// Declare DLQ with 24h TTL so messages are automatically discarded
	// if not consumed within that window.
	_, err = ch.QueueDeclare(dlqName, true, false, false, false, amqp.Table{
		"x-message-ttl": int32(86400000),
	})
	if err != nil {
		ch.Close()
		conn.Close()
		return nil, fmt.Errorf("failed to declare DLQ %s: %w", dlqName, err)
	}

	// Declare main queue with DLX binding.
	args := amqp.Table{
		"x-dead-letter-exchange":    "",
		"x-dead-letter-routing-key": dlqName,
	}
	_, err = ch.QueueDeclare(queue, true, false, false, false, args)
	if err != nil {
		ch.Close()
		conn.Close()
		return nil, fmt.Errorf("failed to declare queue %s: %w", queue, err)
	}

	if err := ch.Qos(prefetch, 0, false); err != nil {
		ch.Close()
		conn.Close()
		return nil, fmt.Errorf("failed to set QoS: %w", err)
	}

	return &Consumer{
		conn:     conn,
		ch:       ch,
		url:      url,
		queue:    queue,
		dlqName:  dlqName,
		prefetch: prefetch,
	}, nil
}

func (c *Consumer) Close() {
	if c.ch != nil {
		c.ch.Close()
	}
	if c.conn != nil {
		c.conn.Close()
	}
}

func (c *Consumer) Consume(queueName string) (<-chan amqp.Delivery, error) {
	return c.ch.Consume(
		queueName,
		"parser",
		false, // manual ack
		false, false, false, nil,
	)
}

// ConsumeWithReconnect wraps RabbitMQ consumption with automatic reconnection.
// If the broker restarts or the channel dies, it reconnects with exponential
// backoff (1s → 30s max) and resumes delivery on the returned channel.
func (c *Consumer) ConsumeWithReconnect(done <-chan struct{}) <-chan amqp.Delivery {
	outCh := make(chan amqp.Delivery, c.prefetch)

	go func() {
		// NOTE: outCh is intentionally NOT closed on exit. The processor
		// exits via ctx.Done(), not via channel close. Keeping outCh open
		// allows the processor to drain any buffered messages during the
		// shutdown window. The channel is GC'd after all goroutines exit.
		backoff := 1 * time.Second

		for {
			select {
			case <-done:
				return
			default:
			}

			cons, err := NewConsumer(c.url, c.queue, c.dlqName, c.prefetch)
			if err != nil {
				logger.Log.Warn("Consumer init failed, retrying",
					zap.Error(err),
					zap.Duration("backoff", backoff))
				time.Sleep(backoff)
				backoff = min(backoff*2, 30*time.Second)
				continue
			}

			msgs, err := cons.Consume(c.queue)
			if err != nil {
				cons.Close()
				logger.Log.Warn("Failed to start consuming, retrying",
					zap.Error(err),
					zap.Duration("backoff", backoff))
				time.Sleep(backoff)
				backoff = min(backoff*2, 30*time.Second)
				continue
			}

			backoff = 1 * time.Second // reset on success

			// Forward messages until channel closes.
			for d := range msgs {
				select {
				case outCh <- d:
				case <-done:
					cons.Close()
					return
				}
			}

			cons.Close()
			time.Sleep(backoff)
			backoff = min(backoff*2, 30*time.Second)
		}
	}()

	return outCh
}
