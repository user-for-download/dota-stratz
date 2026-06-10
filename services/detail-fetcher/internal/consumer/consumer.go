package consumer

import (
	"fmt"

	"github.com/dota-stratz/shared/go-common/mq"
	amqp "github.com/rabbitmq/amqp091-go"
)

// Consumer wraps a RabbitMQ channel for consuming from the match_ids queue.
type Consumer struct {
	conn *amqp.Connection
	ch   *amqp.Channel
}

func NewConsumer(url string, queueName, dlqName string, prefetch int) (*Consumer, error) {
	conn, ch, err := mq.Connect(url)
	if err != nil {
		return nil, err
	}

	// Declare DLQ.
	_, err = ch.QueueDeclare(dlqName, true, false, false, false, amqp.Table{
		"x-message-ttl": int32(86400000), // 24h TTL
	})
	if err != nil {
		ch.Close()
		conn.Close()
		return nil, fmt.Errorf("failed to declare DLQ %s: %w", dlqName, err)
	}

	// Declare main queue with DLQ binding.
	args := amqp.Table{
		"x-dead-letter-exchange":    "",
		"x-dead-letter-routing-key": dlqName,
	}
	_, err = ch.QueueDeclare(queueName, true, false, false, false, args)
	if err != nil {
		ch.Close()
		conn.Close()
		return nil, fmt.Errorf("failed to declare queue %s: %w", queueName, err)
	}

	if err := ch.Qos(prefetch, 0, false); err != nil {
		ch.Close()
		conn.Close()
		return nil, fmt.Errorf("failed to set QoS: %w", err)
	}

	return &Consumer{conn: conn, ch: ch}, nil
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
		"detail-fetcher",
		false, // manual ack
		false, false, false, nil,
	)
}
