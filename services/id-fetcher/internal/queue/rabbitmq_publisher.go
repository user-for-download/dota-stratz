package queue

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/dota-stratz/shared/go-common/mq"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

// Publisher manages RabbitMQ connectivity and publishes match IDs with
// publisher confirms for guaranteed delivery. Supports automatic
// reconnection when the broker connection is lost (rabbitmq restart,
// network partition, etc.).
type Publisher struct {
	mu        sync.Mutex
	url       string
	conn      *amqp.Connection
	matchIDsQ string
	queueCfg  mq.QueueConfig

	closed bool

	// shutdown is closed on Close() to signal reconnect loops to exit
	// immediately instead of retrying forever.
	shutdown  chan struct{}
	closeOnce sync.Once
}

type MatchIDMessage struct {
	MatchID int64 `json:"match_id"`
}

// NewPublisher connects to RabbitMQ, declares the match_ids queue + its DLQ
// using the shared mq.DeclareQueueWithDLQ, and returns a Publisher ready to
// publish batches.
func NewPublisher(url string, matchIDsQueue string) (*Publisher, error) {
	conn, topoCh, err := mq.Connect(url)
	if err != nil {
		return nil, err
	}
	defer topoCh.Close()

	// Declare queues using the shared helper.
	queueCfg := mq.QueueConfig{
		Name:       matchIDsQueue,
		DLQName:    matchIDsQueue + ".dlq",
		MessageTTL: mq.DefaultMessageTTL,
	}
	if err := mq.DeclareQueueWithDLQ(topoCh, queueCfg); err != nil {
		conn.Close()
		return nil, err
	}

	p := &Publisher{
		url:       url,
		conn:      conn,
		matchIDsQ: matchIDsQueue,
		queueCfg:  queueCfg,
		shutdown:  make(chan struct{}),
	}

	// Start listening for connection-level errors so we can reconnect
	// proactively.
	go p.handleConnectionLost()

	return p, nil
}

// handleConnectionLost blocks until the connection's NotifyClose fires,
// then replaces the connection with a new one. It exits when Close() is
// called (signalled by closed flag + conn.Close()).
func (p *Publisher) handleConnectionLost() {
	select {
	case <-p.shutdown:
		return
	default:
	}

	// Snapshot p.conn under mu to avoid a data race when reconnect()
	// swaps the pointer concurrently.
	p.mu.Lock()
	conn := p.conn
	p.mu.Unlock()
	if conn == nil {
		return
	}

	connErr := <-conn.NotifyClose(make(chan *amqp.Error, 1))
	if connErr == nil {
		// Clean close (Close() was called) — nothing to do.
		return
	}
	logger.Log.Warn("RabbitMQ connection lost, reconnecting",
		zap.Error(connErr))

	// Check if Close() was called between NotifyClose and now.
	select {
	case <-p.shutdown:
		return
	default:
	}

	p.reconnect()
}

// reconnect safely replaces the dead connection with a fresh one. Safe to
// call with or without p.mu held — it acquires the lock internally when
// swapping connections.
func (p *Publisher) reconnect() {
	backoff := 1 * time.Second
	const maxBackoff = 30 * time.Second

	for attempt := 1; ; attempt++ {
		select {
		case <-p.shutdown:
			logger.Log.Debug("Reconnect aborted during shutdown")
			return
		default:
		}
		conn, topoCh, err := mq.Connect(p.url)
		if err == nil {
			// Re-declare queues on the new connection to handle
			// Mnesia corruption, manual deletion, or cluster recovery.
			if err := mq.DeclareQueueWithDLQ(topoCh, p.queueCfg); err != nil {
				topoCh.Close()
				conn.Close()
				logger.Log.Warn("Reconnect queue declare failed, retrying",
					zap.Error(err))
				select {
				case <-time.After(backoff):
				case <-p.shutdown:
					return
				}
				backoff = min(backoff*2, maxBackoff)
				continue
			}
			topoCh.Close()

			p.mu.Lock()
			if p.closed {
				conn.Close()
				p.mu.Unlock()
				return
			}
			oldConn := p.conn
			p.conn = conn
			p.mu.Unlock()

			// Close the old connection AFTER swapping p.conn so the old
			// NotifyClose goroutine unblocks and exits cleanly instead of
			// leaking.
			if oldConn != nil {
				oldConn.Close()
			}

			logger.Log.Info("RabbitMQ reconnected",
				zap.Int("attempt", attempt))
			go p.handleConnectionLost()
			return
		}

		logger.Log.Warn("RabbitMQ reconnect attempt failed",
			zap.Int("attempt", attempt),
			zap.Duration("backoff", backoff),
			zap.Error(err))

		select {
		case <-time.After(backoff):
		case <-p.shutdown:
			logger.Log.Debug("Reconnect backoff aborted during shutdown")
			return
		}
		if backoff < maxBackoff {
			backoff *= 2
		}
	}
}

func (p *Publisher) Close() {
	p.closeOnce.Do(func() {
		// Close shutdown FIRST to unblock any in-progress reconnect loops
		// before acquiring p.mu.
		close(p.shutdown)

		p.mu.Lock()
		p.closed = true
		if p.conn != nil {
			p.conn.Close()
		}
		p.mu.Unlock()
	})
}

// publishChannel creates a fresh confirm-mode AMQP channel for one batch.
// Closed after the batch so NotifyPublish listeners are cleaned up and
// don't accumulate.
//
// If the underlying connection is dead (e.g. broker restart), this method
// reconnects transparently before creating the channel.
func (p *Publisher) publishChannel() (*amqp.Channel, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return nil, fmt.Errorf("publisher is closed")
	}

	// Try to reconnect if the connection is no longer healthy.
	if err := p.reconnectIfNeeded(); err != nil {
		return nil, fmt.Errorf("reconnect failed: %w", err)
	}

	ch, err := p.conn.Channel()
	if err != nil {
		return nil, fmt.Errorf("publish channel: %w", err)
	}
	if err := ch.Confirm(false); err != nil {
		ch.Close()
		return nil, fmt.Errorf("publish channel confirm: %w", err)
	}
	return ch, nil
}

// reconnectIfNeeded checks whether the connection is still alive (by
// acquiring a channel) and reconnects if not. Releases p.mu before
// reconnecting to avoid deadlock (reconnect acquires p.mu internally).
func (p *Publisher) reconnectIfNeeded() error {
	testCh, err := p.conn.Channel()
	if err == nil {
		testCh.Close()
		return nil
	}

	logger.Log.Warn("Publisher connection dead, reconnecting",
		zap.Error(err))
	p.mu.Unlock()
	p.reconnect()
	p.mu.Lock()

	// Verify the new connection is alive after reconnect
	if p.conn == nil {
		return fmt.Errorf("reconnect failed: connection is nil (publisher may be closed)")
	}
	if ch, err := p.conn.Channel(); err != nil {
		return fmt.Errorf("reconnected but channel failed: %w", err)
	} else {
		ch.Close()
	}
	return nil
}

// PublishBatch publishes match IDs to the match_ids queue and waits for
// broker confirms. A fresh AMQP channel is created per batch and closed
// when done to prevent NotifyPublish listener leaks.
func (p *Publisher) PublishBatch(ctx context.Context, queueName string, matchIDs []int64) error {
	ch, err := p.publishChannel()
	if err != nil {
		return fmt.Errorf("PublishBatch: %w", err)
	}
	defer ch.Close()

	confirms := ch.NotifyPublish(make(chan amqp.Confirmation, len(matchIDs)))

	for _, id := range matchIDs {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		body, err := json.Marshal(MatchIDMessage{MatchID: id})
		if err != nil {
			return fmt.Errorf("failed to marshal match_id %d: %w", id, err)
		}
		err = ch.PublishWithContext(ctx, "", queueName, false, false, amqp.Publishing{
			ContentType:  "application/json",
			DeliveryMode: amqp.Persistent,
			Body:         body,
			Timestamp:    time.Now(),
		})
		if err != nil {
			return fmt.Errorf("failed to publish match_id %d: %w", id, err)
		}
	}

	for i := 0; i < len(matchIDs); i++ {
		select {
		case confirm := <-confirms:
			if !confirm.Ack {
				return fmt.Errorf("broker rejected message (Nack)")
			}
		case <-time.After(5 * time.Second):
			return fmt.Errorf("timed out waiting for publisher confirm after %d of %d messages",
				i, len(matchIDs))
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}
