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
//
// The trigger-queue consumer and coordinator wiring have been removed —
// the id-fetcher now owns its own cron schedule and no longer needs to
// listen for external trigger messages.
type Publisher struct {
	mu        sync.Mutex
	url       string
	conn      *amqp.Connection
	matchIDsQ string

	closed bool

	// shutdown is closed on Close() to signal reconnect loops to exit
	// immediately instead of retrying forever (fixes deadlock where
	// Close() blocks on mu while handleConnectionLost holds mu in an
	// infinite reconnect loop).
	shutdown  chan struct{}
	closeOnce sync.Once
}

type MatchIDMessage struct {
	MatchID int64 `json:"match_id"`
}

// NewPublisher connects to RabbitMQ, declares the match_ids queue + its DLQ,
// and returns a Publisher ready to publish batches.
func NewPublisher(url string, matchIDsQueue string) (*Publisher, error) {
	conn, topoCh, err := mq.Connect(url)
	if err != nil {
		return nil, err
	}
	defer topoCh.Close()

	// Declare the Dead-Letter Queue for match_ids. Poisoned payloads
	// auto-expire after 24 hours instead of accumulating.
	dlqMatchIDs := matchIDsQueue + ".dlq"
	_, err = topoCh.QueueDeclare(dlqMatchIDs, true, false, false, false, amqp.Table{
		"x-message-ttl": int32(86400000), // 24 hours
	})
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to declare DLQ %s: %w", dlqMatchIDs, err)
	}

	// Declare the match_ids queue with DLQ binding.
	_, err = topoCh.QueueDeclare(matchIDsQueue, true, false, false, false, amqp.Table{
		"x-dead-letter-exchange":    "",
		"x-dead-letter-routing-key": dlqMatchIDs,
	})
	if err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to declare queue %s: %w", matchIDsQueue, err)
	}

	p := &Publisher{
		url:       url,
		conn:      conn,
		matchIDsQ: matchIDsQueue,
		shutdown:  make(chan struct{}),
	}

	// Start listening for connection-level errors so we can reconnect
	// proactively. Without this, the first publish after a broker restart
	// triggers the error and we reconnect then, but having the reconnection
	// goroutine means we're ready before the next batch.
	go p.handleConnectionLost()

	return p, nil
}

// handleConnectionLost blocks until the connection's NotifyClose fires,
// then replaces the connection with a new one. It exits when Close() is
// called (signalled by closed flag + conn.Close()).
//
// CRITICAL FIX: The mutex is released BEFORE calling reconnect() so a
// concurrent Close() call is not blocked forever by the infinite reconnect
// loop. See Bug #1.
func (p *Publisher) handleConnectionLost() {
	select {
	case <-p.shutdown:
		return
	default:
	}

	// Snapshot p.conn under mu to avoid a data race when reconnect()
	// swaps the pointer concurrently (finding #9).
	p.mu.Lock()
	conn := p.conn
	p.mu.Unlock()
	if conn == nil {
		return
	}

	connErr := <-conn.NotifyClose(make(chan *amqp.Error))
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

	// NOTE: we do NOT hold p.mu here. reconnect() uses the shutdown
	// channel to break out of its retry loop, and reconnectLocked()
	// is only called when p.mu is already held (from reconnectIfNeeded).
	p.reconnect()
}

// reconnect safely replaces the dead connection with a fresh one. Safe to
// call with or without p.mu held — it acquires the lock internally when
// swapping connections. The reconnect loop is interruptible via the shutdown
// channel so Close() cannot be blocked forever (fixes Bug #1/#4).
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
			// leaking (audit finding #1: goroutine fan-out on reconnect).
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

// reconnectLocked is identical to reconnect but assumes p.mu is already
// held. Used by reconnectIfNeeded which is always called under mu.
func (p *Publisher) reconnectLocked() {
	backoff := 1 * time.Second
	const maxBackoff = 30 * time.Second

	for attempt := 1; ; attempt++ {
		select {
		case <-p.shutdown:
			logger.Log.Debug("ReconnectLocked aborted during shutdown")
			return
		default:
		}

		conn, topoCh, err := mq.Connect(p.url)
		if err == nil {
			topoCh.Close()
			if p.closed {
				conn.Close()
				return
			}
			oldConn := p.conn
			p.conn = conn

			// Close the old connection AFTER swapping p.conn so the old
			// NotifyClose goroutine unblocks and exits cleanly instead of
			// leaking (audit finding #1: goroutine fan-out on reconnect).
			if oldConn != nil {
				oldConn.Close()
			}

			logger.Log.Info("RabbitMQ reconnected (locked)",
				zap.Int("attempt", attempt))
			go p.handleConnectionLost()
			return
		}

		logger.Log.Warn("RabbitMQ reconnect attempt failed (locked)",
			zap.Int("attempt", attempt),
			zap.Duration("backoff", backoff),
			zap.Error(err))

		select {
		case <-time.After(backoff):
		case <-p.shutdown:
			logger.Log.Debug("ReconnectLocked backoff aborted during shutdown")
			return
		}
		if backoff < maxBackoff {
			backoff *= 2
		}
	}
}

// reconnectIfNeeded checks whether the connection is still alive (by
// acquiring a channel) and reconnects if not. Must be called with p.mu held.
func (p *Publisher) reconnectIfNeeded() error {
	// Quick liveness check: if we can open a channel, the connection is fine.
	testCh, err := p.conn.Channel()
	if err == nil {
		testCh.Close()
		return nil
	}

	logger.Log.Warn("Publisher connection dead, reconnecting",
		zap.Error(err))
	p.reconnectLocked()
	return nil
}

func (p *Publisher) Close() {
	p.closeOnce.Do(func() {
		// Close shutdown FIRST to unblock any in-progress reconnect loops
		// before acquiring p.mu. This prevents the deadlock where
		// handleConnectionLost holds mu during reconnect and Close() blocks
		// on mu forever (Bug #1).
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
// don't accumulate (rabbitmq/amqp091-go#109).
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
		// reconnectIfNeeded already reconnected on error, but if the
		// reconnect itself failed, the inner reconnect() handles
		// exponential backoff internally.
	}

	ch, err := p.conn.Channel()
	if err != nil {
		return nil, fmt.Errorf("publish channel: %w", err)
	}
	if err := ch.Confirm(false); err != nil {
		ch.Close()
		return nil, fmt.Errorf("publish channel confirm: %w", err)
	}
	if _, err := ch.QueueDeclarePassive(p.matchIDsQ, true, false, false, false, nil); err != nil {
		ch.Close()
		return nil, fmt.Errorf("publish channel queue declare: %w", err)
	}
	return ch, nil
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
