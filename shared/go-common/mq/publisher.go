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

// Publisher sends messages to a RabbitMQ queue with publisher confirms
// for at-least-once delivery. Supports automatic reconnection when the
// broker connection is lost.
type Publisher struct {
	closeMu   sync.Mutex // serializes Close so it doesn't race with reconnect
	publishMu sync.Mutex // serializes Publish so the goroutine that sends reads its own ack

	// reconnectMu serializes reconnection attempts so handleConnectionLost
	// (background) and Publish (worker) cannot race in exchange() — only
	// one goroutine swaps connections at a time.
	reconnectMu sync.Mutex

	url      string
	queueCfg *QueueConfig // optional — if set, queues are re-declared on reconnect
	conn     *amqp.Connection
	ch       *amqp.Channel
	confirms <-chan amqp.Confirmation
	closed   bool

	// shutdown is closed on Close() to abort any in-progress reconnect
	// loops so the background goroutine does not leak.
	shutdown  chan struct{}
	closeOnce sync.Once
}

// NewPublisher connects to RabbitMQ, enables publisher confirms, optionally
// declares queues, and starts listening for connection errors.
//
// If queueCfg is non-nil, the queue + DLQ are declared on initial connect
// and re-declared on each reconnect.
func NewPublisher(url string, queueCfg *QueueConfig) (*Publisher, error) {
	conn, ch, err := Connect(url)
	if err != nil {
		return nil, err
	}

	if err := ch.Confirm(false); err != nil {
		ch.Close()
		conn.Close()
		return nil, fmt.Errorf("failed to enable publisher confirms: %w", err)
	}

	// Register confirms ONCE during construction. A shared channel + Mutex
	// serializes publish-and-wait so the goroutine that published the message
	// is guaranteed to read its own confirmation.
	confirms := ch.NotifyPublish(make(chan amqp.Confirmation, 100))

	p := &Publisher{
		url:       url,
		queueCfg:  queueCfg,
		conn:      conn,
		ch:        ch,
		confirms:  confirms,
		shutdown:  make(chan struct{}),
	}

	if queueCfg != nil {
		if err := DeclareQueueWithDLQ(ch, *queueCfg); err != nil {
			conn.Close()
			return nil, fmt.Errorf("failed to declare queues: %w", err)
		}
	}

	// Start listening for connection errors so we reconnect proactively.
	go p.handleConnectionLost()

	return p, nil
}

// Publish sends a single message to the given queue with publisher confirms.
// Thread-safe — uses internal mutex to serialise publishes so the calling
// goroutine reads its own confirmation.
func (p *Publisher) Publish(ctx context.Context, queueName string, body []byte) error {
	p.publishMu.Lock()
	defer p.publishMu.Unlock()

	p.closeMu.Lock()
	ch := p.ch
	confirms := p.confirms
	p.closeMu.Unlock()

	if err := ch.PublishWithContext(ctx,
		"",        // default exchange
		queueName, // routing key = queue name
		false,     // mandatory
		false,     // immediate
		amqp.Publishing{
			ContentType:  "application/json",
			DeliveryMode: amqp.Persistent,
			Body:         body,
			Timestamp:    time.Now(),
		},
	); err != nil {
		// Connection/channel may be dead. Try reconnecting once and retry.
		logger.Log.Warn("Publish failed, attempting reconnect and retry",
			zap.Error(err))
		p.reconnect()

		p.closeMu.Lock()
		ch = p.ch
		confirms = p.confirms
		p.closeMu.Unlock()

		if err := ch.PublishWithContext(ctx,
			"", queueName, false, false,
			amqp.Publishing{
				ContentType:  "application/json",
				DeliveryMode: amqp.Persistent,
				Body:         body,
				Timestamp:    time.Now(),
			},
		); err != nil {
			return fmt.Errorf("publish failed after reconnect: %w", err)
		}
	}

	select {
	case confirm, ok := <-confirms:
		if !ok {
			return fmt.Errorf("confirm channel closed; delivery status unknown")
		}
		if !confirm.Ack {
			return fmt.Errorf("broker nacked message")
		}
	case <-time.After(5 * time.Second):
		return fmt.Errorf("publish confirm timeout")
	case <-ctx.Done():
		return ctx.Err()
	}

	return nil
}

// isConnectionAlive returns true if the current connection is non-nil and
// not closed by the broker.
func (p *Publisher) isConnectionAlive() bool {
	p.closeMu.Lock()
	defer p.closeMu.Unlock()
	if p.conn == nil {
		return false
	}
	return !p.conn.IsClosed()
}

// handleConnectionLost blocks until the connection's NotifyClose fires,
// then replaces the connection with a new one.
func (p *Publisher) handleConnectionLost() {
	select {
	case <-p.shutdown:
		return
	default:
	}

	// Snapshot p.conn under closeMu to avoid a data race when exchange()
	// (called by reconnect()) swaps the pointer concurrently.
	p.closeMu.Lock()
	conn := p.conn
	p.closeMu.Unlock()
	if conn == nil {
		return
	}

	connErr := <-conn.NotifyClose(make(chan *amqp.Error, 1))
	if connErr == nil {
		// Clean close (Close() was called) — nothing to do.
		return
	}

	// Check if Close() was called between NotifyClose and now.
	select {
	case <-p.shutdown:
		return
	default:
	}

	logger.Log.Warn("Publisher RabbitMQ connection lost, reconnecting",
		zap.Error(connErr))

	p.reconnect()
}

// reconnect replaces the dead connection+channel with fresh ones. Uses
// reconnectMu to ensure only one goroutine is reconnecting at a time,
// preventing the exchange race where two concurrent goroutines each close
// the other's fresh connection.
func (p *Publisher) reconnect() {
	if p.isConnectionAlive() {
		return
	}

	p.reconnectMu.Lock()
	defer p.reconnectMu.Unlock()

	// Double-check: the connection may have been re-established while we
	// were waiting for reconnectMu.
	if p.isConnectionAlive() {
		return
	}

	backoff := 1 * time.Second
	const maxBackoff = 30 * time.Second

	for attempt := 1; ; attempt++ {
		select {
		case <-p.shutdown:
			logger.Log.Debug("Publisher reconnect aborted during shutdown")
			return
		default:
		}

		conn, ch, err := Connect(p.url)
		if err == nil {
			if err := ch.Confirm(false); err != nil {
				ch.Close()
				conn.Close()
				logger.Log.Warn("Reconnect confirm failed, retrying",
					zap.Error(err))
				p.backoffOrShutdown(backoff)
				backoff = min(backoff*2, maxBackoff)
				continue
			}

			if p.queueCfg != nil {
				if err := DeclareQueueWithDLQ(ch, *p.queueCfg); err != nil {
					ch.Close()
					conn.Close()
					logger.Log.Warn("Reconnect queue declare failed, retrying",
						zap.Error(err))
					p.backoffOrShutdown(backoff)
					backoff = min(backoff*2, maxBackoff)
					continue
				}
			}

			if err := p.exchange(conn, ch); err != nil {
				logger.Log.Warn("Reconnect exchange failed, retrying",
					zap.Error(err))
				p.backoffOrShutdown(backoff)
				backoff = min(backoff*2, maxBackoff)
				continue
			}

			logger.Log.Info("Publisher reconnected to RabbitMQ",
				zap.Int("attempt", attempt))
			go p.handleConnectionLost()
			return
		}

		logger.Log.Warn("Publisher reconnect attempt failed",
			zap.Int("attempt", attempt),
			zap.Duration("backoff", backoff),
			zap.Error(err))
		p.backoffOrShutdown(backoff)
		backoff = min(backoff*2, maxBackoff)
	}
}

// backoffOrShutdown sleeps for the given duration or returns early if
// shutdown is signalled.
func (p *Publisher) backoffOrShutdown(d time.Duration) {
	select {
	case <-time.After(d):
	case <-p.shutdown:
	}
}

// exchange atomically replaces the connection and channel under closeMu.
// closeMu is released BEFORE calling oldConn.Close() to prevent a deadlock
// when the old connection is in a blackholed/stalled TCP state.
func (p *Publisher) exchange(conn *amqp.Connection, ch *amqp.Channel) error {
	p.closeMu.Lock()

	if p.closed {
		p.closeMu.Unlock()
		conn.Close()
		ch.Close()
		return fmt.Errorf("publisher is closed")
	}

	oldConn := p.conn
	oldCh := p.ch

	p.conn = conn
	p.ch = ch
	p.confirms = ch.NotifyPublish(make(chan amqp.Confirmation, 100))

	// Release closeMu BEFORE closing the old connection so Publish() and
	// Close() are not blocked by a stalled TCP Close().
	p.closeMu.Unlock()

	if oldCh != nil {
		oldCh.Close()
	}
	if oldConn != nil {
		oldConn.Close()
	}
	return nil
}

// Close cleanly shuts down the publisher, aborting any in-progress
// reconnect loops before closing the connection.
func (p *Publisher) Close() {
	p.closeOnce.Do(func() {
		// Close shutdown FIRST to abort any in-progress reconnect loops
		// before acquiring closeMu.
		close(p.shutdown)

		p.closeMu.Lock()
		p.closed = true
		if p.ch != nil {
			p.ch.Close()
		}
		if p.conn != nil {
			p.conn.Close()
		}
		p.closeMu.Unlock()
	})
}
