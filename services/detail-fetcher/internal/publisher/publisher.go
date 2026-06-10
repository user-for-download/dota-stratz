package publisher

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

// RawMatchMessage is the payload published to the raw_matches queue for the
// parser service to consume.
//
// RawJSON is typed as json.RawMessage (which is also []byte under the hood)
// so it is marshaled as the inner JSON, not as a base64-encoded string.
// Using plain []byte here would cause the parser service to receive a
// string-typed field ("raw_json":"eyJ...") that fails to unmarshal
// directly into models.OpenDotaMatch — see Issue #32.
type RawMatchMessage struct {
	MatchID   int64           `json:"match_id"`
	RawJSON   json.RawMessage `json:"raw_json"`
	FetchedAt time.Time       `json:"fetched_at"`
}

// Publisher sends match JSON to the raw_matches queue with publisher confirms
// for at-least-once delivery. Supports automatic reconnection when the
// broker connection is lost (broker restart, network partition, etc.).
type Publisher struct {
	closeMu   sync.Mutex // serializes Close so it doesn't race with reconnect
	publishMu sync.Mutex // serializes Publish so the goroutine that sends reads its own ack

	// reconnectMu serializes reconnection attempts so handleConnectionLost
	// (background) and Publish (worker) cannot race in exchange() — only
	// one goroutine swaps connections at a time (fixes Bug #3).
	reconnectMu sync.Mutex

	url         string
	rawMatchesQ string
	dlqQ        string
	conn        *amqp.Connection
	ch          *amqp.Channel
	confirms    <-chan amqp.Confirmation
	closed      bool

	// shutdown is closed on Close() to abort any in-progress reconnect
	// loops so the background goroutine does not leak (fixes Bug #2).
	shutdown  chan struct{}
	closeOnce sync.Once
}

func NewPublisher(url string, rawMatchesQueue, dlqQueue string) (*Publisher, error) {
	conn, ch, err := mq.Connect(url)
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
	// is guaranteed to read its own confirmation. (See the concurrency trap
	// in the detail-fetcher review.)
	confirms := ch.NotifyPublish(make(chan amqp.Confirmation, 100))

	p := &Publisher{
		url:         url,
		rawMatchesQ: rawMatchesQueue,
		dlqQ:        dlqQueue,
		conn:        conn,
		ch:          ch,
		confirms:    confirms,
		shutdown:    make(chan struct{}),
	}

	if err := p.declareQueues(); err != nil {
		conn.Close()
		return nil, err
	}

	// Start listening for connection errors so we reconnect proactively.
	go p.handleConnectionLost()

	return p, nil
}

func (p *Publisher) declareQueues() error {
	// Declare DLQ.
	_, err := p.ch.QueueDeclare(p.dlqQ, true, false, false, false, amqp.Table{
		"x-message-ttl": int32(86400000), // 24h TTL
	})
	if err != nil {
		return fmt.Errorf("failed to declare DLQ %s: %w", p.dlqQ, err)
	}

	// Declare main queue with DLQ binding.
	args := amqp.Table{
		"x-dead-letter-exchange":    "",
		"x-dead-letter-routing-key": p.dlqQ,
	}
	_, err = p.ch.QueueDeclare(p.rawMatchesQ, true, false, false, false, args)
	if err != nil {
		return fmt.Errorf("failed to declare queue %s: %w", p.rawMatchesQ, err)
	}

	return nil
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
	// (called by reconnect()) swaps the pointer concurrently (finding #9).
	p.closeMu.Lock()
	conn := p.conn
	p.closeMu.Unlock()
	if conn == nil {
		return
	}

	connErr := <-conn.NotifyClose(make(chan *amqp.Error))
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

// isConnectionAlive returns true if the current connection is non-nil and
// not closed by the broker. Used for short-circuit checks before entering
// the expensive full reconnect loop.
func (p *Publisher) isConnectionAlive() bool {
	p.closeMu.Lock()
	defer p.closeMu.Unlock()
	if p.conn == nil {
		return false
	}
	return !p.conn.IsClosed()
}

// reconnect replaces the dead connection+channel with fresh ones. Uses
// reconnectMu to ensure only one goroutine is reconnecting at a time,
// preventing the exchange race where two concurrent goroutines each close
// the other's fresh connection (fixes Bug #3).
//
// A fast liveness check before acquiring reconnectMu avoids unnecessary
// reconnects when the connection was already re-established by another
// goroutine (double-checked locking pattern).
//
// All blocking operations (time.Sleep) are interruptible via the shutdown
// channel so the background goroutine does not leak after Close() (fixes
// Bug #2).
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

		conn, ch, err := mq.Connect(p.url)
		if err == nil {
			if err := ch.Confirm(false); err != nil {
				ch.Close()
				conn.Close()
				logger.Log.Warn("Reconnect channel confirm failed, retrying",
					zap.Error(err))

				select {
				case <-time.After(backoff):
				case <-p.shutdown:
					logger.Log.Debug("Reconnect confirm backoff aborted during shutdown")
					return
				}
				if backoff < maxBackoff {
					backoff *= 2
				}
				continue
			}

			if err := p.exchange(conn, ch); err != nil {
				logger.Log.Warn("Reconnect exchange failed, retrying",
					zap.Error(err))

				select {
				case <-time.After(backoff):
				case <-p.shutdown:
					logger.Log.Debug("Reconnect exchange backoff aborted during shutdown")
					return
				}
				if backoff < maxBackoff {
					backoff *= 2
				}
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

		select {
		case <-time.After(backoff):
		case <-p.shutdown:
			logger.Log.Debug("Publisher reconnect backoff aborted during shutdown")
			return
		}
		if backoff < maxBackoff {
			backoff *= 2
		}
	}
}

// exchange atomically replaces the connection and channel under closeMu.
// Must only be called from reconnect() which holds no locks.
func (p *Publisher) exchange(conn *amqp.Connection, ch *amqp.Channel) error {
	p.closeMu.Lock()
	defer p.closeMu.Unlock()

	if p.closed {
		conn.Close()
		ch.Close()
		return fmt.Errorf("publisher is closed")
	}

	oldConn := p.conn
	oldCh := p.ch

	p.conn = conn
	p.ch = ch
	p.confirms = ch.NotifyPublish(make(chan amqp.Confirmation, 100))

	if oldCh != nil {
		oldCh.Close()
	}
	if oldConn != nil {
		oldConn.Close()
	}
	return nil
}

func (p *Publisher) Close() {
	p.closeOnce.Do(func() {
		// Close shutdown FIRST to abort any in-progress reconnect loops
		// before acquiring closeMu. This prevents background goroutine
		// leaks (Bug #2).
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

// Publish sends a single message to the given queue and waits for the broker
// to confirm delivery. Serialized via sync.Mutex so the calling goroutine
// reads its own confirmation from the shared channel.
//
// If the underlying connection is dead (e.g. broker restart), this method
// reconnects transparently before retrying the publish.
func (p *Publisher) Publish(ctx context.Context, queueName string, msg RawMatchMessage) error {
	p.publishMu.Lock()
	defer p.publishMu.Unlock()

	body, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	logger.Log.Debug("Publishing raw match",
		zap.Int64("match_id", msg.MatchID),
		zap.Int("json_bytes", len(msg.RawJSON)))

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
			zap.Int64("match_id", msg.MatchID),
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
			return fmt.Errorf("confirm channel closed during reconnect; delivery status unknown for match_id %d", msg.MatchID)
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
