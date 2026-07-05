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

type Publisher struct {
	closeMu     sync.Mutex
	publishMu   sync.Mutex
	reconnectMu sync.Mutex

	url      string
	queueCfg *QueueConfig
	conn     *amqp.Connection
	ch       *amqp.Channel
	confirms <-chan amqp.Confirmation
	closed   bool

	pendingConfirms sync.Map

	shutdown  chan struct{}
	closeOnce sync.Once
}

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

	confirms := ch.NotifyPublish(make(chan amqp.Confirmation, 1000))

	p := &Publisher{
		url:      url,
		queueCfg: queueCfg,
		conn:     conn,
		ch:       ch,
		confirms: confirms,
		shutdown: make(chan struct{}),
	}

	go p.confirmListener(confirms)

	if queueCfg != nil {
		if err := DeclareQueueWithDLQ(ch, *queueCfg); err != nil {
			conn.Close()
			return nil, fmt.Errorf("failed to declare queues: %w", err)
		}
	}

	go p.handleConnectionLost()

	return p, nil
}

func (p *Publisher) confirmListener(confirms <-chan amqp.Confirmation) {
	for c := range confirms {
		if v, ok := p.pendingConfirms.LoadAndDelete(c.DeliveryTag); ok {
			waitCh := v.(chan error)
			if c.Ack {
				waitCh <- nil
			} else {
				waitCh <- fmt.Errorf("broker nacked message")
			}
		}
	}
}

func (p *Publisher) Publish(ctx context.Context, queueName string, body []byte) error {
	p.closeMu.Lock()
	ch := p.ch
	p.closeMu.Unlock()

	if ch == nil {
		p.reconnect()
		p.closeMu.Lock()
		ch = p.ch
		p.closeMu.Unlock()
		if ch == nil {
			return fmt.Errorf("publish failed: no channel available")
		}
	}

	waitCh := make(chan error, 1)

	p.publishMu.Lock()
	seqNo := ch.GetNextPublishSeqNo()
	p.pendingConfirms.Store(seqNo, waitCh)

	err := ch.PublishWithContext(ctx,
		"", queueName, false, false,
		amqp.Publishing{
			ContentType:  "application/json",
			DeliveryMode: amqp.Persistent,
			Body:         body,
			Timestamp:    time.Now(),
		},
	)
	p.publishMu.Unlock()

	if err != nil {
		p.pendingConfirms.Delete(seqNo)
		logger.Log.Warn("Publish failed, attempting reconnect and retry", zap.Error(err))

		p.reconnect()
		p.closeMu.Lock()
		ch = p.ch
		p.closeMu.Unlock()
		if ch == nil {
			return fmt.Errorf("publish failed after reconnect: no channel")
		}

		waitCh = make(chan error, 1)
		p.publishMu.Lock()
		seqNo = ch.GetNextPublishSeqNo()
		p.pendingConfirms.Store(seqNo, waitCh)

		err = ch.PublishWithContext(ctx, "", queueName, false, false, amqp.Publishing{
			ContentType: "application/json", DeliveryMode: amqp.Persistent,
			Body: body, Timestamp: time.Now(),
		})
		p.publishMu.Unlock()

		if err != nil {
			p.pendingConfirms.Delete(seqNo)
			return fmt.Errorf("publish failed after reconnect: %w", err)
		}
	}

	select {
	case err := <-waitCh:
		return err
	case <-time.After(5 * time.Second):
		p.pendingConfirms.Delete(seqNo)
		return fmt.Errorf("publish confirm timeout")
	case <-ctx.Done():
		p.pendingConfirms.Delete(seqNo)
		return ctx.Err()
	}
}

func (p *Publisher) isConnectionAlive() bool {
	p.closeMu.Lock()
	defer p.closeMu.Unlock()
	return p.conn != nil && !p.conn.IsClosed()
}

func (p *Publisher) handleConnectionLost() {
	select {
	case <-p.shutdown:
		return
	default:
	}
	p.closeMu.Lock()
	conn := p.conn
	p.closeMu.Unlock()
	if conn == nil {
		return
	}
	connErr := <-conn.NotifyClose(make(chan *amqp.Error, 1))
	if connErr == nil {
		return
	}

	select {
	case <-p.shutdown:
		return
	default:
	}

	logger.Log.Warn("Publisher RabbitMQ connection lost, reconnecting", zap.Error(connErr))
	p.reconnect()
}

func (p *Publisher) reconnect() {
	if p.isConnectionAlive() {
		return
	}
	p.reconnectMu.Lock()
	defer p.reconnectMu.Unlock()
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
			if err := ch.Confirm(false); err == nil {
				if p.queueCfg != nil {
					if err := DeclareQueueWithDLQ(ch, *p.queueCfg); err == nil {
						if err := p.exchange(conn, ch); err == nil {
							logger.Log.Info("Publisher reconnected", zap.Int("attempt", attempt))
							go p.handleConnectionLost()
							return
						}
					}
				} else {
					if err := p.exchange(conn, ch); err == nil {
						logger.Log.Info("Publisher reconnected", zap.Int("attempt", attempt))
						go p.handleConnectionLost()
						return
					}
				}
			}
			ch.Close()
			conn.Close()
		}

		p.backoffOrShutdown(backoff)
		backoff = min(backoff*2, maxBackoff)
	}
}

func (p *Publisher) backoffOrShutdown(d time.Duration) {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
	case <-p.shutdown:
	}
}

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
	p.confirms = ch.NotifyPublish(make(chan amqp.Confirmation, 1000))
	go p.confirmListener(p.confirms)
	p.closeMu.Unlock()

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
		p.closeMu.Lock()
		close(p.shutdown)
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
