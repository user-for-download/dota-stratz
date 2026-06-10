package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/dota-stratz/services/detail-fetcher/internal/api"
	"github.com/dota-stratz/services/detail-fetcher/internal/config"
	"github.com/dota-stratz/services/detail-fetcher/internal/consumer"
	"github.com/dota-stratz/services/detail-fetcher/internal/publisher"
	"github.com/dota-stratz/services/detail-fetcher/internal/worker"
	"github.com/dota-stratz/shared/go-common/cache"
	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/dota-stratz/shared/go-common/proxypool"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	amqp "github.com/rabbitmq/amqp091-go"
	"go.uber.org/zap"
)

func main() {
	logger.InitLogger()
	defer logger.Sync()

	cfg, err := config.Load("config/config.yaml")
	if err != nil {
		logger.Log.Fatal("Failed to load config", zap.Error(err))
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 1. Redis & Proxy Pool
	rdb, err := cache.Connect(cfg.Redis.Addr, cfg.Redis.Password, cfg.Redis.DB)
	if err != nil {
		logger.Log.Fatal("Redis connection failed", zap.Error(err))
	}
	defer rdb.Close()

	// Register the Redis-sourced proxy pool collector BEFORE the metrics
	// server starts so /metrics includes dota2_proxy_pool_available and
	// dota2_proxy_pool_leased on the first scrape. P0-4: replaces the
	// per-process promauto.NewGauge which diverged across services
	// (e.g. detail-fetcher reported -322 while Redis ground truth was
	// 129) because the background lease reaper bypassed the gauge.
	prometheus.MustRegister(
		proxypool.NewRedisPoolCollector(rdb, "dota2:proxies", "dota2:proxies:leases"),
	)

	proxyPool, err := proxypool.New(rdb, proxypool.Config{
		Strategy:          "timestamp",
		SoftFailThreshold: 3,
		CooldownDuration:  5 * time.Minute,
		LeaseDuration:     60 * time.Second,
		SoftRetryDelay:    30 * time.Second,
		FailureCounterTTL: 60 * time.Minute,
	})
	if err != nil {
		logger.Log.Fatal("Proxy Pool init failed", zap.Error(err))
	}

	// 2. OpenDota API Client (rate-limited proxy lifecycle)
	odClient := api.NewClient(cfg.OpenDota.APIURL, proxyPool, cfg.OpenDota.TimeoutSec, cfg.OpenDota.MaxReqPerMin, cfg.OpenDota.UserAgent)

	// 3. RabbitMQ Publisher (raw_matches queue)
	pub, err := publisher.NewPublisher(
		cfg.RabbitMQ.URL,
		cfg.RabbitMQ.Queues.RawMatches,
		cfg.RabbitMQ.Queues.RawMatchesDLQ,
	)
	if err != nil {
		logger.Log.Fatal("Publisher init failed", zap.Error(err))
	}
	defer pub.Close()

	// 4. Metrics & Health HTTP Server
	go startMetricsServer(ctx, cfg.Worker.MetricsPort)

	// 5. RabbitMQ Consumer with reconnection loop (survives broker restarts)
	msgs, closeConsumer := consumeWithReconnect(ctx, cfg)

	// 6. Worker Pool
	w := worker.NewWorker(odClient, pub, cfg.RabbitMQ.Queues.RawMatches, cfg.Worker.MaxRetries, cfg.Worker.RetryDelaySec)
	var wg sync.WaitGroup

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	logger.Log.Info("Detail Fetcher started",
		zap.Int("concurrency", cfg.Worker.Concurrency),
		zap.Int("metrics_port", cfg.Worker.MetricsPort))

	for range cfg.Worker.Concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case d, ok := <-msgs:
					if !ok {
						// The channel only closes on shutdown (not on
						// reconnect — consumeWithReconnect reuses the
						// same outCh for its whole lifetime). If the
						// invariant ever breaks and we DO see a closed
						// channel, break out of the loop to avoid a
						// 100%-CPU spin: a closed channel is always
						// ready for receive, so the select would keep
						// selecting this branch forever.
						if ctx.Err() != nil {
							return
						}
						logger.Log.Warn("Message channel closed unexpectedly; worker exiting")
						return
					}
					if ctx.Err() != nil {
						// Shutdown — requeue the in-flight message so
						// the next instance can pick it up.
						if err := d.Nack(false, true); err != nil { // requeue
							logger.Log.Error("Failed to Nack delivery during shutdown",
								zap.Uint64("delivery_tag", d.DeliveryTag),
								zap.Error(err))
						}
						continue
					}
					switch w.Process(ctx, d) {
					case worker.Ack:
						if err := d.Ack(false); err != nil {
							logger.Log.Error("Failed to Ack delivery",
								zap.Uint64("delivery_tag", d.DeliveryTag),
								zap.Error(err))
						}
					case worker.NackDLQ:
						if err := d.Nack(false, false); err != nil { // DLQ (no requeue)
							logger.Log.Error("Failed to Nack delivery to DLQ",
								zap.Uint64("delivery_tag", d.DeliveryTag),
								zap.Error(err))
						}
					case worker.NackRequeue:
						if err := d.Nack(false, true); err != nil { // requeue for retry
							logger.Log.Error("Failed to Nack delivery for requeue",
								zap.Uint64("delivery_tag", d.DeliveryTag),
								zap.Error(err))
						}
					}
				}
			}
		}()
	}

	// 7. Graceful Shutdown (two-phase: stop consuming first, then drain workers)
	<-quit
	logger.Log.Info("Shutting down Detail Fetcher...")

	// Phase 1: Stop accepting new deliveries. The consumer goroutine
	// sees ctx cancellation and stops forwarding messages to outCh.
	// Crucially, it does NOT close the AMQP connection — that would
	// prevent in-flight workers from Ack/Nack-ing their deliveries.
	cancel()

	// Phase 2: Wait for in-flight workers to finish their current work
	// and Ack/Nack each delivery on the still-open AMQP connection.
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
		logger.Log.Info("In-flight workers drained")
	case <-time.After(30 * time.Second):
		logger.Log.Warn("Shutdown timeout exceeded, forcing exit")
	}

	// Phase 3: Now that all workers have finished, safely close the
	// consumer connection. No goroutines are trying to Ack/Nack anymore.
	closeConsumer()
}

// startMetricsServer exposes /metrics and /healthz for Prometheus scraping and
// orchestration readiness probes.
func startMetricsServer(ctx context.Context, port int) {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	srv := &http.Server{
		Addr:              fmt.Sprintf(":%d", port),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()

	logger.Log.Info("Metrics server listening", zap.Int("port", port))
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		logger.Log.Error("Metrics server error", zap.Error(err))
	}
}

// consumeWithReconnect wraps RabbitMQ consumption with automatic reconnection.
// If the broker restarts or the channel dies, it reconnects with exponential
// backoff (1s → 30s max) and resumes delivery on the returned channel.
//
// On shutdown (ctx cancelled), the forward loop stops but the consumer
// connection is NOT closed — the caller must invoke the returned close
// function AFTER all workers have drained their in-flight messages and
// Ack/Nack-ed them. This ordering prevents a race where cons.Close()
// closes the AMQP channel while a worker is mid-Ack.
func consumeWithReconnect(ctx context.Context, cfg *config.Config) (<-chan amqp.Delivery, func()) {
	outCh := make(chan amqp.Delivery, 100)

	// Track the current consumer so the close func can close it.
	// Protected by consMu because the reconnect goroutine writes it
	// (setConsumer) and the main goroutine reads it (closeConsumer)
	// concurrently during the shutdown window (Bug #5).
	var (
		currentCons *consumer.Consumer
		consMu      sync.Mutex
	)
	setConsumer := func(c *consumer.Consumer) {
		consMu.Lock()
		currentCons = c
		consMu.Unlock()
	}

	go func() {
		// NOTE: outCh is intentionally NOT closed on exit. Workers
		// exit via <-ctx.Done(), not via channel close. Keeping outCh
		// open allows workers to drain any buffered messages during
		// the shutdown window. The channel is GC'd after all
		// goroutines exit.
		backoff := 1 * time.Second

		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			cons, err := consumer.NewConsumer(
				cfg.RabbitMQ.URL,
				cfg.RabbitMQ.Queues.MatchIDs,
				cfg.RabbitMQ.Queues.MatchIDsDLQ,
				cfg.Worker.Prefetch,
			)
			if err != nil {
				logger.Log.Error("Consumer init failed, retrying",
					zap.Error(err),
					zap.Duration("backoff", backoff))
				time.Sleep(backoff)
				backoff = min(backoff*2, 30*time.Second)
				continue
			}

			msgs, err := cons.Consume(cfg.RabbitMQ.Queues.MatchIDs)
			if err != nil {
				logger.Log.Error("Failed to start consuming, retrying",
					zap.Error(err))
				cons.Close()
				time.Sleep(backoff)
				backoff = min(backoff*2, 30*time.Second)
				continue
			}

			logger.Log.Info("Consumer connected successfully")
			backoff = 1 * time.Second // reset on success
			setConsumer(cons)         // save for deferred close by caller

			// Forward messages until channel closes.
			// NOTE: we do NOT close outCh when this loop exits because
			// ctx.Done() is the worker exit signal. Closing outCh early
			// would drop buffered messages. The channel is GC'd after
			// all goroutines exit during shutdown.
		forward:
			for d := range msgs {
				select {
				case outCh <- d:
				case <-ctx.Done():
					break forward
				}
			}

			// On the shutdown path (ctx cancelled), return without
			// closing the consumer. The caller's closeConsumer() func
			// will close it after all workers have finished their
			// in-flight Ack/Nack operations.
			if ctx.Err() != nil {
				logger.Log.Debug("Consumer disconnected during shutdown")
				return
			}

			// Normal reconnect (consumer channel died, not shutdown):
			// close the old consumer and retry.
			cons.Close()
			// Nil currentCons so closeConsumer() cannot double-close
			// this consumer later (Bug #6).
			consMu.Lock()
			currentCons = nil
			consMu.Unlock()
			logger.Log.Warn("Consumer channel closed, reconnecting...")
			time.Sleep(backoff)
			backoff = min(backoff*2, 30*time.Second)
		}
	}()

	closeConsumer := func() {
		consMu.Lock()
		c := currentCons
		currentCons = nil
		consMu.Unlock()

		if c != nil {
			c.Close()
		}
	}
	return outCh, closeConsumer
}
