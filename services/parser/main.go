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

	"github.com/dota-stratz/services/parser/internal/config"
	"github.com/dota-stratz/services/parser/internal/consumer"
	"github.com/dota-stratz/services/parser/internal/repository"
	"github.com/dota-stratz/services/parser/internal/worker"
	"github.com/dota-stratz/shared/go-common/db"
	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/prometheus/client_golang/prometheus/promhttp"
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

	// 1. Postgres Connection
	pool, err := db.Connect(ctx, cfg.Postgres.DSN)
	if err != nil {
		logger.Log.Fatal("Postgres connection failed", zap.Error(err))
	}
	defer pool.Close()
	logger.Log.Info("Connected to Postgres")

	repo := repository.NewRepository(pool)

	// 2. RabbitMQ Consumer (with auto-reconnection via ConsumeWithReconnect).
	//    NOTE: The Consumer struct is created for DLQ-channel management only;
	//    the actual message stream is obtained through ConsumeWithReconnect
	//    which survives broker restarts by recreating the consumer under the
	//    hood.
	cons, err := consumer.NewConsumer(
		cfg.RabbitMQ.URL,
		cfg.RabbitMQ.Queues.RawMatches,
		cfg.RabbitMQ.Queues.RawMatchesDLQ,
		cfg.Worker.Prefetch,
	)
	if err != nil {
		logger.Log.Fatal("RabbitMQ consumer init failed", zap.Error(err))
	}
	defer cons.Close()
	logger.Log.Info("Connected to RabbitMQ, consuming from",
		zap.String("queue", cfg.RabbitMQ.Queues.RawMatches))

	// ConsumeWithReconnect wraps the raw Consume call with automatic
	// reconnection. If the broker restarts or the channel dies, it
	// reconnects with exponential backoff (1s → 30s max) and resumes
	// delivery on the returned channel. The channel is never closed on
	// reconnect — only on permanent shutdown via the done channel.
	done := make(chan struct{})
	defer close(done)
	msgs := cons.ConsumeWithReconnect(done)

	// 3. Metrics & Health Server
	go startMetricsServer(ctx, cfg.Worker.MetricsPort, repo)

	// 4. Processor — batch-unmarshals match JSON, validates, and writes to
	//     Postgres. Invalid/poison messages are Nack'd without requeue and
	//     routed to the DLQ via the queue's x-dead-letter-exchange binding.
	processor := worker.NewProcessor(
		repo,
		msgs,
		cfg.Worker.BatchSize,
		cfg.FetchTimeout(),
	)

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	logger.Log.Info("Parser started",
		zap.Int("batch_size", cfg.Worker.BatchSize),
		zap.Int("prefetch", cfg.Worker.Prefetch),
		zap.Duration("fetch_timeout", cfg.FetchTimeout()),
	)

	var processorWG sync.WaitGroup
	processorWG.Add(1)
	go func() {
		defer processorWG.Done()
		processor.Run(ctx)
	}()

	<-quit
	logger.Log.Info("Shutting down Parser...")
	cancel()

	// Wait for the processor to drain its current batch. The batch
	// write uses context.WithoutCancel so it won't be aborted by
	// cancel() — we just need to give it time to finish committing.
	// 30s is enough for the largest reasonable batch (batch_size=100
	// matches × ~30ms each ≈ 3s) plus headroom for an active write.
	drained := make(chan struct{})
	go func() {
		processorWG.Wait()
		close(drained)
	}()
	select {
	case <-drained:
		logger.Log.Info("Parser shutdown complete (processor drained)")
	case <-time.After(30 * time.Second):
		logger.Log.Warn("Processor did not drain in 30s, exiting anyway")
	}
}

func startMetricsServer(ctx context.Context, port int, repo *repository.Repository) {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())

	// Enhanced health check: pings Postgres on every probe.
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if err := repo.Ping(r.Context()); err != nil {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte("db_down"))
			return
		}
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
