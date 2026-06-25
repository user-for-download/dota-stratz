package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/dota-stratz/services/id-fetcher/internal/api"
	"github.com/dota-stratz/services/id-fetcher/internal/config"
	"github.com/dota-stratz/services/id-fetcher/internal/queue"

	"github.com/dota-stratz/shared/go-common/cache"
	"github.com/dota-stratz/shared/go-common/checkpoint"
	"github.com/dota-stratz/shared/go-common/db"
	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/dota-stratz/shared/go-common/proxypool"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/robfig/cron/v3"
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

	// 1. Connect to RabbitMQ (only needs the match_ids publish queue now —
	//    no trigger queue, no coordinator).
	mqPub, err := queue.NewPublisher(cfg.RabbitMQ.URL, cfg.RabbitMQ.Queues.MatchIDs)
	if err != nil {
		logger.Log.Fatal("Failed to connect to RabbitMQ", zap.Error(err))
	}
	defer mqPub.Close()

	// 2. Connect to Redis & initialise Proxy Pool.
	rdb, err := cache.Connect(cfg.Redis.Addr, cfg.Redis.Password, cfg.Redis.DB)
	if err != nil {
		logger.Log.Fatal("Failed to connect to Redis", zap.Error(err))
	}
	defer rdb.Close()

	// Register the Redis-sourced proxy pool collector BEFORE the metrics
	// server starts so /metrics includes dota2_proxy_pool_available and
	// dota2_proxy_pool_leased on the first scrape. P0-4: replaces the
	// per-process promauto.NewGauge which diverged across services
	// (e.g. id-fetcher reported 0 because it does balanced inc/dec per
	// cron run, while Redis ground truth was 129).
	prometheus.MustRegister(
		proxypool.NewRedisPoolCollector(rdb, "dota2:proxies", "dota2:proxies:leases"),
	)

	pool, err := proxypool.New(rdb, proxypool.Config{
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

	// 3. Build client and fetcher.
	odClient := api.NewOpenDotaClient(
		cfg.OpenDota.APIURL,
		pool,
		cfg.OpenDota.FetchLastCountDay,
		cfg.OpenDota.FetchLobbyTypes,
	)
	fetcher := api.NewFetcher(odClient, mqPub, cfg.RabbitMQ.Queues.MatchIDs, cfg.OpenDota.BatchSize, rdb)

	// 3b. Postgres bootstrap — keep a persistent pool for the full
	// lifecycle so the fetcher can query existing match IDs (Layer 3
	// filter) on every Run() call.
	var dbPool *pgxpool.Pool
	if cfg.Postgres.DSN != "" {
		var err error
		dbPool, err = db.Connect(ctx, cfg.Postgres.DSN, 4) // small pool: 4 conns
		if err != nil {
			logger.Log.Warn("Postgres connection failed, watermark + DB filter disabled",
				zap.String("pipeline", checkpoint.CheckpointPipelineIDFetcher),
				zap.Error(err))
		}
	}
	if dbPool != nil {
		defer dbPool.Close()

		// Watermark bootstrap — read the parser's last_parsed_match_id
		// from ingestion_checkpoints. If the row exists and is > 0, switch
		// the fetcher into watermark mode so we don't re-fetch matches the
		// parser has already committed.
		if err := bootstrapCheckpoint(ctx, cfg, fetcher, dbPool); err != nil {
			logger.Log.Warn("Checkpoint bootstrap failed, using rolling-window path",
				zap.String("pipeline", checkpoint.CheckpointPipelineIDFetcher),
				zap.Error(err))
		}

		// Layer 3 filter: skip match IDs already committed to the matches
		// table. Controlled by FORCE_DOWNLOAD_REWRITE env var.
		ec := api.NewPostgresExistenceChecker(dbPool)
		fetcher.SetExistenceChecker(ec, cfg.Postgres.ForceDownloadRewrite)
		logger.Log.Info("DB existence filter configured",
			zap.Bool("force_download_rewrite", cfg.Postgres.ForceDownloadRewrite))
	} else {
		logger.Log.Warn("No Postgres connection — Layer 3 DB filter disabled")
	}

	// 4. Metrics server.
	metricsMux := http.NewServeMux()
	metricsMux.Handle("/metrics", promhttp.Handler())
	metricsMux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	metricsServer := &http.Server{
		Addr:              fmt.Sprintf(":%d", cfg.Worker.MetricsPort),
		Handler:           metricsMux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		logger.Log.Info("Starting metrics server", zap.Int("port", cfg.Worker.MetricsPort))
		if err := metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Log.Error("Metrics server failed", zap.Error(err))
		}
	}()
	defer func() {
		shutdownCtx, c := context.WithTimeout(context.Background(), 5*time.Second)
		defer c()
		_ = metricsServer.Shutdown(shutdownCtx)
	}()

	// 5. Cron scheduler — runs the fetch job on the configured schedule.
	//    A mutex flag prevents a slow run from overlapping the next tick
	//    (e.g. if the proxy pool is exhausted and retries take >24h).
	running := make(chan struct{}, 1) // capacity-1 = non-blocking trylock
	running <- struct{}{}             // pre-fill: first tick is allowed immediately

	scheduler := cron.New()
	_, err = scheduler.AddFunc(cfg.OpenDota.FetchSchedule, func() {
		select {
		case token := <-running:
			defer func() { running <- token }() // release when done
		default:
			logger.Log.Warn("Fetch already in progress, skipping this tick")
			return
		}

		logger.Log.Info("Cron: starting scheduled fetch run")
		if err := fetcher.Run(ctx); err != nil {
			if ctx.Err() != nil {
				return // shutdown in progress, not a real error
			}
			logger.Log.Error("Scheduled fetch run failed", zap.Error(err))
		}
	})
	if err != nil {
		logger.Log.Fatal("Failed to register cron job", zap.Error(err))
	}

	scheduler.Start()
	logger.Log.Info("ID Fetcher started",
		zap.Int("fetch_last_n_days", cfg.OpenDota.FetchLastCountDay),
		zap.Ints("lobby_types", cfg.OpenDota.FetchLobbyTypes),
		zap.String("schedule", cfg.OpenDota.FetchSchedule),
		zap.Bool("start_run", cfg.OpenDota.StartRun),
		zap.Int("start_run_min_pool_size", cfg.OpenDota.StartRunMinPoolSize),
		zap.Int64("watermark", fetcher.Watermark()),
		zap.Bool("watermark_path", fetcher.Watermark() > 0),
		zap.Int("watermark_lookback_days", cfg.OpenDota.WatermarkLookbackDays))

	// 5b. Opt-in startup fetch — wait for the proxy pool to reach the
	// configured minimum, then run fetcher.Run once. Shares the same
	// trylock (`running`) with the cron job so the startup fetch cannot
	// overlap the first scheduled tick. If the pool does not fill within
	// StartRunMaxWait, the startup fetch is skipped and the regular
	// schedule takes over.
	if cfg.OpenDota.StartRun {
		go runStartupFetch(ctx, fetcher, pool, running, cfg.OpenDota.StartRunMinPoolSize, cfg.OpenDota.StartRunMaxWait)
	}

	// 6. Graceful shutdown.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Log.Info("Shutting down ID Fetcher...")
	cancel() // signal any in-flight fetch to stop

	// Wait for the cron job to finish, bounded so a hung fetch (e.g. an
	// infinite proxy retry loop with no proxies available) cannot block
	// shutdown forever. cron.Stop() returns a context that is Done once
	// running jobs exit, but provides no upper bound.
	stopCtx, stopCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer stopCancel()
	select {
	case <-scheduler.Stop().Done():
	case <-stopCtx.Done():
		logger.Log.Warn("Scheduler stop timed out after 30s, forcing exit")
	}

	logger.Log.Info("ID Fetcher shutdown complete")
}

// runStartupFetch polls the proxy pool until it reaches minPoolSize, then
// triggers a one-shot fetch through the same `running` trylock the cron job
// uses (so it cannot overlap the first scheduled tick). If the pool does not
// reach the threshold within maxWait, the startup fetch is skipped — the
// regular cron schedule still runs.
//
// Honours ctx.Done() in every wait so SIGTERM during the pool-wait phase
// does not block shutdown.
func runStartupFetch(
	ctx context.Context,
	fetcher *api.Fetcher,
	pool *proxypool.Pool,
	running chan struct{},
	minPoolSize int,
	maxWait time.Duration,
) {
	logger.Log.Info("Startup fetch: waiting for proxy pool to fill",
		zap.Int("min_pool_size", minPoolSize),
		zap.Duration("max_wait", maxWait))

	if !waitForPool(ctx, pool, minPoolSize, maxWait) {
		return
	}

	// Reuse the cron job's trylock. If the cron job is mid-run when we
	// wake up (unlikely — we'd be racing the very first tick), skip and
	// let the cron tick take over.
	select {
	case token := <-running:
		defer func() { running <- token }()
	default:
		logger.Log.Warn("Startup fetch: trylock busy (cron already running), skipping")
		return
	}

	if ctx.Err() != nil {
		return
	}
	logger.Log.Info("Startup fetch: starting")
	if err := fetcher.Run(ctx); err != nil {
		if ctx.Err() != nil {
			return
		}
		logger.Log.Error("Startup fetch: run failed", zap.Error(err))
		return
	}
	logger.Log.Info("Startup fetch: complete")
}

// waitForPool polls Redis until the proxy pool has at least minPoolSize
// proxies or maxWait elapses. Returns true if the pool is ready, false if
// the deadline expired or ctx was cancelled.
//
// Extracted as a separate function to avoid goto (golang-pro best practice).
func waitForPool(ctx context.Context, pool *proxypool.Pool, minPoolSize int, maxWait time.Duration) bool {
	const pollInterval = 5 * time.Second
	deadline := time.Now().Add(maxWait)

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			logger.Log.Info("Startup fetch: cancelled during pool wait (shutdown)")
			return false
		case <-ticker.C:
			if time.Now().After(deadline) {
				logger.Log.Warn("Startup fetch: pool did not reach minimum within max_wait, skipping",
					zap.Int("min_pool_size", minPoolSize),
					zap.Duration("max_wait", maxWait))
				return false
			}
			avail, err := pool.Available(ctx)
			if err != nil {
				logger.Log.Debug("Startup fetch: pool size check failed, retrying",
					zap.Error(err))
				continue
			}
			if avail >= int64(minPoolSize) {
				logger.Log.Info("Startup fetch: pool reached minimum, acquiring trylock",
					zap.Int64("available", avail),
					zap.Int("min_pool_size", minPoolSize))
				return true
			}
			logger.Log.Debug("Startup fetch: pool not yet ready, waiting",
				zap.Int64("available", avail),
				zap.Int("min_pool_size", minPoolSize))
		}
	}
}

// bootstrapCheckpoint reads ingestion_checkpoints.last_parsed_match_id
// from Postgres and configures the fetcher's watermark if the value is
// > 0.
//
// The caller provides a persistent dbPool that stays alive for the
// entire id-fetcher lifecycle so the Layer 3 existence filter can query
// the matches table on every Run() call.
//
// Returns nil on the happy paths (watermark applied OR row missing →
// rolling window) and a non-nil error only when something genuinely
// went wrong that the caller should log as a warning. The id-fetcher
// continues in rolling-window mode on any non-nil error so a transient
// DB outage at startup does not block fetches — a service restart is
// required to re-read the checkpoint, which is acceptable for a
// daily-schedule pipeline.
func bootstrapCheckpoint(ctx context.Context, cfg *config.Config, fetcher *api.Fetcher, pool *pgxpool.Pool) error {
	// Use a short timeout so a slow or unreachable DB does not block
	// service startup. 5s is plenty for a single indexed PK lookup.
	bootCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	watermark, ok, err := checkpoint.ReadWatermark(bootCtx, pool)
	if err != nil {
		return fmt.Errorf("read watermark: %w", err)
	}
	if !ok {
		logger.Log.Info("Checkpoint bootstrap: row missing, using rolling window",
			zap.String("pipeline", checkpoint.CheckpointPipelineIDFetcher))
		return nil
	}

	fetcher.SetWatermark(watermark, cfg.OpenDota.WatermarkLookbackDays)
	logger.Log.Info("Checkpoint bootstrap: watermark applied",
		zap.String("pipeline", checkpoint.CheckpointPipelineIDFetcher),
		zap.Int64("last_parsed_match_id", watermark),
		zap.Int("watermark_lookback_days", cfg.OpenDota.WatermarkLookbackDays),
		zap.Int("batch_size", cfg.OpenDota.BatchSize))
	return nil
}
