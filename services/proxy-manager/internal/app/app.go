// Package app wires together the proxy-manager service: config loading,
// Redis connection, proxy pool initialization, bootstrap from file + remote
// source, and background loops (refresh, lease reaper, metrics server).
package app

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/dota-stratz/services/proxy-manager/internal/config"
	"github.com/dota-stratz/services/proxy-manager/internal/source"
	"github.com/dota-stratz/services/proxy-manager/internal/validator"
	"github.com/dota-stratz/shared/go-common/cache"
	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/dota-stratz/shared/go-common/proxypool"
	"github.com/joho/godotenv"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
)

// sourceFetchLimiter prevents redundant calls to the remote proxy API within
// a configured cooldown window. This avoids getting instantly HTTP 429
// rate-limited on boot when bootstrap + topUpIfBelowMin both try to fetch
// from the same remote source in rapid succession (Issue #27).
//
// BUG-015: moved from package-level vars (which caused test pollution) to
// an explicitly constructed struct owned by Run().
type sourceFetchLimiter struct {
	mu        sync.Mutex
	lastFetch time.Time
	cooldown  time.Duration
}

func newSourceFetchLimiter(cooldown time.Duration) *sourceFetchLimiter {
	return &sourceFetchLimiter{cooldown: cooldown}
}

// Run starts the proxy-manager service and blocks until SIGINT/SIGTERM.
func Run() {
	logger.InitLogger()
	defer logger.Sync()

	if err := godotenv.Load("deploy/.env"); err != nil {
		logger.Log.Warn("No .env file found, relying on process environment", zap.Error(err))
	}

	cfg, err := config.Load()
	if err != nil {
		logger.Log.Fatal("Config load failed", zap.Error(err))
	}

	logger.Log.Debug("Starting proxy-manager",
		zap.String("proxy_file", cfg.ProxyFilePath),
		zap.String("redis_addr", cfg.RedisAddr),
		zap.String("validation_target", cfg.ValidationTargetURL))

	// Create context and signal listener before any operations that may
	// need cancellation (such as Redis connection backoff, pool
	// detection/health checks).
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-quit
		logger.Log.Info("Shutdown signal received")
		cancel()
	}()

	rdb, err := cache.Connect(ctx, cfg.RedisAddr, cfg.RedisPassword, int(cfg.RedisDB))
	if err != nil {
		logger.Log.Fatal("Redis connection failed", zap.Error(err))
	}
	defer rdb.Close()

	// Register the Redis-sourced proxy pool collector BEFORE the metrics
	// server starts so the first Prometheus scrape (typically within
	// 15s of startup) sees a registered collector rather than no metric.
	// P0-4: replaces the old per-process promauto.NewGauge which
	// diverged across services (e.g. -322, 451, 0 vs Redis ground
	// truth 129). See shared/go-common/proxypool/redis_collector.go.
	prometheus.MustRegister(
		proxypool.NewRedisPoolCollector(rdb, "dota2:proxies", "dota2:proxies:leases"),
	)

	proxyPool, err := proxypool.New(rdb, proxypool.Config{
		Strategy:          cfg.RotationStrategy,
		SoftFailThreshold: int(cfg.SoftFailThreshold),
		CooldownDuration:  time.Duration(cfg.CooldownMin) * time.Minute,
		LeaseDuration:     time.Duration(cfg.LeaseDurationSec) * time.Second,
		SoftRetryDelay:    time.Duration(cfg.SoftRetryDelaySec) * time.Second,
		FailureCounterTTL: time.Duration(cfg.FailureCounterTTLMin) * time.Minute,
	})
	if err != nil {
		logger.Log.Fatal("Pool init failed", zap.Error(err))
	}

	// Pool/lease gauges are now sourced from Redis at scrape time by
	// the registered RedisPoolCollector — no startup initialisation
	// needed (and no InitGauges method to call). The first Prometheus
	// scrape will read the live ZCARD/HLEN.

	val, err := validator.New(cfg.ValidationTargetURL, cfg.ValidationUserAgent, int(cfg.ValidationTimeoutSec), int(cfg.Concurrency))
	if err != nil {
		logger.Log.Fatal("Validator init failed", zap.Error(err))
	}

	var wg sync.WaitGroup

	// Start metrics server FIRST so /healthz and /metrics are immediately scrapeable
	wg.Add(1)
	go startMetricsServer(ctx, &wg, int(cfg.MetricsPort), proxyPool, cfg.PoolMinSize)

	// BUG-015: create a dedicated limiter so its state is owned by Run()
	// rather than package-level vars that cause test pollution.
	sourceFetchLimiter := newSourceFetchLimiter(
		time.Duration(cfg.SourceFetchCooldownMin) * time.Minute,
	)

	// Bootstrap: load from local file + remote GET source, validate together,
	// then top-up from the same combined list before considering a re-fetch.
	// This ensures the pool starts populated regardless of which source is healthy.
	if ctx.Err() != nil {
		logger.Log.Info("Shutdown signal before bootstrap, exiting cleanly")
	} else {
		bootstrap(ctx, cfg, val, proxyPool, sourceFetchLimiter)
	}

	// Background loops (skip if already cancelled during bootstrap)
	if ctx.Err() != nil {
		logger.Log.Info("Shutdown during bootstrap, skipping background loops")
	} else {
		wg.Add(1)
		go refreshLoop(ctx, &wg, cfg, val, proxyPool, sourceFetchLimiter)
		wg.Add(1)
		go leaseReaperLoop(ctx, &wg, cfg, proxyPool)
		wg.Add(1)
		go revalidationLoop(ctx, &wg, cfg, val, proxyPool)
	}

	// Block until context is cancelled (signal received during or after bootstrap)
	<-ctx.Done()
	logger.Log.Info("Shutdown: draining workers...")

	// Wait for loops to exit, bounded by ShutdownGraceMs
	if err := waitWithTimeout(&wg, time.Duration(cfg.ShutdownGraceMs)*time.Millisecond); err != nil {
		logger.Log.Warn("Shutdown grace period exceeded, forcing exit",
			zap.Duration("grace", time.Duration(cfg.ShutdownGraceMs)*time.Millisecond))
	} else {
		logger.Log.Info("All workers drained cleanly")
	}
	logger.Log.Info("Shutdown complete")
}

// limitedFetchWithRetry wraps fetchWithRetry with a cooldown guard: if a
// source fetch succeeded within the limiter's cooldown window, subsequent
// calls are skipped and the last result is returned as empty. This prevents
// redundant API calls that would trigger HTTP 429 rate-limiting.
//
// NOTE: lastFetch is updated ONLY on success so that a transient network
// failure does not silence the source for the entire cooldown window
// (Issue #27 refinement).
func limitedFetchWithRetry(ctx context.Context, cfg *config.Config, limiter *sourceFetchLimiter) ([]string, error) {
	limiter.mu.Lock()
	elapsed := time.Since(limiter.lastFetch)
	if !limiter.lastFetch.IsZero() && elapsed < limiter.cooldown {
		limiter.mu.Unlock()
		logger.Log.Debug("Source fetch rate-limited (cooldown active)",
			zap.Duration("elapsed", elapsed),
			zap.Duration("cooldown", limiter.cooldown))
		return nil, nil
	}
	limiter.mu.Unlock()

	proxies, err := fetchWithRetry(ctx, cfg)
	if err == nil {
		limiter.mu.Lock()
		limiter.lastFetch = time.Now()
		limiter.mu.Unlock()
	}
	return proxies, err
}

// waitWithTimeout blocks until wg.Wait() returns or the timeout elapses.
// Returns nil on clean drain, context.DeadlineExceeded on timeout.
//
// NOTE: When the timeout fires before wg.Wait() completes, the helper
// goroutine that calls wg.Wait() is abandoned (it continues blocking on
// wg.Wait() until all tracked goroutines finish). This is a bounded
// goroutine leak — the goroutine is guaranteed to eventually complete
// when the remaining workers finish, so it does not grow unboundedly.
func waitWithTimeout(wg *sync.WaitGroup, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return context.DeadlineExceeded
	}
}

// startMetricsServer runs the HTTP server for /metrics, /healthz and
// /debug/pool. Exits when ctx is cancelled.
func startMetricsServer(
	ctx context.Context,
	wg *sync.WaitGroup,
	port int,
	pool *proxypool.Pool,
	poolMinSize int64,
) {
	defer wg.Done()

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		avail, err := pool.Available(r.Context())
		if err != nil || avail < poolMinSize {
			http.Error(w, "pool not ready", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	mux.HandleFunc("/debug/pool", func(w http.ResponseWriter, r *http.Request) {
		avail, _ := pool.Available(r.Context())
		inUse, _ := pool.InUse(r.Context())
		_, _ = fmt.Fprintf(w, "available=%d in_use=%d\n", avail, inUse)
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

// refreshLoop periodically fetches new proxies from the remote source
// (PROXY_REFRESH_SOURCE_URL), validates them, and records the validated
// set in Redis via the pool. Runs once at startup, then every
// PROXY_REFRESH_TIME minutes until ctx is cancelled.
func refreshLoop(
	ctx context.Context,
	wg *sync.WaitGroup,
	cfg *config.Config,
	val *validator.Validator,
	proxyPool *proxypool.Pool,
	limiter *sourceFetchLimiter,
) {
	defer wg.Done()

	if cfg.RefreshSourceURL == "" {
		logger.Log.Debug("Refresh loop disabled (no source URL configured)")
		return
	}
	ticker := time.NewTicker(time.Duration(cfg.RefreshIntervalMin) * time.Minute)
	defer ticker.Stop()

	logger.Log.Debug("Refresh loop started",
		zap.String("source_url", cfg.RefreshSourceURL),
		zap.Int64("interval_min", cfg.RefreshIntervalMin))

	// Run once at startup so the pool can be re-seeded without waiting
	// a full interval (e.g. when bootstrap was empty).
	runRefresh(ctx, cfg, val, proxyPool, limiter)
	for {
		select {
		case <-ctx.Done():
			logger.Log.Debug("Refresh loop stopped")
			return
		case <-ticker.C:
			runRefresh(ctx, cfg, val, proxyPool, limiter)
		}
	}
}

// leaseReaperLoop periodically reclaims expired proxy leases.
func leaseReaperLoop(
	ctx context.Context,
	wg *sync.WaitGroup,
	cfg *config.Config,
	proxyPool *proxypool.Pool,
) {
	defer wg.Done()

	ticker := time.NewTicker(time.Duration(cfg.LeaseReaperIntervalSec) * time.Second)
	defer ticker.Stop()
	logger.Log.Debug("Lease reaper loop started",
		zap.Int64("interval_sec", cfg.LeaseReaperIntervalSec))
	for {
		select {
		case <-ctx.Done():
			logger.Log.Debug("Lease reaper loop stopped")
			return
		case <-ticker.C:
			if reaped, err := proxyPool.ReapExpiredLeases(ctx); err == nil && reaped > 0 {
				logger.Log.Info("Lease reaper recovered abandoned proxies", zap.Int("count", reaped))
			}
		}
	}
}

// revalidationLoop periodically validates ALL proxies in the Redis pool
// and removes dead ones. This prevents stale proxies from accumulating
// and consuming pool slots.
func revalidationLoop(
	ctx context.Context,
	wg *sync.WaitGroup,
	cfg *config.Config,
	val *validator.Validator,
	proxyPool *proxypool.Pool,
) {
	defer wg.Done()

	if cfg.RevalidationIntervalMin <= 0 {
		logger.Log.Debug("Revalidation loop disabled (interval <= 0)")
		return
	}

	ticker := time.NewTicker(time.Duration(cfg.RevalidationIntervalMin) * time.Minute)
	defer ticker.Stop()
	logger.Log.Info("Revalidation loop started",
		zap.Int64("interval_min", cfg.RevalidationIntervalMin))

	for {
		select {
		case <-ctx.Done():
			logger.Log.Debug("Revalidation loop stopped")
			return
		case <-ticker.C:
			runRevalidation(ctx, val, proxyPool)
		}
	}
}

// runRevalidation fetches all proxies from Redis, validates them concurrently,
// and removes dead ones. This keeps the pool healthy by evicting proxies that
// have become unreachable since they were last validated.
func runRevalidation(ctx context.Context, val *validator.Validator, proxyPool *proxypool.Pool) {
	// Collect all proxies: available (ZSET) + leased (HASH).
	available, err := proxyPool.Members(ctx)
	if err != nil {
		logger.Log.Error("Revalidation: failed to fetch available proxies", zap.Error(err))
		proxypool.ProxyRevalidationRunsTotal.WithLabelValues("error").Inc()
		return
	}

	leased, err := proxyPool.LeasedMembers(ctx)
	if err != nil {
		logger.Log.Warn("Revalidation: failed to fetch leased proxies, validating available only", zap.Error(err))
		leased = nil
	}

	all := make([]string, 0, len(available)+len(leased))
	all = append(all, available...)
	all = append(all, leased...)

	if len(all) == 0 {
		logger.Log.Debug("Revalidation: no proxies in pool")
		proxypool.ProxyRevalidationRunsTotal.WithLabelValues("success").Inc()
		return
	}

	logger.Log.Info("Revalidation: starting full pool validation",
		zap.Int("total", len(all)),
		zap.Int("available", len(available)),
		zap.Int("leased", len(leased)))

	// Track which proxies passed validation.
	seen := make(map[string]struct{}, len(all))

	stats := val.ValidateStream(ctx, all, func(_ context.Context, r validator.Result) {
		if r.OK {
			seen[r.Proxy] = struct{}{}
		}
	})

	// Remove dead proxies (ones not in seen).
	removed := 0
	for _, p := range all {
		if _, ok := seen[p]; !ok {
			// Report as hard failure to remove from pool.
			if err := proxyPool.Report(ctx, p, proxypool.ReasonHardFailure); err != nil {
				logger.Log.Warn("Revalidation: failed to remove dead proxy",
					zap.String("proxy", p), zap.Error(err))
			} else {
				removed++
			}
		}
	}

	proxypool.ProxyRevalidationRunsTotal.WithLabelValues("success").Inc()
	proxypool.ProxyRevalidationRemovedTotal.Add(float64(removed))

	logger.Log.Info("Revalidation: complete",
		zap.Int("validated", stats.Total),
		zap.Int("alive", stats.OK),
		zap.Int("dead", stats.Failed),
		zap.Int("removed", removed),
		zap.Duration("elapsed", stats.Elapsed))
}

// bootstrap loads proxies from all available sources at startup (local file
// AND remote GET URL), deduplicates them, and runs a single combined
// validation pass. Either source failing is non-fatal — we proceed with
// whatever we got. If both fail, the pool starts empty and the refresh loop
// will populate it on the next tick.
func bootstrap(
	ctx context.Context,
	cfg *config.Config,
	val *validator.Validator,
	proxyPool *proxypool.Pool,
	limiter *sourceFetchLimiter,
) {
	if ctx.Err() != nil {
		return
	}

	combined := make([]string, 0, 1024)
	seen := make(map[string]struct{}, 1024)

	// --- Source 1: local file ---
	if fileProxies, err := source.FromFile(cfg.ProxyFilePath); err != nil {
		logger.Log.Warn("Bootstrap: local proxy file unavailable",
			zap.String("path", cfg.ProxyFilePath), zap.Error(err))
	} else {
		for _, p := range fileProxies {
			if _, dup := seen[p]; !dup {
				seen[p] = struct{}{}
				combined = append(combined, p)
			}
		}
		logger.Log.Info("Bootstrap: loaded from file",
			zap.String("path", cfg.ProxyFilePath),
			zap.Int("count", len(fileProxies)))
	}

	// --- Source 2: remote GET ---
	// Use limitedFetchWithRetry (not raw fetchWithRetry) so that
	// lastSourceFetch is updated and the subsequent topUpIfBelowMin
	// call respects the cooldown — otherwise the remote API receives
	// two back-to-back requests milliseconds apart, guaranteeing an
	// instant HTTP 429 ban (issue #27).
	if cfg.RefreshSourceURL == "" {
		logger.Log.Debug("Bootstrap: remote source URL not configured, skipping")
	} else if urlProxies, err := limitedFetchWithRetry(ctx, cfg, limiter); err != nil {
		logger.Log.Warn("Bootstrap: remote source fetch failed", zap.Error(err))
	} else {
		added := 0
		for _, p := range urlProxies {
			if _, dup := seen[p]; !dup {
				seen[p] = struct{}{}
				combined = append(combined, p)
				added++
			}
		}
		logger.Log.Info("Bootstrap: loaded from remote source",
			zap.String("url", cfg.RefreshSourceURL),
			zap.Int("fetched", len(urlProxies)),
			zap.Int("new_after_dedup", added))
	}

	if len(combined) == 0 {
		logger.Log.Warn("Bootstrap: no proxies obtained from any source")
		return
	}

	logger.Log.Info("Bootstrap: starting combined validation",
		zap.Int("total_candidates", len(combined)))
	_ = runValidation(ctx, val, proxyPool, combined, cfg.PoolMaxSize)

	proxypool.ProxyTopUpRunsTotal.WithLabelValues("bootstrap").Inc()

	// Top-up: runValidation already added validated proxies to the pool,
	// so just check if we still need more from the remote source.
	topUpIfBelowMin(ctx, cfg, val, proxyPool, limiter)
}

// runRefresh fetches, filters, validates and tops-up the pool. The
// validated set is recorded in Redis (proxypool) for downstream services
// to consume. Records outcome in ProxyRefreshRunsTotal.
func runRefresh(ctx context.Context, cfg *config.Config, val *validator.Validator, proxyPool *proxypool.Pool, limiter *sourceFetchLimiter) {
	if cfg.RefreshSourceURL == "" {
		logger.Log.Debug("Refresh: no source URL configured, skipping")
		return
	}

	logger.Log.Debug("Refresh: fetching new proxies from source",
		zap.String("url", cfg.RefreshSourceURL))

	proxies, err := limitedFetchWithRetry(ctx, cfg, limiter)
	if err != nil {
		logger.Log.Error("Refresh fetch exhausted retries", zap.Error(err))
		proxypool.ProxyRefreshRunsTotal.WithLabelValues("fetch_failed").Inc()
		return
	}

	fresh := filterNew(ctx, proxyPool, proxies)
	logger.Log.Debug("Refresh: filtering complete",
		zap.Int("candidates", len(proxies)),
		zap.Int("fresh", len(fresh)))
	if len(fresh) > 0 {
		_ = runValidation(ctx, val, proxyPool, fresh, cfg.PoolMaxSize)
	} else {
		logger.Log.Debug("Refresh: no new proxies to validate")
	}

	// Top-up: runValidation already added validated proxies to the pool,
	// so just check if we still need more from the remote source.
	topUpIfBelowMin(ctx, cfg, val, proxyPool, limiter)

	proxypool.ProxyRefreshRunsTotal.WithLabelValues("success").Inc()
}

// fetchWithRetry fetches proxies from the configured source URL with
// exponential backoff (1s, 2s, 4s). Returns the parsed proxy list or the
// last error after exhausting retries.
func fetchWithRetry(ctx context.Context, cfg *config.Config) ([]string, error) {
	const maxRetries = 3
	var lastErr error
	for attempt := range maxRetries {
		proxies, err := source.FromURL(ctx, cfg.RefreshSourceURL,
			time.Duration(cfg.SourceFetchTimeoutSec)*time.Second, cfg.SourceUserAgent)
		if err == nil {
			return proxies, nil
		}
		lastErr = err
		logger.Log.Warn("Source fetch attempt failed",
			zap.Int("attempt", attempt+1),
			zap.Int("max", maxRetries),
			zap.Error(err))
		if attempt < maxRetries-1 {
			backoff := time.Duration(1<<attempt) * time.Second
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
	}
	return nil, lastErr
}

// filterNew returns only those proxies not already present in the pool
// (either in the ZSET or currently leased).
func filterNew(ctx context.Context, proxyPool *proxypool.Pool, proxies []string) []string {
	existingZSET, err := proxyPool.Members(ctx)
	if err != nil {
		logger.Log.Warn("filterNew: batch fetch of ZSET failed, falling back to individual checks",
			zap.Error(err))
		return filterNewIndividual(ctx, proxyPool, proxies)
	}

	// Also fetch leased members — leased proxies are removed from the ZSET
	// (via ZPOPMIN) and placed in the lease HASH. Without this check they
	// would appear as "new" and be re-added, causing state corruption.
	existingLeased, err := proxyPool.LeasedMembers(ctx)
	if err != nil {
		logger.Log.Warn("filterNew: batch fetch of leases failed, including ZSET only",
			zap.Error(err))
		existingLeased = nil
	}

	totalExisting := len(existingZSET) + len(existingLeased)
	poolSet := make(map[string]struct{}, totalExisting)
	for _, p := range existingZSET {
		poolSet[p] = struct{}{}
	}
	for _, p := range existingLeased {
		poolSet[p] = struct{}{}
	}

	fresh := make([]string, 0, len(proxies))
	for _, p := range proxies {
		if _, dup := poolSet[p]; !dup {
			fresh = append(fresh, p)
		}
	}
	return fresh
}

// filterNewIndividual is the fallback: checks each proxy via EXISTS/HExists.
func filterNewIndividual(ctx context.Context, proxyPool *proxypool.Pool, proxies []string) []string {
	fresh := make([]string, 0, len(proxies))
	for _, p := range proxies {
		exists, err := proxyPool.Exists(ctx, p)
		if err != nil {
			// On error, assume "already exists" (conservative). Treating it
			// as new would re-validate the proxy and waste a worker. The
			// original code's zero-value exists=false caused duplicates.
			logger.Log.Warn("filterNewIndividual: Exists check failed, assuming exists",
				zap.String("proxy", p), zap.Error(err))
			continue
		}
		if !exists {
			fresh = append(fresh, p)
		}
	}
	return fresh
}

// topUpIfBelowMin checks the current pool size and, if below PoolMinSize,
// fetches a fresh batch from the GET source, validates them, and adds the
// survivors to the pool.
func topUpIfBelowMin(
	ctx context.Context,
	cfg *config.Config,
	val *validator.Validator,
	proxyPool *proxypool.Pool,
	limiter *sourceFetchLimiter,
) {
	if ctx.Err() != nil {
		return
	}

	available, err := proxyPool.Available(ctx)
	if err != nil {
		logger.Log.Warn("Top-up: failed to read pool size", zap.Error(err))
		return
	}

	if available >= cfg.PoolMinSize {
		proxypool.ProxyTopUpRunsTotal.WithLabelValues("skipped").Inc()
		logger.Log.Debug("Top-up: pool above minimum, skipping",
			zap.Int64("available", available),
			zap.Int64("min", cfg.PoolMinSize))
		return
	}

	logger.Log.Info("Top-up: pool below minimum, adding more proxies",
		zap.Int64("available", available),
		zap.Int64("min", cfg.PoolMinSize))

	if cfg.RefreshSourceURL == "" {
		logger.Log.Debug("Top-up: no source URL configured, skipping fetch")
		return
	}

	// No fresh proxies in the recent batches — fetch from the remote source.
	logger.Log.Info("Top-up: fetching new GET proxies from source",
		zap.String("url", cfg.RefreshSourceURL))

	proxies, err := limitedFetchWithRetry(ctx, cfg, limiter)
	if err != nil {
		proxypool.ProxyTopUpRunsTotal.WithLabelValues("fetch_failed").Inc()
		logger.Log.Error("Top-up: fetch failed", zap.Error(err))
		return
	}

	fresh := filterNew(ctx, proxyPool, proxies)
	if len(fresh) == 0 {
		proxypool.ProxyTopUpRunsTotal.WithLabelValues("no_new").Inc()
		logger.Log.Info("Top-up: source returned no new proxies",
			zap.Int("candidates", len(proxies)))
		return
	}

	proxypool.ProxyTopUpRunsTotal.WithLabelValues("triggered").Inc()
	logger.Log.Info("Top-up: validating fresh proxies from source",
		zap.Int("candidates", len(fresh)))
	runValidation(ctx, val, proxyPool, fresh, cfg.PoolMaxSize)
}

// runValidation executes a validation pass over the given proxy candidates
// and returns the list of proxies that passed (were successfully added to the
// pool).
func runValidation(
	ctx context.Context,
	val *validator.Validator,
	proxyPool *proxypool.Pool,
	proxies []string,
	poolMax int64,
) []string {
	var (
		added     atomic.Int64
		validated = make([]string, 0, len(proxies))
		mu        sync.Mutex
	)

	sink := func(ctx context.Context, r validator.Result) {
		if !r.OK {
			return
		}
		ok, err := proxyPool.Add(ctx, r.Proxy)
		if err != nil {
			logger.Log.Warn("Pool add failed",
				zap.String("proxy", r.Proxy), zap.Error(err))
			return
		}
		if ok {
			added.Add(1)
			mu.Lock()
			validated = append(validated, r.Proxy)
			mu.Unlock()
		}
	}

	stats := val.ValidateStream(ctx, proxies, sink)

	if ctx.Err() != nil {
		logger.Log.Info("Validation run cancelled",
			zap.Error(ctx.Err()),
			zap.Int("completed", stats.Total),
			zap.Int("ok", stats.OK),
			zap.Int("failed", stats.Failed))
		return validated
	}

	if err := proxyPool.Trim(ctx, poolMax); err != nil {
		logger.Log.Warn("Pool trim failed", zap.Error(err))
	}
	available, _ := proxyPool.Available(ctx)

	logger.Log.Info("Validation run complete",
		zap.Int("candidates", stats.Total),
		zap.Int("ok", stats.OK),
		zap.Int("failed", stats.Failed),
		zap.Float64("success_rate", stats.SuccessRate()),
		zap.Duration("elapsed", stats.Elapsed),
		zap.Int64("avg_latency_ms", stats.AvgLatMs),
		zap.Int64("newly_added", added.Load()),
		zap.Int64("pool_available", available))

	return validated
}
