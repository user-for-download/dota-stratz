package proxypool

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Note: `dota2_proxy_pool_available` and `dota2_proxy_pool_leased` are
// NOT declared here. They used to be per-process promauto.NewGauge
// variables, but each Go service had its own registry, so values
// diverged across services (e.g. -322, 451, 0 vs Redis ground truth
// 129). They are now emitted by RedisPoolCollector, which reads
// ZCARD/HLEN from Redis at scrape time so every service reports the
// same value. See redis_collector.go for the implementation.
var (
	ProxyRemovedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dota2_proxy_removed_total",
		Help: "Total number of proxies removed from the pool, by reason.",
	}, []string{"reason"})

	ProxyCooldownTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dota2_proxy_cooldown_total",
		Help: "Total number of times a proxy was rate-limited and sent to cooldown.",
	})

	ProxyReapedTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dota2_proxy_reaped_total",
		Help: "Total number of expired leases reclaimed by the reaper.",
	})

	ProxyAcquireDurationSec = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "dota2_proxy_acquire_duration_seconds",
		Help:    "Time to successfully acquire a proxy from the pool (including retries).",
		Buckets: prometheus.ExponentialBuckets(0.001, 2, 10), // 1ms..1s
	})

	// --- Validation metrics ---

	ValidationDurationSec = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "dota2_proxy_validation_duration_seconds",
		Help:    "Wall-clock time of a full validation run.",
		Buckets: prometheus.ExponentialBuckets(1, 2, 10), // 1s..1024s
	})

	ValidationLatencyMs = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "dota2_proxy_validation_latency_ms",
		Help:    "Per-proxy validation latency in ms (successes only).",
		Buckets: prometheus.ExponentialBuckets(50, 2, 10), // 50ms..51s
	})

	ValidationResultTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dota2_proxy_validation_result_total",
		Help: "Total proxy validation outcomes by result.",
	}, []string{"result"}) // "ok" | "fail"

	ValidationInFlight = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dota2_proxy_validation_in_flight",
		Help: "Current number of proxies being validated.",
	})

	// --- Top-up metrics ---

	ProxyTopUpRunsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dota2_proxy_topup_runs_total",
		Help: "Total number of top-up cycles run at end of refresh, by outcome.",
	}, []string{"result"}) // "triggered" | "skipped" | "fetch_failed" | "no_new"

	// --- Refresh cycle metrics ---

	ProxyRefreshRunsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dota2_proxy_refresh_runs_total",
		Help: "Total number of periodic refresh cycles (URL fetch + validate), by outcome.",
	}, []string{"result"}) // "success" | "fetch_failed"

	// --- Rate-limit metrics ---

	ProxyRateLimitedTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dota2_proxy_rate_limited_total",
		Help: "Total number of times a proxy was skipped due to proactive per-minute rate limiting.",
	})

	ProxyFailuresOtherTotal = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dota2_proxy_removed_other_total",
		Help: "Total number of proxy removals with an unrecognized reason (cardinality guard).",
	})
)
