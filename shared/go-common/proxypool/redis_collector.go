package proxypool

import (
	"context"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/redis/go-redis/v9"
)

// scrapeTimeout bounds each individual Redis call done by the collector.
// 100ms is short enough to keep Prometheus scrapes from stalling on a
// sick Redis (Prometheus itself has a default scrape_timeout of 10s) and
// long enough that a healthy Redis on the same network returns well
// within budget (typical round-trip is sub-millisecond on a local socket).
const scrapeTimeout = 100 * time.Millisecond

// RedisPoolCollector is a custom prometheus.Collector that derives pool
// metrics directly from Redis at scrape time. It is the source of truth
// for `dota2_proxy_pool_available` and `dota2_proxy_pool_leased`.
//
// Why a custom collector instead of promauto.NewGauge:
// The previous implementation used a per-process gauge, which each Go
// service updated independently. Across services, values diverged
// wildly (e.g. -322, 451, 0 vs Redis ground truth 129) because Inc/Dec
// calls are not synchronized with Redis and background goroutines
// (e.g. the lease reaper) bypassed the gauge entirely. Two of the
// currently-firing `ProxyPoolDepleted` alerts were false positives for
// exactly this reason.
//
// This collector reads ZCARD on the proxies ZSET and HLEN on the leases
// HASH on every scrape, so every service that registers it reports the
// same value (ground truth) and there is no per-process drift.
//
// Concurrency: Prometheus may scrape the same collector from multiple
// goroutines. *redis.Client is safe for concurrent use and the
// collector holds no mutable per-scrape state, so no synchronisation
// is needed.
type RedisPoolCollector struct {
	rdb       *redis.Client
	poolKey   string
	leasesKey string

	availableDesc *prometheus.Desc
	leasedDesc    *prometheus.Desc
}

// NewRedisPoolCollector builds a collector that reads pool state from
// rdb using the given key names. The collector emits two metrics with
// the names `dota2_proxy_pool_available` and `dota2_proxy_pool_leased`
// — these names are preserved so existing Grafana dashboards and
// Prometheus alert rules keep working.
//
// Each scrape is bounded by scrapeTimeout (100ms). On any error
// (connection refused, timeout, pipeline failure) the collector emits
// no metrics for that cycle. This is preferable to emitting a
// misleading 0 when Redis is unreachable — Grafana would show a flat
// line at 0, and alerts like `dota2_proxy_pool_available < 20` would
// misfire for a transport issue rather than a real pool depletion.
func NewRedisPoolCollector(rdb *redis.Client, poolKey, leasesKey string) *RedisPoolCollector {
	return &RedisPoolCollector{
		rdb:       rdb,
		poolKey:   poolKey,
		leasesKey: leasesKey,
		availableDesc: prometheus.NewDesc(
			"dota2_proxy_pool_available",
			"Current number of proxies available in the pool (proxies ZSET ZCARD).",
			nil, nil,
		),
		leasedDesc: prometheus.NewDesc(
			"dota2_proxy_pool_leased",
			"Current number of proxies currently leased out (leases HASH HLEN).",
			nil, nil,
		),
	}
}

// Describe implements prometheus.Collector. It is invoked once at
// registration time so Prometheus knows the metric descriptors up
// front; it must NOT be called per scrape.
func (c *RedisPoolCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.availableDesc
	ch <- c.leasedDesc
}

// Collect implements prometheus.Collector. It is invoked on every
// Prometheus scrape. Both ZCARD and HLEN run in a single Redis pipeline
// so the total wall-clock time is one round-trip, bounded by
// scrapeTimeout. The pipeline is used so we issue one network round
// trip and one context timer per scrape (two sequential 100ms calls
// would be wasteful and could double the worst-case scrape latency).
//
// If either command errors (connection refused, timeout, redis.Nil
// from a missing key — though ZCARD/HLEN on a missing key return 0
// with err=nil, we still defend against it), no metrics are emitted
// for that scrape. This is the safe "garbage in, nothing out" choice.
func (c *RedisPoolCollector) Collect(ch chan<- prometheus.Metric) {
	ctx, cancel := context.WithTimeout(context.Background(), scrapeTimeout)
	defer cancel()

	pipe := c.rdb.Pipeline()
	availCmd := pipe.ZCard(ctx, c.poolKey)
	leasedCmd := pipe.HLen(ctx, c.leasesKey)
	if _, err := pipe.Exec(ctx); err != nil {
		// Any pipeline-level error (timeout, connection refused) is
		// treated as "no data for this scrape" rather than emitting a
		// zero or partial reading.
		return
	}

	avail, availErr := availCmd.Result()
	leased, leasedErr := leasedCmd.Result()
	if availErr != nil || leasedErr != nil {
		// Per-command errors (e.g. a command succeeded at the wire
		// level but result decoding failed). Skip emission rather
		// than emit a partial or zero value.
		return
	}

	ch <- prometheus.MustNewConstMetric(c.availableDesc, prometheus.GaugeValue, float64(avail))
	ch <- prometheus.MustNewConstMetric(c.leasedDesc, prometheus.GaugeValue, float64(leased))
}
