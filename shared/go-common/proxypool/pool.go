package proxypool

import (
	"context"
	"errors"
	"fmt"
	"hash/crc64"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/redis/go-redis/v9"
	"go.uber.org/zap"
)

const (
	RedisKey                = "dota2:proxies"
	LeaseKey                = "dota2:proxies:leases"
	FailureCounterKeyPrefix = "dota2:proxies:failures:"
	CooldownKeyPrefix       = "dota2:proxies:cooldown:"
	RateLimitKeyPrefix      = "dota2:proxies:ratelimit:"
)

var ErrNoProxyAvailable = errors.New("no proxy available")

type FailureReason string

const (
	ReasonHardFailure FailureReason = "hard"
	ReasonTimeout     FailureReason = "timeout"
	ReasonBadStatus   FailureReason = "bad_status"
	ReasonRateLimited FailureReason = "rate_limit"
)

// knownReasons is the allowlist of valid failure reasons used for metrics
// label sanitization. Any reason not in this set is collapsed to "other"
// to prevent Prometheus cardinality explosion.
var knownReasons = map[FailureReason]bool{
	ReasonHardFailure: true,
	ReasonTimeout:     true,
	ReasonBadStatus:   true,
	ReasonRateLimited: true,
}

// sanitizeReason collapses unknown failure reasons to "other" for safe use
// as a Prometheus label value. Returns the original string if known.
// This is a pure function — metric side effects belong at the call site.
func sanitizeReason(reason FailureReason) string {
	if knownReasons[reason] {
		return string(reason)
	}
	return "other"
}

type Pool struct {
	rdb               *redis.Client
	softFailThreshold int
	cooldownDuration  time.Duration
	leaseDuration     time.Duration
	softRetryDelay    time.Duration
	failureCounterTTL time.Duration
}

type Config struct {
	Strategy          string
	SoftFailThreshold int
	CooldownDuration  time.Duration
	LeaseDuration     time.Duration
	SoftRetryDelay    time.Duration
	FailureCounterTTL time.Duration
}

// New constructs a Pool. All Config fields are REQUIRED — no defaults.
// The caller (e.g. service config layer) owns env parsing and defaulting.
func New(rdb *redis.Client, cfg Config) (*Pool, error) {
	if cfg.SoftFailThreshold <= 0 {
		return nil, fmt.Errorf("SoftFailThreshold must be > 0")
	}
	if cfg.CooldownDuration <= 0 {
		return nil, fmt.Errorf("CooldownDuration must be > 0")
	}
	if cfg.LeaseDuration <= 0 {
		return nil, fmt.Errorf("LeaseDuration must be > 0")
	}
	if cfg.SoftRetryDelay <= 0 {
		return nil, fmt.Errorf("SoftRetryDelay must be > 0")
	}
	if cfg.FailureCounterTTL <= 0 {
		return nil, fmt.Errorf("FailureCounterTTL must be > 0")
	}
	if cfg.Strategy != "timestamp" && cfg.Strategy != "random" {
		return nil, fmt.Errorf("strategy must be 'timestamp' or 'random', got %q", cfg.Strategy)
	}

	return &Pool{
		rdb:               rdb,
		softFailThreshold: cfg.SoftFailThreshold,
		cooldownDuration:  cfg.CooldownDuration,
		leaseDuration:     cfg.LeaseDuration,
		softRetryDelay:    cfg.SoftRetryDelay,
		failureCounterTTL: cfg.FailureCounterTTL,
	}, nil
}

// --- Lua Scripts (atomic operations) ---

// acquireScript atomically pops the lowest-scored proxy and leases it.
// Returns {status, proxy} where status: 0=empty, 1=in_cooldown, 2=leased.
// KEYS[1] = proxies ZSET, KEYS[2] = leases HASH, KEYS[3] = cooldown prefix
// ARGV[1] = lease expiration, ARGV[2] = deferred score (for cooldown re-add)
var acquireScript = redis.NewScript(`
	local popped = redis.call('ZPOPMIN', KEYS[1], 1)
	if #popped == 0 then return {0, ''} end
	local proxy = popped[1]
	if redis.call('EXISTS', KEYS[3] .. proxy) == 1 then
		redis.call('ZADD', KEYS[1], ARGV[2], proxy)
		return {1, proxy}
	end
	redis.call('HSET', KEYS[2], proxy, ARGV[1])
	return {2, proxy}
`)

var releaseScript = redis.NewScript(`
	local removed = redis.call('HDEL', KEYS[2], ARGV[1])
	if removed == 0 then return 0 end
	redis.call('ZADD', KEYS[1], ARGV[2], ARGV[1])
	return 1
`)

// reapReleaseScript is like releaseScript but takes an expected lease
// expiry as an extra argument. Before HDEL-ing, it HGET s the current
// value and compares it to the caller's snapshot. If they differ, the
// lease was re-acquired by a different caller between the reaper's
// snapshot read and this script execution (TOCTOU race). In that case
// the script does nothing and returns 0, preventing the reaper from
// stealing an active lease.
var reapReleaseScript = redis.NewScript(`
	local current = redis.call('HGET', KEYS[2], ARGV[1])
	if current == false then return 0 end
	if current ~= ARGV[3] then return 0 end
	redis.call('HDEL', KEYS[2], ARGV[1])
	redis.call('ZADD', KEYS[1], ARGV[2], ARGV[1])
	return 1
`)

// rateLimitScript atomically increments a rolling 60-second counter and
// returns 1 if the proxy is under the limit, 0 if over. When over, the
// increment is undone via DECR so the counter only reflects accepted usage.
//
// EXPIRE is set only on the first INCR (count == 1) so the key's lifetime
// is always 60 seconds from the first request in the window, regardless of
// request volume. This avoids the "minute boundary crossing" race (where
// time.Now().Unix()/60 changes between consecutive calls) — see Issue #34.
// Previously the script extended TTL unconditionally, which created an
// infinite sliding window for proxies receiving continuous traffic —
// they would reach the rate limit and never get a fresh window (issue #12).
//
// KEYS[1] = ratelimit:{proxy_hash}
// ARGV[1] = max per minute (integer string)
var rateLimitScript = redis.NewScript(`
	local count = redis.call('INCR', KEYS[1])
	-- Only set EXPIRE on the first INCR (count == 1) so the 60-second
	-- window always starts from the first request. Previously the script
	-- extended TTL unconditionally, which created an infinite sliding
	-- window for proxies receiving continuous traffic — they would reach
	-- the rate limit and never get a fresh window (issue #12).
	if count == 1 then
		redis.call('EXPIRE', KEYS[1], 60)
	end
	if count > tonumber(ARGV[1]) then
		redis.call('DECR', KEYS[1])
		return 0
	end
	return 1
`)

// --- Producer API ---

func (p *Pool) Add(ctx context.Context, proxy string) (bool, error) {
	n, err := p.rdb.ZAddNX(ctx, RedisKey, redis.Z{
		Score:  p.nowScore(),
		Member: proxy,
	}).Result()
	if err != nil {
		return false, err
	}
	if n > 0 {
		logger.Log.Debug("Proxy added to pool", zap.String("proxy", proxy))
		return true, nil
	}
	logger.Log.Debug("Proxy already in pool", zap.String("proxy", proxy))
	return false, nil
}

// --- Consumer API ---

func (p *Pool) Acquire(ctx context.Context) (string, error) {
	defer func(start time.Time) {
		ProxyAcquireDurationSec.Observe(time.Since(start).Seconds())
	}(time.Now())

	// When the Lua script returns cooldown (status=1), the proxy was already
	// re-added to the ZSET with a deferred score. Retry a few times — the
	// cooldown proxy will be buried behind fresh ones.
	const maxRetries = 3

	for attempt := range maxRetries {
		expiresAt := time.Now().Add(p.leaseDuration).Unix()
		// Use UnixMicro() to match initialScore/nextScore — otherwise the
		// cooldown proxy's score (seconds ~1.7e9) is ~6 orders of magnitude
		// smaller than fresh proxies (microseconds ~1.7e15), causing ZPOPMIN
		// to pick the cooldown proxy FIRST instead of last.
		deferredScore := float64(time.Now().Add(p.cooldownDuration).UnixMicro())
		result, err := acquireScript.Run(ctx, p.rdb,
			[]string{RedisKey, LeaseKey, CooldownKeyPrefix}, expiresAt, deferredScore).Result()
		if err != nil {
			return "", fmt.Errorf("acquire script failed on %s: %w", p.rdb.Options().Addr, err)
		}

		arr, ok := result.([]interface{})
		if !ok || len(arr) < 2 {
			return "", ErrNoProxyAvailable
		}

		// Lua returns {status, proxy} where status: 0=empty, 1=cooldown, 2=leased
		status, statusOK := arr[0].(int64)
		proxy, proxyOK := arr[1].(string)
		if !statusOK || !proxyOK {
			return "", ErrNoProxyAvailable
		}

		switch status {
		case 0:
			// Check context before returning "no proxy" so a concurrent
			// shutdown doesn't get a misleading error (Bug #7).
			if ctx.Err() != nil {
				return "", ctx.Err()
			}
			logger.Log.Debug("Acquire: no proxy available")
			return "", ErrNoProxyAvailable
		case 1:
			logger.Log.Debug("Acquire: proxy in cooldown, retrying",
				zap.String("proxy", proxy),
				zap.Int("attempt", attempt+1))
			if attempt < maxRetries-1 {
				// Brief pause so we don't busy-spin against Redis.
				// Check ctx so shutdown doesn't wait unnecessarily.
				select {
				case <-time.After(50 * time.Millisecond):
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}
			continue
		case 2:
			logger.Log.Debug("Acquire: proxy leased",
				zap.String("proxy", proxy),
				zap.Int64("lease_expires_at", expiresAt))
			return proxy, nil
		default:
			return "", fmt.Errorf("unexpected acquire status: %d", status)
		}
	}
	return "", ErrNoProxyAvailable
}

// AcquireWithRateLimit wraps Acquire with a proactive rolling 60-second
// rate-limit check. If the proxy has already been used >= maxPerMinute times
// in the current window, it is released and a different proxy is attempted.
// This prevents wasting requests that would get a 429 response.
//
// The 60-second window starts when the first request hits the proxy. The key
// expires 60s after that, resetting the counter. Using a stable key (without
// a minute-window suffix) avoids the boundary-crossing race condition where
// time.Now().Unix()/60 changes between consecutive AcquireWithRateLimit calls.
func (p *Pool) AcquireWithRateLimit(ctx context.Context, maxPerMinute int) (string, error) {
	if maxPerMinute <= 0 {
		return p.Acquire(ctx)
	}
	const maxRetries = 5
	for attempt := range maxRetries {
		proxy, err := p.Acquire(ctx)
		if err != nil {
			return "", err
		}

		// Rolling window key = ratelimit:{proxy_hash} (no minute suffix).
		// The Lua script sets EXPIRE 60 only on the first INCR, so the
		// window is always 60s from the first request regardless of clock
		// boundaries.
		key := fmt.Sprintf("%s%s", RateLimitKeyPrefix, proxyHashKey(proxy))

		ok, err := rateLimitScript.Run(ctx, p.rdb, []string{key}, maxPerMinute).Result()
		if err != nil {
			_ = p.Release(ctx, proxy)
			return "", err
		}

		if ok.(int64) == 1 {
			// Rate limit counter was INCR'd. Return the proxy to the caller
			// — they'll Release() it when done, which decrements the lease.
			// If Release() fails, the counter stays incremented for the 60s
			// window, slightly shrinking the effective rate limit for this
			// proxy (Bug #16 — edge case, low impact).
			return proxy, nil
		}

		// Over limit — release with deferred score so this proxy sinks to
		// the back of the pool instead of getting a fresh (current-time)
		// timestamp, which would make it the next candidate and cause
		// busy-spin on the same proxy (audit finding #5).
		deferredScore := float64(time.Now().Add(p.cooldownDuration).UnixMicro())
		_, _ = releaseScript.Run(ctx, p.rdb, []string{RedisKey, LeaseKey}, proxy, deferredScore).Result()
		ProxyRateLimitedTotal.Inc()

		if attempt < maxRetries-1 {
			select {
			case <-time.After(50 * time.Millisecond):
			case <-ctx.Done():
				return "", ctx.Err()
			}
		}
	}
	return "", ErrNoProxyAvailable
}

func (p *Pool) Release(ctx context.Context, proxy string) error {
	newScore := p.nowScore()
	res, err := releaseScript.Run(ctx, p.rdb, []string{RedisKey, LeaseKey}, proxy, newScore).Result()
	if err != nil {
		return err
	}

	// Only log the return-to-pool if HDEL actually removed something
	// (i.e. lease was live). Pool/lease sizes are now derived by the
	// RedisPoolCollector at scrape time — no per-process gauge updates.
	if n, ok := res.(int64); ok && n == 1 {
		logger.Log.Debug("Release: proxy returned to pool",
			zap.String("proxy", proxy),
			zap.Float64("new_score", newScore))
	} else {
		logger.Log.Debug("Release: lease already gone", zap.String("proxy", proxy))
	}
	return nil
}

func (p *Pool) Report(ctx context.Context, proxy string, reason FailureReason) error {
	// Clear lease unconditionally; track whether it existed for the
	// remove/increment paths so they can decide whether to ZADD back.
	leaseExisted, hdelErr := p.rdb.HDel(ctx, LeaseKey, proxy).Result()
	if hdelErr != nil {
		logger.Log.Warn("Report: HDel lease failed",
			zap.String("proxy", proxy), zap.Error(hdelErr))
	}
	wasLeased := leaseExisted > 0

	logger.Log.Debug("Report: proxy failed",
		zap.String("proxy", proxy),
		zap.String("reason", string(reason)),
		zap.Int64("lease_was_live", leaseExisted))

	switch reason {
	case ReasonHardFailure, ReasonBadStatus:
		return p.remove(ctx, proxy, string(reason), wasLeased)
	case ReasonRateLimited:
		return p.cooldown(ctx, proxy)
	case ReasonTimeout:
		return p.incrementAndMaybeRemove(ctx, proxy, wasLeased)
	default:
		return fmt.Errorf("unknown failure reason: %s", reason)
	}
}

func (p *Pool) ReportSuccess(ctx context.Context, proxy string) error {
	return p.rdb.Del(ctx, failureKey(proxy)).Err()
}

// --- Internal state mutations ---

func (p *Pool) remove(ctx context.Context, proxy, reason string, wasLeased bool) error {
	pipe := p.rdb.Pipeline()
	pipe.ZRem(ctx, RedisKey, proxy)
	pipe.Del(ctx, failureKey(proxy))
	pipe.Del(ctx, CooldownKeyPrefix+proxy)
	if _, err := pipe.Exec(ctx); err != nil {
		return err
	}

	// `wasLeased` is still passed in to preserve the Report() call
	// chain contract, even though the gauge-update branch that used
	// it has been removed. A leased proxy was already ZPOPMIN'd
	// during Acquire, so Report only clears its lease; an unleased
	// proxy was still in the ZSET, so we ZRem it here. The pool size
	// itself is now derived by RedisPoolCollector from ZCARD at
	// scrape time.
	label := sanitizeReason(FailureReason(reason))
	if label == "other" {
		ProxyFailuresOtherTotal.Inc()
	}
	ProxyRemovedTotal.WithLabelValues(label).Inc()
	logger.Log.Debug("Proxy removed from pool",
		zap.String("proxy", proxy),
		zap.Bool("was_leased", wasLeased),
		zap.String("reason", reason))
	return nil
}

func (p *Pool) incrementAndMaybeRemove(ctx context.Context, proxy string, wasLeased bool) error {
	pipe := p.rdb.Pipeline()
	countIncr := pipe.Incr(ctx, failureKey(proxy))
	pipe.Expire(ctx, failureKey(proxy), p.failureCounterTTL)
	if _, err := pipe.Exec(ctx); err != nil {
		return err
	}
	count := countIncr.Val()

	logger.Log.Debug("Increment fail count",
		zap.String("proxy", proxy),
		zap.Int64("count", count),
		zap.Int("threshold", p.softFailThreshold))

	if int(count) >= p.softFailThreshold {
		// Soft-threshold exceeded — remove the proxy permanently.
		// Report() already cleared the lease, so pass wasLeased through.
		return p.remove(ctx, proxy, "soft_threshold_exceeded", wasLeased)
	}

	// Return to pool with delayed score so it's deprioritized briefly.
	// Report() already cleared the lease, so just re-add to the ZSET.
	added, err := p.rdb.ZAdd(ctx, RedisKey, redis.Z{
		// Use UnixMicro() to match initialScore/nextScore (see Acquire's
		// deferredScore for the same reason).
		Score:  float64(time.Now().Add(p.softRetryDelay).UnixMicro()),
		Member: proxy,
	}).Result()
	if err != nil {
		return err
	}
	if added > 0 {
		logger.Log.Debug("Soft fail: proxy returned to pool with delay",
			zap.String("proxy", proxy),
			zap.Duration("delay", p.softRetryDelay))
	}
	return nil
}

func (p *Pool) cooldown(ctx context.Context, proxy string) error {
	pipe := p.rdb.Pipeline()
	pipe.Set(ctx, CooldownKeyPrefix+proxy, "1", p.cooldownDuration)
	pipe.ZAdd(ctx, RedisKey, redis.Z{
		// Use UnixMicro() to match initialScore/nextScore (see Acquire's
		// deferredScore for the same reason).
		Score:  float64(time.Now().Add(p.cooldownDuration).UnixMicro()),
		Member: proxy,
	})
	if _, err := pipe.Exec(ctx); err != nil {
		return err
	}
	// Proxy re-added to the ZSET with a deferred score so ZPOPMIN
	// deprioritises it for the cooldown window. Pool size itself
	// is derived by RedisPoolCollector from ZCARD at scrape time.
	ProxyCooldownTotal.Inc()
	logger.Log.Debug("Proxy sent to cooldown",
		zap.String("proxy", proxy),
		zap.Duration("duration", p.cooldownDuration))
	return nil
}

// --- Reaper ---

func (p *Pool) ReapExpiredLeases(ctx context.Context) (int, error) {
	now := time.Now().Unix()
	leases, err := p.rdb.HGetAll(ctx, LeaseKey).Result()
	if err != nil {
		return 0, err
	}

	reaped := 0
	for proxy, expiresAtStr := range leases {
		expiresAt, parseErr := strconv.ParseInt(expiresAtStr, 10, 64)
		if parseErr != nil {
			continue
		}
		if expiresAt > now {
			continue
		}

		res, runErr := reapReleaseScript.Run(ctx, p.rdb,
			[]string{RedisKey, LeaseKey}, proxy, p.nowScore(), expiresAtStr).Result()
		if runErr != nil {
			logger.Log.Warn("Reap: failed to release expired lease",
				zap.String("proxy", proxy), zap.Error(runErr))
			continue
		}
		if n, ok := res.(int64); ok && n == 1 {
			reaped++
			ProxyReapedTotal.Inc()
		}
	}
	if reaped > 0 {
		logger.Log.Debug("Reaped expired leases", zap.Int("count", reaped))
	}
	return reaped, nil
}

// --- Read-only helpers ---

func (p *Pool) Exists(ctx context.Context, proxy string) (bool, error) {
	pipe := p.rdb.Pipeline()
	inPool := pipe.ZScore(ctx, RedisKey, proxy)
	inLease := pipe.HExists(ctx, LeaseKey, proxy)
	if _, err := pipe.Exec(ctx); err != nil && !errors.Is(err, redis.Nil) {
		return false, err
	}

	if _, err := inPool.Result(); err == nil {
		return true, nil
	}
	if leased, err := inLease.Result(); err == nil && leased {
		return true, nil
	}
	return false, nil
}

func (p *Pool) Available(ctx context.Context) (int64, error) {
	return p.rdb.ZCard(ctx, RedisKey).Result()
}

func (p *Pool) InUse(ctx context.Context) (int64, error) {
	return p.rdb.HLen(ctx, LeaseKey).Result()
}

// Members returns all proxy members currently in the pool ZSET (not leased).
// This is used to batch-check existence locally instead of O(N) round-trips.
func (p *Pool) Members(ctx context.Context) ([]string, error) {
	return p.rdb.ZRange(ctx, RedisKey, 0, -1).Result()
}

// LeasedMembers returns all proxy members currently in the lease HASH.
// Used in combination with Members to get a complete picture of all proxies
// managed by the pool (both available and in-use).
func (p *Pool) LeasedMembers(ctx context.Context) ([]string, error) {
	m, err := p.rdb.HKeys(ctx, LeaseKey).Result()
	if err != nil {
		return nil, err
	}
	return m, nil
}

// InCooldown checks whether a proxy is currently cooling down.
// Useful for diagnostic endpoints and pre-flight checks in downstream services.
func (p *Pool) InCooldown(ctx context.Context, proxy string) (bool, error) {
	n, err := p.rdb.Exists(ctx, CooldownKeyPrefix+proxy).Result()
	return n > 0, err
}

func (p *Pool) Trim(ctx context.Context, max int64) error {
	size, err := p.Available(ctx)
	if err != nil {
		return err
	}
	if size <= max {
		return nil
	}
	excess := size - max
	// Remove the highest-scored (cooldown/future-timestamp) proxies instead
	// of the lowest-scored (ready-to-use) ones. Cooldown proxies have scores
	// of time.Now().Add(cooldownDuration).UnixMicro() which makes them the
	// highest-ranked members — removing by negative rank targets those.
	_, err = p.rdb.ZRemRangeByRank(ctx, RedisKey, -excess, -1).Result()
	if err != nil {
		return err
	}
	logger.Log.Debug("Trimmed pool",
		zap.Int64("excess", excess),
		zap.Int64("max", max))
	return nil
}

// InitGauges was removed in P0-4. The pool/lease gauges are now derived
// from Redis at scrape time by RedisPoolCollector, so there is nothing
// to "initialise" at startup — the first scrape after Redis is
// reachable will read the current ZCARD/HLEN.

func failureKey(proxy string) string {
	return FailureCounterKeyPrefix + proxy
}

// nowScore returns a monotonic timestamp suitable for Redis ZSET scores.
// We use UnixMicro() instead of UnixNano() because float64 only guarantees
// 53 bits of mantissa (~9e15 integer precision).  Current UnixNano()
// values (~1.7e18) exceed this range and lose sub-microsecond precision.
// UnixMicro() values (~1.7e15) fit safely within float64's exact integer
// range (Issue #29).
//
// BUG-016: collapsed identical initialScore / nextScore into this single
// function to eliminate dead abstraction.
func (p *Pool) nowScore() float64 {
	return float64(time.Now().UnixMicro())
}

// proxyHashKey returns a deterministic compact key for a proxy URL suitable for
// use in Redis keys. This avoids characters ({, }, :) that can interfere with
// Redis Cluster hash-slot distribution and keeps keys a bounded length.
//
// Uses CRC64 (ECMA-182) instead of SHA256+hex to avoid the heap allocation
// from hex.EncodeToString — CRC64 is ~10× faster and the collision risk for
// Redis keys is negligible (Issue #17).
var crc64Table = crc64.MakeTable(crc64.ECMA)

func proxyHashKey(proxy string) string {
	h := crc64.Checksum([]byte(proxy), crc64Table)
	return strconv.FormatUint(h, 36) // base-36: up to 13 chars, no special chars
}

// --- Safe consumer wrapper ---

// MaxWithProxyBytes bounds the response body read through WithProxy so a
// misbehaving/compromised proxy cannot exhaust service memory. Callers that
// need larger payloads should read directly (no WithProxy).
const MaxWithProxyBytes = 16 * 1024 * 1024 // 16 MiB

func (p *Pool) WithProxy(ctx context.Context, fn func(proxy string) (*http.Response, error)) ([]byte, error) {
	proxy, err := p.Acquire(ctx)
	if err != nil {
		return nil, err
	}

	resp, err := fn(proxy)

	// Guard: nil response with no error is a pathological callback that
	// returns resp == nil, err == nil — prevent a nil-deref panic below.
	// When resp is nil AND err is set (timeout, connection refused, DNS),
	// delegate to ClassifyError so timeouts get ReasonTimeout (soft retry)
	// instead of ReasonHardFailure (permanent ban).
	if resp == nil {
		if err != nil {
			if reason, shouldReport := ClassifyError(err, resp); shouldReport {
				_ = p.Report(ctx, proxy, reason)
			} else {
				// Non-reportable error (e.g. context.Canceled): release
				// the lease instead of leaking it until the reaper fires.
				// Fixes audit finding #2: lease leak on non-reportable error.
				_ = p.Release(ctx, proxy)
			}
			return nil, err // preserve the original error
		}
		_ = p.Report(ctx, proxy, ReasonHardFailure)
		return nil, fmt.Errorf("nil response for proxy %s", proxy)
	}
	defer resp.Body.Close()

	if reason, shouldReport := ClassifyError(err, resp); shouldReport {
		logger.Log.Debug("WithProxy: reporting proxy failure",
			zap.String("proxy", proxy),
			zap.String("reason", string(reason)))
		_ = p.Report(ctx, proxy, reason)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("proxy reported: %s", reason)
	}

	// Non-reportable error with a non-nil response (e.g. context.Canceled
	// arriving after a partial response). Release the lease instead of
	// falling through to ReportSuccess/ReadAll, which would treat a
	// cancelled request as a success (audit finding #3).
	if err != nil {
		_ = p.Release(ctx, proxy)
		return nil, err
	}

	body, readErr := io.ReadAll(io.LimitReader(resp.Body, MaxWithProxyBytes+1))
	if readErr != nil {
		logger.Log.Debug("WithProxy: body read failed, reporting timeout",
			zap.String("proxy", proxy),
			zap.Error(readErr))
		// Use ReasonTimeout for transient TCP read failures so the proxy
		// gets a soft retry instead of permanent removal (Bug #6).
		_ = p.Report(ctx, proxy, ReasonTimeout)
		return nil, readErr
	}
	if int64(len(body)) > MaxWithProxyBytes {
		logger.Log.Debug("WithProxy: body too large, reporting hard failure",
			zap.String("proxy", proxy),
			zap.Int("bytes", len(body)),
			zap.Int("max", MaxWithProxyBytes))
		_ = p.Report(ctx, proxy, ReasonHardFailure)
		return nil, fmt.Errorf("response body exceeds %d bytes (proxy %s)", MaxWithProxyBytes, proxy)
	}
	_ = p.ReportSuccess(ctx, proxy)
	_ = p.Release(ctx, proxy)
	logger.Log.Debug("WithProxy: completed successfully",
		zap.String("proxy", proxy),
		zap.Int("body_bytes", len(body)))
	return body, nil
}
