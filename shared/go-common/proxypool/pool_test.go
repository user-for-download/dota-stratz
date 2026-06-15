package proxypool

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// errReader is a helper that returns a fixed error on Read — used to simulate
// body read failures in WithProxy tests.
type errReader struct{ err error }

func (r *errReader) Read([]byte) (int, error) { return 0, r.err }

func TestMain(m *testing.M) {
	_ = os.Setenv("LOG_LEVEL", "error") // suppress debug logging in tests
	logger.InitLogger()
	os.Exit(m.Run())
}

// newTestPoolDefault creates a Pool backed by miniredis for testing with standard config.
// Returns the pool, the miniredis server (for direct state checks), and a cleanup func.
func newTestPoolDefault(t *testing.T) (*Pool, *miniredis.Miniredis, func()) {
	t.Helper()

	mr, err := miniredis.Run()
	if err != nil {
		t.Fatalf("miniredis.Run: %v", err)
	}

	rdb := redis.NewClient(&redis.Options{
		Addr: mr.Addr(),
	})

	pool, err := New(rdb, Config{
		Strategy:          "timestamp",
		SoftFailThreshold: 3,
		CooldownDuration:  1 * time.Minute,
		LeaseDuration:     10 * time.Second,
		SoftRetryDelay:    1 * time.Second,
		FailureCounterTTL: 1 * time.Hour,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	return pool, mr, func() {
		rdb.Close()
		mr.Close()
	}
}

// newTestPool creates a Pool from the given miniredis and config.
// The caller owns the miniredis lifecycle.
func newTestPool(t *testing.T, s *miniredis.Miniredis, cfg Config) *Pool {
	t.Helper()
	rdb := redis.NewClient(&redis.Options{Addr: s.Addr()})
	t.Cleanup(func() { rdb.Close() })
	pool, err := New(rdb, cfg)
	require.NoError(t, err)
	return pool
}

func TestAddAndExists(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	ok, err := pool.Add(ctx, "http://proxy-a:8080")
	if err != nil {
		t.Fatalf("Add: %v", err)
	}
	if !ok {
		t.Fatal("expected Add to return true for new proxy")
	}

	exists, err := pool.Exists(ctx, "http://proxy-a:8080")
	if err != nil {
		t.Fatalf("Exists: %v", err)
	}
	if !exists {
		t.Fatal("expected Exists to return true after Add")
	}

	// Duplicate add should report false
	ok, err = pool.Add(ctx, "http://proxy-a:8080")
	if err != nil {
		t.Fatalf("Add duplicate: %v", err)
	}
	if ok {
		t.Fatal("expected Add duplicate to return false")
	}
}

func TestAcquireRelease(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	proxy, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	if proxy != "http://proxy-a:8080" {
		t.Fatalf("expected proxy-a, got %q", proxy)
	}

	// Release back to pool
	if err := pool.Release(ctx, proxy); err != nil {
		t.Fatalf("Release: %v", err)
	}

	// Should be available again
	avail, err := pool.Available(ctx)
	if err != nil {
		t.Fatalf("Available: %v", err)
	}
	if avail != 1 {
		t.Fatalf("expected 1 available after release, got %d", avail)
	}
}

func TestAcquireEmpty(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	_, err := pool.Acquire(ctx)
	if !errors.Is(err, ErrNoProxyAvailable) {
		t.Fatalf("expected ErrNoProxyAvailable, got %v", err)
	}
}

func TestAcquireAllCooledDown(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	// Add a proxy and rate-limit it into cooldown
	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}
	proxy, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	// Report rate-limited — sends proxy to cooldown with a deferred score
	if err := pool.Report(ctx, proxy, ReasonRateLimited); err != nil {
		t.Fatalf("Report: %v", err)
	}

	// Now the only proxy is in cooldown. Acquire should retry briefly and
	// eventually return ErrNoProxyAvailable (all cooled down).
	_, err = pool.Acquire(ctx)
	if !errors.Is(err, ErrNoProxyAvailable) {
		t.Fatalf("expected ErrNoProxyAvailable when all cooled down, got %v", err)
	}
}

func TestReportHardFailure(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	for _, p := range []string{"http://proxy-a:8080", "http://proxy-b:8080"} {
		if _, err := pool.Add(ctx, p); err != nil {
			t.Fatalf("Add %s: %v", p, err)
		}
	}

	// Acquire proxy-a
	proxy, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	// Report hard failure — removes proxy from pool + lease
	if err := pool.Report(ctx, proxy, ReasonHardFailure); err != nil {
		t.Fatalf("Report: %v", err)
	}

	// Should be removed from pool ZSET
	exists, _ := pool.Exists(ctx, proxy)
	if exists {
		t.Fatalf("expected %q to be removed from pool", proxy)
	}

	// Should still have proxy-b available
	avail, err := pool.Available(ctx)
	if err != nil {
		t.Fatalf("Available: %v", err)
	}
	if avail != 1 {
		t.Fatalf("expected 1 available after removal, got %d", avail)
	}
}

func TestReportTimeout(t *testing.T) {
	pool, mr, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	proxy, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	// Report timeout — increments failure counter, returns to pool
	if err := pool.Report(ctx, proxy, ReasonTimeout); err != nil {
		t.Fatalf("Report: %v", err)
	}

	// Verify failure counter was created (per-proxy key with TTL)
	failKey := failureKey(proxy)
	if !mr.Exists(failKey) {
		t.Fatal("expected failure counter key to exist")
	}
	v, err := mr.Get(failKey)
	if err != nil {
		t.Fatalf("Get %s: %v", failKey, err)
	}
	if v != "1" {
		t.Fatalf("expected failure count 1, got %q", v)
	}

	// Proxy should be back in the pool
	exists, _ := pool.Exists(ctx, proxy)
	if !exists {
		t.Fatalf("expected %q to be returned to pool after timeout", proxy)
	}
}

func TestTrim(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	// Add proxies with varying scores by sleeping briefly between adds
	// (timestamp strategy means each Add gets a slightly higher score)
	for i := 0; i < 5; i++ {
		p := "http://proxy-" + strconv.Itoa(i) + ":8080"
		if _, err := pool.Add(ctx, p); err != nil {
			t.Fatalf("Add %s: %v", p, err)
		}
		time.Sleep(1 * time.Millisecond)
	}

	// Trim to max 2 — should remove the 3 HIGHEST-scored (cooldown/freshest),
	// keep the 2 lowest-scored (ready-to-use/stalest). In production the
	// highest-scored proxies are those in cooldown (future timestamps).
	if err := pool.Trim(ctx, 2); err != nil {
		t.Fatalf("Trim: %v", err)
	}

	avail, err := pool.Available(ctx)
	if err != nil {
		t.Fatalf("Available: %v", err)
	}
	if avail != 2 {
		t.Fatalf("expected 2 available after trim, got %d", avail)
	}

	// The two remaining should be the lowest-scored: proxy-0 and proxy-1 (stalest/oldest)
	for _, expected := range []string{"http://proxy-0:8080", "http://proxy-1:8080"} {
		exists, err := pool.Exists(ctx, expected)
		if err != nil {
			t.Fatalf("Exists %s: %v", expected, err)
		}
		if !exists {
			t.Errorf("expected %s to remain after Trim (lowest-scored proxy)", expected)
		}
	}

	// proxy-3 and proxy-4 (highest-scored) should have been removed
	for _, removed := range []string{"http://proxy-3:8080", "http://proxy-4:8080"} {
		exists, _ := pool.Exists(ctx, removed)
		if exists {
			t.Errorf("expected %s to be removed by Trim (highest-scored proxy)", removed)
		}
	}
}

func TestConcurrency(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	const numProxies = 10
	const numGoroutines = 50

	// Add all proxies
	for i := 0; i < numProxies; i++ {
		if _, err := pool.Add(ctx, "http://proxy-"+strconv.Itoa(i)+":8080"); err != nil {
			t.Fatalf("Add proxy-%d: %v", i, err)
		}
	}

	var wg sync.WaitGroup
	sema := make(chan struct{}, numProxies) // limit to pool size

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			sema <- struct{}{}
			defer func() { <-sema }()

			proxy, err := pool.Acquire(ctx)
			if err != nil {
				if errors.Is(err, ErrNoProxyAvailable) {
					// All proxies in use — legitimate
					return
				}
				t.Errorf("goroutine %d: Acquire error: %v", id, err)
				return
			}

			// Simulate some work, then release
			time.Sleep(time.Duration(id%5) * time.Millisecond)
			if err := pool.Release(ctx, proxy); err != nil {
				t.Errorf("goroutine %d: Release %s error: %v", id, proxy, err)
			}
		}(i)
	}
	wg.Wait()

	// All proxies should eventually be available
	avail, err := pool.Available(ctx)
	if err != nil {
		t.Fatalf("Available: %v", err)
	}
	if avail != numProxies {
		t.Fatalf("expected %d available after concurrency test, got %d", numProxies, avail)
	}
}

func TestReapExpiredLeasesIgnoresLive(t *testing.T) {
	pool, mr, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Acquire — creates a lease expiring in 10s (the LeaseDuration in newTestPool)
	proxy, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	// Run reaper immediately while the lease is still live
	reaped, err := pool.ReapExpiredLeases(ctx)
	if err != nil {
		t.Fatalf("ReapExpiredLeases: %v", err)
	}
	if reaped != 0 {
		t.Fatalf("expected 0 reaped (lease still live), got %d", reaped)
	}

	// Lease should still be in the HASH (reaper left it alone)
	leaseVal := mr.HGet(LeaseKey, proxy)
	if leaseVal == "" {
		t.Fatal("expected lease to still exist after reaper ignored it")
	}

	// The proxy IS tracked in the lease HASH even though it's not in the
	// pool ZSET, so Exists must report true. The original test relied on a
	// miniredis-specific quirk where pipe.Exec propagated redis.Nil from
	// ZScore before the lease check could run; in production (real Redis)
	// the pipeline never returns redis.Nil for a non-empty command list,
	// so the lease check executes correctly.
	exists, err := pool.Exists(ctx, proxy)
	if err != nil && !errors.Is(err, redis.Nil) {
		t.Fatalf("Exists: %v", err)
	}
	if !exists {
		t.Fatal("expected proxy to exist (in lease hash)")
	}
}

func TestReapExpiredLeasesExpired(t *testing.T) {
	pool, mr, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Acquire — creates lease with Unix timestamp expiry
	proxy, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	// The lease HASH entry stores timestamp: "lease_expires_at"
	// Acquire's Lua: HSET KEYS[2] proxy ARGV[1] where ARGV[1] = now + leaseDuration
	// We need to backdate it to simulate expiry
	// Since we know the key structure, we can directly manipulate via miniredis
	oldExpiry := mr.HGet(LeaseKey, proxy)
	t.Logf("Original lease expiry: %s", oldExpiry)

	// Set lease to already expired (1 hour ago)
	past := time.Now().Add(-1 * time.Hour).Unix()
	mr.HSet(LeaseKey, proxy, strconv.FormatInt(past, 10))

	// Reaper should find and restore it
	reaped, err := pool.ReapExpiredLeases(ctx)
	if err != nil {
		t.Fatalf("ReapExpiredLeases: %v", err)
	}
	if reaped != 1 {
		t.Fatalf("expected 1 reaped, got %d", reaped)
	}

	// Proxy should be back in pool
	exists, _ := pool.Exists(ctx, proxy)
	if !exists {
		t.Fatal("expected proxy to be restored to pool after reap")
	}

	avail, err := pool.Available(ctx)
	if err != nil {
		t.Fatalf("Available: %v", err)
	}
	if avail != 1 {
		t.Fatalf("expected 1 available after reap, got %d", avail)
	}
}

// --- AcquireWithRateLimit tests ---

func TestAcquireWithRateLimit_UnderLimit(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// maxPerMinute=50 with 1 proxy: first acquire should succeed
	proxy, err := pool.AcquireWithRateLimit(ctx, 50)
	if err != nil {
		t.Fatalf("AcquireWithRateLimit: %v", err)
	}
	if proxy != "http://proxy-a:8080" {
		t.Fatalf("expected proxy-a, got %q", proxy)
	}
}

func TestAcquireWithRateLimit_OverLimit(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Set maxPerMinute=0 (bypass check) → acquires and releases to "warm" the
	// rate-limit counter up to the limit via direct Redis INCR. Then set limit
	// to 1 and verify subsequent acquires fail.
	limit := 1

	// Use a separate goroutine-orchestrated approach: acquire once (succeeds
	// since counter=0 <= 1), then the counter is at 1. The next acquire on the
	// same proxy should exceed the limit.
	proxy, err := pool.AcquireWithRateLimit(ctx, limit)
	if err != nil {
		t.Fatalf("first AcquireWithRateLimit should succeed: %v", err)
	}
	_ = pool.Release(ctx, proxy) // return to pool so it can be re-acquired

	// Now counter=1, limit=1, next attempt should bump to 2 > 1 and fail
	_, err = pool.AcquireWithRateLimit(ctx, limit)
	if !errors.Is(err, ErrNoProxyAvailable) {
		t.Fatalf("expected ErrNoProxyAvailable when over limit, got %v", err)
	}
}

func TestAcquireWithRateLimit_Disabled(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// maxPerMinute <= 0 should bypass rate limiting entirely
	proxy, err := pool.AcquireWithRateLimit(ctx, 0)
	if err != nil {
		t.Fatalf("AcquireWithRateLimit(0): %v", err)
	}
	if proxy != "http://proxy-a:8080" {
		t.Fatalf("expected proxy-a, got %q", proxy)
	}
}

func TestAcquireWithRateLimit_MinuteBoundary(t *testing.T) {
	pool, mr, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	// Single proxy so the retry loop has no alternative — it must be this one.
	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add proxy-a: %v", err)
	}

	// First call under limit=1 succeeds; Lua INCR bumps counter to 1.
	proxy, err := pool.AcquireWithRateLimit(ctx, 1)
	if err != nil {
		t.Fatalf("first AcquireWithRateLimit: %v", err)
	}
	if err := pool.Release(ctx, proxy); err != nil {
		t.Fatalf("Release: %v", err)
	}

	// Without a boundary advance, a second call fails (INCR → 2 > 1 → DECR → 0).
	// (The retry loop exhausts 5 attempts because the only proxy keeps getting
	// rate-limited.)
	_, err = pool.AcquireWithRateLimit(ctx, 1)
	if !errors.Is(err, ErrNoProxyAvailable) {
		t.Fatalf("expected ErrNoProxyAvailable before boundary: %v", err)
	}

	// Advance miniredis time past the 60s TTL so the old rate-limit key expires.
	mr.FastForward(90 * time.Second)

	// The key is now gone, so INCR starts from 1 again → under limit.
	proxy, err = pool.AcquireWithRateLimit(ctx, 1)
	if err != nil {
		t.Fatalf("AcquireWithRateLimit after minute boundary: %v", err)
	}
	if err := pool.Release(ctx, proxy); err != nil {
		t.Fatalf("Release after boundary: %v", err)
	}
}

// --- WithProxy tests ---

func TestWithProxy_AcquireFail(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	// Pool is empty, so Acquire should fail immediately
	_, err := pool.WithProxy(ctx, func(proxy string) (*http.Response, error) {
		t.Fatal("callback should not be called when Acquire fails")
		return nil, nil
	})
	if !errors.Is(err, ErrNoProxyAvailable) {
		t.Fatalf("expected ErrNoProxyAvailable, got %v", err)
	}
}

func TestWithProxy_NilResponse(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Callback returns nil response, nil error → path 2
	_, err := pool.WithProxy(ctx, func(proxy string) (*http.Response, error) {
		return nil, nil
	})
	if err == nil || !strings.Contains(err.Error(), "nil response") {
		t.Fatalf("expected nil response error, got %v", err)
	}

	// Proxy should have been removed (hard failure via Report)
	exists, _ := pool.Exists(ctx, "http://proxy-a:8080")
	if exists {
		t.Fatal("expected proxy to be removed after nil response")
	}
}

func TestWithProxy_TimeoutError(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Return both a 200 response AND context.DeadlineExceeded.
	// ClassifyError sees the timeout first → ReasonTimeout → soft fail.
	_, err := pool.WithProxy(ctx, func(proxy string) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader("ok"))},
			context.DeadlineExceeded
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected DeadlineExceeded, got %v", err)
	}

	// Proxy should still exist (timeout = soft failure, just increments counter)
	exists, _ := pool.Exists(ctx, "http://proxy-a:8080")
	if !exists {
		t.Fatal("expected proxy to remain after timeout (soft fail)")
	}
}

func TestWithProxy_BadStatus(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// 500 with no Go error → path 4: ClassifyError returns (ReasonBadStatus, true),
	// shouldReport=true, err==nil → "proxy reported: bad_status"
	_, err := pool.WithProxy(ctx, func(proxy string) (*http.Response, error) {
		return &http.Response{StatusCode: 500, Body: io.NopCloser(strings.NewReader(""))}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "bad_status") {
		t.Fatalf("expected bad_status error, got %v", err)
	}

	// Proxy should have been removed (bad status = hard removal)
	exists, _ := pool.Exists(ctx, "http://proxy-a:8080")
	if exists {
		t.Fatal("expected proxy to be removed after 500")
	}
}

func TestWithProxy_BodyReadFail(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Body that errors on read → path 5 (now treated as ReasonTimeout per Bug #6)
	failBody := io.NopCloser(&errReader{err: fmt.Errorf("connection reset")})
	_, err := pool.WithProxy(ctx, func(proxy string) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: failBody}, nil
	})
	if err == nil {
		t.Fatal("expected body read error, got nil")
	}

	// Proxy should still exist (body read failure = ReasonTimeout = soft fail,
	// just increments the failure counter instead of removing — Bug #6).
	exists, _ := pool.Exists(ctx, "http://proxy-a:8080")
	if !exists {
		t.Fatal("expected proxy to remain after body read failure (soft fail)")
	}
}

func TestWithProxy_Success(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	body, err := pool.WithProxy(ctx, func(proxy string) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader("hello world"))}, nil
	})
	if err != nil {
		t.Fatalf("expected success, got %v", err)
	}
	if string(body) != "hello world" {
		t.Fatalf("expected body 'hello world', got %q", string(body))
	}

	// Proxy should be back in the pool (ReportSuccess + Release)
	exists, _ := pool.Exists(ctx, "http://proxy-a:8080")
	if !exists {
		t.Fatal("expected proxy to be back in pool after success")
	}
}

func TestWithProxy_429IsCooldown(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	if _, err := pool.Add(ctx, "http://proxy-a:8080"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// 429 → ReasonRateLimited → cooldown (proxy stays but gets deferred score)
	_, err := pool.WithProxy(ctx, func(proxy string) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusTooManyRequests,
			Body: io.NopCloser(strings.NewReader(""))}, nil
	})
	if err == nil || !strings.Contains(err.Error(), "rate_limit") {
		t.Fatalf("expected rate_limit error, got %v", err)
	}

	// Proxy should still be in pool (cooldown re-adds it with deferred score)
	exists, _ := pool.Exists(ctx, "http://proxy-a:8080")
	if !exists {
		t.Fatal("expected proxy to remain after 429 (cooldown)")
	}
}

// TestCooldownScoreNotPickedFirst is a regression test for a bug where the
// cooldown/soft-fail re-add used Unix seconds (1.7e9) for the ZSET score while
// fresh Add/Release used Unix microseconds (1.7e15). This caused ZPOPMIN to
// pick the cooldown proxy FIRST (~6 orders of magnitude smaller score) — the
// exact opposite of the intended deprioritization. All ZSET operations must
// use the same time unit.
func TestCooldownScoreNotPickedFirst(t *testing.T) {
	pool, mr, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	// Add 3 proxies, each 1ms apart so their microsecond scores are distinct
	for i, p := range []string{"http://a:8080", "http://b:8080", "http://c:8080"} {
		if _, err := pool.Add(ctx, p); err != nil {
			t.Fatalf("Add %s: %v", p, err)
		}
		if i < 2 {
			time.Sleep(2 * time.Millisecond)
		}
	}

	// Acquire 'a' and rate-limit it → cooldown re-adds it with a deferred score
	proxy, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	if proxy != "http://a:8080" {
		t.Fatalf("expected a, got %s", proxy)
	}
	if err := pool.Report(ctx, proxy, ReasonRateLimited); err != nil {
		t.Fatalf("Report: %v", err)
	}

	// Bump miniredis time forward so 'a' is "in cooldown" (less than CooldownDuration
	// in the future). With the bug, 'a' would have a seconds-scale score (~1.7e9),
	// which is smaller than the microsecond scores of b/c (~1.7e15), making ZPOPMIN
	// pick 'a' first.
	mr.FastForward(1 * time.Second)

	// Acquire again — must NOT return 'a' first, since it's in cooldown.
	got, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire after cooldown: %v", err)
	}
	if got == "http://a:8080" {
		t.Errorf("BUG: cooldown proxy 'a' was picked first; expected b or c (scoring units inconsistent)")
	}
}

// TestSoftFailScoreNotPickedFirst is a regression test for the same scoring
// bug as TestCooldownScoreNotPickedFirst, but in the incrementAndMaybeRemove
// (soft-fail) path.
func TestSoftFailScoreNotPickedFirst(t *testing.T) {
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	for i, p := range []string{"http://a:8080", "http://b:8080", "http://c:8080"} {
		if _, err := pool.Add(ctx, p); err != nil {
			t.Fatalf("Add %s: %v", p, err)
		}
		if i < 2 {
			time.Sleep(2 * time.Millisecond)
		}
	}

	// Acquire 'a' and soft-fail it (ReasonTimeout → increment + delayed score)
	proxy, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	if err := pool.Report(ctx, proxy, ReasonTimeout); err != nil {
		t.Fatalf("Report: %v", err)
	}

	// 'a' has been re-added with a deferred (future) score. Read the ZSET
	// directly to verify the score unit matches the other proxies.
	rdb := redis.NewClient(&redis.Options{Addr: pool.rdb.Options().Addr})
	defer rdb.Close()
	zs, err := rdb.ZRangeWithScores(ctx, "dota2:proxies", 0, -1).Result()
	if err != nil {
		t.Fatalf("ZRangeWithScores: %v", err)
	}
	if len(zs) != 3 {
		t.Fatalf("expected 3 proxies in ZSET, got %d", len(zs))
	}

	// Find the score for 'a' and the max score of b/c.
	var aScore float64
	var maxBC float64
	for _, z := range zs {
		if z.Member == "http://a:8080" {
			aScore = z.Score
		} else {
			if z.Score > maxBC {
				maxBC = z.Score
			}
		}
	}
	if aScore <= maxBC {
		t.Errorf("BUG: soft-failed proxy 'a' has score %v, not > max(b,c)=%v; "+
			"all ZSET scores must use the same time unit", aScore, maxBC)
	}
}

// =============================================================================
// Additional pool tests
// =============================================================================

// waitForZSET polls until pool.Available() returns expected, with a 5s timeout.
func waitForZSET(ctx context.Context, pool *Pool, expected int) error {
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		n, err := pool.Available(ctx)
		if err != nil {
			return err
		}
		if n == int64(expected) {
			return nil
		}
		select {
		case <-time.After(50 * time.Millisecond):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	n, _ := pool.Available(ctx)
	return fmt.Errorf("timed out waiting for ZCARD=%d, last value=%d", expected, n)
}

func TestAcquireLeaseAndZSET(t *testing.T) {
	t.Parallel()
	pool, mr, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	// Seed 1 proxy
	_, err := pool.Add(ctx, "http://proxy-a:8080")
	require.NoError(t, err)

	// Acquire
	proxy, err := pool.Acquire(ctx)
	require.NoError(t, err)
	assert.Equal(t, "http://proxy-a:8080", proxy)

	// Assert lease is set in Redis HASH
	leaseVal := mr.HGet(LeaseKey, proxy)
	assert.NotEmpty(t, leaseVal, "expected lease HGET to return a timestamp")
	// Assert ZCARD=0 (popped from ZSET)
	zcard, err := pool.Available(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(0), zcard, "expected ZSET to be empty after acquire")
}

func TestAcquireSkipsCooldownProxy(t *testing.T) {
	t.Parallel()
	pool, mr, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	// Seed 2 proxies
	_, err := pool.Add(ctx, "http://proxy-a:8080")
	require.NoError(t, err)
	_, err = pool.Add(ctx, "http://proxy-b:8080")
	require.NoError(t, err)

	// SET cooldown key for the first proxy
	mr.Set(CooldownKeyPrefix+"http://proxy-a:8080", "1")

	// Acquire should skip proxy-a (in cooldown) and return proxy-b
	proxy, err := pool.Acquire(ctx)
	require.NoError(t, err)
	assert.Equal(t, "http://proxy-b:8080", proxy)
}

func TestReportHardFailureRemoves(t *testing.T) {
	t.Parallel()
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	// Seed 1 proxy
	_, err := pool.Add(ctx, "http://proxy-a:8080")
	require.NoError(t, err)

	// Acquire
	proxy, err := pool.Acquire(ctx)
	require.NoError(t, err)

	// Report hard failure
	err = pool.Report(ctx, proxy, ReasonHardFailure)
	require.NoError(t, err)

	// Assert ZCARD=0 and HLEN=0
	zcard, err := pool.Available(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(0), zcard, "expected ZCARD=0 after hard failure")

	hlen, err := pool.InUse(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(0), hlen, "expected HLEN=0 after hard failure")

	// Proxy should not exist in pool or lease
	exists, err := pool.Exists(ctx, proxy)
	require.NoError(t, err)
	assert.False(t, exists, "expected proxy to be fully removed")
}

func TestReportTimeoutRemovesAfterThreshold(t *testing.T) {
	t.Parallel()
	pool, mr, _ := newTestPoolDefault(t)
	// Override config pool to use lower threshold
	mr.Close()
	mr2, err := miniredis.Run()
	require.NoError(t, err)

	rdb := redis.NewClient(&redis.Options{Addr: mr2.Addr()})
	pool, err = New(rdb, Config{
		Strategy:          "timestamp",
		SoftFailThreshold: 2, // lower threshold
		CooldownDuration:  1 * time.Minute,
		LeaseDuration:     10 * time.Second,
		SoftRetryDelay:    1 * time.Second,
		FailureCounterTTL: 1 * time.Hour,
	})
	require.NoError(t, err)
	defer func() {
		rdb.Close()
		mr2.Close()
	}()
	ctx := context.Background()

	_, err = pool.Add(ctx, "http://proxy-a:8080")
	require.NoError(t, err)

	// First timeout: should increment counter and return to pool
	proxy, err := pool.Acquire(ctx)
	require.NoError(t, err)
	err = pool.Report(ctx, proxy, ReasonTimeout)
	require.NoError(t, err)

	exists, err := pool.Exists(ctx, proxy)
	require.NoError(t, err)
	assert.True(t, exists, "proxy should still exist after first timeout")

	// Second timeout: threshold exceeded, should remove
	proxy, err = pool.Acquire(ctx)
	require.NoError(t, err)
	err = pool.Report(ctx, proxy, ReasonTimeout)
	require.NoError(t, err)

	exists, err = pool.Exists(ctx, proxy)
	require.NoError(t, err)
	assert.False(t, exists, "proxy should be removed after exceeding soft-fail threshold")

	zcard, err := pool.Available(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(0), zcard, "expected ZCARD=0 after removal")
}

func TestReportRateLimitedCooldown(t *testing.T) {
	t.Parallel()
	pool, mr, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	_, err := pool.Add(ctx, "http://proxy-a:8080")
	require.NoError(t, err)

	proxy, err := pool.Acquire(ctx)
	require.NoError(t, err)

	// Report rate-limited
	err = pool.Report(ctx, proxy, ReasonRateLimited)
	require.NoError(t, err)

	// Assert cooldown key exists
	cooldownKey := CooldownKeyPrefix + proxy
	assert.True(t, mr.Exists(cooldownKey), "expected cooldown key to exist")

	// Assert cooldown key has a TTL
	ttl := mr.TTL(cooldownKey)
	assert.Greater(t, ttl, time.Duration(0), "expected cooldown key to have TTL")
	assert.LessOrEqual(t, ttl, 1*time.Minute, "expected cooldown TTL <= 1m")

	// Assert proxy is back in the ZSET (re-added with deferred score)
	exists, err := pool.Exists(ctx, proxy)
	require.NoError(t, err)
	assert.True(t, exists, "expected proxy to be re-added to pool after cooldown")
}

func TestAcquireWithRateLimitExhaustion(t *testing.T) {
	t.Parallel()
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	_, err := pool.Add(ctx, "http://proxy-a:8080")
	require.NoError(t, err)

	// maxPerMinute=2: we can acquire the same proxy 2 times, 3rd should fail
	proxy, err := pool.AcquireWithRateLimit(ctx, 2)
	require.NoError(t, err)
	assert.Equal(t, "http://proxy-a:8080", proxy)
	err = pool.Release(ctx, proxy)
	require.NoError(t, err)

	proxy, err = pool.AcquireWithRateLimit(ctx, 2)
	require.NoError(t, err)
	assert.Equal(t, "http://proxy-a:8080", proxy)
	err = pool.Release(ctx, proxy)
	require.NoError(t, err)

	// Third call should exceed rate limit
	_, err = pool.AcquireWithRateLimit(ctx, 2)
	assert.ErrorIs(t, err, ErrNoProxyAvailable)
}

func TestTrimRemovesExcess(t *testing.T) {
	t.Parallel()
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	// Seed 5 proxies with distinct scores (small sleep between adds)
	for i := 0; i < 5; i++ {
		p := "http://proxy-" + strconv.Itoa(i) + ":8080"
		_, err := pool.Add(ctx, p)
		require.NoError(t, err)
		if i < 4 {
			time.Sleep(1 * time.Millisecond)
		}
	}

	// Trim to 3
	err := pool.Trim(ctx, 3)
	require.NoError(t, err)

	zcard, err := pool.Available(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(3), zcard, "expected ZCARD=3 after Trim to 3")
}

func TestNowScore(t *testing.T) {
	t.Parallel()
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()

	before := float64(time.Now().UnixMicro())
	got := pool.nowScore()
	after := float64(time.Now().UnixMicro())

	// nowScore must be within [before, after] range
	assert.GreaterOrEqual(t, got, before, "nowScore should be >= before timestamp")
	assert.LessOrEqual(t, got, after, "nowScore should be <= after timestamp")

	// Verify it's in microsecond range (not seconds, not nanoseconds)
	// Current UnixMicro is ~1.7e15, seconds is ~1.7e9
	assert.Greater(t, got, float64(1e12), "nowScore should be in microsecond range (>= 1e12)")
	assert.Less(t, got, float64(1e18), "nowScore should be in microsecond range (< 1e18)")
}

func TestReapExpiredLeasesRestoresToPool(t *testing.T) {
	t.Parallel()
	pool, mr, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	// Seed 1 proxy and acquire it
	_, err := pool.Add(ctx, "http://proxy-a:8080")
	require.NoError(t, err)

	proxy, err := pool.Acquire(ctx)
	require.NoError(t, err)

	// Manually backdate the lease entry to 1 hour ago
	past := strconv.FormatInt(time.Now().Add(-1*time.Hour).Unix(), 10)
	mr.HSet(LeaseKey, proxy, past)

	// Reap
	reaped, err := pool.ReapExpiredLeases(ctx)
	require.NoError(t, err)
	assert.Equal(t, 1, reaped)

	// Assert proxy back in ZSET
	exists, err := pool.Exists(ctx, proxy)
	require.NoError(t, err)
	assert.True(t, exists, "expected proxy to be restored to pool after reap")

	// Assert lease cleared
	hlen, err := pool.InUse(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(0), hlen, "expected HLEN=0 after reap")

	// Assert ZCARD=1
	zcard, err := pool.Available(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(1), zcard)
}

// waitFor polls pool.Available until it reaches the expected value.
// Used in tests that need to wait for Redis state to converge.
func waitFor(t *testing.T, ctx context.Context, pool *Pool, expected int) {
	t.Helper()
	for i := 0; i < 50; i++ {
		n, err := pool.Available(ctx)
		if err == nil && n == int64(expected) {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestAcquire seeds 1 proxy and verifies Acquire returns it with a lease.
func TestAcquire(t *testing.T) {
	t.Parallel()
	pool, mr, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	_, err := pool.Add(ctx, "http://proxy-a:8080")
	require.NoError(t, err)

	proxy, err := pool.Acquire(ctx)
	require.NoError(t, err)
	assert.Equal(t, "http://proxy-a:8080", proxy)

	// Lease exists (HGET returns non-empty)
	leaseVal := mr.HGet(LeaseKey, proxy)
	assert.NotEmpty(t, leaseVal, "expected lease HGET to return a timestamp")

	// ZCARD=0 (popped from ZSET)
	zcard, err := pool.Available(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(0), zcard, "expected ZSET to be empty after acquire")
}

// TestAcquireSkipsCooldown verifies Acquire skips a cooldown proxy.
func TestAcquireSkipsCooldown(t *testing.T) {
	t.Parallel()
	pool, mr, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	// Seed 2 proxies
	_, err := pool.Add(ctx, "http://proxy-a:8080")
	require.NoError(t, err)
	_, err = pool.Add(ctx, "http://proxy-b:8080")
	require.NoError(t, err)

	// SET cooldown key for the first proxy
	mr.Set(CooldownKeyPrefix+"http://proxy-a:8080", "1")

	// Acquire should skip proxy-a (in cooldown) and return proxy-b
	proxy, err := pool.Acquire(ctx)
	require.NoError(t, err)
	assert.Equal(t, "http://proxy-b:8080", proxy)
}

// TestAcquireReturnsErrNoProxyAvailable verifies Acquire fails on empty pool.
func TestAcquireReturnsErrNoProxyAvailable(t *testing.T) {
	t.Parallel()
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	_, err := pool.Acquire(ctx)
	assert.ErrorIs(t, err, ErrNoProxyAvailable)
}

// TestReleaseReturnsProxy verifies Release returns a proxy to the ZSET.
func TestReleaseReturnsProxy(t *testing.T) {
	t.Parallel()
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	_, err := pool.Add(ctx, "http://proxy-a:8080")
	require.NoError(t, err)

	proxy, err := pool.Acquire(ctx)
	require.NoError(t, err)
	assert.Equal(t, "http://proxy-a:8080", proxy)

	// Release back
	err = pool.Release(ctx, proxy)
	require.NoError(t, err)

	// ZCARD should be back to 1
	avail, err := pool.Available(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(1), avail, "expected ZCARD=1 after release")
}

// TestAcquireWithRateLimit verifies AcquireWithRateLimit exhausts after maxPerMinute.
func TestAcquireWithRateLimit(t *testing.T) {
	t.Parallel()
	pool, _, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	_, err := pool.Add(ctx, "http://proxy-a:8080")
	require.NoError(t, err)

	// maxPerMinute=2: we can acquire the same proxy 2 times, 3rd should fail
	proxy, err := pool.AcquireWithRateLimit(ctx, 2)
	require.NoError(t, err)
	assert.Equal(t, "http://proxy-a:8080", proxy)
	err = pool.Release(ctx, proxy)
	require.NoError(t, err)

	proxy, err = pool.AcquireWithRateLimit(ctx, 2)
	require.NoError(t, err)
	assert.Equal(t, "http://proxy-a:8080", proxy)
	err = pool.Release(ctx, proxy)
	require.NoError(t, err)

	// Third call should exceed rate limit
	_, err = pool.AcquireWithRateLimit(ctx, 2)
	assert.ErrorIs(t, err, ErrNoProxyAvailable)
}

// TestReapExpiredLeases verifies expired leases are returned to the ZSET.
func TestReapExpiredLeases(t *testing.T) {
	t.Parallel()
	pool, mr, cleanup := newTestPoolDefault(t)
	defer cleanup()
	ctx := context.Background()

	// Seed 1 proxy and acquire it
	_, err := pool.Add(ctx, "http://proxy-a:8080")
	require.NoError(t, err)

	proxy, err := pool.Acquire(ctx)
	require.NoError(t, err)

	// Manually backdate the lease entry to 1 hour ago
	past := strconv.FormatInt(time.Now().Add(-1*time.Hour).Unix(), 10)
	mr.HSet(LeaseKey, proxy, past)

	// Reap
	reaped, err := pool.ReapExpiredLeases(ctx)
	require.NoError(t, err)
	assert.Equal(t, 1, reaped)

	// Assert proxy back in ZSET
	exists, err := pool.Exists(ctx, proxy)
	require.NoError(t, err)
	assert.True(t, exists, "expected proxy to be restored to pool after reap")

	// Assert lease cleared
	hlen, err := pool.InUse(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(0), hlen, "expected HLEN=0 after reap")

	// Assert ZCARD=1
	zcard, err := pool.Available(ctx)
	require.NoError(t, err)
	assert.Equal(t, int64(1), zcard)
}
