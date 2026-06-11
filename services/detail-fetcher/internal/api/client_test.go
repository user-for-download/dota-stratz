package api

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/dota-stratz/shared/go-common/proxypool"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

func init() {
	logger.Log = zap.NewNop()
}

// --- fake pool ---

type fakePool struct {
	t               *testing.T
	proxyURL        string
	acquiredProxy   string
	reportedReason  proxypool.FailureReason
	reportedSuccess bool
	released        bool
	mu              sync.Mutex
}

func (p *fakePool) AcquireWithRateLimit(ctx context.Context, maxPerMin int) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.acquiredProxy = p.proxyURL
	return p.proxyURL, nil
}

func (p *fakePool) Report(ctx context.Context, proxy string, reason proxypool.FailureReason) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.reportedReason = reason
	return nil
}

func (p *fakePool) ReportSuccess(ctx context.Context, proxy string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.reportedSuccess = true
	return nil
}

func (p *fakePool) Release(ctx context.Context, proxy string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.released = true
	return nil
}

func (p *fakePool) reset() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.acquiredProxy = ""
	p.reportedReason = ""
	p.reportedSuccess = false
	p.released = false
}

// newTestClient creates a Client with a fake pool pointed at the given
// httptest server. The fake pool returns the httptest server's address as
// the proxy URL so that the HTTP client routes all requests through the
// httptest server (acting as its own HTTP forward proxy).
func newTestClient(t *testing.T, ts *httptest.Server, pool *fakePool) *Client {
	t.Helper()
	pool.proxyURL = "http://" + ts.Listener.Addr().String()
	pool.reset()
	return &Client{
		baseURL:      ts.URL,
		pool:         pool,
		timeout:      5 * time.Second,
		maxReqPerMin: 10,
		userAgent:    "test-agent",
	}
}

// --- tests ---

func TestFetchRaw_200OK(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"match_id":123}`)
	}))
	defer ts.Close()

	pool := &fakePool{t: t}
	client := newTestClient(t, ts, pool)

	body, err := client.FetchRaw(context.Background(), 123)
	assert.NoError(t, err)
	assert.JSONEq(t, `{"match_id":123}`, string(body))

	pool.mu.Lock()
	assert.True(t, pool.reportedSuccess, "ReportSuccess should be called on 200 OK")
	assert.True(t, pool.released, "Release should be called on 200 OK")
	assert.Empty(t, pool.reportedReason, "Report should NOT be called on 200 OK")
	pool.mu.Unlock()
}

func TestFetchRaw_InvalidJSON(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `not-json`)
	}))
	defer ts.Close()

	pool := &fakePool{t: t}
	client := newTestClient(t, ts, pool)

	body, err := client.FetchRaw(context.Background(), 123)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid JSON response")
	assert.Nil(t, body)

	pool.mu.Lock()
	assert.Equal(t, proxypool.ReasonHardFailure, pool.reportedReason,
		"Report(ReasonHardFailure) should be called for invalid JSON")
	assert.False(t, pool.reportedSuccess, "ReportSuccess should NOT be called")
	assert.False(t, pool.released, "Release should NOT be called — Report already handled the proxy")
	pool.mu.Unlock()
}

func TestFetchRaw_MatchNotFound(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"error":"Match ID not found"}`)
	}))
	defer ts.Close()

	pool := &fakePool{t: t}
	client := newTestClient(t, ts, pool)

	body, err := client.FetchRaw(context.Background(), 123)
	assert.ErrorIs(t, err, ErrMatchNotFound)
	assert.Nil(t, body)

	pool.mu.Lock()
	assert.True(t, pool.released, "Release should be called")
	assert.False(t, pool.reportedSuccess, "ReportSuccess should NOT be called")
	assert.Empty(t, pool.reportedReason, "Report should NOT be called")
	pool.mu.Unlock()
}

func TestFetchRaw_UnknownAPIError_BUG008(t *testing.T) {
	// REGRESSION: for unknown API errors (not "Match ID not found" or
	// "private match"), the client must call ReportSuccess BEFORE Release
	// so the proxy's failure counter does not accumulate incorrectly.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"error":"some other error"}`)
	}))
	defer ts.Close()

	pool := &fakePool{t: t}
	client := newTestClient(t, ts, pool)

	body, err := client.FetchRaw(context.Background(), 123)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "opendota error: some other error")
	assert.Nil(t, body)

	pool.mu.Lock()
	assert.True(t, pool.reportedSuccess, "BUG-008: ReportSuccess MUST be called before Release"+
		" for unknown API errors so proxy failure counters are not incorrectly incremented")
	assert.True(t, pool.released, "Release should be called after ReportSuccess")
	assert.Empty(t, pool.reportedReason, "Report should NOT be called")
	pool.mu.Unlock()
}

func TestFetchRaw_429(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer ts.Close()

	pool := &fakePool{t: t}
	client := newTestClient(t, ts, pool)

	body, err := client.FetchRaw(context.Background(), 123)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "rate limited")
	assert.Nil(t, body)

	pool.mu.Lock()
	assert.Equal(t, proxypool.ReasonRateLimited, pool.reportedReason,
		"Report(ReasonRateLimited) should be called on 429")
	assert.False(t, pool.reportedSuccess, "ReportSuccess should NOT be called")
	assert.False(t, pool.released, "Release should NOT be called — Report already handled the proxy")
	pool.mu.Unlock()
}

func TestFetchRaw_500(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer ts.Close()

	pool := &fakePool{t: t}
	client := newTestClient(t, ts, pool)

	body, err := client.FetchRaw(context.Background(), 123)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "non-200 status: 500")
	assert.Nil(t, body)

	pool.mu.Lock()
	assert.Equal(t, proxypool.ReasonBadStatus, pool.reportedReason,
		"Report(ReasonBadStatus) should be called on 5xx")
	assert.False(t, pool.reportedSuccess, "ReportSuccess should NOT be called")
	assert.False(t, pool.released, "Release should NOT be called — Report already handled the proxy")
	pool.mu.Unlock()
}

func TestFetchRaw_BodyTooLarge(t *testing.T) {
	const seventeenMB = 17 * 1024 * 1024
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		// Write more than the 16 MiB client limit.
		data := make([]byte, seventeenMB)
		for i := range data {
			data[i] = 'A'
		}
		w.Write(data)
	}))
	defer ts.Close()

	pool := &fakePool{t: t}
	client := newTestClient(t, ts, pool)

	body, err := client.FetchRaw(context.Background(), 123)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "response body exceeds")
	assert.Nil(t, body)

	pool.mu.Lock()
	assert.Equal(t, proxypool.ReasonHardFailure, pool.reportedReason,
		"Report(ReasonHardFailure) should be called when body exceeds limit")
	assert.False(t, pool.reportedSuccess, "ReportSuccess should NOT be called")
	assert.False(t, pool.released, "Release should NOT be called — Report already handled the proxy")
	pool.mu.Unlock()
}


