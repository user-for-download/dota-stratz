package validator

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"sync/atomic"
	"testing"
	"time"

	"github.com/dota-stratz/shared/go-common/logger"
)

func TestMain(m *testing.M) {
	_ = os.Setenv("LOG_LEVEL", "error")
	logger.InitLogger()
	os.Exit(m.Run())
}

// sinkSpy records all results delivered to a Sink for assertions.
// Fields are atomic to avoid data races in the concurrent validator.
type sinkSpy struct {
	total  atomic.Int64
	ok     atomic.Int64
	failed atomic.Int64
}

// newTestServer creates an httptest server that responds with the given status and body.
func newTestServer(status int, body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
}

func TestValidateStream(t *testing.T) {
	ts := newTestServer(http.StatusOK, "OK")
	defer ts.Close()

	v, err := New(ts.URL, "test-agent", 5, 5)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Use the target server's URL as the "proxy" address so the request
	// goes through successfully — the transport connects to ts via its own
	// URL, sends the absolute target URL, and ts responds with 200.
	proxies := []string{ts.URL}

	spy := &sinkSpy{}
	stats := v.ValidateStream(context.Background(), proxies, func(ctx context.Context, r Result) {
		spy.total.Add(1)
		if r.OK {
			spy.ok.Add(1)
		} else {
			spy.failed.Add(1)
		}
	})

	if stats.Total != 1 {
		t.Fatalf("expected 1 total, got %d", stats.Total)
	}
	if stats.OK != 1 {
		t.Fatalf("expected 1 ok, got %d", stats.OK)
	}
	if stats.Failed != 0 {
		t.Fatalf("expected 0 failed, got %d", stats.Failed)
	}
	if spy.total.Load() != 1 {
		t.Fatalf("expected sink 1 total, got %d", spy.total.Load())
	}
	if spy.ok.Load() != 1 {
		t.Fatalf("expected sink 1 ok, got %d", spy.ok.Load())
	}
	if stats.Elapsed <= 0 {
		t.Fatal("expected positive elapsed duration")
	}
	// Note: AvgLatMs is NOT asserted here because on fast hardware the
	// single-proxy round-trip may complete in <1ms, rounding to 0.
	if rate := stats.SuccessRate(); rate != 1.0 {
		t.Fatalf("expected 100%% success rate, got %f", rate)
	}
}

func TestValidateStreamFailures(t *testing.T) {
	// A target that returns 500
	ts := newTestServer(http.StatusInternalServerError, "fail")
	defer ts.Close()

	v, err := New(ts.URL, "test-agent", 5, 5)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	proxies := []string{ts.URL, ts.URL, ts.URL}

	spy := &sinkSpy{}
	stats := v.ValidateStream(context.Background(), proxies, func(ctx context.Context, r Result) {
		spy.total.Add(1)
		if r.OK {
			spy.ok.Add(1)
		} else {
			spy.failed.Add(1)
		}
	})

	if stats.Total != 3 {
		t.Fatalf("expected 3 total, got %d", stats.Total)
	}
	if stats.OK != 0 {
		t.Fatalf("expected 0 ok (all 500), got %d", stats.OK)
	}
	if stats.Failed != 3 {
		t.Fatalf("expected 3 failed, got %d", stats.Failed)
	}
	if spy.failed.Load() != 3 {
		t.Fatalf("expected sink 3 failed, got %d", spy.failed.Load())
	}
}

func TestValidateStreamMixed(t *testing.T) {
	// Two servers: one returning 200, one returning 500
	okSrv := newTestServer(http.StatusOK, "OK")
	defer okSrv.Close()
	failSrv := newTestServer(http.StatusInternalServerError, "fail")
	defer failSrv.Close()

	v, err := New(okSrv.URL, "test-agent", 5, 5)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// First 5 "proxies" connect through the OK server, last 5 through the fail server
	proxies := make([]string, 10)
	for i := 0; i < 5; i++ {
		proxies[i] = okSrv.URL
		proxies[i+5] = failSrv.URL
	}

	spy := &sinkSpy{}
	stats := v.ValidateStream(context.Background(), proxies, func(ctx context.Context, r Result) {
		spy.total.Add(1)
		if r.OK {
			spy.ok.Add(1)
		} else {
			spy.failed.Add(1)
		}
	})

	if stats.Total != 10 {
		t.Fatalf("expected 10 total, got %d", stats.Total)
	}
	if stats.OK != 5 {
		t.Fatalf("expected 5 ok, got %d", stats.OK)
	}
	if stats.Failed != 5 {
		t.Fatalf("expected 5 failed, got %d", stats.Failed)
	}
}

func TestValidateStreamContextCancel(t *testing.T) {
	// Slow server that hangs long enough for us to cancel.
	// The handler listens on r.Context().Done() so it unblocks immediately
	// when the client disconnects, preventing ts.Close() from blocking.
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(5 * time.Second):
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("OK"))
		case <-r.Context().Done():
		}
	}))
	defer ts.Close()

	v, err := New(ts.URL, "test-agent", 10, 2)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	proxies := []string{ts.URL, ts.URL, ts.URL}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	stats := v.ValidateStream(ctx, proxies, func(ctx context.Context, r Result) {})

	// Should have partial stats (some proxies may have been processed or not)
	// Key assertion: not an empty Stats{} with zero elapsed
	if stats.Total == 0 && stats.Elapsed == 0 {
		t.Fatal("expected partial stats on cancel, got empty Stats{}")
	}
	if stats.Elapsed <= 0 {
		t.Fatal("expected positive elapsed on cancelled run")
	}

	t.Logf("Cancelled run: total=%d ok=%d failed=%d elapsed=%v",
		stats.Total, stats.OK, stats.Failed, stats.Elapsed)
}

func TestValidateStreamEmpty(t *testing.T) {
	ts := newTestServer(http.StatusOK, "OK")
	defer ts.Close()

	v, err := New(ts.URL, "test-agent", 5, 5)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	stats := v.ValidateStream(context.Background(), nil, func(ctx context.Context, r Result) {})

	if stats.Total != 0 {
		t.Fatalf("expected 0 total for empty input, got %d", stats.Total)
	}
	if stats.SuccessRate() != 0 {
		t.Fatalf("expected 0 success rate for empty input, got %f", stats.SuccessRate())
	}
}

func TestNewValidatorValidation(t *testing.T) {
	tests := []struct {
		name        string
		targetURL   string
		userAgent   string
		timeout     int
		concurrency int
		wantErr     bool
	}{
		{"valid", "http://example.com", "agent", 5, 3, false},
		{"empty URL", "", "agent", 5, 3, true},
		{"invalid URL (space in host)", "http://a b.com", "agent", 5, 3, true},
		{"zero timeout", "http://example.com", "agent", 0, 3, true},
		{"negative timeout", "http://example.com", "agent", -1, 3, true},
		{"zero concurrency", "http://example.com", "agent", 5, 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := New(tt.targetURL, tt.userAgent, tt.timeout, tt.concurrency)
			if (err != nil) != tt.wantErr {
				t.Fatalf("New() err=%v, wantErr=%v", err, tt.wantErr)
			}
		})
	}
}
