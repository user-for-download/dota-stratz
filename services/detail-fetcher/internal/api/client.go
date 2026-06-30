package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/dota-stratz/services/detail-fetcher/internal/metrics"
	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/dota-stratz/shared/go-common/proxypool"
	"go.uber.org/zap"
)

var ErrMatchNotFound = errors.New("match not found or private")

// proxyPool is the subset of *proxypool.Pool used by Client. Defined as an
// interface so tests can inject deterministic fakes without a Redis-backed
// proxy pool.
type proxyPool interface {
	AcquireWithRateLimit(ctx context.Context, maxPerMin int) (string, error)
	Report(ctx context.Context, proxy string, reason proxypool.FailureReason) error
	ReportSuccess(ctx context.Context, proxy string) error
	Release(ctx context.Context, proxy string) error
}

// Compile-time check that *proxypool.Pool satisfies proxyPool.
var _ proxyPool = (*proxypool.Pool)(nil)

type Client struct {
	baseURL      string
	pool         proxyPool
	timeout      time.Duration
	maxReqPerMin int
	userAgent    string
}

func NewClient(baseURL string, pool proxyPool, timeoutSec, maxReqPerMin int, userAgent string) *Client {
	return &Client{
		baseURL:      baseURL,
		pool:         pool,
		timeout:      time.Duration(timeoutSec) * time.Second,
		maxReqPerMin: maxReqPerMin,
		userAgent:    userAgent,
	}
}

func (c *Client) FetchRaw(ctx context.Context, matchID int64) ([]byte, error) {
	start := time.Now()
	defer func() {
		metrics.FetchDuration.Observe(time.Since(start).Seconds())
	}()

	reqURL := fmt.Sprintf("%s/%d", c.baseURL, matchID)
	const maxBodyBytes = 16 * 1024 * 1024

	proxy, err := c.pool.AcquireWithRateLimit(ctx, c.maxReqPerMin)
	if err != nil {
		return nil, fmt.Errorf("no proxy available: %w", err)
	}

	transport, transportErr := proxypool.MakeTransport(proxy, c.timeout)
	if transportErr != nil {
		_ = c.pool.Report(ctx, proxy, proxypool.ReasonHardFailure)
		_ = c.pool.Release(ctx, proxy)
		return nil, transportErr
	}
	defer transport.CloseIdleConnections()

	client := &http.Client{
		Transport: transport,
		Timeout:   c.timeout,
	}

	req, reqErr := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if reqErr != nil {
		_ = c.pool.Release(ctx, proxy)
		return nil, reqErr
	}
	req.Header.Set("User-Agent", c.userAgent)

	resp, err := client.Do(req)
	if err != nil {
		reason, shouldReport := proxypool.ClassifyError(err, resp)
		if shouldReport {
			_ = c.pool.Report(ctx, proxy, reason)
		} else {
			_ = c.pool.Release(ctx, proxy)
		}
		return nil, fmt.Errorf("fetch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusTooManyRequests {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = c.pool.Report(ctx, proxy, proxypool.ReasonRateLimited)
		return nil, fmt.Errorf("rate limited on proxy %s", proxy)
	}

	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, resp.Body)
		reason, shouldReport := proxypool.ClassifyError(nil, resp)
		if shouldReport {
			_ = c.pool.Report(ctx, proxy, reason)
		} else {
			_ = c.pool.Release(ctx, proxy)
		}
		return nil, fmt.Errorf("non-200 status: %d", resp.StatusCode)
	}

	body, readErr := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes+1))
	if readErr != nil {
		_ = c.pool.Report(ctx, proxy, proxypool.ReasonHardFailure)
		return nil, fmt.Errorf("body read failed: %w", readErr)
	}
	if len(body) > maxBodyBytes {
		_ = c.pool.Report(ctx, proxy, proxypool.ReasonHardFailure)
		return nil, fmt.Errorf("response body exceeds %d bytes (proxy %s)", maxBodyBytes, proxy)
	}

	var rawCheck json.RawMessage
	if err := json.Unmarshal(body, &rawCheck); err != nil {
		_ = c.pool.Report(ctx, proxy, proxypool.ReasonHardFailure)
		logger.Log.Warn("OpenDota returned invalid JSON",
			zap.Int64("match_id", matchID),
			zap.Error(err))
		return nil, fmt.Errorf("invalid JSON response: %w", err)
	}

	var errResp struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(body, &errResp) == nil && errResp.Error != "" {
		logger.Log.Debug("OpenDota returned API error",
			zap.Int64("match_id", matchID),
			zap.String("error", errResp.Error))
		if errResp.Error == "Match ID not found" || errResp.Error == "private match" {
			_ = c.pool.Release(ctx, proxy)
			return nil, ErrMatchNotFound
		}
		// Unknown API error (not "not found" or "private") — the proxy
		// itself worked fine, so report success to reset its failure
		// counter, then release (BUG-008).
		_ = c.pool.ReportSuccess(ctx, proxy)
		_ = c.pool.Release(ctx, proxy)
		return nil, fmt.Errorf("opendota error: %s", errResp.Error)
	}

	_ = c.pool.ReportSuccess(ctx, proxy)
	_ = c.pool.Release(ctx, proxy)
	return body, nil
}

// FetchRawDirect fetches match data from OpenDota directly (no proxy).
// Used as a fallback when all proxy-based retries are exhausted.
func (c *Client) FetchRawDirect(ctx context.Context, matchID int64) ([]byte, error) {
	start := time.Now()
	defer func() {
		metrics.FetchDuration.Observe(time.Since(start).Seconds())
	}()

	reqURL := fmt.Sprintf("%s/%d", c.baseURL, matchID)
	const maxBodyBytes = 16 * 1024 * 1024

	client := &http.Client{
		Timeout: c.timeout,
	}

	req, reqErr := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if reqErr != nil {
		return nil, reqErr
	}
	req.Header.Set("User-Agent", c.userAgent)

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("direct fetch failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusTooManyRequests {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil, fmt.Errorf("rate limited on direct connection")
	}

	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil, fmt.Errorf("non-200 status on direct: %d", resp.StatusCode)
	}

	body, readErr := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes+1))
	if readErr != nil {
		return nil, fmt.Errorf("direct body read failed: %w", readErr)
	}
	if len(body) > maxBodyBytes {
		return nil, fmt.Errorf("response body exceeds %d bytes (direct)", maxBodyBytes)
	}

	var rawCheck json.RawMessage
	if err := json.Unmarshal(body, &rawCheck); err != nil {
		logger.Log.Warn("OpenDota returned invalid JSON (direct)",
			zap.Int64("match_id", matchID),
			zap.Error(err))
		return nil, fmt.Errorf("invalid JSON response (direct): %w", err)
	}

	var errResp struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(body, &errResp) == nil && errResp.Error != "" {
		logger.Log.Debug("OpenDota returned API error (direct)",
			zap.Int64("match_id", matchID),
			zap.String("error", errResp.Error))
		if errResp.Error == "Match ID not found" || errResp.Error == "private match" {
			return nil, ErrMatchNotFound
		}
		return nil, fmt.Errorf("opendota error (direct): %s", errResp.Error)
	}

	return body, nil
}
