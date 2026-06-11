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

type Client struct {
	baseURL      string
	pool         *proxypool.Pool
	timeout      time.Duration
	maxReqPerMin int
	userAgent    string
}

func NewClient(baseURL string, pool *proxypool.Pool, timeoutSec, maxReqPerMin int, userAgent string) *Client {
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
		return nil, transportErr
	}

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
		_ = c.pool.Report(ctx, proxy, proxypool.ReasonRateLimited)
		return nil, fmt.Errorf("rate limited on proxy %s", proxy)
	}

	if resp.StatusCode != http.StatusOK {
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
