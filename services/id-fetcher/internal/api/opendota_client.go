package api

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/dota-stratz/services/id-fetcher/internal/metrics"
	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/dota-stratz/shared/go-common/proxypool"
	"go.uber.org/zap"
)

//go:embed matches.sql
var fetchMatchesQuery string

//go:embed matches_watermark.sql
var fetchMatchesWatermarkQuery string

// OpenDotaClient executes SQL queries against the OpenDota Explorer API
// through the proxy pool for automatic IP rotation on rate-limits and bans.
type OpenDotaClient struct {
	url        string
	pool       *proxypool.Pool
	lastNDays  int
	lobbyTypes []int
}

func NewOpenDotaClient(apiURL string, pool *proxypool.Pool, lastNDays int, lobbyTypes []int) *OpenDotaClient {
	return &OpenDotaClient{
		url:        apiURL,
		pool:       pool,
		lastNDays:  lastNDays,
		lobbyTypes: lobbyTypes,
	}
}

// MatchNode represents a single row from the OpenDota Explorer results.
type MatchNode struct {
	MatchID   int64 `json:"match_id"`
	StartTime int64 `json:"start_time"`
}

// ExplorerResponse wraps the JSON structure returned by OpenDota Explorer.
type ExplorerResponse struct {
	Err      *string     `json:"err"`
	RowCount int         `json:"rowCount"`
	Rows     []MatchNode `json:"rows"`
}

// FetchMatches executes a single SQL query against OpenDota Explorer that
// returns all matches from the last FETCH_LAST_COUNT_DAY days whose
// lobby_type is in FETCH_LOBBY_TYPES. No pagination — one request, all
// results.
//
// Retries indefinitely as long as the pool has available proxies, stopping
// only when:
//   - the request succeeds
//   - the context is cancelled (shutdown / parent timeout)
//   - the pool is exhausted (ErrNoProxyAvailable)
//
// Each failed attempt rotates to a fresh proxy automatically via WithProxy.
// Backoff is capped at 8s so we don't stall longer than one proxy-manager
// validation cycle (~10s).
//
// Each callback defers transport.CloseIdleConnections() so OS file
// descriptors are released regardless of outcome (fixes the "Too many open
// connections" error caused by leaking http.Transport instances).
func (c *OpenDotaClient) FetchMatches(ctx context.Context) ([]MatchNode, error) {
	// Build lobby_type IN (...) list from config, e.g. "1,2,6"
	parts := make([]string, len(c.lobbyTypes))
	for i, lt := range c.lobbyTypes {
		parts[i] = fmt.Sprintf("%d", lt)
	}
	lobbyList := strings.Join(parts, ",")

	query := fmt.Sprintf(fetchMatchesQuery, c.lastNDays, lobbyList)
	reqURL := c.url + "?sql=" + url.QueryEscape(query)

	logger.Log.Info("OpenDota: executing fetch query",
		zap.Int("last_n_days", c.lastNDays),
		zap.String("lobby_types", lobbyList))

	body, err := c.executeWithRetry(ctx, reqURL)
	if err != nil {
		return nil, err
	}
	return c.parseResponse(body)
}

// parseResponse decodes an OpenDota Explorer JSON body and validates the
// `err` field.
func (c *OpenDotaClient) parseResponse(body []byte) ([]MatchNode, error) {
	var response ExplorerResponse
	if err := json.Unmarshal(body, &response); err != nil {
		metrics.APICallsTotal.WithLabelValues("error").Inc()
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	if response.Err != nil {
		metrics.APICallsTotal.WithLabelValues("error").Inc()
		return nil, fmt.Errorf("opendota sql error: %s", *response.Err)
	}

	metrics.APICallsTotal.WithLabelValues("ok").Inc()
	logger.Log.Info("OpenDota: query returned results",
		zap.Int("row_count", len(response.Rows)))

	return response.Rows, nil
}

// FetchMatchesSince fetches matches with match_id > watermark, returning
// at most maxResults rows (newest first by match_id).
//
// OpenDota's public Explorer API does not support `match_id > X` as a
// server-side filter, so this method issues the watermark-based query
// (matches_watermark.sql) against a wider lookback window
// (lookbackDays days, ≥ FetchLastCountDay) and filters + truncates
// the result in Go. The query is sorted by match_id DESC so the Go
// filter can stop as soon as the cursor drops below the watermark
// (avoiding scanning the full result set).
//
// Parameters:
//   - ctx:              cancelled on graceful shutdown or per-request
//     timeout
//   - watermark:        the parser's last_parsed_match_id; only matches
//     with strictly greater IDs are returned
//   - lookbackDays:     rolling window in days (must be > 0); wider
//     than the bootstrap path to bound the result set
//     while still covering periods of low traffic
//   - maxResults:       hard cap on returned rows (caller passes e.g.
//     5 × batch_size to allow for filtering slack)
//
// Returns the filtered, truncated slice (already sorted by match_id
// DESC). Retries on transient errors exactly like FetchMatches — the
// proxy rotation, capped backoff, and pool-exhaustion handling are
// identical.
func (c *OpenDotaClient) FetchMatchesSince(
	ctx context.Context,
	watermark int64,
	lookbackDays int,
	maxResults int,
) ([]MatchNode, error) {
	if watermark <= 0 {
		return nil, fmt.Errorf("FetchMatchesSince: watermark must be > 0, got %d", watermark)
	}
	if lookbackDays <= 0 {
		return nil, fmt.Errorf("FetchMatchesSince: lookbackDays must be > 0, got %d", lookbackDays)
	}
	if maxResults <= 0 {
		return nil, fmt.Errorf("FetchMatchesSince: maxResults must be > 0, got %d", maxResults)
	}

	// Build lobby_type IN (...) list from config, e.g. "1,2,6".
	parts := make([]string, len(c.lobbyTypes))
	for i, lt := range c.lobbyTypes {
		parts[i] = fmt.Sprintf("%d", lt)
	}
	lobbyList := strings.Join(parts, ",")

	query := fmt.Sprintf(fetchMatchesWatermarkQuery, lookbackDays, lobbyList)
	reqURL := c.url + "?sql=" + url.QueryEscape(query)

	logger.Log.Info("OpenDota: executing watermark fetch query",
		zap.Int64("watermark", watermark),
		zap.Int("lookback_days", lookbackDays),
		zap.Int("max_results", maxResults),
		zap.String("lobby_types", lobbyList))

	body, err := c.executeWithRetry(ctx, reqURL)
	if err != nil {
		return nil, err
	}

	all, err := c.parseResponse(body)
	if err != nil {
		return nil, err
	}

	// Filter + truncate. The query is ordered by match_id DESC (see
	// matches_watermark.sql), so we could stop as soon as we see a row
	// at or below the watermark. However, if the ordering guarantee is
	// ever compromised (e.g. schema change in matches_watermark.sql),
	// a break would silently drop valid newer matches that appear after
	// an older one. Use continue for defensive correctness — the O(n)
	// cost is negligible over a LIMIT 50000 result set.
	filtered := make([]MatchNode, 0, min(len(all), maxResults))
	for _, m := range all {
		if m.MatchID <= watermark {
			continue
		}
		filtered = append(filtered, m)
		if len(filtered) >= maxResults {
			break
		}
	}

	logger.Log.Info("OpenDota: watermark filter applied",
		zap.Int("rows_before_filter", len(all)),
		zap.Int("rows_after_filter", len(filtered)),
		zap.Int64("watermark", watermark))

	return filtered, nil
}

// executeWithRetry wraps the proxy-rotation retry loop so both
// FetchMatches (rolling window) and FetchMatchesSince (watermark) can
// share the exact same proxy/backoff/cancellation semantics.
func (c *OpenDotaClient) executeWithRetry(ctx context.Context, reqURL string) ([]byte, error) {
	attempt := 0
	for {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		attempt++
		body, err := c.pool.WithProxy(ctx, func(proxyStr string) (*http.Response, error) {
			transport, err := proxypool.MakeTransport(proxyStr, 15*time.Second)
			if err != nil {
				return nil, err
			}
			defer transport.CloseIdleConnections()

			client := &http.Client{
				Transport: transport,
				Timeout:   60 * time.Second,
			}

			req, reqErr := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
			if reqErr != nil {
				return nil, reqErr
			}
			req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

			return client.Do(req)
		})

		if err == nil {
			if attempt > 1 {
				logger.Log.Info("OpenDota fetch succeeded after retries",
					zap.Int("attempts", attempt))
			}
			return body, nil
		}

		if errors.Is(err, proxypool.ErrNoProxyAvailable) {
			metrics.APICallsTotal.WithLabelValues("error").Inc()
			return nil, fmt.Errorf("proxy pool exhausted after %d attempt(s): %w", attempt, err)
		}

		// Only exit on caller cancellation — the per-request HTTP
		// Client.Timeout also produces context.DeadlineExceeded but
		// that should rotate to a fresh proxy, not stop retrying.
		// The caller's deadline is already checked at the top of the
		// loop and in the backoff select below.
		if errors.Is(err, context.Canceled) {
			return nil, err
		}

		backoff := time.Duration(1<<min(attempt-1, 3)) * time.Second // 1s,2s,4s,8s,8s,...
		logger.Log.Warn("OpenDota fetch attempt failed, retrying with next proxy",
			zap.Int("attempt", attempt),
			zap.Duration("backoff", backoff),
			zap.Error(err))

		select {
		case <-time.After(backoff):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}
