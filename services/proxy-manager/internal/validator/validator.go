package validator

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/dota-stratz/shared/go-common/proxypool"
	"go.uber.org/zap"
)

type Result struct {
	Proxy     string
	OK        bool
	LatencyMs int64
	Err       error
}

// Sink is called for every result as it completes. It MUST be safe for
// concurrent use, and MUST NOT block — slow sinks will throttle validation.
type Sink func(ctx context.Context, r Result)

type Validator struct {
	targetURL   string
	userAgent   string
	timeout     time.Duration
	concurrency int
}

func New(targetURL, userAgent string, timeoutSec, concurrency int) (*Validator, error) {
	if targetURL == "" {
		return nil, fmt.Errorf("targetURL required")
	}
	if _, err := url.Parse(targetURL); err != nil {
		return nil, fmt.Errorf("invalid targetURL: %w", err)
	}
	if timeoutSec <= 0 {
		return nil, fmt.Errorf("timeoutSec must be > 0")
	}
	if concurrency <= 0 {
		return nil, fmt.Errorf("concurrency must be > 0")
	}
	return &Validator{
		targetURL:   targetURL,
		userAgent:   userAgent,
		timeout:     time.Duration(timeoutSec) * time.Second,
		concurrency: concurrency,
	}, nil
}

// ValidateStream runs a fully concurrent, streaming validation over `proxies`.
func (v *Validator) ValidateStream(ctx context.Context, proxies []string, sink Sink) Stats {
	if len(proxies) == 0 {
		return Stats{}
	}

	start := time.Now()
	defer func() {
		proxypool.ValidationDurationSec.Observe(time.Since(start).Seconds())
	}()

	workers := v.concurrency
	if workers > len(proxies) {
		workers = len(proxies)
	}

	jobs := make(chan string, workers*2)
	var (
		wg         sync.WaitGroup
		okCount    atomic.Int64
		failCount  atomic.Int64
		totalLatMs atomic.Int64
	)

	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for proxy := range jobs {
				if ctx.Err() != nil {
					return
				}

				proxypool.ValidationInFlight.Inc()
				res := v.checkOne(ctx, proxy)
				proxypool.ValidationInFlight.Dec()

				if res.OK {
					okCount.Add(1)
					totalLatMs.Add(res.LatencyMs)
					proxypool.ValidationLatencyMs.Observe(float64(res.LatencyMs))
					proxypool.ValidationResultTotal.WithLabelValues("ok").Inc()
				} else {
					failCount.Add(1)
					proxypool.ValidationResultTotal.WithLabelValues("fail").Inc()
				}

				sink(ctx, res)
			}
		}()
	}

	progressDone := make(chan struct{})
	defer close(progressDone)
	go v.reportProgress(ctx, len(proxies), &okCount, &failCount, progressDone)

	for _, p := range proxies {
		select {
		case <-ctx.Done():
			close(jobs)
			wg.Wait()
			return buildStats(&okCount, &failCount, &totalLatMs, start)
		case jobs <- p:
		}
	}

	close(jobs)
	wg.Wait()
	return buildStats(&okCount, &failCount, &totalLatMs, start)
}

type Stats struct {
	Total    int
	OK       int
	Failed   int
	Elapsed  time.Duration
	AvgLatMs int64
}

func (s Stats) SuccessRate() float64 {
	if s.Total == 0 {
		return 0
	}
	return float64(s.OK) / float64(s.Total)
}

func buildStats(okCount, failCount, totalLatMs *atomic.Int64, start time.Time) Stats {
	ok := okCount.Load()
	fail := failCount.Load()
	s := Stats{
		Total:    int(ok + fail),
		OK:       int(ok),
		Failed:   int(fail),
		Elapsed:  time.Since(start),
		AvgLatMs: 0,
	}
	if ok > 0 {
		s.AvgLatMs = totalLatMs.Load() / ok
	}
	return s
}

func (v *Validator) reportProgress(
	ctx context.Context,
	total int,
	ok, fail *atomic.Int64,
	done <-chan struct{},
) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-done:
			return
		case <-ticker.C:
			o, f := ok.Load(), fail.Load()
			processed := o + f
			pct := float64(processed) / float64(total) * 100

			remaining := int64(total) - processed
			if remaining < 0 {
				remaining = 0
			}

			logger.Log.Info("Validation progress",
				zap.Int64("processed", processed),
				zap.Int("total", total),
				zap.Float64("pct", pct),
				zap.Int64("ok", o),
				zap.Int64("fail", f),
				zap.Int64("remaining", remaining))
		}
	}
}

func (v *Validator) checkOne(ctx context.Context, proxyStr string) Result {
	start := time.Now()
	res := Result{Proxy: proxyStr}

	if _, err := url.Parse(proxyStr); err != nil {
		res.Err = err
		logger.Log.Debug("Validator: failed to parse proxy URL",
			zap.String("proxy", proxyStr), zap.Error(err))
		return res
	}

	transport, err := proxypool.MakeTransport(proxyStr, v.timeout)
	if err != nil {
		res.Err = err
		logger.Log.Debug("Validator: failed to build transport",
			zap.String("proxy", proxyStr), zap.Error(err))
		return res
	}
	defer transport.CloseIdleConnections()

	client := &http.Client{
		Transport: transport,
		Timeout:   v.timeout,
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, v.targetURL, nil)
	if err != nil {
		res.Err = err
		logger.Log.Debug("Validator: failed to create request",
			zap.String("proxy", proxyStr), zap.Error(err))
		return res
	}
	req.Header.Set("User-Agent", v.userAgent)

	resp, err := client.Do(req)
	if err != nil {
		res.Err = err
		logger.Log.Debug("Validator: request failed (timeout/refused)",
			zap.String("proxy", proxyStr), zap.Error(err))
		return res
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		io.Copy(io.Discard, resp.Body)
		res.Err = fmt.Errorf("non-200 status: %d", resp.StatusCode)
		logger.Log.Debug("Validator: non-200 status",
			zap.String("proxy", proxyStr), zap.Int("status", resp.StatusCode))
		return res
	}

	if _, err := io.Copy(io.Discard, io.LimitReader(resp.Body, 256)); err != nil {
		res.Err = err
		logger.Log.Debug("Validator: body drain failed",
			zap.String("proxy", proxyStr), zap.Error(err))
		return res
	}

	res.OK = true
	res.LatencyMs = time.Since(start).Milliseconds()
	logger.Log.Debug("Validator: proxy OK",
		zap.String("proxy", proxyStr), zap.Int64("latency_ms", res.LatencyMs))
	return res
}
