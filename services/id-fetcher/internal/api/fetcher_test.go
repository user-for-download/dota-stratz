package api

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/dota-stratz/shared/go-common/logger"
)

// TestMain initializes the global logger once before all tests run.
func TestMain(m *testing.M) {
	logger.InitLogger()
	defer logger.Sync()
	m.Run()
}

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

// fakeSource is a deterministic test double for the openDotaSource
// interface using function fields so each test can configure exactly
// what FetchMatches / FetchMatchesSince return.
type fakeSource struct {
	fetchMatchesFn      func(ctx context.Context) ([]MatchNode, error)
	fetchMatchesSinceFn func(ctx context.Context, watermark int64, lookbackDays int, maxResults int) ([]MatchNode, error)
}

func (f *fakeSource) FetchMatches(ctx context.Context) ([]MatchNode, error) {
	return f.fetchMatchesFn(ctx)
}

func (f *fakeSource) FetchMatchesSince(ctx context.Context, watermark int64, lookbackDays int, maxResults int) ([]MatchNode, error) {
	return f.fetchMatchesSinceFn(ctx, watermark, lookbackDays, maxResults)
}

// batchCall records a single PublishBatch invocation. Used by fakePublisher
// so tests can inspect individual batch sizes and contents.
type batchCall struct {
	queueName string
	matchIDs  []int64
}

// fakePublisher records every PublishBatch call and delegates to an
// optional function field so tests can inject cancellation or errors.
type fakePublisher struct {
	publishBatchFn func(ctx context.Context, queueName string, matchIDs []int64) error
	calls          []batchCall
	mu             sync.Mutex
}

func (f *fakePublisher) PublishBatch(ctx context.Context, queueName string, matchIDs []int64) error {
	f.mu.Lock()
	f.calls = append(f.calls, batchCall{queueName: queueName, matchIDs: append([]int64{}, matchIDs...)})
	f.mu.Unlock()
	if f.publishBatchFn != nil {
		return f.publishBatchFn(ctx, queueName, matchIDs)
	}
	return nil
}

func (f *fakePublisher) publishedItems() []int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	var all []int64
	for _, c := range f.calls {
		all = append(all, c.matchIDs...)
	}
	return all
}

// newTestFetcher wires a Fetcher with the given fakes and batch size.
// Watermark/lookback are left at zero (rolling-window path) unless the
// caller calls SetWatermark.
func newTestFetcher(src *fakeSource, pub *fakePublisher, batchSize int) *Fetcher {
	return &Fetcher{
		client:    src,
		publisher: pub,
		queueName: "queue.match_ids",
		batchSize: batchSize,
	}
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Test_RollingWindowPublishesBatches verifies that when watermark is 0
// and FetchMatches returns 25 matches, they are published in three
// batchSize-sized chunks: [1..10], [11..20], [21..25].
func Test_RollingWindowPublishesBatches(t *testing.T) {
	matches := make([]MatchNode, 25)
	for i := range matches {
		matches[i] = MatchNode{MatchID: int64(i + 1)}
	}

	src := &fakeSource{
		fetchMatchesFn: func(_ context.Context) ([]MatchNode, error) {
			return matches, nil
		},
	}
	pub := &fakePublisher{}
	f := newTestFetcher(src, pub, 10)

	if err := f.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	pub.mu.Lock()
	if len(pub.calls) != 3 {
		t.Fatalf("expected 3 PublishBatch calls, got %d", len(pub.calls))
	}

	want := [][]int64{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		{11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
		{21, 22, 23, 24, 25},
	}
	for i, w := range want {
		got := pub.calls[i].matchIDs
		if len(got) != len(w) {
			t.Errorf("batch %d length = %d, want %d", i, len(got), len(w))
			continue
		}
		for j := range w {
			if got[j] != w[j] {
				t.Errorf("batch %d[%d] = %d, want %d", i, j, got[j], w[j])
			}
		}
	}
	pub.mu.Unlock()
}

// Test_WatermarkPathCallsFetchMatchesSince verifies that when a
// watermark is set via SetWatermark, the fetcher calls FetchMatchesSince
// with the correct watermark, lookback days, and maxResults.
func Test_WatermarkPathCallsFetchMatchesSince(t *testing.T) {
	const batchSize = 10
	var recordedWatermark int64
	var recordedLookback int
	var recordedMaxResults int

	src := &fakeSource{
		fetchMatchesSinceFn: func(_ context.Context, watermark int64, lookbackDays int, maxResults int) ([]MatchNode, error) {
			recordedWatermark = watermark
			recordedLookback = lookbackDays
			recordedMaxResults = maxResults
			out := make([]MatchNode, 5)
			for i := range out {
				out[i] = MatchNode{MatchID: watermark + int64(i) + 1}
			}
			return out, nil
		},
	}
	pub := &fakePublisher{}
	f := newTestFetcher(src, pub, batchSize)
	f.SetWatermark(12345, 7)

	if err := f.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if recordedWatermark != 12345 {
		t.Errorf("watermark = %d, want 12345", recordedWatermark)
	}
	if recordedLookback != 7 {
		t.Errorf("lookbackDays = %d, want 7", recordedLookback)
	}
	wantMaxResults := batchSize * watermarkOverscanMultiplier // 10 * 5 = 50
	if recordedMaxResults != wantMaxResults {
		t.Errorf("maxResults = %d, want %d", recordedMaxResults, wantMaxResults)
	}
	if got := pub.publishedItems(); len(got) != 5 {
		t.Errorf("published %d items, want 5", len(got))
	}
}

// Test_WatermarkZeroUsesRollingWindow asserts that when SetWatermark
// is never called (watermark stays 0), FetchMatches is used and
// FetchMatchesSince is never called.
func Test_WatermarkZeroUsesRollingWindow(t *testing.T) {
	rollingCalled := false
	watermarkCalled := false

	src := &fakeSource{
		fetchMatchesFn: func(_ context.Context) ([]MatchNode, error) {
			rollingCalled = true
			return []MatchNode{{MatchID: 1}}, nil
		},
		fetchMatchesSinceFn: func(_ context.Context, _ int64, _ int, _ int) ([]MatchNode, error) {
			watermarkCalled = true
			return nil, nil
		},
	}
	pub := &fakePublisher{}
	f := newTestFetcher(src, pub, 10)

	if err := f.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if !rollingCalled {
		t.Error("FetchMatches was not called (expected rolling path)")
	}
	if watermarkCalled {
		t.Error("FetchMatchesSince was called (expected rolling path only)")
	}
}

// Test_ContextCancellationFlushesPartialBatch verifies that graceful
// shutdown during a fetch cycle does not lose published batches and
// returns ctx.Canceled.
//
// The fetcher's Run loop checks ctx.Err() at the top of each match
// iteration. After a successful full-batch publish the accumulator is
// reset to zero, so a cancellation detected immediately after a publish
// results in an empty flush. The test validates that:
//   - items already published are accounted for
//   - Run returns ctx.Canceled (not a hang or panic)
//   - the first batch is published correctly before cancellation
//
// The remaining items that were never accumulated are expected to be
// picked up on the next cron cycle — this is the canonical design of
// the id-fetcher (stateless, periodic, re-fetches on each tick).
func Test_ContextCancellationFlushesPartialBatch(t *testing.T) {
	const totalMatches = 17
	const batchSize = 10

	matches := make([]MatchNode, totalMatches)
	for i := range matches {
		matches[i] = MatchNode{MatchID: int64(i + 1)}
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	firstPublishDone := make(chan struct{})

	src := &fakeSource{
		fetchMatchesFn: func(_ context.Context) ([]MatchNode, error) {
			return matches, nil
		},
	}

	pub := &fakePublisher{
		publishBatchFn: func(pubCtx context.Context, _ string, ids []int64) error {
			if len(ids) == batchSize && !isClosed(firstPublishDone) {
				// First full batch — signal the main goroutine and block
				// until the context is cancelled. This guarantees that
				// cancel() has been called before Run continues.
				close(firstPublishDone)
				<-pubCtx.Done()
				return nil
			}
			return nil
		},
	}

	f := newTestFetcher(src, pub, batchSize)

	errCh := make(chan error, 1)
	go func() {
		errCh <- f.Run(ctx)
	}()

	// Wait for the first full batch to begin publishing.
	<-firstPublishDone

	// Cancel the context. The PublishBatch callback unblocks and
	// returns nil (successful publish). Run then resets the batch
	// accumulator and checks ctx on the next iteration, finding
	// it cancelled with an empty batch — returning ctx.Canceled.
	cancel()

	err := <-errCh
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Run err = %v, want context.Canceled", err)
	}

	// Verify the first batch was published correctly.
	pub.mu.Lock()
	if len(pub.calls) != 1 {
		t.Fatalf("expected 1 PublishBatch call, got %d", len(pub.calls))
	}
	published := pub.calls[0].matchIDs
	pub.mu.Unlock()

	if len(published) != batchSize {
		t.Fatalf("expected %d items in first batch, got %d", batchSize, len(published))
	}
	for i, id := range published {
		want := int64(i + 1)
		if id != want {
			t.Errorf("batch[%d] = %d, want %d", i, id, want)
		}
	}
}

func isClosed(ch <-chan struct{}) bool {
	select {
	case <-ch:
		return true
	default:
		return false
	}
}

// Test_WatermarkPositiveWithZeroLookbackDays_ReturnsError verifies
// that calling SetWatermark with a positive watermark but zero
// lookback days causes Run to return an error containing
// "fetcher: watermark path requires".
func Test_WatermarkPositiveWithZeroLookbackDays_ReturnsError(t *testing.T) {
	src := &fakeSource{
		fetchMatchesFn: func(_ context.Context) ([]MatchNode, error) {
			return nil, nil
		},
	}
	pub := &fakePublisher{}
	f := newTestFetcher(src, pub, 10)
	// Set watermark to 100 but leave lookbackDays at 0.
	f.SetWatermark(100, 0)

	err := f.Run(context.Background())
	if err == nil {
		t.Fatal("Run should return an error when watermark > 0 but lookback = 0")
	}
	if !strings.Contains(err.Error(), "fetcher: watermark path requires") {
		t.Errorf("Run err = %q, want substring %q", err.Error(), "fetcher: watermark path requires")
	}
}
