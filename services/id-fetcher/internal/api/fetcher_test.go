package api

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/dota-stratz/shared/go-common/logger"
)

// TestMain initializes the global logger once before all tests run.
// Without this, Fetcher.Run (and anything else that calls logger.Log.*)
// panics with a nil-pointer dereference in test binaries, because the
// production main() never runs in `go test`. The logger output is
// discarded unless a test fails.
func TestMain(m *testing.M) {
	logger.InitLogger()
	defer logger.Sync()
	m.Run()
}

// fakeSource is a deterministic test double for the openDotaSource
// interface used by Fetcher. It returns a pre-canned list of matches
// and records the parameters passed to FetchMatchesSince so tests can
// assert the watermark path was taken with the right arguments.
type fakeSource struct {
	mu sync.Mutex

	// rolling matches returned by FetchMatches
	rolling []MatchNode
	// watermark matches returned by FetchMatchesSince
	watermark []MatchNode

	// recorded calls
	rollingCalled     int
	watermarkCalled   int
	lastWatermarkArg  int64
	lastLookbackArg   int
	lastMaxResultsArg int
}

func (f *fakeSource) FetchMatches(_ context.Context) ([]MatchNode, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.rollingCalled++
	return f.rolling, nil
}

func (f *fakeSource) FetchMatchesSince(_ context.Context, watermark int64, lookbackDays int, maxResults int) ([]MatchNode, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.watermarkCalled++
	f.lastWatermarkArg = watermark
	f.lastLookbackArg = lookbackDays
	f.lastMaxResultsArg = maxResults
	// Apply the same match_id > watermark filter the real helper would
	// apply so tests exercise the same code path the fetcher sees.
	out := make([]MatchNode, 0, len(f.watermark))
	for _, m := range f.watermark {
		if m.MatchID > watermark {
			out = append(out, m)
		}
	}
	return out, nil
}

// fakePublisher is a deterministic test double for the
// matchIDPublisher interface. It records every batch it receives so
// tests can assert the published stream.
type fakePublisher struct {
	mu      sync.Mutex
	batches [][]int64
	err     error
}

func (f *fakePublisher) PublishBatch(_ context.Context, _ string, ids []int64) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.err != nil {
		return f.err
	}
	// Copy the slice so later appends by the fetcher can't mutate
	// the captured batch.
	cp := make([]int64, len(ids))
	copy(cp, ids)
	f.batches = append(f.batches, cp)
	return nil
}

func (f *fakePublisher) allPublished() []int64 {
	f.mu.Lock()
	defer f.mu.Unlock()
	var all []int64
	for _, b := range f.batches {
		all = append(all, b...)
	}
	return all
}

// newTestFetcher wires a Fetcher with a fake source and fake publisher
// and the given batch size. The watermark/lookback are left at zero
// (rolling-window path) unless the caller sets them via SetWatermark.
func newTestFetcher(src *fakeSource, pub *fakePublisher, batchSize int) *Fetcher {
	return &Fetcher{
		client:    src,
		publisher: pub,
		queueName: "queue.match_ids",
		batchSize: batchSize,
	}
}

// TestRun_RollingWindowPath: Watermark=0 → Fetcher calls
// FetchMatches (the rolling window) and never calls FetchMatchesSince.
func TestRun_RollingWindowPath(t *testing.T) {
	src := &fakeSource{
		rolling: []MatchNode{
			{MatchID: 1, StartTime: 100},
			{MatchID: 2, StartTime: 200},
			{MatchID: 3, StartTime: 300},
		},
	}
	pub := &fakePublisher{}
	f := newTestFetcher(src, pub, 10)
	// No SetWatermark → watermark stays 0 → rolling path.

	if err := f.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if src.rollingCalled != 1 {
		t.Errorf("FetchMatches called %d times, want 1", src.rollingCalled)
	}
	if src.watermarkCalled != 0 {
		t.Errorf("FetchMatchesSince called %d times, want 0 (rolling path should not be used)", src.watermarkCalled)
	}
	got := pub.allPublished()
	if len(got) != 3 {
		t.Fatalf("published %d ids, want 3: %v", len(got), got)
	}
}

// TestRun_WatermarkPath: Watermark>0 → Fetcher calls FetchMatchesSince
// with the right watermark, lookback, and 5x overscan, and never calls
// the rolling FetchMatches.
func TestRun_WatermarkPath(t *testing.T) {
	src := &fakeSource{
		watermark: []MatchNode{
			{MatchID: 105, StartTime: 100},
			{MatchID: 104, StartTime: 100},
			{MatchID: 103, StartTime: 100},
		},
	}
	pub := &fakePublisher{}
	f := newTestFetcher(src, pub, 100) // batchSize 100 → maxResults 500
	f.SetWatermark(100, 30)

	if err := f.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if src.watermarkCalled != 1 {
		t.Errorf("FetchMatchesSince called %d times, want 1", src.watermarkCalled)
	}
	if src.rollingCalled != 0 {
		t.Errorf("FetchMatches called %d times, want 0 (watermark path should not be used)", src.rollingCalled)
	}
	if src.lastWatermarkArg != 100 {
		t.Errorf("lastWatermarkArg = %d, want 100", src.lastWatermarkArg)
	}
	if src.lastLookbackArg != 30 {
		t.Errorf("lastLookbackArg = %d, want 30", src.lastLookbackArg)
	}
	// 5x overscan: batchSize 100 * 5 = 500.
	if src.lastMaxResultsArg != 500 {
		t.Errorf("lastMaxResultsArg = %d, want 500 (batchSize * 5)", src.lastMaxResultsArg)
	}
	got := pub.allPublished()
	if len(got) != 3 {
		t.Fatalf("published %d ids, want 3: %v", len(got), got)
	}
}

// TestRun_WatermarkPath_NoNewMatches: when the watermark path
// returns zero matches, nothing should be published (the pipeline is
// caught up).
func TestRun_WatermarkPath_NoNewMatches(t *testing.T) {
	src := &fakeSource{
		// All matches are ≤ watermark 1000 → fakeSource's filter
		// strips them all.
		watermark: []MatchNode{
			{MatchID: 999, StartTime: 100},
			{MatchID: 500, StartTime: 100},
		},
	}
	pub := &fakePublisher{}
	f := newTestFetcher(src, pub, 100)
	f.SetWatermark(1000, 30)

	if err := f.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := pub.allPublished(); len(got) != 0 {
		t.Errorf("published %d ids, want 0 (no new matches): %v", len(got), got)
	}
}

// TestRun_BatchBoundary: with batchSize=3, publishing exactly 3 matches
// (or any multiple of 3) should produce one full batch with no
// remainder. Tests the batch-publish boundary so we don't leave a
// partial batch sitting in memory if the cron job is killed.
func TestRun_BatchBoundary(t *testing.T) {
	const batchSize = 3
	src := &fakeSource{
		rolling: []MatchNode{
			{MatchID: 1}, {MatchID: 2}, {MatchID: 3},
		},
	}
	pub := &fakePublisher{}
	f := newTestFetcher(src, pub, batchSize)

	if err := f.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	got := pub.allPublished()
	if len(got) != batchSize {
		t.Fatalf("published %d ids, want %d: %v", len(got), batchSize, got)
	}
	if len(pub.batches) != 1 {
		t.Errorf("expected exactly 1 batch, got %d: %v", len(pub.batches), pub.batches)
	}
}

// TestRun_BatchMultipleFlushes: with batchSize=2 and 5 matches, we
// expect 2 published (batch 1) + 2 published (batch 2) + 1 published
// (remainder flush) = 3 PublishBatch calls.
func TestRun_BatchMultipleFlushes(t *testing.T) {
	const batchSize = 2
	src := &fakeSource{
		rolling: []MatchNode{
			{MatchID: 1}, {MatchID: 2}, {MatchID: 3}, {MatchID: 4}, {MatchID: 5},
		},
	}
	pub := &fakePublisher{}
	f := newTestFetcher(src, pub, batchSize)

	if err := f.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if got := pub.allPublished(); len(got) != 5 {
		t.Errorf("published %d ids, want 5: %v", len(got), got)
	}
	if len(pub.batches) != 3 {
		t.Errorf("expected 3 batches, got %d", len(pub.batches))
	}
	// Verify the contents of each batch.
	wantBatches := [][]int64{{1, 2}, {3, 4}, {5}}
	for i, want := range wantBatches {
		if i >= len(pub.batches) {
			t.Errorf("missing batch %d", i)
			continue
		}
		got := pub.batches[i]
		if len(got) != len(want) {
			t.Errorf("batch %d length = %d, want %d", i, len(got), len(want))
			continue
		}
		for j := range want {
			if got[j] != want[j] {
				t.Errorf("batch %d[%d] = %d, want %d", i, j, got[j], want[j])
			}
		}
	}
}

// TestRun_PublisherError_Propagates: a publish failure is treated as
// a fetch-run failure (PaginationRunsTotal=error) and the error
// bubbles up. This is the "fail loud" behaviour operators rely on.
func TestRun_PublisherError_Propagates(t *testing.T) {
	src := &fakeSource{
		rolling: []MatchNode{{MatchID: 1}},
	}
	pub := &fakePublisher{err: errors.New("broker rejected")}
	f := newTestFetcher(src, pub, 1)

	err := f.Run(context.Background())
	if err == nil {
		t.Fatal("Run should return error when publisher fails")
	}
	if err.Error() != "broker rejected" {
		t.Errorf("Run error = %q, want %q", err.Error(), "broker rejected")
	}
}

// TestRun_ContextCancelled: when the caller's ctx is cancelled mid-run,
// Run returns ctx.Err() and flushes any accumulated batch with a
// fresh context (so the IDs aren't lost).
func TestRun_ContextCancelled(t *testing.T) {
	// We can't actually cancel the fake's ctx mid-flight because the
	// fake is synchronous, so we test the simpler property: Run
	// returns ctx.Err() when the context is already cancelled.
	src := &fakeSource{
		rolling: []MatchNode{{MatchID: 1}},
	}
	pub := &fakePublisher{}
	f := newTestFetcher(src, pub, 10)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := f.Run(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("Run err = %v, want context.Canceled", err)
	}
}

// TestSetWatermark_BoundsCheck: SetWatermark clamps negative inputs
// to 0 so a misconfigured caller cannot accidentally enable the
// watermark path with a negative ID (which would never match
// anything).
func TestSetWatermark_BoundsCheck(t *testing.T) {
	f := newTestFetcher(&fakeSource{}, &fakePublisher{}, 10)
	f.SetWatermark(-1, 30)
	if f.Watermark() != 0 {
		t.Errorf("Watermark = %d, want 0 (negative input should clamp)", f.Watermark())
	}

	f.SetWatermark(42, -1)
	if f.Watermark() != 42 {
		t.Errorf("Watermark = %d, want 42 (negative lookback should not affect watermark)", f.Watermark())
	}
	if f.watermarkLookbackDays != 0 {
		t.Errorf("lookback = %d, want 0 (negative lookback should clamp)", f.watermarkLookbackDays)
	}
}

// TestFetch_RequiresLookbackForWatermark: defensive guard — calling
// Run with a positive watermark but no lookback should return an
// error rather than silently use the rolling path (which would
// regress to the pre-P1-1 behaviour).
func TestFetch_RequiresLookbackForWatermark(t *testing.T) {
	f := newTestFetcher(&fakeSource{}, &fakePublisher{}, 10)
	f.watermark = 100
	// f.watermarkLookbackDays left at 0

	_, err := f.fetch(context.Background())
	if err == nil {
		t.Fatal("fetch should return error when watermark>0 but lookback=0")
	}
}

// TestFetch_ContextDeadlinePropagation: ctx cancellation propagates
// into the fetch layer. This is the contract the cron shutdown path
// relies on (see main.go's stopCtx).
func TestFetch_ContextDeadlinePropagation(t *testing.T) {
	// Build a source whose FetchMatchesSince blocks until ctx is done.
	src := &blockingSource{}
	f := newTestFetcher(&fakeSource{rolling: nil}, &fakePublisher{}, 10)
	f.client = src
	f.watermark = 100
	f.watermarkLookbackDays = 30

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	_, err := f.fetch(ctx)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("fetch err = %v, want context.DeadlineExceeded", err)
	}
}

// blockingSource is an openDotaSource whose FetchMatchesSince blocks
// on ctx.Done(). Used to verify the fetcher's ctx propagation.
type blockingSource struct{}

func (b *blockingSource) FetchMatches(ctx context.Context) ([]MatchNode, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func (b *blockingSource) FetchMatchesSince(ctx context.Context, _ int64, _ int, _ int) ([]MatchNode, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}
