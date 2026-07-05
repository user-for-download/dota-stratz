package worker

import (
	"context"
	"errors"
	"testing"

	"github.com/dota-stratz/services/detail-fetcher/internal/api"
	"github.com/dota-stratz/services/detail-fetcher/internal/publisher"
	"github.com/dota-stratz/shared/go-common/logger"
	amqp "github.com/rabbitmq/amqp091-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func init() {
	logger.Log = zap.NewNop()
}

// --- fakes ---

type fakeFetcher struct {
	fetchRawFn       func(ctx context.Context, matchID int64) ([]byte, error)
	fetchRawDirectFn func(ctx context.Context, matchID int64) ([]byte, error)
}

func (f *fakeFetcher) FetchRaw(ctx context.Context, matchID int64) ([]byte, error) {
	return f.fetchRawFn(ctx, matchID)
}

func (f *fakeFetcher) FetchRawDirect(ctx context.Context, matchID int64) ([]byte, error) {
	if f.fetchRawDirectFn != nil {
		return f.fetchRawDirectFn(ctx, matchID)
	}
	return nil, errors.New("direct fallback not wired in test")
}

type fakePublisher struct {
	publishFn func(ctx context.Context, queueName string, msg publisher.RawMatchMessage) error
}

func (f *fakePublisher) Publish(ctx context.Context, queueName string, msg publisher.RawMatchMessage) error {
	return f.publishFn(ctx, queueName, msg)
}

// fakeExistenceChecker implements matchExistenceChecker for tests.
type fakeExistenceChecker struct {
	matchExistsFn func(ctx context.Context, matchID int64) (bool, error)
}

func (f *fakeExistenceChecker) MatchExists(ctx context.Context, matchID int64) (bool, error) {
	return f.matchExistsFn(ctx, matchID)
}

// --- tests ---

func TestProcess_Success(t *testing.T) {
	var fetchCalled, publishCalled bool

	fetcher := &fakeFetcher{
		fetchRawFn: func(ctx context.Context, matchID int64) ([]byte, error) {
			fetchCalled = true
			assert.Equal(t, int64(123), matchID)
			return []byte(`{"match_id":123}`), nil
		},
	}

	pub := &fakePublisher{
		publishFn: func(ctx context.Context, queueName string, msg publisher.RawMatchMessage) error {
			publishCalled = true
			assert.Equal(t, "raw_matches", queueName)
			assert.Equal(t, int64(123), msg.MatchID)
			assert.Equal(t, `{"match_id":123}`, string(msg.RawJSON))
			assert.False(t, msg.FetchedAt.IsZero())
			return nil
		},
	}

	w := NewWorker(fetcher, pub, "raw_matches", 3, 0)
	d := amqp.Delivery{Body: []byte(`{"match_id":123}`)}
	result := w.Process(context.Background(), d)
	assert.Equal(t, Ack, result)
	assert.True(t, fetchCalled, "FetchRaw should have been called")
	assert.True(t, publishCalled, "Publish should have been called")
}

func TestProcess_SkipWhenExistsInDB(t *testing.T) {
	var fetchCalled bool
	var publishCalled bool

	fetcher := &fakeFetcher{
		fetchRawFn: func(ctx context.Context, matchID int64) ([]byte, error) {
			fetchCalled = true
			t.Error("FetchRaw should NOT be called when match already exists in DB")
			return nil, nil
		},
	}

	pub := &fakePublisher{
		publishFn: func(ctx context.Context, queueName string, msg publisher.RawMatchMessage) error {
			publishCalled = true
			t.Error("Publish should NOT be called when match already exists in DB")
			return nil
		},
	}

	w := NewWorker(fetcher, pub, "raw_matches", 3, 0)
	w.SetDBExistenceChecker(&fakeExistenceChecker{
		matchExistsFn: func(ctx context.Context, matchID int64) (bool, error) {
			return true, nil
		},
	})

	d := amqp.Delivery{Body: []byte(`{"match_id":123}`)}
	result := w.Process(context.Background(), d)
	assert.Equal(t, Ack, result)
	assert.False(t, fetchCalled, "FetchRaw should NOT have been called")
	assert.False(t, publishCalled, "Publish should NOT have been called")
}

func TestProcess_DBCheckErrorProceedsWithFetch(t *testing.T) {
	var fetchCalled, publishCalled bool

	fetcher := &fakeFetcher{
		fetchRawFn: func(ctx context.Context, matchID int64) ([]byte, error) {
			fetchCalled = true
			return []byte(`{"match_id":123}`), nil
		},
	}

	pub := &fakePublisher{
		publishFn: func(ctx context.Context, queueName string, msg publisher.RawMatchMessage) error {
			publishCalled = true
			return nil
		},
	}

	w := NewWorker(fetcher, pub, "raw_matches", 3, 0)
	w.SetDBExistenceChecker(&fakeExistenceChecker{
		matchExistsFn: func(ctx context.Context, matchID int64) (bool, error) {
			return false, errors.New("simulated DB error")
		},
	})

	d := amqp.Delivery{Body: []byte(`{"match_id":123}`)}
	result := w.Process(context.Background(), d)
	assert.Equal(t, Ack, result)
	assert.True(t, fetchCalled, "FetchRaw should still be called when DB check errors")
	assert.True(t, publishCalled, "Publish should still be called when DB check errors")
}

func TestProcess_MatchNotFound(t *testing.T) {
	var fetchCalled bool
	publishCalled := false

	fetcher := &fakeFetcher{
		fetchRawFn: func(ctx context.Context, matchID int64) ([]byte, error) {
			fetchCalled = true
			return nil, api.ErrMatchNotFound
		},
	}

	pub := &fakePublisher{
		publishFn: func(ctx context.Context, queueName string, msg publisher.RawMatchMessage) error {
			publishCalled = true
			return nil
		},
	}

	w := NewWorker(fetcher, pub, "raw_matches", 3, 0)
	d := amqp.Delivery{Body: []byte(`{"match_id":123}`)}
	result := w.Process(context.Background(), d)
	assert.Equal(t, Ack, result)
	assert.True(t, fetchCalled, "FetchRaw should have been called")
	assert.False(t, publishCalled, "Publish should NOT have been called when match is not found")
}

func TestProcess_FetchErrorRetriesThenDLQ(t *testing.T) {
	fetchAttempts := 0
	expectedErr := errors.New("fetch failed")

	fetcher := &fakeFetcher{
		fetchRawFn: func(ctx context.Context, matchID int64) ([]byte, error) {
			fetchAttempts++
			return nil, expectedErr
		},
	}

	pub := &fakePublisher{
		publishFn: func(ctx context.Context, queueName string, msg publisher.RawMatchMessage) error {
			t.Error("Publish should never be called when fetch always fails")
			return nil
		},
	}

	w := NewWorker(fetcher, pub, "raw_matches", 3, 0)
	d := amqp.Delivery{Body: []byte(`{"match_id":123}`)}
	result := w.Process(context.Background(), d)
	assert.Equal(t, NackDLQ, result)
	assert.Equal(t, 3, fetchAttempts, "Should have attempted fetch exactly 3 times (maxRetries=3)")
}

func TestProcess_PublishErrorRetriesWithoutReFetch(t *testing.T) {
	fetchCount := 0

	fetcher := &fakeFetcher{
		fetchRawFn: func(ctx context.Context, matchID int64) ([]byte, error) {
			fetchCount++
			if fetchCount > 1 {
				panic("FetchRaw should not be called more than once when rawJSON is already set")
			}
			return []byte(`{"match_id":123}`), nil
		},
	}

	publishAttempt := 0
	pub := &fakePublisher{
		publishFn: func(ctx context.Context, queueName string, msg publisher.RawMatchMessage) error {
			publishAttempt++
			if publishAttempt <= 2 {
				return errors.New("publish transient failure")
			}
			return nil
		},
	}

	w := NewWorker(fetcher, pub, "raw_matches", 3, 0)
	d := amqp.Delivery{Body: []byte(`{"match_id":123}`)}
	result := w.Process(context.Background(), d)
	assert.Equal(t, Ack, result)
	assert.Equal(t, 1, fetchCount, "FetchRaw should be called exactly once")
	assert.Equal(t, 3, publishAttempt, "Publish should have been attempted 3 times (2 failures, 1 success)")
}

func TestProcess_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	fetcher := &fakeFetcher{
		fetchRawFn: func(ctx context.Context, matchID int64) ([]byte, error) {
			// Cancel context inside the fetch call so that when Process
			// enters the backoff select, ctx.Done() fires immediately
			// instead of waiting for the full retry delay.
			cancel()
			return nil, errors.New("fetch failed")
		},
	}

	pub := &fakePublisher{
		publishFn: func(ctx context.Context, queueName string, msg publisher.RawMatchMessage) error {
			t.Error("Publish should never be called")
			return nil
		},
	}

	// Use a long retry delay (10s) to ensure ctx.Done() fires before
	// the backoff timer; since cancel() is called from the fetcher,
	// the context is already done when the select is reached.
	w := NewWorker(fetcher, pub, "raw_matches", 3, 10)
	d := amqp.Delivery{Body: []byte(`{"match_id":123}`)}
	result := w.Process(ctx, d)
	assert.Equal(t, NackRequeue, result)
}

func TestProcess_PanicRecovery(t *testing.T) {
	fetcher := &fakeFetcher{
		fetchRawFn: func(ctx context.Context, matchID int64) ([]byte, error) {
			panic("simulated transport bug")
		},
	}

	pub := &fakePublisher{
		publishFn: func(ctx context.Context, queueName string, msg publisher.RawMatchMessage) error {
			t.Error("Publish should never be called after a panic")
			return nil
		},
	}

	w := NewWorker(fetcher, pub, "raw_matches", 3, 0)
	d := amqp.Delivery{Body: []byte(`{"match_id":123}`)}
	result := w.Process(context.Background(), d)
	assert.Equal(t, NackDLQ, result)

	// Also verify that Process does not panic (the defer/recover handles it).
	require.NotPanics(t, func() {
		w.Process(context.Background(), d)
	}, "Process should recover from panics")
}
