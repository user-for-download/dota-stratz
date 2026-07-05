package worker

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/dota-stratz/services/parser/internal/models"
	"github.com/dota-stratz/shared/go-common/logger"
	"github.com/jackc/pgx/v5/pgconn"
	amqp "github.com/rabbitmq/amqp091-go"
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

// fakeAcknowledger implements amqp.Acknowledger so tests can intercept
// Ack / Nack calls without a real RabbitMQ broker.
type fakeAcknowledger struct {
	mu          sync.Mutex
	ackCalled   bool
	nackRequeue bool // records the `requeue` parameter from the last Nack call
	nackCalled  bool // true if Nack was ever called
	ackCount    int
}

func (f *fakeAcknowledger) Ack(tag uint64, multiple bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.ackCalled = true
	f.ackCount++
	return nil
}

func (f *fakeAcknowledger) Nack(tag uint64, multiple bool, requeue bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.nackCalled = true
	f.nackRequeue = requeue
	return nil
}

func (f *fakeAcknowledger) Reject(tag uint64, requeue bool) error {
	return nil
}

// fakeRepo implements matchWriter for deterministic testing.
type fakeRepo struct {
	mu           sync.Mutex
	callCount    int
	writeBatchFn func(ctx context.Context, matches []models.OpenDotaMatch) error
}

func (f *fakeRepo) WriteBatch(ctx context.Context, matches []models.OpenDotaMatch) error {
	f.mu.Lock()
	f.callCount++
	f.mu.Unlock()
	if f.writeBatchFn != nil {
		return f.writeBatchFn(ctx, matches)
	}
	return nil
}

// newDelivery builds a minimal amqp.Delivery whose JSON body will
// successfully unmarshal through RawMatchMessage → OpenDotaMatch.
func newDelivery(matchID int64, ack amqp.Acknowledger) amqp.Delivery {
	body := fmt.Sprintf(
		`{"match_id":%d,"raw_json":{"match_id":%d,"duration":120}}`,
		matchID, matchID,
	)
	return amqp.Delivery{
		Acknowledger: ack,
		DeliveryTag:  uint64(matchID),
		Body:         []byte(body),
	}
}

// sendDeliveries populates a channel with N deliveries using the
// provided acknowledgers slice.
func sendDeliveries(ch chan<- amqp.Delivery, acknowledgers []*fakeAcknowledger) {
	for i, ack := range acknowledgers {
		ch <- newDelivery(int64(i+1), ack)
	}
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Test_consecutiveBatchFailuresResetAfterFKFallback_BUG010 is a
// regression test for Bug #10: when a batch write fails with a foreign
// key violation and falls back to individual inserts, the
// consecutiveBatchFailures counter must be reset to 0 so that the next
// non-FK error starts fresh instead of inheriting the previous streak.
//
// Scenario:
//   - consecutiveBatchFailures starts at 2 (previous failures)
//   - WriteBatch returns pgconn.PgError{Code: "23503"} (FK violation)
//   - Individual fallback succeeds for all matches
//   - After one iteration, consecutiveBatchFailures must be 0
func Test_consecutiveBatchFailuresResetAfterFKFallback_BUG010(t *testing.T) {
	const batchSize = 3

	acknowledgers := make([]*fakeAcknowledger, batchSize)
	for i := range acknowledgers {
		acknowledgers[i] = &fakeAcknowledger{}
	}

	msgChan := make(chan amqp.Delivery, batchSize)
	sendDeliveries(msgChan, acknowledgers)

	var repoCallCount int
	signalDone := make(chan struct{})

	repo := &fakeRepo{}
	repo.writeBatchFn = func(ctx context.Context, matches []models.OpenDotaMatch) error {
		repo.mu.Lock()
		repoCallCount++
		count := repoCallCount
		repo.mu.Unlock()

		if count == 1 {
			// First call is the batch — return FK violation.
			return &pgconn.PgError{Code: "23503"}
		}
		// Subsequent calls are individual fallback writes.
		if count == batchSize+1 {
			close(signalDone) // all individual writes done
		}
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	proc := &Processor{
		repo:                     repo,
		batchSize:                batchSize,
		fetchTimeout:             10 * time.Millisecond,
		msgs:                     msgChan,
		consecutiveBatchFailures: 2, // pre-set to test the reset
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		proc.Run(ctx)
	}()

	// Wait for individual fallback to complete.
	<-signalDone

	// Cancel so Run exits cleanly.
	cancel()
	wg.Wait()

	// consecutiveBatchFailures must be 0 after FK fallback.
	if proc.consecutiveBatchFailures != 0 {
		t.Errorf("consecutiveBatchFailures = %d, want 0 (BUG-010 regression)", proc.consecutiveBatchFailures)
	}

	// Verify WriteBatch was called 1 (batch) + N (individual) times.
	repo.mu.Lock()
	wantCalls := batchSize + 1
	if repoCallCount != wantCalls {
		t.Errorf("WriteBatch call count = %d, want %d", repoCallCount, wantCalls)
	}
	repo.mu.Unlock()

	// All individual writes succeeded → deliveries are Ack'd, not Nack'd.
	for i, a := range acknowledgers {
		a.mu.Lock()
		if !a.ackCalled {
			t.Errorf("delivery %d was not Ack'd", i+1)
		}
		if a.ackCount != 1 {
			t.Errorf("delivery %d Ack count = %d, want 1", i+1, a.ackCount)
		}
		a.mu.Unlock()
	}

	// Drain remaining messages from the channel.
	go func() {
		for range msgChan {
		}
	}()
	close(msgChan)
}

// Test_BatchEscalatesToDLQ verifies that after maxConsecutiveBatchFailures
// consecutive non-FK write failures, the processor Nacks deliveries without
// requeue, routing them to the Dead-Letter Queue.
//
// The test starts with consecutiveBatchFailures = 2 so a single additional
// failure triggers the DLQ threshold (maxConsecutiveBatchFailures = 3).
func Test_BatchEscalatesToDLQ(t *testing.T) {
	const batchSize = 3

	acknowledgers := make([]*fakeAcknowledger, batchSize)
	for i := range acknowledgers {
		acknowledgers[i] = &fakeAcknowledger{}
	}

	msgChan := make(chan amqp.Delivery, batchSize)
	sendDeliveries(msgChan, acknowledgers)

	writeCalled := make(chan struct{})
	repo := &fakeRepo{}
	repo.writeBatchFn = func(ctx context.Context, matches []models.OpenDotaMatch) error {
		close(writeCalled)
		return errors.New("generic DB error (non-FK)")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	proc := &Processor{
		repo:                     repo,
		batchSize:                batchSize,
		fetchTimeout:             10 * time.Millisecond,
		msgs:                     msgChan,
		consecutiveBatchFailures: maxConsecutiveBatchFailures - 1, // start at 2
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		proc.Run(ctx)
	}()

	// Wait for WriteBatch to be called (triggering the DLQ path).
	<-writeCalled
	// Give the processor a moment to execute the DLQ Nack.
	time.Sleep(20 * time.Millisecond)
	cancel()
	wg.Wait()

	// All deliveries should be Nack'd without requeue (→ DLQ).
	for i, a := range acknowledgers {
		a.mu.Lock()
		if a.ackCalled {
			t.Errorf("delivery %d was Ack'd, expected Nack to DLQ", i+1)
		}
		if !a.nackCalled {
			t.Errorf("delivery %d was never Nack'd (expected DLQ routing)", i+1)
		} else if a.nackRequeue {
			t.Errorf("delivery %d was Nack'd with requeue=true, want requeue=false (DLQ)", i+1)
		}
		a.mu.Unlock()
	}

	// Counter resets after DLQ escalation.
	if proc.consecutiveBatchFailures != 0 {
		t.Errorf("consecutiveBatchFailures = %d, want 0 after DLQ", proc.consecutiveBatchFailures)
	}

	go func() {
		for range msgChan {
		}
	}()
	close(msgChan)
}

// Test_FKViolationTriggersIndividualFallback verifies that when a batch
// write fails with a foreign key violation (SQLSTATE 23503), the
// processor falls back to individual inserts and Ack's all successful
// deliveries.
//
// The user's acceptance criteria require:
//   - deliveries are Ack'd, not Nack'd
//   - repo.WriteBatch called exactly 1 (batch) + N (individual) times
func Test_FKViolationTriggersIndividualFallback(t *testing.T) {
	const batchSize = 3

	acknowledgers := make([]*fakeAcknowledger, batchSize)
	for i := range acknowledgers {
		acknowledgers[i] = &fakeAcknowledger{}
	}

	msgChan := make(chan amqp.Delivery, batchSize)
	sendDeliveries(msgChan, acknowledgers)

	var repoCallCount int
	signalDone := make(chan struct{})

	repo := &fakeRepo{}
	repo.writeBatchFn = func(ctx context.Context, matches []models.OpenDotaMatch) error {
		repo.mu.Lock()
		repoCallCount++
		count := repoCallCount
		repo.mu.Unlock()

		if count == 1 {
			// Batch write → FK violation.
			return &pgconn.PgError{Code: "23503"}
		}
		// Individual fallback — all succeed.
		if count == batchSize+1 {
			close(signalDone)
		}
		return nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	proc := &Processor{
		repo:         repo,
		batchSize:    batchSize,
		fetchTimeout: 10 * time.Millisecond,
		msgs:         msgChan,
	}

	go func() {
		proc.Run(ctx)
	}()

	<-signalDone
	cancel()
	time.Sleep(20 * time.Millisecond)

	// WriteBatch calls: 1 batch + N individual.
	repo.mu.Lock()
	wantCalls := batchSize + 1
	if repoCallCount != wantCalls {
		t.Errorf("WriteBatch call count = %d, want %d", repoCallCount, wantCalls)
	}
	repo.mu.Unlock()

	// All deliveries should be Ack'd (individual writes succeeded).
	for i, a := range acknowledgers {
		a.mu.Lock()
		if !a.ackCalled {
			t.Errorf("delivery %d was not Ack'd", i+1)
		}
		a.mu.Unlock()
	}

	go func() {
		for range msgChan {
		}
	}()
	close(msgChan)
}
