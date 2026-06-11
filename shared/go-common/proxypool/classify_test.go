package proxypool

import (
	"context"
	"net"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestClassifyError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		err            error
		resp           *http.Response
		wantReason     FailureReason
		wantShouldFail bool
	}{
		{
			name:           "context deadline exceeded",
			err:            context.DeadlineExceeded,
			resp:           nil,
			wantReason:     ReasonTimeout,
			wantShouldFail: true,
		},
		{
			name:           "context canceled (non-reportable)",
			err:            context.Canceled,
			resp:           nil,
			wantReason:     "",
			wantShouldFail: false,
		},
		{
			name: "net.Error timeout",
			err: &net.DNSError{
				IsTimeout: true,
			},
			resp:           nil,
			wantReason:     ReasonTimeout,
			wantShouldFail: true,
		},
		{
			name:           "generic error",
			err:            assert.AnError,
			resp:           nil,
			wantReason:     ReasonHardFailure,
			wantShouldFail: true,
		},
		{
			name:           "429 rate limited",
			err:            nil,
			resp:           &http.Response{StatusCode: http.StatusTooManyRequests},
			wantReason:     ReasonRateLimited,
			wantShouldFail: true,
		},
		{
			name:           "500 bad status",
			err:            nil,
			resp:           &http.Response{StatusCode: http.StatusInternalServerError},
			wantReason:     ReasonBadStatus,
			wantShouldFail: true,
		},
		{
			name:           "403 forbidden (hard failure)",
			err:            nil,
			resp:           &http.Response{StatusCode: http.StatusForbidden},
			wantReason:     ReasonHardFailure,
			wantShouldFail: true,
		},
		{
			name:           "404 bad status (non-reportable)",
			err:            nil,
			resp:           &http.Response{StatusCode: http.StatusNotFound},
			wantReason:     ReasonBadStatus,
			wantShouldFail: false,
		},
		{
			name:           "200 success no error",
			err:            nil,
			resp:           &http.Response{StatusCode: http.StatusOK},
			wantReason:     "",
			wantShouldFail: false,
		},
		{
			name:           "nil response nil error",
			err:            nil,
			resp:           nil,
			wantReason:     ReasonHardFailure,
			wantShouldFail: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			reason, shouldReport := ClassifyError(tt.err, tt.resp)
			assert.Equal(t, tt.wantReason, reason)
			assert.Equal(t, tt.wantShouldFail, shouldReport)
		})
	}
}
