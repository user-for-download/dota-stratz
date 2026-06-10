package proxypool

import (
	"context"
	"errors"
	"net"
	"net/http"
)

// ClassifyError inspects an HTTP response/error and returns the appropriate
// FailureReason, along with a boolean indicating whether the proxy should be
// reported as failed.
func ClassifyError(err error, resp *http.Response) (FailureReason, bool) {
	if err != nil {
		// Use standard library error inspection instead of fragile string matching.
		// This covers context.DeadlineExceeded, net.Error timeouts (including TLS),
		// and any wrapped timeout errors.
		if errors.Is(err, context.DeadlineExceeded) {
			return ReasonTimeout, true
		}
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			return ReasonTimeout, true
		}

		// Context cancellation is an intentional application shutdown signal,
		// NOT a proxy failure. Do not report it — otherwise every deploy or
		// graceful restart permanently bans the active proxy from the pool.
		if errors.Is(err, context.Canceled) {
			return "", false
		}

		// Everything else — DNS, connection refused, TLS cert errors — is a hard failure
		return ReasonHardFailure, true
	}

	if resp == nil {
		return ReasonHardFailure, true
	}

	switch {
	case resp.StatusCode == http.StatusTooManyRequests:
		return ReasonRateLimited, true
	case resp.StatusCode >= 500:
		return ReasonBadStatus, true
	case resp.StatusCode == http.StatusForbidden,
		resp.StatusCode == http.StatusProxyAuthRequired:
		// 403 and 407 indicate this proxy cannot authenticate — hard failure.
		return ReasonHardFailure, true
	case resp.StatusCode >= 400:
		// Other 4xx errors are usually client errors, not proxy failures.
		return ReasonBadStatus, false
	}

	return "", false
}
