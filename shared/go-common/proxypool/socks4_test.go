package proxypool

import (
	"bytes"
	"context"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// funcDialer wraps a dial function to implement the tcpDialer interface.
// This allows tests to inject net.Pipe connections without real TCP.
type funcDialer struct {
	fn func(ctx context.Context, network, addr string) (net.Conn, error)
}

func (d *funcDialer) DialContext(ctx context.Context, network, addr string) (net.Conn, error) {
	return d.fn(ctx, network, addr)
}

// testSocks4Dialer creates a socks4Dialer whose TCP dialer is backed by
// the given dialFn (typically returning a net.Pipe client conn).
func testSocks4Dialer(t *testing.T, dialFn func(ctx context.Context, network, addr string) (net.Conn, error)) *socks4Dialer {
	t.Helper()
	d, err := newSocks4Dialer("127.0.0.1:9999", "test", time.Second)
	require.NoError(t, err)
	d.tcpDialer = &funcDialer{fn: dialFn}
	return d
}

// TestPartialReadRegression_BUG009 is a CRITICAL regression test for a bug
// where the SOCKS4 response reader used plain Read() instead of io.ReadFull.
// A plain Read() can return fewer than 8 bytes on a fragmented TCP stream,
// causing the version/status check to read bytes from the wrong offset or
// fail entirely. The fix was to use io.ReadFull to guarantee all 8 response
// bytes are read atomically.
//
// This test simulates a fragmented TCP write by writing the 8-byte grant
// response in two 4-byte chunks separated by a 10ms delay. io.ReadFull must
// wait for all bytes and return success. A plain Read() would return only
// the first 4 bytes and leave the connection in an inconsistent state.
//
// WARNING: net.Pipe is synchronous — a Write on one end blocks until the
// other end does a Read. The goroutine MUST read the request first to
// unblock the dialer's Write, then write the fragmented response.
func TestPartialReadRegression_BUG009(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()

	// Goroutine reads the request (to unblock the dialer's Write), then writes
	// the SOCKS4 response in two 4-byte chunks with a 10ms delay between them.
	go func() {
		// Read request first — net.Pipe is synchronous: Write blocks until Read.
		buf := make([]byte, 1024)
		_, _ = server.Read(buf)

		resp := []byte{0x00, 0x5A, 0, 0, 0, 0, 0, 0}
		_, _ = server.Write(resp[:4])
		time.Sleep(10 * time.Millisecond)
		_, _ = server.Write(resp[4:])
		server.Close()
	}()

	dialer := testSocks4Dialer(t, func(ctx context.Context, network, addr string) (net.Conn, error) {
		return client, nil
	})

	// DialContext must succeed — io.ReadFull correctly waits for all 8 bytes
	conn, err := dialer.DialContext(context.Background(), "tcp", "203.0.113.1:80")
	require.NoError(t, err, "BUG009: io.ReadFull must wait for fragmented SOCKS4 response")
	require.NotNil(t, conn)
	conn.Close()
}

// TestGrantedResponse verifies that a full 8-byte grant response (0x5A)
// returns a valid connection and no error.
func TestGrantedResponse(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()

	go func() {
		// Read request first — net.Pipe is synchronous.
		buf := make([]byte, 1024)
		_, _ = server.Read(buf)

		resp := []byte{0x00, 0x5A, 0, 0, 0, 0, 0, 0}
		_, _ = server.Write(resp)
		server.Close()
	}()

	dialer := testSocks4Dialer(t, func(ctx context.Context, network, addr string) (net.Conn, error) {
		return client, nil
	})

	conn, err := dialer.DialContext(context.Background(), "tcp", "203.0.113.1:80")
	require.NoError(t, err)
	require.NotNil(t, conn)
	conn.Close()
}

// TestRejectedResponse verifies that a rejected response (0x5B) returns an
// error containing "rejected".
func TestRejectedResponse(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()

	go func() {
		// Read request first — net.Pipe is synchronous.
		buf := make([]byte, 1024)
		_, _ = server.Read(buf)

		resp := []byte{0x00, 0x5B, 0, 0, 0, 0, 0, 0}
		_, _ = server.Write(resp)
		server.Close()
	}()

	dialer := testSocks4Dialer(t, func(ctx context.Context, network, addr string) (net.Conn, error) {
		return client, nil
	})

	_, err := dialer.DialContext(context.Background(), "tcp", "203.0.113.1:80")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "rejected")
}

// TestNonIPv4Address verifies that an IPv6 destination address returns
// an error containing "non-IPv4".
func TestNonIPv4Address(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()

	dialer := testSocks4Dialer(t, func(ctx context.Context, network, addr string) (net.Conn, error) {
		return client, nil
	})

	// [::1]:80 is an IPv6 loopback address — SOCKS4 only supports IPv4.
	// The error occurs during request building, before any write to the
	// proxy connection, so no response from the server side is needed.
	_, err := dialer.DialContext(context.Background(), "tcp", "[::1]:80")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "non-IPv4")
}

// TestSOCKS4aDomainMode verifies that when a domain name is passed as the
// destination (instead of an IPv4 address), the dialer uses SOCKS4a mode.
// Bytes 4-7 of the request must be 0,0,0,1 and the domain must appear
// after the null-terminated userID field.
func TestSOCKS4aDomainMode(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()

	var reqBytes []byte
	done := make(chan struct{})

	go func() {
		defer close(done)
		// Read the SOCKS4 request from the server end of the pipe.
		buf := make([]byte, 1024)
		n, err := server.Read(buf)
		if err != nil {
			return
		}
		reqBytes = make([]byte, n)
		copy(reqBytes, buf[:n])
		// Write a grant response so DialContext completes.
		_, _ = server.Write([]byte{0x00, 0x5A, 0, 0, 0, 0, 0, 0})
		server.Close()
	}()

	dialer := testSocks4Dialer(t, func(ctx context.Context, network, addr string) (net.Conn, error) {
		return client, nil
	})

	conn, err := dialer.DialContext(context.Background(), "tcp", "example.com:80")
	require.NoError(t, err)
	require.NotNil(t, conn)
	conn.Close()
	<-done

	// Verify we captured the request
	require.GreaterOrEqual(t, len(reqBytes), 8, "request must be at least 8 bytes")

	// Bytes 4-7 must be 0,0,0,1 (SOCKS4a domain indicator)
	assert.Equal(t, byte(0), reqBytes[4], "byte 4 must be 0 for SOCKS4a")
	assert.Equal(t, byte(0), reqBytes[5], "byte 5 must be 0 for SOCKS4a")
	assert.Equal(t, byte(0), reqBytes[6], "byte 6 must be 0 for SOCKS4a")
	assert.Equal(t, byte(1), reqBytes[7], "byte 7 must be 1 for SOCKS4a")

	// After the null-terminated userID, the domain name should appear.
	// Request layout: [ver, cmd, port(2), IP(4), userID, 0x00, domain, 0x00]
	// Header = 8 bytes, userID = "test" + null = 5 bytes
	offset := 8 + len("test") + 1
	require.Less(t, offset, len(reqBytes), "request must contain userID and domain")
	domainEnd := bytes.IndexByte(reqBytes[offset:], 0)
	require.Greater(t, domainEnd, 0, "domain must be null-terminated")
	domain := string(reqBytes[offset : offset+domainEnd])
	assert.Equal(t, "example.com", domain)
}
