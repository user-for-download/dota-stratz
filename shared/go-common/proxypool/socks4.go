package proxypool

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"net"
	"strconv"
	"time"
)

// SOCKS4 status codes.
const (
	socks4Granted       = 0x5A
	socks4Rejected      = 0x5B
	socks4NoIdentd      = 0x5C
	socks4IdentdDiffers = 0x5D
)

// socks4Dialer implements a minimal SOCKS4/SOCKS4a dialer.
// The golang.org/x/net/proxy package does not include SOCKS4,
// so this fills the gap for proxy sources that return socks4:// URLs.
type socks4Dialer struct {
	host      string
	userID    string
	timeout   time.Duration
	tcpDialer *net.Dialer
}

func newSocks4Dialer(host string, userID string, timeout time.Duration) *socks4Dialer {
	return &socks4Dialer{
		host:    host,
		userID:  userID,
		timeout: timeout,
		tcpDialer: &net.Dialer{
			Timeout:   timeout,
			KeepAlive: -1,
		},
	}
}

// DialContext implements proxy.ContextDialer.
func (d *socks4Dialer) DialContext(ctx context.Context, network, addr string) (net.Conn, error) {
	if network != "tcp" {
		return nil, fmt.Errorf("socks4: unsupported network %q", network)
	}

	// Connect to the SOCKS4 proxy server.
	proxyConn, err := d.tcpDialer.DialContext(ctx, "tcp", d.host)
	if err != nil {
		return nil, fmt.Errorf("socks4: connect to proxy %s: %w", d.host, err)
	}
	proxyConn.SetDeadline(time.Now().Add(d.timeout))

	destHost, destPortStr, err := net.SplitHostPort(addr)
	if err != nil {
		proxyConn.Close()
		return nil, fmt.Errorf("socks4: split %q: %w", addr, err)
	}
	destPort, err := strconv.Atoi(destPortStr)
	if err != nil {
		proxyConn.Close()
		return nil, fmt.Errorf("socks4: port %q: %w", destPortStr, err)
	}

	// Build the SOCKS4 request header.
	//   version=4, command=1(CONNECT), port, IP, userID, [domain for SOCKS4a]
	var req []byte
	ip := net.ParseIP(destHost)
	if ip != nil {
		// SOCKS4: direct IP address.
		ip4 := ip.To4()
		if ip4 == nil {
			proxyConn.Close()
			return nil, fmt.Errorf("socks4: non-IPv4 address %q", destHost)
		}
		req = make([]byte, 0, 9+len(d.userID)+1)
		req = append(req, 0x04, 0x01)                              // version, command
		req = binary.BigEndian.AppendUint16(req, uint16(destPort)) // port
		req = append(req, ip4...)                                  // IP
		req = append(req, []byte(d.userID)...)                     // user ID
		req = append(req, 0x00)                                    // null terminator
	} else {
		// SOCKS4a: domain name (indicated by 0.0.0.x IP prefix).
		req = make([]byte, 0, 9+len(d.userID)+1+len(destHost)+1)
		req = append(req, 0x04, 0x01)                              // version, command
		req = binary.BigEndian.AppendUint16(req, uint16(destPort)) // port
		req = append(req, 0, 0, 0, 1)                              // 0.0.0.1 = domain follows
		req = append(req, []byte(d.userID)...)                     // user ID
		req = append(req, 0x00)                                    // null terminator
		req = append(req, []byte(destHost)...)                     // domain name
		req = append(req, 0x00)                                    // null terminator
	}

	if _, err := proxyConn.Write(req); err != nil {
		proxyConn.Close()
		return nil, fmt.Errorf("socks4: write request: %w", err)
	}

	// Read response: 8 bytes.
	resp := make([]byte, 8)
	if _, err := proxyConn.Read(resp); err != nil {
		proxyConn.Close()
		return nil, fmt.Errorf("socks4: read response: %w", err)
	}

	if resp[0] != 0x00 {
		proxyConn.Close()
		return nil, fmt.Errorf("socks4: bad version byte in response: 0x%02x", resp[0])
	}

	switch resp[1] {
	case socks4Granted:
		// Success — clear deadline so the caller sets its own.
		proxyConn.SetDeadline(time.Time{})
		return proxyConn, nil
	case socks4Rejected:
		proxyConn.Close()
		return nil, errors.New("socks4: request rejected or failed")
	case socks4NoIdentd:
		proxyConn.Close()
		return nil, errors.New("socks4: request failed (identd not running)")
	case socks4IdentdDiffers:
		proxyConn.Close()
		return nil, errors.New("socks4: request failed (identd differs)")
	default:
		proxyConn.Close()
		return nil, fmt.Errorf("socks4: unknown status 0x%02x", resp[1])
	}
}
