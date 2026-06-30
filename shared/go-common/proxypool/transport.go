package proxypool

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"time"

	"golang.org/x/net/proxy"
)

// MakeTransport returns an *http.Transport that routes through the given
// proxy URL. Supported schemes:
//   - http:// / https://  → http.ProxyURL (CONNECT tunnel)
//   - socks5:// / socks:// → golang.org/x/net/proxy SOCKS5 dialer
//   - socks4://            → native SOCKS4/SOCKS4a dialer (issue #5)
//
// The timeout is applied at the TCP-dial layer so it also covers SOCKS
// handshake time, not just the TLS handshake.
func MakeTransport(proxyStr string, timeout time.Duration) (*http.Transport, error) {
	proxyURL, err := url.Parse(proxyStr)
	if err != nil {
		return nil, fmt.Errorf("parse proxy URL: %w", err)
	}

	switch proxyURL.Scheme {
	case "socks4":
		// SOCKS4 and SOCKS4a (domain names sent inline).
		// golang.org/x/net/proxy does NOT provide a SOCKS4 dialer, so
		// we use our own implementation. Forcing SOCKS4 through the
		// SOCKS5 dialer would cause a handshake failure because the
		// SOCKS5 greeting (0x05, ...) is not understood by SOCKS4-only
		// servers, resulting in all SOCKS4 proxies being classified as
		// hard failures (issue #5).
		var userID string
		if proxyURL.User != nil {
			userID = proxyURL.User.Username()
		}
		dialer, err := newSocks4Dialer(proxyURL.Host, userID, timeout)
		if err != nil {
			return nil, fmt.Errorf("socks4 dialer: %w", err)
		}
		return &http.Transport{
			DialContext:           dialer.DialContext,
			DisableKeepAlives:     true,
			TLSHandshakeTimeout:   timeout,
			ResponseHeaderTimeout: timeout,
			ExpectContinueTimeout: 1 * time.Second,
		}, nil

	case "socks5", "socks":
		var auth *proxy.Auth
		if proxyURL.User != nil {
			pass, _ := proxyURL.User.Password()
			auth = &proxy.Auth{
				User:     proxyURL.User.Username(),
				Password: pass,
			}
		}
		dialer, err := proxy.SOCKS5("tcp", proxyURL.Host, auth,
			&net.Dialer{Timeout: timeout, KeepAlive: -1})
		if err != nil {
			return nil, fmt.Errorf("socks5 dialer: %w", err)
		}
		dc, ok := dialer.(proxy.ContextDialer)
		if !ok {
			return nil, fmt.Errorf("socks5 dialer does not implement ContextDialer")
		}
		return &http.Transport{
			DialContext:           dc.DialContext,
			DisableKeepAlives:     true,
			TLSHandshakeTimeout:   timeout,
			ResponseHeaderTimeout: timeout,
			ExpectContinueTimeout: 1 * time.Second,
		}, nil

	default: // http, https, or bare
		return &http.Transport{
			Proxy: http.ProxyURL(proxyURL),
			DialContext: (&net.Dialer{
				Timeout:   timeout,
				KeepAlive: -1,
			}).DialContext,
			DisableKeepAlives:     true,
			TLSHandshakeTimeout:   timeout,
			ResponseHeaderTimeout: timeout,
			ExpectContinueTimeout: 1 * time.Second,
		}, nil
	}
}
