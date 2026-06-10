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
//   - socks5:// / socks4:// / socks:// → golang.org/x/net/proxy dialer
//
// The timeout is applied at the TCP-dial layer so it also covers SOCKS
// handshake time, not just the TLS handshake.
func MakeTransport(proxyStr string, timeout time.Duration) (*http.Transport, error) {
	proxyURL, err := url.Parse(proxyStr)
	if err != nil {
		return nil, fmt.Errorf("parse proxy URL: %w", err)
	}

	switch proxyURL.Scheme {
	case "socks5", "socks4", "socks":
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
			return nil, fmt.Errorf("socks dialer: %w", err)
		}
		dc, ok := dialer.(proxy.ContextDialer)
		if !ok {
			return nil, fmt.Errorf("socks dialer does not implement ContextDialer")
		}
		t := &http.Transport{
			DialContext:           dc.DialContext,
			DisableKeepAlives:     true,
			TLSHandshakeTimeout:   timeout,
			ResponseHeaderTimeout: timeout,
			ExpectContinueTimeout: 1 * time.Second,
		}
		return t, nil

	default: // http, https, or bare
		t := &http.Transport{
			Proxy: http.ProxyURL(proxyURL),
			DialContext: (&net.Dialer{
				Timeout:   timeout,
				KeepAlive: -1,
			}).DialContext,
			DisableKeepAlives:     true,
			TLSHandshakeTimeout:   timeout,
			ResponseHeaderTimeout: timeout,
			ExpectContinueTimeout: 1 * time.Second,
		}
		return t, nil
	}
}
