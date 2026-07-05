package source

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/dota-stratz/shared/go-common/logger"
	"go.uber.org/zap"
)

func FromFile(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	lines := parseLines(f)
	logger.Log.Debug("Source: loaded proxies from file",
		zap.String("path", path),
		zap.Int("count", len(lines)))
	return lines, nil
}

func FromURL(ctx context.Context, url string, timeout time.Duration, userAgent string) ([]string, error) {
	if url == "" {
		return nil, fmt.Errorf("empty source URL")
	}

	logger.Log.Debug("Source: fetching from URL",
		zap.String("url", url),
		zap.Duration("timeout", timeout))

	client := &http.Client{Timeout: timeout}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		io.Copy(io.Discard, resp.Body)
		return nil, fmt.Errorf("source returned status %d", resp.StatusCode)
	}

	lines := parseLines(resp.Body)
	logger.Log.Debug("Source: fetched proxies from URL",
		zap.String("url", url),
		zap.Int("count", len(lines)))
	return lines, nil
}

func parseLines(r io.Reader) []string {
	scanner := bufio.NewScanner(r)
	// Buffer up to 1MB lines
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)

	out := make([]string, 0, 256)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Basic normalization: bare host:port is assumed HTTP.
		if !strings.Contains(line, "://") {
			logger.Log.Debug("Source: bare host:port assumed http://",
				zap.String("raw", line))
			line = "http://" + line
		}
		out = append(out, line)
	}

	if err := scanner.Err(); err != nil {
		logger.Log.Warn("Source: scanner error while parsing",
			zap.Error(err))
	}

	return out
}
