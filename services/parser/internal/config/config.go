package config

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// envPattern matches ${VAR} and ${VAR:-default} syntax.
var envPattern = regexp.MustCompile(`\$\{([^}]+)\}`)

// expandEnv replaces ${VAR} with os.Getenv("VAR") and ${VAR:-default} with
// os.Getenv("VAR") falling back to "default" when unset or empty.
// Unlike os.ExpandEnv, this correctly handles the :-default suffix.
func expandEnv(s string) string {
	return envPattern.ReplaceAllStringFunc(s, func(match string) string {
		inner := match[2 : len(match)-1] // strip ${ and }
		parts := strings.SplitN(inner, ":-", 2)
		key := parts[0]
		val := os.Getenv(key)
		if val == "" && len(parts) == 2 {
			return parts[1]
		}
		return val
	})
}

type Config struct {
	App struct {
		Environment string `yaml:"environment"`
		LogLevel    string `yaml:"log_level"`
	} `yaml:"app"`

	Postgres struct {
		DSN          string `yaml:"dsn"`
		PoolMaxConns int    `yaml:"pool_max_conns"`
	} `yaml:"postgres"`

	RabbitMQ struct {
		URL    string `yaml:"url"`
		Queues struct {
			RawMatches    string `yaml:"raw_matches"`
			RawMatchesDLQ string `yaml:"raw_matches_dlq"`
		} `yaml:"queues"`
	} `yaml:"rabbitmq"`

	Worker struct {
		BatchSize      int `yaml:"batch_size"`
		FetchTimeoutMs int `yaml:"fetch_timeout_ms"`
		Prefetch       int `yaml:"prefetch"`
		MetricsPort    int `yaml:"metrics_port"`
	} `yaml:"worker"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	// Expand environment variables (e.g. ${POSTGRES_USER} or ${VAR:-default})
	expanded := expandEnv(string(data))

	var cfg Config
	if err := yaml.Unmarshal([]byte(expanded), &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	// Apply defaults
	if cfg.Postgres.PoolMaxConns <= 0 {
		cfg.Postgres.PoolMaxConns = 25 // sufficient for batch parser workload
	}
	if cfg.Worker.BatchSize == 0 {
		cfg.Worker.BatchSize = 100
	}
	if cfg.Worker.FetchTimeoutMs == 0 {
		cfg.Worker.FetchTimeoutMs = 2000
	}
	if cfg.Worker.Prefetch == 0 {
		cfg.Worker.Prefetch = 100
	}
	if cfg.Worker.MetricsPort == 0 {
		cfg.Worker.MetricsPort = 9093
	}

	return &cfg, nil
}

// FetchTimeout returns the fetch timeout as a time.Duration.
func (c *Config) FetchTimeout() time.Duration {
	return time.Duration(c.Worker.FetchTimeoutMs) * time.Millisecond
}
