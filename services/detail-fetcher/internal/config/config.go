package config

import (
	"fmt"
	"os"
	"regexp"
	"strings"

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
	OpenDota struct {
		APIURL       string `yaml:"api_url"`
		TimeoutSec   int    `yaml:"timeout_sec"`
		MaxReqPerMin int    `yaml:"max_req_per_min"`
		UserAgent    string `yaml:"user_agent"`
	} `yaml:"opendota"`
	Redis struct {
		Addr     string `yaml:"addr"`
		Password string `yaml:"password"`
		DB       int    `yaml:"db"`
	} `yaml:"redis"`
	RabbitMQ struct {
		URL    string `yaml:"url"`
		Queues struct {
			MatchIDs      string `yaml:"match_ids"`
			RawMatches    string `yaml:"raw_matches"`
			MatchIDsDLQ   string `yaml:"match_ids_dlq"`
			RawMatchesDLQ string `yaml:"raw_matches_dlq"`
		} `yaml:"queues"`
	} `yaml:"rabbitmq"`
	Postgres struct {
		DSN string `yaml:"dsn"`
	} `yaml:"postgres"`
	Worker struct {
		Concurrency   int `yaml:"concurrency"`
		Prefetch      int `yaml:"prefetch"`
		MaxRetries    int `yaml:"max_retries"`
		RetryDelaySec int `yaml:"retry_delay_sec"`
		MetricsPort   int `yaml:"metrics_port"`
	} `yaml:"worker"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	expandedData := expandEnv(string(data))
	var cfg Config
	if err := yaml.NewDecoder(strings.NewReader(expandedData)).Decode(&cfg); err != nil {
		return nil, err
	}

	if cfg.OpenDota.APIURL == "" {
		return nil, fmt.Errorf("opendota.api_url is required")
	}
	if cfg.Worker.Concurrency <= 0 {
		cfg.Worker.Concurrency = 5
	}
	if cfg.Worker.Prefetch <= 0 {
		cfg.Worker.Prefetch = 10
	}
	if cfg.Worker.MaxRetries <= 0 {
		cfg.Worker.MaxRetries = 3
	}
	if cfg.Worker.MetricsPort <= 0 {
		cfg.Worker.MetricsPort = 9091
	}
	if cfg.OpenDota.MaxReqPerMin <= 0 {
		cfg.OpenDota.MaxReqPerMin = 50
	}

	return &cfg, nil
}
