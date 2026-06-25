package config

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

var envPattern = regexp.MustCompile(`\$\{([^}]+)\}`)

func expandEnv(s string) string {
	return envPattern.ReplaceAllStringFunc(s, func(match string) string {
		inner := match[2 : len(match)-1]
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
		APIURL             string `yaml:"api_url"`
		BatchSize          int    `yaml:"batch_size"`
		FetchLastCountDay  int    `yaml:"fetch_last_count_day"`
		FetchLobbyTypesRaw string `yaml:"fetch_lobby_types"`
		FetchLobbyTypes    []int  `yaml:"-"`
		// WatermarkLookbackDays is the rolling window (in days) used by
		// the watermark-based query (see matches_watermark.sql). It must
		// be ≥ FetchLastCountDay so we never query a window narrower than
		// the bootstrap path. Default 30.
		WatermarkLookbackDays int `yaml:"watermark_lookback_days"`
		// Cron schedule for the fetch job, e.g. "@every 24h" or "0 3 * * *"
		FetchSchedule string `yaml:"fetch_schedule"`
		// StartRun triggers a one-shot fetch on application startup, after
		// the proxy pool reaches StartRunMinPoolSize healthy entries.
		// Useful for "warm up the pipeline" deployments where the first
		// cron tick would otherwise be hours away.
		StartRun bool `yaml:"start_run"`
		// StartRunMinPoolSize is the minimum proxy pool size required
		// before the startup fetch will run. Defaults to 20 to match the
		// proxy-manager's default PROXY_POOL_MIN_SIZE.
		StartRunMinPoolSize int `yaml:"start_run_min_pool_size"`
		// StartRunMaxWait is the maximum time the startup fetch will
		// wait for the pool to fill. After this, the startup fetch is
		// skipped (the regular cron schedule still runs). Default 5m.
		StartRunMaxWait time.Duration `yaml:"-"`
		// StartRunMaxWaitRaw is the YAML/env string form (e.g. "5m").
		StartRunMaxWaitRaw string `yaml:"start_run_max_wait"`
	} `yaml:"opendota"`
	Redis struct {
		Addr     string `yaml:"addr"`
		Password string `yaml:"password"`
		DB       int    `yaml:"db"`
	} `yaml:"redis"`
	RabbitMQ struct {
		URL    string `yaml:"url"`
		Queues struct {
			MatchIDs string `yaml:"match_ids"`
		} `yaml:"queues"`
	} `yaml:"rabbitmq"`
	// Postgres is used to read ingestion_checkpoints.last_parsed_match_id
	// on startup so the watermark-based fetch path can be used. It is also
	// used to skip match IDs already in the matches table (Layer 3 filter).
	// The connection is optional: if DSN is empty or the ping fails, the
	// id-fetcher logs a warning and falls back to the rolling-window
	// path (watermark=0).
	Postgres struct {
		DSN                  string `yaml:"dsn"`
		ForceDownloadRewrite bool   `yaml:"force_download_rewrite"`
	} `yaml:"postgres"`
	Worker struct {
		MetricsPort int `yaml:"metrics_port"`
	} `yaml:"worker"`
}

func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := yaml.NewDecoder(strings.NewReader(expandEnv(string(data)))).Decode(&cfg); err != nil {
		return nil, err
	}

	if cfg.OpenDota.APIURL == "" {
		return nil, fmt.Errorf("opendota.api_url is required")
	}
	if cfg.OpenDota.BatchSize <= 0 {
		cfg.OpenDota.BatchSize = 500
	}
	if cfg.OpenDota.FetchLastCountDay <= 0 {
		return nil, fmt.Errorf("FETCH_LAST_COUNT_DAY must be > 0")
	}
	// Default watermark lookback to fetch_last_count_day so the
	// watermark query always covers at least as wide a window as the
	// bootstrap path, regardless of how large FETCH_LAST_COUNT_DAY is.
	// Previously hard-coded to 30, which broke when FETCH_LAST_COUNT_DAY
	// exceeded that value.
	if cfg.OpenDota.WatermarkLookbackDays <= 0 {
		cfg.OpenDota.WatermarkLookbackDays = cfg.OpenDota.FetchLastCountDay
	}
	if cfg.OpenDota.WatermarkLookbackDays < cfg.OpenDota.FetchLastCountDay {
		return nil, fmt.Errorf(
			"watermark_lookback_days (%d) must be >= fetch_last_count_day (%d)",
			cfg.OpenDota.WatermarkLookbackDays, cfg.OpenDota.FetchLastCountDay)
	}
	if cfg.OpenDota.FetchSchedule == "" {
		cfg.OpenDota.FetchSchedule = "@every 24h"
	}

	if cfg.OpenDota.FetchLobbyTypesRaw == "" {
		return nil, fmt.Errorf("FETCH_LOBBY_TYPES is required (comma-separated, e.g. '1,2,6')")
	}
	for _, part := range strings.Split(cfg.OpenDota.FetchLobbyTypesRaw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid lobby type %q in FETCH_LOBBY_TYPES: %w", part, err)
		}
		cfg.OpenDota.FetchLobbyTypes = append(cfg.OpenDota.FetchLobbyTypes, n)
	}
	if len(cfg.OpenDota.FetchLobbyTypes) == 0 {
		return nil, fmt.Errorf("FETCH_LOBBY_TYPES must contain at least one value")
	}

	// Defaults for the opt-in startup fetch.
	if cfg.OpenDota.StartRunMinPoolSize <= 0 {
		cfg.OpenDota.StartRunMinPoolSize = 20
	}
	if cfg.OpenDota.StartRunMaxWaitRaw == "" {
		cfg.OpenDota.StartRunMaxWait = 5 * time.Minute
	} else {
		d, err := time.ParseDuration(cfg.OpenDota.StartRunMaxWaitRaw)
		if err != nil {
			return nil, fmt.Errorf("invalid start_run_max_wait %q: %w", cfg.OpenDota.StartRunMaxWaitRaw, err)
		}
		if d <= 0 {
			return nil, fmt.Errorf("start_run_max_wait must be > 0, got %q", cfg.OpenDota.StartRunMaxWaitRaw)
		}
		cfg.OpenDota.StartRunMaxWait = d
	}

	if cfg.Redis.Addr == "" {
		return nil, fmt.Errorf("redis.addr is required")
	}
	if cfg.Worker.MetricsPort <= 0 {
		cfg.Worker.MetricsPort = 9094
	}

	return &cfg, nil
}
