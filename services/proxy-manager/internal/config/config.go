package config

import (
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	// Redis
	RedisAddr     string
	RedisPassword string
	RedisDB       int64

	// Sources
	ProxyFilePath    string
	RefreshSourceURL string
	SourceUserAgent  string

	// Validation
	ValidationTargetURL   string
	ValidationUserAgent   string
	ValidationTimeoutSec  int64
	SourceFetchTimeoutSec int64
	Concurrency           int64

	// Refresh & pool
	RefreshIntervalMin     int64
	SourceFetchCooldownMin int64
	RotationStrategy       string
	PoolMaxSize            int64
	PoolMinSize            int64

	// Invalidation & leasing
	SoftFailThreshold      int64
	SoftRetryDelaySec      int64
	CooldownMin            int64
	LeaseDurationSec       int64
	LeaseReaperIntervalSec int64
	FailureCounterTTLMin   int64

	// Lifecycle
	ShutdownGraceMs int64

	// Observability
	MetricsPort int64
}

func Load() (*Config, error) {
	redisHost, err := requireEnv("REDIS_HOST")
	if err != nil {
		return nil, err
	}
	redisPort, err := requireEnv("REDIS_PORT")
	if err != nil {
		return nil, err
	}

	cfg := &Config{}
	cfg.RedisAddr = redisHost + ":" + redisPort
	cfg.RedisPassword = os.Getenv("REDIS_PASSWORD") // allowed empty

	// Load all required env vars with proper error propagation.
	pairs := []struct {
		key string
		dst *string
	}{
		{"PROXY_FILE_PATH", &cfg.ProxyFilePath},
		{"PROXY_SOURCE_USER_AGENT", &cfg.SourceUserAgent},
		{"PROXY_VALIDATION_TARGET_URL", &cfg.ValidationTargetURL},
		{"PROXY_VALIDATOR_USER_AGENT", &cfg.ValidationUserAgent},
		{"PROXY_ROTATION_STRATEGY", &cfg.RotationStrategy},
	}
	for _, p := range pairs {
		v, err := requireEnv(p.key)
		if err != nil {
			return nil, err
		}
		*p.dst = v
	}
	cfg.RefreshSourceURL = os.Getenv("PROXY_REFRESH_SOURCE_URL") // empty disables loop

	// Load all required int env vars.
	intPairs := []struct {
		key string
		dst *int64
	}{
		{"REDIS_DB", &cfg.RedisDB},
		{"PROXY_VALIDATION_TIMEOUT_SEC", &cfg.ValidationTimeoutSec},
		{"PROXY_SOURCE_FETCH_TIMEOUT_SEC", &cfg.SourceFetchTimeoutSec},
		{"PROXY_VALIDATION_CONCURRENCY", &cfg.Concurrency},
		{"PROXY_REFRESH_INTERVAL_MIN", &cfg.RefreshIntervalMin},
		{"PROXY_POOL_MAX_SIZE", &cfg.PoolMaxSize},
		{"PROXY_POOL_MIN_SIZE", &cfg.PoolMinSize},
		{"PROXY_SOFT_FAIL_THRESHOLD", &cfg.SoftFailThreshold},
		{"PROXY_SOFT_RETRY_DELAY_SEC", &cfg.SoftRetryDelaySec},
		{"PROXY_COOLDOWN_MIN", &cfg.CooldownMin},
		{"PROXY_LEASE_DURATION_SEC", &cfg.LeaseDurationSec},
		{"PROXY_LEASE_REAPER_INTERVAL_SEC", &cfg.LeaseReaperIntervalSec},
		{"PROXY_FAILURE_COUNTER_TTL_MIN", &cfg.FailureCounterTTLMin},
		{"PROXY_SHUTDOWN_GRACE_MS", &cfg.ShutdownGraceMs},
		{"PROXY_METRICS_PORT", &cfg.MetricsPort},
	}
	for _, p := range intPairs {
		n, err := requireEnvInt(p.key)
		if err != nil {
			return nil, err
		}
		*p.dst = n
	}

	// Source fetch cooldown (optional, defaults to 10 min)
	cfg.SourceFetchCooldownMin = 10
	if v := os.Getenv("PROXY_SOURCE_FETCH_COOLDOWN_MIN"); v != "" {
		n, err := strconv.ParseInt(v, 10, 64)
		if err == nil {
			cfg.SourceFetchCooldownMin = n
		}
	}

	// Validate enums / ranges centrally
	if cfg.RotationStrategy != "timestamp" && cfg.RotationStrategy != "random" {
		return nil, fmt.Errorf("PROXY_ROTATION_STRATEGY must be 'timestamp' or 'random', got %q", cfg.RotationStrategy)
	}
	if cfg.PoolMinSize >= cfg.PoolMaxSize {
		return nil, fmt.Errorf("PROXY_POOL_MIN_SIZE (%d) must be < PROXY_POOL_MAX_SIZE (%d)",
			cfg.PoolMinSize, cfg.PoolMaxSize)
	}
	return cfg, nil
}

func requireEnv(key string) (string, error) {
	v := os.Getenv(key)
	if v == "" {
		return "", fmt.Errorf("required env var %s is not set", key)
	}
	return v, nil
}

func requireEnvInt(key string) (int64, error) {
	v, err := requireEnv(key)
	if err != nil {
		return 0, err
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("env var %s must be int, got %q", key, v)
	}
	return n, nil
}
