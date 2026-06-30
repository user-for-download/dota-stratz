package logger

import (
	"os"
	"strings"
	"sync"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var (
	Log     *zap.Logger
	logOnce sync.Once
)

// InitLogger initializes a global zap logger.
// Log level is read from the LOG_LEVEL env var (debug, info, warn, error).
// Defaults to "info" if unset or invalid. Safe to call multiple times — only
// the first invocation initialises the logger.
func InitLogger() {
	logOnce.Do(func() {
		level := zap.NewAtomicLevelAt(zap.InfoLevel)
		switch strings.ToLower(os.Getenv("LOG_LEVEL")) {
		case "debug":
			level = zap.NewAtomicLevelAt(zap.DebugLevel)
		case "info":
			level = zap.NewAtomicLevelAt(zap.InfoLevel)
		case "warn":
			level = zap.NewAtomicLevelAt(zap.WarnLevel)
		case "error":
			level = zap.NewAtomicLevelAt(zap.ErrorLevel)
		}

		config := zap.NewProductionConfig()
		config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
		config.Level = level

		logger, err := config.Build()
		if err != nil {
			panic("failed to initialize logger: " + err.Error())
		}

		Log = logger
	})
}

// Sync flushes any buffered log entries. Call this on main() exit.
func Sync() {
	if Log != nil {
		_ = Log.Sync()
	}
}
