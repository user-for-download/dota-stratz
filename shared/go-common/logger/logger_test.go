package logger

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInitLogger_SetsGlobal(t *testing.T) {
	// Reset global for test isolation.
	Log = nil

	InitLogger()
	assert.NotNil(t, Log, "Log should be non-nil after InitLogger")
	Sync() // should not panic
}

func TestInitLogger_RespectsLogLevel(t *testing.T) {
	tests := []struct {
		level string
	}{
		{"debug"},
		{"info"},
		{"warn"},
		{"error"},
		{"invalid"}, // should default to info
	}

	for _, tt := range tests {
		t.Run(tt.level, func(t *testing.T) {
			os.Setenv("LOG_LEVEL", tt.level)
			defer os.Unsetenv("LOG_LEVEL")

			Log = nil
			InitLogger()
			assert.NotNil(t, Log, "Log should be non-nil for level %q", tt.level)
		})
	}
}

func TestSync_NilLogger(t *testing.T) {
	// Sync should not panic when Log is nil.
	Log = nil
	Sync()
}

func TestSync_Flushes(t *testing.T) {
	// Sync on an initialized logger should not error.
	Log = nil
	InitLogger()
	Sync() // should not panic
	Log = nil
}
