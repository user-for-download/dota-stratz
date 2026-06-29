package db

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConnect_InvalidDSN(t *testing.T) {
	// An invalid DSN should return a parse error without attempting a connection.
	pool, err := Connect(context.Background(), "invalid://not-a-valid-dsn", 0)
	require.Error(t, err)
	assert.Nil(t, pool)
	assert.Contains(t, err.Error(), "unable to parse database DSN")
}

func TestConnect_EmptyDSN(t *testing.T) {
	pool, err := Connect(context.Background(), "", 0)
	require.Error(t, err)
	assert.Nil(t, pool)
}

func TestConnect_RejectsUnknownHost(t *testing.T) {
	// A syntactically valid DSN that points to nowhere should fail to connect.
	// This tests the Ping path (the second error source in Connect).
	pool, err := Connect(context.Background(), "postgres://localhost:1/dota2?sslmode=disable", 5)
	require.Error(t, err)
	assert.Nil(t, pool)
}
