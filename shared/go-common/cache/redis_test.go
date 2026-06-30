package cache

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestConnect_RetriesOnFailure(t *testing.T) {
	// No Redis server running on this address — should fail after retries.
	rdb, err := Connect(context.Background(), "127.0.0.1:1", "", 0)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to connect to Redis")
	assert.Nil(t, rdb)
}

func TestConnect_InvalidHost(t *testing.T) {
	rdb, err := Connect(context.Background(), "invalid:abc", "", 0)
	require.Error(t, err)
	assert.Nil(t, rdb)
}

func TestConnect_EmptyAddr(t *testing.T) {
	rdb, err := Connect(context.Background(), "", "", 0)
	require.Error(t, err)
	assert.Nil(t, rdb)
}
