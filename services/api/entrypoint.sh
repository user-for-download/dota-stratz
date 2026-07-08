#!/bin/sh
# Entrypoint wrapper for API service.
# Runs as root to fix volume permissions, then drops to appuser via gosu.

set -e

# Fix /models permissions (we are root here)
if [ -d /models ]; then
    chown -R appuser:appgroup /models
    chmod -R 775 /models
fi

# Drop privileges and execute the main command as appuser
exec gosu appuser "$@"
