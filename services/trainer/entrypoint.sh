#!/bin/sh
# Entrypoint: runs as root to fix volume permissions, then exec's the CMD.

set -e

# Fix /models permissions (container starts as root)
if [ -d /models ]; then
    chown -R appuser:appgroup /models 2>/dev/null || true
    chmod -R 775 /models 2>/dev/null || true
fi

# If gosu is available, drop privileges; otherwise exec directly
if command -v gosu >/dev/null 2>&1; then
    exec gosu appuser "$@"
else
    exec "$@"
fi
