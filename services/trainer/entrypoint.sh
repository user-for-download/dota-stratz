#!/bin/sh
# Entrypoint wrapper for trainer service.
# Ensures /models directory is writable before running the actual command.
# Handles permission issues when the volume is first created.

set -e

# Try to ensure /models is writable
# If we can't chown (not root), that's okay — the volume might already have correct permissions
if [ -d /models ]; then
    # Check if we can write to /models
    if ! touch /models/.write_test 2>/dev/null; then
        echo "Warning: Cannot write to /models, attempting to fix permissions..."
        # Try to chown (will fail if not root, but that's okay)
        chown -R appuser:appgroup /models 2>/dev/null || true
        chmod -R 775 /models 2>/dev/null || true
    else
        rm -f /models/.write_test
    fi
fi

# Execute the actual command
exec "$@"
