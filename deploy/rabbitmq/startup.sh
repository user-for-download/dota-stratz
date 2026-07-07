#!/bin/bash
# ==============================================================================
# RabbitMQ startup wrapper.
#
# Runs as root (CMD doesn't match "rabbitmq*") so we can fix volume
# permissions, then starts RabbitMQ via the official entrypoint (as the
# rabbitmq user for proper setup), waits for it to be ready with retries,
# and runs init.sh to create per-service users automatically.
# ==============================================================================

set -euo pipefail

# 1. Fix data directory ownership — fresh volumes are root-owned, and the
#    official entrypoint only does this when CMD is exactly "rabbitmq-server".
find /var/lib/rabbitmq \! -user rabbitmq -exec chown rabbitmq '{}' +

# 2. Start RabbitMQ via the official entrypoint (handles cookie, nodename,
#    env vars, then exec's rabbitmq-server) in the background.
#    Using the entrypoint ensures all setup steps are executed correctly.
su-exec rabbitmq /usr/local/bin/docker-entrypoint.sh rabbitmq-server "$@" &
RABBITMQ_PID=$!

# 3. Wait for RabbitMQ to be fully started, with retries.
#    rabbitmqctl await_startup can fail if the Erlang VM hasn't created the
#    cookie yet, so we loop.
echo "Waiting for RabbitMQ to start..."
for i in $(seq 1 60); do
    if su-exec rabbitmq rabbitmqctl await_startup 2>/dev/null; then
        echo "RabbitMQ is up — running init.sh"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: RabbitMQ failed to start within 60 seconds"
        kill "$RABBITMQ_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# 4. Create per-service users (idempotent — safe to run every boot).
su-exec rabbitmq /workspace/init.sh

# 5. Bring RabbitMQ back to the foreground for proper signal handling.
wait "$RABBITMQ_PID"
