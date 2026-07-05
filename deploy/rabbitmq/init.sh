#!/bin/sh
# ==============================================================================
# RabbitMQ user initialisation script.
# Creates one user per microservice with granular permissions.
#
# Usage: run this AFTER RabbitMQ is started and before any service connects.
#   docker exec -e RABBITMQ_USER_PARSER_PASS=parser_secret \
#                -e RABBITMQ_USER_ID_FETCHER_PASS=idfetcher_secret \
#                -e RABBITMQ_USER_DETAIL_FETCHER_PASS=detailfetcher_secret \
#                dota2-rabbitmq /init.sh
#
# The actual password values should match the per-service env vars in
# deploy/.env:
#   RABBITMQ_USER_ID_FETCHER_PASS
#   RABBITMQ_USER_DETAIL_FETCHER_PASS
#   RABBITMQ_USER_PARSER_PASS
# ==============================================================================

set -e

# Passwords come from env vars (must be passed via docker exec -e).
# Fallbacks here only for local dev convenience.
ID_FETCHER_PASS="${RABBITMQ_USER_ID_FETCHER_PASS:-idfetcher_secret}"
DETAIL_FETCHER_PASS="${RABBITMQ_USER_DETAIL_FETCHER_PASS:-detailfetcher_secret}"
PARSER_PASS="${RABBITMQ_USER_PARSER_PASS:-parser_secret}"

# Create users (idempotent — rabbitmqctl errors on duplicate, so we check first)
for user in id-fetcher detail-fetcher parser; do
    if ! rabbitmqctl list_users 2>/dev/null | grep -q "^$user[[:space:]]"; then
        pass_var="RABBITMQ_USER_$(echo "$user" | tr '[:lower:]-' '[:upper:]_')_PASS"
        eval "pass=\${$pass_var:-}"
        if [ -z "$pass" ]; then
            echo "ERROR: $pass_var is not set. Cannot create user $user."
            exit 1
        fi
        rabbitmqctl add_user "$user" "$pass"
        echo "Created RabbitMQ user: $user"
    else
        echo "RabbitMQ user already exists: $user"
    fi
done

# Set granular permissions
# RabbitMQ 4.x behaviour: queue declaration on the default exchange checks
# write permission against the exchange name "amq.default", not the routing
# key.  Using .* for write is the simplest fix; configure/read still limit
# each service to its own queues.
# id-fetcher: configures and writes to queue.match_ids; read needed for QueueDeclare (RabbitMQ 4.x)
rabbitmqctl set_permissions -p / id-fetcher "queue\.match_ids(\.dlq)?" ".*" "queue\.match_ids(\.dlq)?"

# detail-fetcher: configures/writes/reads match_ids; configures/writes raw_matches
rabbitmqctl set_permissions -p / detail-fetcher "queue\.(match_ids|raw_matches)(\.dlq)?" ".*" "queue\.(match_ids|raw_matches)(\.dlq)?"

# parser: configure and read queue.raw_matches and its DLQ
rabbitmqctl set_permissions -p / parser "queue\.raw_matches(\.dlq)?" ".*" "queue\.raw_matches(\.dlq)?"

echo "RabbitMQ user initialisation complete."
