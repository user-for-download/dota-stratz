#!/bin/sh
# ==============================================================================
# RabbitMQ user initialisation script.
# Creates one user per microservice with granular permissions.
#
# Usage: run this AFTER RabbitMQ is started and before any service connects.
#   docker exec dota2-rabbitmq ./init.sh
#
# Or mount this into the container at /docker-entrypoint-initdb.d/init.sh
# (RabbitMQ does NOT support this natively; use a post-start hook instead).
#
# The actual password values should match the per-service env vars in
# deploy/.env:
#   RABBITMQ_USER_ID_FETCHER_PASS
#   RABBITMQ_USER_DETAIL_FETCHER_PASS
#   RABBITMQ_USER_PARSER_PASS
# ==============================================================================

set -e

ID_FETCHER_PASS="${RABBITMQ_USER_ID_FETCHER_PASS:-idfetcher_secret}"
DETAIL_FETCHER_PASS="${RABBITMQ_USER_DETAIL_FETCHER_PASS:-detailfetcher_secret}"
PARSER_PASS="${RABBITMQ_USER_PARSER_PASS:-parser_secret}"

# Create users (idempotent — rabbitmqctl errors on duplicate, so we check first)
for user in id-fetcher detail-fetcher parser; do
    if ! rabbitmqctl list_users 2>/dev/null | grep -q "^$user\t"; then
        pass_var="RABBITMQ_USER_$(echo "$user" | tr '[:lower:]-' '[:upper:]_')_PASS"
        eval "pass=\${$pass_var}"
        rabbitmqctl add_user "$user" "$pass"
        echo "Created RabbitMQ user: $user"
    else
        echo "RabbitMQ user already exists: $user"
    fi
done

# Set granular permissions
# id-fetcher: write-only to queue.match_ids
rabbitmqctl set_permissions -p / id-fetcher "" "queue\.match_ids" ""

# detail-fetcher: read from queue.match_ids, write to queue.raw_matches
rabbitmqctl set_permissions -p / detail-fetcher "" "queue\.raw_matches" "queue\.match_ids"

# parser: read-only from queue.raw_matches
rabbitmqctl set_permissions -p / parser "" "" "queue\.raw_matches"

echo "RabbitMQ user initialisation complete."
