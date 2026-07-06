#!/usr/bin/env bash
# ============================================================================
# replay-dlq.sh — Drain queue.match_ids.dlq, dedupe against the matches
# table, and republish only the truly-missing match_ids to queue.match_ids.
#
# Idempotent: safe to run multiple times. Matches already in the database
# (whether from a previous successful fetch or from a previous replay) are
# skipped.
#
# Usage:
#   replay-dlq.sh [MAX_PER_RUN] [OPTIONS]
#
# Arguments:
#   MAX_PER_RUN            max messages to drain from the DLQ (default: 500)
#
# Options:
#   --dry-run              show what would be done; do not republish
#   --no-skip-existing     republish every drained match_id (use with caution)
#   -h, --help             show this help
#
# Environment overrides:
#   DLQ, TARGET_QUEUE                  queue names
#   RABBITMQ_HOST/PORT, RABBITMQ_DEFAULT_USER/PASS
#   POSTGRES_HOST, POSTGRES_USER, POSTGRES_DB
#   POSTGRES_CONTAINER (default: dota2-postgres)
#
# Recommended cron schedule (every 6 hours):
#   0 */6 * * * /opt/dota2-stratz/deploy/scripts/replay-dlq.sh 500
# ============================================================================

set -euo pipefail

# ---------- Defaults & arg parsing -------------------------------------------

MAX_PER_RUN=500
DRY_RUN=false
SKIP_EXISTING=true

usage() {
  sed -n '2,/^# ====/p' "$0" | sed 's/^# \?//' | head -25
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)             DRY_RUN=true; shift ;;
    --skip-existing)       SKIP_EXISTING=true; shift ;;
    --no-skip-existing)    SKIP_EXISTING=false; shift ;;
    -h|--help)             usage; exit 0 ;;
    [0-9]*)                MAX_PER_RUN="$1"; shift ;;
    *)                     echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# ---------- Config (env-overridable) -----------------------------------------

DLQ="${DLQ:-queue.match_ids.dlq}"
TARGET_QUEUE="${TARGET_QUEUE:-queue.match_ids}"
RABBIT_HOST="${RABBITMQ_HOST:-localhost}"
RABBITMQ_MANAGEMENT_PORT="${RABBITMQ_MANAGEMENT_PORT:-${RABBITMQ_PORT:-15672}}"
RABBIT_PORT="$RABBITMQ_MANAGEMENT_PORT"
RABBIT_USER="${RABBITMQ_DEFAULT_USER:-guest}"
RABBIT_PASS="${RABBITMQ_DEFAULT_PASS:-guest}"
RABBIT_API="http://${RABBIT_HOST}:${RABBIT_PORT}/api"
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-dota2-postgres}"
POSTGRES_USER="${POSTGRES_USER:-dota2}"
POSTGRES_DB="${POSTGRES_DB:-dota2}"

# ---------- Helpers ----------------------------------------------------------

log()  { echo "[$(date -u +%H:%M:%S)] $*"; }
warn() { echo "[$(date -u +%H:%M:%S)] WARNING: $*" >&2; }
fail() { echo "[$(date -u +%H:%M:%S)] ERROR: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command '$1' not found in PATH"
}

require_cmd curl
require_cmd docker
require_cmd python3

# ---------- Preflight --------------------------------------------------------

log "=== DLQ replay tool ==="
log "  DLQ:              $DLQ"
log "  Target queue:     $TARGET_QUEUE"
log "  Max per run:      $MAX_PER_RUN"
log "  Skip existing:    $SKIP_EXISTING"
log "  Dry run:          $DRY_RUN"
log "  RabbitMQ API:     $RABBIT_API"
log "  Postgres ctr:     $POSTGRES_CONTAINER / db: $POSTGRES_DB"
echo ""

# Sanity check RabbitMQ reachability
if ! curl -sf -o /dev/null -u "${RABBIT_USER}:${RABBIT_PASS}" \
     "${RABBIT_API}/overview"; then
  fail "RabbitMQ management API not reachable at ${RABBIT_API}. Check RABBITMQ_HOST."
fi
log "RabbitMQ API reachable."

# Sanity check Postgres
if ! docker exec "$POSTGRES_CONTAINER" pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
     >/dev/null 2>&1; then
  fail "Postgres container '$POSTGRES_CONTAINER' not ready."
fi
log "Postgres ready."

# ---------- 1. Drain the DLQ -------------------------------------------------
#
# IMPORTANT: In dry-run mode we use ackmode=ack_requeue_true so the DLQ is
# left untouched. In real-run mode we use ackmode=ack_requeue_false to
# actually drain the messages. The trade-off: in real-run mode, messages
# that we fail to republish are LOST — the cron will re-fetch them on the
# next run, but if you need stronger delivery guarantees, increase
# max_retries on the detail-fetcher instead.
# -----------------------------------------------------------------------------

if [[ "$DRY_RUN" == "true" ]]; then
  ACK_MODE="ack_requeue_true"
  log "[1/4] Peeking at up to $MAX_PER_RUN messages from $DLQ (dry run)..."
else
  ACK_MODE="ack_requeue_false"
  log "[1/4] Draining up to $MAX_PER_RUN messages from $DLQ (real run)..."
fi
DRAIN_RESPONSE=$(curl -sf -u "${RABBIT_USER}:${RABBIT_PASS}" \
  "${RABBIT_API}/queues/%2F/${DLQ}/get" \
  -H 'content-type: application/json' \
  -d "{\"count\":${MAX_PER_RUN},\"ackmode\":\"${ACK_MODE}\",\"encoding\":\"auto\"}")

# Parse the response — if queue is empty, RabbitMQ returns []
MSG_COUNT=$(printf '%s' "$DRAIN_RESPONSE" | python3 -c "
import json, sys
try:
  print(len(json.load(sys.stdin)))
except Exception:
  print(0)
" 2>/dev/null)
log "  Drained $MSG_COUNT message(s)."

if [[ "$MSG_COUNT" -eq 0 ]]; then
  log "Nothing to replay. Exiting."
  exit 0
fi

# ---------- 2. Extract match_ids from payloads -------------------------------

log "[2/4] Extracting match_id from each payload..."
MATCH_IDS_CSV=$(printf '%s' "$DRAIN_RESPONSE" | python3 -c "
import json, sys
msgs = json.load(sys.stdin)
ids = []
for m in msgs:
  try:
    payload = m.get('payload') or '{}'
    # Payload might already be a dict (some clients publish dicts) or a JSON string
    if isinstance(payload, str):
      p = json.loads(payload)
    else:
      p = payload
    mid = p.get('match_id') or p.get('matchId')
    if mid is not None:
      ids.append(str(int(mid)))   # normalize to int-as-string
  except Exception as e:
    print(f'  parse error on message: {e}', file=sys.stderr)
print(','.join(ids))
")
MATCH_ID_COUNT=$(printf '%s' "$MATCH_IDS_CSV" | tr ',' '\n' | grep -c '^[0-9]' || true)
log "  Extracted $MATCH_ID_COUNT unique-or-not match_id(s)."

if [[ -z "$MATCH_IDS_CSV" ]]; then
  warn "No parseable match_ids in the drained messages. They may use an unknown schema."
  log "Sample of first drained payload:"
  printf '%s' "$DRAIN_RESPONSE" | python3 -c "
import json, sys
msgs = json.load(sys.stdin)
if msgs:
  print(json.dumps(msgs[0], indent=2)[:500])
"
  exit 0
fi

# ---------- 3. Bulk SQL — which IDs are already in DB? -----------------------
#
# Optimization (per pro-tip): one bulk SQL query for the dedupe step. Avoids
# the ~100ms-per-call overhead of docker exec + psql for each individual
# match_id. For very large DLQs (>>10K) this single query is still fine; for
# 100K+ you'd want to batch the IN clause.
# -----------------------------------------------------------------------------

EXISTING_FILE=$(mktemp)
TO_PUBLISH_FILE=$(mktemp)
trap 'rm -f "$EXISTING_FILE" "$TO_PUBLISH_FILE"' EXIT

if [[ "$SKIP_EXISTING" == "true" ]]; then
  log "[3/4] Checking which match_ids are already in the matches table (single bulk query)..."
  docker exec -i "$POSTGRES_CONTAINER" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
    "SELECT match_id FROM matches WHERE match_id IN (${MATCH_IDS_CSV})" \
    > "$EXISTING_FILE" 2>/dev/null
  EXISTING_COUNT=$(grep -c '^[0-9]' "$EXISTING_FILE" 2>/dev/null || true)
  EXISTING_COUNT=${EXISTING_COUNT:-0}
  log "  Found $EXISTING_COUNT in DB (will be skipped)."

  # Set difference: IDs in MATCH_IDS_CSV that are NOT in EXISTING_FILE
  # comm requires sorted unique inputs
  printf '%s\n' "${MATCH_IDS_CSV//,/$'\n'}" | sort -un > "$TO_PUBLISH_FILE.unsorted"
  comm -23 "$TO_PUBLISH_FILE.unsorted" <(sort -un "$EXISTING_FILE") > "$TO_PUBLISH_FILE" \
    || true
  rm -f "$TO_PUBLISH_FILE.unsorted"
  TO_PUBLISH_COUNT=$(grep -c '^[0-9]' "$TO_PUBLISH_FILE" 2>/dev/null || true)
  TO_PUBLISH_COUNT=${TO_PUBLISH_COUNT:-0}
else
  log "[3/4] Skipping dedupe (--no-skip-existing). Will republish all $MATCH_ID_COUNT."
  printf '%s\n' "${MATCH_IDS_CSV//,/$'\n'}" | sort -un > "$TO_PUBLISH_FILE"
  TO_PUBLISH_COUNT=$MATCH_ID_COUNT
  EXISTING_COUNT=0
fi

# ---------- 4. Republish the missing ones -----------------------------------

log "[4/4] Republishing $TO_PUBLISH_COUNT match_id(s) to $TARGET_QUEUE..."
if [[ "$TO_PUBLISH_COUNT" -eq 0 ]]; then
  log "  Nothing to publish (all drained messages are already in the DB)."
  log ""
  log "=== Summary ==="
  log "  Drained:        $MSG_COUNT"
  log "  Already in DB:  $EXISTING_COUNT (skipped)"
  log "  Republished:    0"
  log "  Failed:         0"
  exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
  log "  (dry run — would publish the following $TO_PUBLISH_COUNT match_id(s))"
  head -20 "$TO_PUBLISH_FILE" | sed 's/^/    /'
  if [[ "$TO_PUBLISH_COUNT" -gt 20 ]]; then
    log "    ... and $((TO_PUBLISH_COUNT - 20)) more"
  fi
  log ""
  log "=== Summary (DRY RUN) ==="
  log "  Drained:        $MSG_COUNT"
  log "  Already in DB:  $EXISTING_COUNT (skipped)"
  log "  Would publish:  $TO_PUBLISH_COUNT"
  log "  Failed:         0"
  exit 0
fi

PUBLISHED=0
FAILED=0
FAILED_IDS=""
while IFS= read -r MID; do
  [[ -z "$MID" ]] && continue
  JSON_BODY=$(python3 -c "
import json
mid = ${MID}
print(json.dumps({
    'properties': {},
    'routing_key': '${TARGET_QUEUE}',
    'payload': json.dumps({'match_id': mid}),
    'payload_encoding': 'string'
}))
")
  RESP=$(curl -sf -u "${RABBIT_USER}:${RABBIT_PASS}" \
    "${RABBIT_API}/exchanges/%2F/amq.default/publish" \
    -H 'content-type: application/json' \
    -d "$JSON_BODY" \
    2>&1) || { FAILED=$((FAILED+1)); FAILED_IDS="$FAILED_IDS $MID"; continue; }
  if echo "$RESP" | grep -q '"routed":true'; then
    PUBLISHED=$((PUBLISHED+1))
  else
    FAILED=$((FAILED+1))
    FAILED_IDS="$FAILED_IDS $MID"
  fi
done < "$TO_PUBLISH_FILE"

log ""
log "=== Summary ==="
log "  Drained:        $MSG_COUNT"
log "  Already in DB:  $EXISTING_COUNT (skipped)"
log "  Republished:    $PUBLISHED"
log "  Failed:         $FAILED"
if [[ -n "$FAILED_IDS" ]]; then
  warn "Failed match_ids:$FAILED_IDS"
fi
log ""
log "DLQ depth after run:"
sleep 1
DLQ_NOW=$(curl -sf -u "${RABBIT_USER}:${RABBIT_PASS}" \
  "${RABBIT_API}/queues/%2F/${DLQ}" | python3 -c "
import json, sys
try:
  print(json.load(sys.stdin).get('messages', '?'))
except: print('?')
")
TARGET_NOW=$(curl -sf -u "${RABBIT_USER}:${RABBIT_PASS}" \
  "${RABBIT_API}/queues/%2F/${TARGET_QUEUE}" | python3 -c "
import json, sys
try:
  print(json.load(sys.stdin).get('messages', '?'))
except: print('?')
")
log "  $DLQ: $DLQ_NOW msg(s)"
log "  $TARGET_QUEUE: $TARGET_NOW msg(s)"

# Exit non-zero if anything failed to publish
if [[ "$FAILED" -gt 0 ]]; then
  exit 2
fi
