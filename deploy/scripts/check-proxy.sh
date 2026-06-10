#!/usr/bin/env bash
set -o pipefail

# ---------- Configuration ----------
PROXY_LIST_FILE="proxy.txt"    # Checks local file first (can be .txt or .json)
# PROXY_LIST_URL="https://raw.githubusercontent.com/iplocate/free-proxy-list/refs/heads/main/all-proxies.txt"
PROXY_LIST_URL="https://raw.githubusercontent.com/wiki/gfpcom/free-proxy-list/lists/socks5.txt"
TARGET_URL="https://api.ipify.org/"
# TARGET_URL="https://api.opendota.com/api/health"
OUTPUT_DIR="deploy"
OUTPUT_FILE="$OUTPUT_DIR/proxy.txt"
THREADS=300
CONNECT_TIMEOUT=10
TIMEOUT=5

# ---------- Setup ----------
mkdir -p "$OUTPUT_DIR"
> "$OUTPUT_FILE"

TEMP_LIST="$(mktemp)"

cleanup() {
    rm -f "$TEMP_LIST"
}
trap cleanup EXIT INT TERM

export TARGET_URL OUTPUT_FILE CONNECT_TIMEOUT TIMEOUT

# ---------- Proxy check function ----------
check_proxy() {
    local input="$1"

    # Seed random to ensure true randomness across simultaneous subshells
    RANDOM=$(( $$ + $SRANDOM + $SECONDS ))

    local uas=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15"
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0"
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1"
    )
    local user_agent="${uas[$((RANDOM % 6))]}"

    local targets=()

    # If the protocol is already known (from JSON or Text), only test that ONE protocol.
    # Otherwise, brute-force test all 3 standard protocols.
    if [[ "$input" == *"://"* ]]; then
        targets=("$input")
    else
        targets=("http://${input}" "socks5://${input}" "socks4://${input}")
    fi

    for proxy in "${targets[@]}"; do
        echo -e "[\e[32m-CHECK-\e[0m] $proxy"
        # Test the proxy (silent, fast fail, ipv4 only)
        if [ "$(curl -4 -s -x "$proxy" -A "$user_agent" --connect-timeout "$CONNECT_TIMEOUT" -m "$TIMEOUT" -o /dev/null -w "%{http_code}" "$TARGET_URL" 2>/dev/null)" = "200" ]; then

            # Native >> is atomic. Safe for parallel threads.
            echo "$proxy" >> "$OUTPUT_FILE"
            echo -e "[\e[32mVALID\e[0m] $proxy"

            return 0 # Proxy found, stop checking other protocols for this IP
        fi
    done
}
export -f check_proxy

# ---------- Pre-Process List (Blazing Fast JSON & Text parsing) ----------
{
    if [[ -n "$PROXY_LIST_FILE" ]] && [[ -f "$PROXY_LIST_FILE" ]]; then
        echo "Reading proxies from local file: $PROXY_LIST_FILE..." >&2
        cat "$PROXY_LIST_FILE"
    else
        echo "Local file not found. Downloading proxies from URL..." >&2
        curl -fsSL "$PROXY_LIST_URL"
    fi
} | tr -d '\r' | \
sed 's/,/\n/g; s/{/\n{\n/g; s/}/\n}\n/g' | \
awk '
BEGIN { ip=""; port=""; proto=""; in_json=0; grab_proto=0 }
{
    # JSON object boundaries
    if ($0 ~ /\{/) { in_json=1; ip=""; port=""; proto=""; grab_proto=0; next }
    if ($0 ~ /\}/) {
        if (in_json && ip != "" && port != "") {
            # Attach protocol if we found one in the JSON
            if (proto != "") print proto "://" ip ":" port
            else print ip ":" port
        }
        in_json=0; ip=""; port=""; proto=""; grab_proto=0; next
    }

    if (in_json) {
        # 1. Grab IP
        if (match($0, /"ip"[ \t]*:[ \t]*"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+"/)) {
            s = substr($0, RSTART, RLENGTH)
            split(s, arr, "\"")
            ip = arr[4]
        }
        # 2. Grab Port
        else if (match($0, /"port"[ \t]*:[ \t]*"?[0-9]+"?[ \t]*/)) {
            s = substr($0, RSTART, RLENGTH)
            match(s, /[0-9]+/)
            port = substr(s, RSTART, RLENGTH)
        }
        # 3. Grab Protocol (e.g., "protocols": ["socks5"])
        else if (match($0, /"protocols"/)) {
            grab_proto=1
            # If the protocol is on the exact same line as the array
            if (match($0, /"protocols"[ \t]*:[ \t]*\[?[ \t]*"[a-zA-Z0-9]+"/)) {
                s = substr($0, RSTART, RLENGTH)
                split(s, arr, "\"")
                proto = arr[4]
                grab_proto=0
            }
        }
        # If the protocol was pushed to the next line down in the JSON array
        else if (grab_proto && match($0, /"[a-zA-Z0-9]+"/)) {
            s = substr($0, RSTART, RLENGTH)
            proto = substr(s, 2, length(s)-2)
            grab_proto=0
        }
    } else {
        # Plain Text format: optionally matches "socks5://" and "1.2.3.4:8080"
        if (match($0, /([a-zA-Z0-9]+:\/\/)?[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+/)) {
            print substr($0, RSTART, RLENGTH)
        }
    }
}' > "$TEMP_LIST"

# De-duplicate the list completely
sort -u -o "$TEMP_LIST" "$TEMP_LIST"

TOTAL_PROXIES=$(wc -l < "$TEMP_LIST")
if [ "$TOTAL_PROXIES" -eq 0 ]; then
    echo "ERROR: Proxy list is empty or failed to load." >&2
    exit 1
fi

echo "Loaded $TOTAL_PROXIES unique proxies."
echo "Running aggressively with $THREADS threads..."
echo "------------------------------------------------------"

# ---------- Run in parallel ----------
xargs -a "$TEMP_LIST" -n 1 -P "$THREADS" bash -c 'check_proxy "$1"' _

# ---------- Report ----------
echo "------------------------------------------------------"
VALID_COUNT=$(wc -l < "$OUTPUT_FILE" 2>/dev/null | awk '{print $1}' || echo 0)
echo "Checked IPs: $TOTAL_PROXIES"
echo "Valid Found: $VALID_COUNT"
echo "Saved to:    $OUTPUT_FILE"
