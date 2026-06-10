#!/bin/bash
# Fetch OpenDota constants and seed the const_* tables.
# Idempotent: uses ON CONFLICT (id) DO NOTHING.
#
# Usage: ./deploy/scripts/seed-constants.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Use docker exec to run psql inside the postgres container
DOCKER_PSQL="docker exec -i dota2-postgres psql -U dota2 -d dota2"

# Escape single quotes for SQL safety: ' -> ''
sql_escape() {
    printf "%s" "$1" | sed "s/'/''/g"
}

echo "Fetching OpenDota constants..."

# Heroes
curl -s 'https://api.opendota.com/api/heroes' -o /tmp/heroes.json
echo "Seeding const_hero..."
python3 -c "
import json
heroes = json.load(open('/tmp/heroes.json'))
def esc(s): return str(s).replace(chr(39), chr(39)*2)
for h in heroes:
    roles_arr = h.get('roles') or []
    # Build PostgreSQL array literal: {Carry,Escape,Nuker}
    roles_lit = '{' + ','.join(esc(r) for r in roles_arr) + '}' if roles_arr else '{}'
    name = esc(h['name'])
    localized = esc(h['localized_name'])
    pa = esc(h.get('primary_attr',''))
    at = esc(h.get('attack_type',''))
    print(f\"INSERT INTO const_hero (id, name, localized_name, primary_attr, attack_type, roles, legs) VALUES ({h['id']}, '{name}', '{localized}', '{pa}', '{at}', '{roles_lit}', {h.get('legs', 0)}) ON CONFLICT (id) DO NOTHING;\")
" > /tmp/heroes_seed.sql
$DOCKER_PSQL < /tmp/heroes_seed.sql 2>&1 | tail -3
echo "Heroes seeded."

# Items (if table exists)
curl -s 'https://api.opendota.com/api/constants/items' -o /tmp/items.json
echo "Seeding const_item..."
python3 -c "
import json
items = json.load(open('/tmp/items.json'))
for k, v in items.items():
    if not v.get('id'):
        continue
    name = v.get('dname', k).replace(\"'\", \"''\")
    img = v.get('img', '')
    print(f\"INSERT INTO const_item (id, name, img, cost) VALUES ({v['id']}, '{name}', '{img}', {v.get('cost', 0)}) ON CONFLICT (id) DO NOTHING;\")
" > /tmp/items_seed.sql
$DOCKER_PSQL < /tmp/items_seed.sql 2>&1 | tail -3
echo "Items seeded."

# Abilities
curl -s 'https://api.opendota.com/api/constants/abilities' -o /tmp/abilities.json
echo "Seeding const_ability..."
python3 -c "
import json
abs = json.load(open('/tmp/abilities.json'))
for k, v in abs.items():
    if not v.get('id'):
        continue
    name = v.get('dname', k).replace(\"'\", \"''\")
    print(f\"INSERT INTO const_ability (id, name) VALUES ({v['id']}, '{name}') ON CONFLICT (id) DO NOTHING;\")
" > /tmp/abilities_seed.sql
$DOCKER_PSQL < /tmp/abilities_seed.sql 2>&1 | tail -3
echo "Abilities seeded."

echo "Done."
