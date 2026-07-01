#!/bin/bash
# Fetch OpenDota constants and seed the const_* tables.
# Idempotent: uses ON CONFLICT (id) DO NOTHING.
#
# Usage: ./deploy/scripts/seed-constants.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Use docker exec to run psql inside the postgres container
POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-dota2-postgres}"
POSTGRES_USER="${POSTGRES_USER:-dota2}"
POSTGRES_DB="${POSTGRES_DB:-dota2}"
DOCKER_PSQL="docker exec -i $POSTGRES_CONTAINER psql -U $POSTGRES_USER -d $POSTGRES_DB"

# Escape single quotes for SQL safety: ' -> ''
sql_escape() {
    printf "%s" "$1" | sed "s/'/''/g"
}

echo "Fetching OpenDota constants..."

# ── Heroes ──────────────────────────────────────────────────────────────────
curl -s 'https://api.opendota.com/api/heroes' -o /tmp/heroes.json
echo "Seeding const_hero..."
python3 << 'PYEOF' > /tmp/heroes_seed.sql
import json
heroes = json.load(open('/tmp/heroes.json'))
def esc(s): return str(s).replace(chr(39), chr(39)*2)
for h in heroes:
    roles_arr = h.get('roles') or []
    roles_lit = '{' + ','.join(esc(r) for r in roles_arr) + '}' if roles_arr else '{}'
    name = esc(h['name'])
    localized = esc(h['localized_name'])
    pa = esc(h.get('primary_attr',''))
    at = esc(h.get('attack_type',''))
    print(f"INSERT INTO const_hero (id, name, localized_name, primary_attr, attack_type, roles, legs) VALUES ({h['id']}, '{name}', '{localized}', '{pa}', '{at}', '{roles_lit}', {h.get('legs', 0)}) ON CONFLICT (id) DO NOTHING;")
PYEOF
$DOCKER_PSQL < /tmp/heroes_seed.sql 2>&1 | tail -3
echo "Heroes seeded."

# ── Items ───────────────────────────────────────────────────────────────────
curl -s 'https://api.opendota.com/api/constants/items' -o /tmp/items.json
echo "Seeding const_item..."
python3 << 'PYEOF' > /tmp/items_seed.sql
import json
items = json.load(open('/tmp/items.json'))
for k, v in items.items():
    if not v.get('id'):
        continue
    name = v.get('dname', k).replace("'", "''")
    img = v.get('img', '').replace("'", "''")
    print(f"INSERT INTO const_item (id, name, img, cost) VALUES ({v['id']}, '{name}', '{img}', {v.get('cost', 0)}) ON CONFLICT (id) DO NOTHING;")
PYEOF
$DOCKER_PSQL < /tmp/items_seed.sql 2>&1 | tail -3
echo "Items seeded."

# ── Abilities ───────────────────────────────────────────────────────────────
curl -s 'https://api.opendota.com/api/constants/ability_ids' -o /tmp/ability_ids.json
curl -s 'https://api.opendota.com/api/constants/abilities' -o /tmp/abilities.json
echo "Seeding const_ability..."
python3 << 'PYEOF' > /tmp/abilities_seed.sql
import json
ids = json.load(open('/tmp/ability_ids.json'))
abs = json.load(open('/tmp/abilities.json'))
for id_str, ability_key in ids.items():
    if ability_key.startswith('empty') or ability_key in ('dota_base_ability', 'courier_autodeliver'):
        continue
    info = abs.get(ability_key, {})
    key_esc = ability_key.replace("'", "''")
    dname_esc = info.get('dname', ability_key).replace("'", "''")
    img_esc = info.get('img', '').replace("'", "''")
    print(f"INSERT INTO const_ability (name, dname, img) VALUES ('{key_esc}', '{dname_esc}', '{img_esc}') ON CONFLICT (name) DO NOTHING;")
PYEOF
$DOCKER_PSQL < /tmp/abilities_seed.sql 2>&1 | tail -3
echo "Abilities seeded."

# ── Ability IDs ─────────────────────────────────────────────────────────────
echo "Seeding const_ability_id..."
python3 << 'PYEOF' > /tmp/ability_ids_seed.sql
import json
ids = json.load(open('/tmp/ability_ids.json'))
for id_str, ability_key in ids.items():
    try:
        aid = int(id_str)
    except ValueError:
        continue  # skip malformed keys like '3060,1617'
    key_esc = ability_key.replace("'", "''")
    print(f"INSERT INTO const_ability_id (id, name) VALUES ({aid}, '{key_esc}') ON CONFLICT (id) DO NOTHING;")
PYEOF
$DOCKER_PSQL < /tmp/ability_ids_seed.sql 2>&1 | tail -3
echo "Ability IDs seeded."

# ── Item IDs ────────────────────────────────────────────────────────────────
curl -s 'https://api.opendota.com/api/constants/item_ids' -o /tmp/item_ids.json
echo "Seeding const_item_id..."
python3 << 'PYEOF' > /tmp/item_ids_seed.sql
import json
ids = json.load(open('/tmp/item_ids.json'))
for id_str, item_key in ids.items():
    aid = int(id_str)
    key_esc = item_key.replace("'", "''")
    print(f"INSERT INTO const_item_id (id, name) VALUES ({aid}, '{key_esc}') ON CONFLICT (id) DO NOTHING;")
PYEOF
$DOCKER_PSQL < /tmp/item_ids_seed.sql 2>&1 | tail -3
echo "Item IDs seeded."

# ── Hero abilities + talents ────────────────────────────────────────────────
curl -s 'https://api.opendota.com/api/constants/hero_abilities' -o /tmp/hero_abilities.json
echo "Seeding const_hero_ability & const_hero_talent..."
python3 << 'PYEOF' > /tmp/hero_abilities_seed.sql
import json
data = json.load(open('/tmp/hero_abilities.json'))
for hero_key, hdata in data.items():
    hero_esc = hero_key.replace("'", "''")
    # Abilities (some entries are lists for multi-form heroes like Monkey King)
    for idx, ab_entry in enumerate(hdata.get('abilities', [])):
        if isinstance(ab_entry, list):
            ab_name = ab_entry[0]  # use first entry
        else:
            ab_name = ab_entry
        ab_esc = ab_name.replace("'", "''")
        print(f"INSERT INTO const_hero_ability (hero_name, ability_name, ability_order) VALUES ('{hero_esc}', '{ab_esc}', {idx}) ON CONFLICT (hero_name, ability_name) DO NOTHING;")
    # Talents
    for idx, talent in enumerate(hdata.get('talents', [])):
        tn = talent.get('name', '')
        tl = talent.get('level', 0)
        tn_esc = tn.replace("'", "''")
        print(f"INSERT INTO const_hero_talent (hero_name, talent_name, talent_level, talent_order) VALUES ('{hero_esc}', '{tn_esc}', {tl}, {idx}) ON CONFLICT (hero_name, talent_order) DO NOTHING;")
PYEOF
$DOCKER_PSQL < /tmp/hero_abilities_seed.sql 2>&1 | tail -3
echo "Hero abilities & talents seeded."

echo "Done."
