-- 003_const.sql
-- Dota 2 static reference data tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS const_game_mode (id INT PRIMARY KEY, name VARCHAR NOT NULL, balanced BOOLEAN NOT NULL);
CREATE TABLE IF NOT EXISTS const_lobby_type (id INT PRIMARY KEY, name VARCHAR NOT NULL, balanced BOOLEAN NOT NULL);
CREATE TABLE IF NOT EXISTS const_region (id INT PRIMARY KEY, name VARCHAR NOT NULL);
CREATE TABLE IF NOT EXISTS const_patch (id INT PRIMARY KEY, name VARCHAR NOT NULL, release_date TIMESTAMPTZ);

CREATE TABLE IF NOT EXISTS const_hero (
    id INT PRIMARY KEY, name VARCHAR NOT NULL, localized_name VARCHAR, primary_attr VARCHAR, attack_type VARCHAR, roles TEXT[], img VARCHAR, icon VARCHAR,
    base_health INT, base_health_regen FLOAT, base_mana INT, base_mana_regen FLOAT, base_armor FLOAT, base_mr FLOAT, base_attack_min INT, base_attack_max INT,
    base_str INT, base_agi INT, base_int INT, str_gain FLOAT, agi_gain FLOAT, int_gain FLOAT, attack_range INT, projectile_speed INT, attack_rate FLOAT,
    base_attack_time FLOAT, attack_point FLOAT, move_speed INT, turn_rate FLOAT, cm_enabled BOOLEAN, legs INT, day_vision INT, night_vision INT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_const_hero_name ON const_hero(name);
CREATE INDEX IF NOT EXISTS idx_const_hero_primary_attr ON const_hero(primary_attr);

-- Unique constraint needed if referenced by other const_* tables
DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'uq_const_hero_name') THEN ALTER TABLE const_hero ADD CONSTRAINT uq_const_hero_name UNIQUE (name); END IF; END $$;

CREATE TABLE IF NOT EXISTS const_item (
    id INT PRIMARY KEY, name VARCHAR NOT NULL, dname VARCHAR, qual VARCHAR, cost INT, behavior VARCHAR, dmg_type VARCHAR, bkbpierce VARCHAR, notes TEXT,
    lore TEXT, created BOOLEAN, charges BOOLEAN, cd FLOAT, mc FLOAT, hc FLOAT, img VARCHAR, components TEXT[], attrib JSONB, abilities JSONB
);
CREATE INDEX IF NOT EXISTS idx_const_item_name ON const_item(name);
CREATE INDEX IF NOT EXISTS idx_const_item_dname ON const_item(dname);

CREATE TABLE IF NOT EXISTS const_item_id (id INT PRIMARY KEY, name VARCHAR NOT NULL);
CREATE INDEX IF NOT EXISTS idx_const_item_id_name ON const_item_id(name);

CREATE TABLE IF NOT EXISTS const_ability_id (id INT PRIMARY KEY, name VARCHAR NOT NULL);
CREATE INDEX IF NOT EXISTS idx_const_ability_id_name ON const_ability_id(name);

CREATE TABLE IF NOT EXISTS const_ability (name VARCHAR PRIMARY KEY, dname VARCHAR, behavior VARCHAR, dmg_type VARCHAR, bkbpierce VARCHAR, description TEXT, lore TEXT, img VARCHAR, attrib JSONB);

CREATE TABLE IF NOT EXISTS const_hero_ability (hero_name VARCHAR NOT NULL, ability_name VARCHAR NOT NULL, ability_order INT NOT NULL, PRIMARY KEY (hero_name, ability_name));
CREATE INDEX IF NOT EXISTS idx_const_hero_ability_hero ON const_hero_ability(hero_name);

CREATE TABLE IF NOT EXISTS const_hero_talent (hero_name VARCHAR NOT NULL, talent_name VARCHAR NOT NULL, talent_level INT NOT NULL, talent_order INT NOT NULL, PRIMARY KEY (hero_name, talent_order));
CREATE INDEX IF NOT EXISTS idx_const_hero_talent_hero ON const_hero_talent(hero_name);

-- Note: const_hero_facet was removed in 007_cleanup.sql
-- (not referenced by any Go code or ETL script).

-- ============================================================================
-- INTERNAL CONSTANTS FOREIGN KEYS
-- These are safe because the constants ETL script updates them together.
-- Fact-to-Dimension FKs (like players -> const_hero) have been removed
-- to prevent ingestion pipelines from crashing on missing/new heroes.
-- ============================================================================
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_ha_hero') THEN
        ALTER TABLE const_hero_ability ADD CONSTRAINT fk_ha_hero FOREIGN KEY (hero_name) REFERENCES const_hero(name) ON DELETE CASCADE; END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_ht_hero') THEN
        ALTER TABLE const_hero_talent ADD CONSTRAINT fk_ht_hero FOREIGN KEY (hero_name) REFERENCES const_hero(name) ON DELETE CASCADE; END IF;
    -- fk_hf_hero was removed alongside const_hero_facet in 007_cleanup.sql
END $$;
