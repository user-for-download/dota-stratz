-- 003_static.sql
-- Dota 2 static reference data: table definitions + seed data.
-- Merges 002_constants.

-- 1. GAME MODE
CREATE TABLE IF NOT EXISTS const_game_mode (id INT PRIMARY KEY, name VARCHAR NOT NULL, balanced BOOLEAN NOT NULL);
INSERT INTO const_game_mode (id, name, balanced) VALUES
    (0,  'game_mode_unknown',                    true),
    (1,  'game_mode_all_pick',                   true),
    (2,  'game_mode_captains_mode',              true),
    (3,  'game_mode_random_draft',               true),
    (4,  'game_mode_single_draft',               true),
    (5,  'game_mode_all_random',                 true),
    (6,  'game_mode_intro',                      false),
    (7,  'game_mode_diretide',                   false),
    (8,  'game_mode_reverse_captains_mode',      false),
    (9,  'game_mode_greeviling',                 false),
    (10, 'game_mode_tutorial',                   false),
    (11, 'game_mode_mid_only',                   false),
    (12, 'game_mode_least_played',               true),
    (13, 'game_mode_limited_heroes',             false),
    (14, 'game_mode_compendium_matchmaking',     false),
    (15, 'game_mode_custom',                     false),
    (16, 'game_mode_captains_draft',             true),
    (17, 'game_mode_balanced_draft',             true),
    (18, 'game_mode_ability_draft',              false),
    (19, 'game_mode_event',                      false),
    (20, 'game_mode_all_random_death_match',     false),
    (21, 'game_mode_1v1_mid',                    false),
    (22, 'game_mode_all_draft',                  true),
    (23, 'game_mode_turbo',                      false),
    (24, 'game_mode_mutation',                   false),
    (25, 'game_mode_coaches_challenge',          false)
ON CONFLICT (id) DO NOTHING;

-- 2. LOBBY TYPE
CREATE TABLE IF NOT EXISTS const_lobby_type (id INT PRIMARY KEY, name VARCHAR NOT NULL, balanced BOOLEAN NOT NULL);
INSERT INTO const_lobby_type (id, name, balanced) VALUES
    (0,  'lobby_type_normal',           true),
    (1,  'lobby_type_practice',         true),
    (2,  'lobby_type_tournament',       true),
    (3,  'lobby_type_tutorial',         false),
    (4,  'lobby_type_coop_bots',        false),
    (5,  'lobby_type_ranked_team_mm',   true),
    (6,  'lobby_type_ranked_solo_mm',   true),
    (7,  'lobby_type_ranked',           true),
    (8,  'lobby_type_1v1_mid',          false),
    (9,  'lobby_type_battle_cup',       true),
    (10, 'lobby_type_local_bots',       true),
    (11, 'lobby_type_spectator',        true),
    (12, 'lobby_type_event',            false),
    (13, 'lobby_type_gauntlet',         true),
    (14, 'lobby_type_new_player',       false),
    (15, 'lobby_type_featured',         false)
ON CONFLICT (id) DO NOTHING;

-- 3. REGION
CREATE TABLE IF NOT EXISTS const_region (id INT PRIMARY KEY, name VARCHAR NOT NULL);
INSERT INTO const_region (id, name) VALUES
    (1,  'US WEST'), (2,  'US EAST'), (3,  'EUROPE'), (5,  'SINGAPORE'), (6,  'DUBAI'), (7,  'AUSTRALIA'),
    (8,  'STOCKHOLM'), (9,  'AUSTRIA'), (10, 'BRAZIL'), (11, 'SOUTHAFRICA'), (12, 'PW TELECOM SHANGHAI'),
    (13, 'PW UNICOM'), (14, 'CHILE'), (15, 'PERU'), (16, 'INDIA'), (17, 'PW TELECOM GUANGDONG'),
    (18, 'PW TELECOM ZHEJIANG'), (19, 'JAPAN'), (20, 'PW TELECOM WUHAN'), (25, 'PW UNICOM TIANJIN'),
    (37, 'TAIWAN'), (38, 'ARGENTINA')
ON CONFLICT (id) DO NOTHING;

-- 4. PATCH
CREATE TABLE IF NOT EXISTS const_patch (id INT PRIMARY KEY, name VARCHAR NOT NULL, release_date TIMESTAMPTZ);
INSERT INTO const_patch (id, name, release_date) VALUES
    (0, '6.70', '2010-12-24T00:00:00Z'), (1, '6.71', '2011-01-21T00:00:00Z'), (2, '6.72', '2011-04-27T00:00:00Z'),
    (3, '6.73', '2011-12-24T00:00:00Z'), (4, '6.74', '2012-03-10T00:00:00Z'), (5, '6.75', '2012-09-30T00:00:00Z'),
    (6, '6.76', '2012-10-21T00:00:00Z'), (7, '6.77', '2012-12-15T00:00:00Z'), (8, '6.78', '2013-05-30T00:00:00Z'),
    (9, '6.79', '2013-11-24T00:00:00Z'), (10, '6.80', '2014-01-27T00:00:00Z'), (11, '6.81', '2014-04-29T00:00:00Z'),
    (12, '6.82', '2014-09-24T00:00:00Z'), (13, '6.83', '2014-12-17T00:00:00Z'), (14, '6.84', '2015-04-30T21:00:00Z'),
    (15, '6.85', '2015-09-24T20:00:00Z'), (16, '6.86', '2015-12-16T20:00:00Z'), (17, '6.87', '2016-04-26T01:00:00Z'),
    (18, '6.88', '2016-06-12T08:00:00Z'), (19, '7.00', '2016-12-13T00:00:00Z'), (20, '7.01', '2016-12-21T03:00:00Z'),
    (21, '7.02', '2017-02-09T04:00:00Z'), (22, '7.03', '2017-03-16T00:00:00Z'), (23, '7.04', '2017-03-23T18:00:00Z'),
    (24, '7.05', '2017-04-09T22:00:00Z'), (25, '7.06', '2017-05-15T15:00:00Z'), (26, '7.07', '2017-10-31T23:00:00Z'),
    (27, '7.08', '2018-02-01T00:00:00Z'), (28, '7.09', '2018-02-15T00:00:00Z'), (29, '7.10', '2018-03-01T00:00:00Z'),
    (30, '7.11', '2018-03-15T00:00:00Z'), (31, '7.12', '2018-03-29T00:00:00Z'), (32, '7.13', '2018-04-12T00:00:00Z'),
    (33, '7.14', '2018-04-26T00:00:00Z'), (34, '7.15', '2018-05-10T00:00:00Z'), (35, '7.16', '2018-05-27T00:00:00Z'),
    (36, '7.17', '2018-06-10T00:00:00Z'), (37, '7.18', '2018-06-24T00:00:00Z'), (38, '7.19', '2018-07-30T00:00:00Z'),
    (39, '7.20', '2018-11-19T18:00:00Z'), (40, '7.21', '2019-01-29T18:00:00Z'), (41, '7.22', '2019-05-25T00:00:00Z'),
    (42, '7.23', '2019-11-26T18:00:00Z'), (43, '7.24', '2020-01-27T00:00:00Z'), (44, '7.25', '2020-03-17T18:00:00Z'),
    (45, '7.26', '2020-04-18T00:00:00Z'), (46, '7.27', '2020-06-29T00:00:00Z'), (47, '7.28', '2020-12-18T00:00:00Z'),
    (48, '7.29', '2021-04-10T00:00:00Z'), (49, '7.30', '2021-08-18T02:53:21Z'), (50, '7.31', '2022-02-23T23:46:14Z'),
    (51, '7.32', '2022-08-24T02:16:32Z'), (52, '7.33', '2023-04-21T01:22:56Z'), (53, '7.34', '2023-08-09T00:11:15Z'),
    (54, '7.35', '2023-12-14T16:07:43Z'), (55, '7.36', '2024-05-23T05:26:05Z'), (56, '7.37', '2024-08-01T07:30:27Z'),
    (57, '7.38', '2025-02-19T13:48:29Z'), (58, '7.39', '2025-05-22T23:36:01Z'), (59, '7.40', '2025-12-16T00:50:40Z'),
    (60, '7.41', '2026-03-24T00:50:59Z')
ON CONFLICT (id) DO NOTHING;

-- 5. HEROES
CREATE TABLE IF NOT EXISTS const_hero (
    id INT PRIMARY KEY, name VARCHAR NOT NULL, localized_name VARCHAR, primary_attr VARCHAR, attack_type VARCHAR, roles TEXT[], img VARCHAR, icon VARCHAR,
    base_health INT, base_health_regen FLOAT, base_mana INT, base_mana_regen FLOAT, base_armor FLOAT, base_mr FLOAT, base_attack_min INT, base_attack_max INT,
    base_str INT, base_agi INT, base_int INT, str_gain FLOAT, agi_gain FLOAT, int_gain FLOAT, attack_range INT, projectile_speed INT, attack_rate FLOAT,
    base_attack_time FLOAT, attack_point FLOAT, move_speed INT, turn_rate FLOAT, cm_enabled BOOLEAN, legs INT, day_vision INT, night_vision INT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_const_hero_name ON const_hero(name);
CREATE INDEX IF NOT EXISTS idx_const_hero_primary_attr ON const_hero(primary_attr);

-- 6. ITEMS
CREATE TABLE IF NOT EXISTS const_item (
    id INT PRIMARY KEY, name VARCHAR NOT NULL, dname VARCHAR, qual VARCHAR, cost INT, behavior VARCHAR, dmg_type VARCHAR, bkbpierce VARCHAR, notes TEXT,
    lore TEXT, created BOOLEAN, charges BOOLEAN, cd FLOAT, mc FLOAT, hc FLOAT, img VARCHAR, components TEXT[], attrib JSONB, abilities JSONB
);
CREATE INDEX IF NOT EXISTS idx_const_item_name ON const_item(name);
CREATE INDEX IF NOT EXISTS idx_const_item_dname ON const_item(dname);

CREATE TABLE IF NOT EXISTS const_item_id (id INT PRIMARY KEY, name VARCHAR NOT NULL);
CREATE INDEX IF NOT EXISTS idx_const_item_id_name ON const_item_id(name);

-- 7. ABILITIES
CREATE TABLE IF NOT EXISTS const_ability_id (id INT PRIMARY KEY, name VARCHAR NOT NULL);
CREATE INDEX IF NOT EXISTS idx_const_ability_id_name ON const_ability_id(name);

CREATE TABLE IF NOT EXISTS const_ability (name VARCHAR PRIMARY KEY, dname VARCHAR, behavior VARCHAR, dmg_type VARCHAR, bkbpierce VARCHAR, description TEXT, lore TEXT, img VARCHAR, attrib JSONB);

-- 8. HERO-ABILITY & HERO-TALENT MAPPINGS
CREATE TABLE IF NOT EXISTS const_hero_ability (hero_name VARCHAR NOT NULL, ability_name VARCHAR NOT NULL, ability_order INT NOT NULL, PRIMARY KEY (hero_name, ability_name));
CREATE INDEX IF NOT EXISTS idx_const_hero_ability_hero ON const_hero_ability(hero_name);

CREATE TABLE IF NOT EXISTS const_hero_talent (hero_name VARCHAR NOT NULL, talent_name VARCHAR NOT NULL, talent_level INT NOT NULL, talent_order INT NOT NULL, PRIMARY KEY (hero_name, talent_order));
CREATE INDEX IF NOT EXISTS idx_const_hero_talent_hero ON const_hero_talent(hero_name);

-- 9. INTERNAL FKs
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_ha_hero') THEN
        ALTER TABLE const_hero_ability ADD CONSTRAINT fk_ha_hero FOREIGN KEY (hero_name) REFERENCES const_hero(name) ON DELETE CASCADE; END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_ht_hero') THEN
        ALTER TABLE const_hero_talent ADD CONSTRAINT fk_ht_hero FOREIGN KEY (hero_name) REFERENCES const_hero(name) ON DELETE CASCADE; END IF;
END $$;

-- ============================================================================
-- MIGRATIONS TRACKER
-- ============================================================================
CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());
INSERT INTO _migrations (name) VALUES ('003_static.sql') ON CONFLICT DO NOTHING;
