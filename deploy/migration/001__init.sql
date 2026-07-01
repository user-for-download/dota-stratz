-- 001__init.sql
-- Core schema, partition management, ingestion checkpoints.
-- Merges and optimizes: 001_core, 004_partition_verify, 006_postgres_best_practices_fixes (partition part),
-- 008_minute_stats_columns (superseded), 010_team_id_bigint_indexes (core part), 013_separate_time_series_arrays.

CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- ============================================================================
-- 1. MATCHES
-- ============================================================================
CREATE TABLE IF NOT EXISTS matches (
    match_id BIGINT PRIMARY KEY,
    version INT,
    start_time BIGINT,
    duration INT,
    series_id BIGINT,
    series_type INT,
    cluster INT,
    replay_salt BIGINT,
    radiant_win BOOLEAN,
    pre_game_duration INT,
    match_seq_num BIGINT,
    tower_status_radiant INT,
    tower_status_dire INT,
    barracks_status_radiant INT,
    barracks_status_dire INT,
    first_blood_time INT,
    lobby_type INT,
    human_players INT,
    game_mode INT,
    flags INT,
    engine INT,
    radiant_score INT,
    dire_score INT,
    radiant_team_id BIGINT,
    radiant_name VARCHAR,
    radiant_logo BIGINT,
    radiant_team_complete INT,
    dire_team_id BIGINT,
    dire_name VARCHAR,
    dire_logo BIGINT,
    dire_team_complete INT,
    radiant_captain BIGINT,
    dire_captain BIGINT,
    leagueid INT,
    patch INT,
    region INT,
    replay_url VARCHAR,
    throw INT,
    loss INT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_matches_start_time_brin ON matches USING BRIN (start_time) WITH (pages_per_range = 32);
CREATE INDEX IF NOT EXISTS idx_matches_leagueid ON matches(leagueid) WHERE leagueid > 0;
CREATE INDEX IF NOT EXISTS idx_matches_pro_filter ON matches(leagueid, lobby_type, start_time) WHERE leagueid > 0 AND lobby_type IN (1, 2);
CREATE INDEX IF NOT EXISTS idx_matches_start_time_date ON matches(((TIMESTAMP 'epoch' + start_time * INTERVAL '1 second')::date));
CREATE INDEX IF NOT EXISTS idx_matches_patch_result ON matches (patch, match_id) WHERE radiant_win IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_matches_patch_start ON matches (patch, start_time, match_id) WHERE radiant_win IS NOT NULL;

-- ============================================================================
-- 2. PLAYERS (RANGE-partitioned by match_id)
-- ============================================================================
CREATE TABLE IF NOT EXISTS players (
    match_id BIGINT NOT NULL,
    player_slot INT NOT NULL,
    account_id BIGINT,
    hero_id INT,
    hero_variant INT,
    party_id BIGINT,
    party_size INT,
    team_number INT,
    team_slot INT,
    is_radiant BOOLEAN,
    radiant_win BOOLEAN,
    win INT,
    lose INT,
    kills INT,
    deaths INT,
    assists INT,
    leaver_status INT,
    last_hits INT,
    denies INT,
    gold_per_min INT,
    xp_per_min INT,
    level INT,
    net_worth INT,
    gold INT,
    gold_spent INT,
    total_gold INT,
    total_xp INT,
    aghanims_scepter INT,
    aghanims_shard INT,
    moonshard INT,
    hero_damage INT,
    tower_damage INT,
    hero_healing INT,
    kills_per_min FLOAT,
    kda FLOAT,
    abandons INT,
    neutral_kills INT,
    tower_kills INT,
    courier_kills INT,
    lane_kills INT,
    hero_kills INT,
    observer_kills INT,
    sentry_kills INT,
    roshan_kills INT,
    necronomicon_kills INT,
    ancient_kills INT,
    buyback_count INT,
    observer_uses INT,
    sentry_uses INT,
    lane_efficiency FLOAT,
    lane_efficiency_pct INT,
    lane INT,
    lane_role INT,
    is_roaming BOOLEAN,
    actions_per_min INT,
    life_state_dead INT,
    obs_placed INT,
    sen_placed INT,
    creeps_stacked INT,
    camps_stacked INT,
    rune_pickups INT,
    firstblood_claimed INT,
    teamfight_participation FLOAT,
    towers_killed INT,
    roshans_killed INT,
    observers_placed INT,
    stuns FLOAT,
    item_0 INT, item_1 INT, item_2 INT, item_3 INT, item_4 INT, item_5 INT,
    backpack_0 INT, backpack_1 INT, backpack_2 INT,
    item_neutral INT,
    item_neutral2 INT,
    personaname VARCHAR,
    name VARCHAR,
    last_login VARCHAR,
    rank_tier INT,
    computed_mmr FLOAT,
    is_subscriber BOOLEAN,
    ability_targets JSONB,
    damage_targets JSONB,
    gold_reasons JSONB,
    xp_reasons JSONB,
    killed JSONB,
    item_uses JSONB,
    hero_hits JSONB,
    damage JSONB,
    damage_taken JSONB,
    damage_inflictor JSONB,
    runes JSONB,
    killed_by JSONB,
    kill_streaks JSONB,
    multi_kills JSONB,
    life_state JSONB,
    healing JSONB,
    damage_inflictor_received JSONB,
    lane_pos JSONB,
    obs JSONB,
    sen JSONB,
    actions JSONB,
    cosmetics JSONB,
    purchase_time JSONB,
    first_purchase_time JSONB,
    item_win JSONB,
    item_usage JSONB,

    PRIMARY KEY (match_id, player_slot),
    CONSTRAINT chk_player_slot_range CHECK (player_slot BETWEEN 0 AND 255)
) PARTITION BY RANGE (match_id);

ALTER TABLE players
    DROP CONSTRAINT IF EXISTS players_match_id_fkey;
ALTER TABLE players
    ADD CONSTRAINT players_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

CREATE INDEX IF NOT EXISTS idx_players_account_id ON players(account_id);
CREATE INDEX IF NOT EXISTS idx_players_hero_id ON players(hero_id);
CREATE INDEX IF NOT EXISTS idx_players_kda ON players(kda);
CREATE INDEX IF NOT EXISTS idx_players_account_hero_match ON players(account_id, hero_id, match_id);
CREATE INDEX IF NOT EXISTS idx_players_hero_kills ON players(hero_id, kills);
CREATE INDEX IF NOT EXISTS idx_players_account_hero ON players(account_id, hero_id);
CREATE INDEX IF NOT EXISTS idx_players_item_uses_gin ON players USING GIN (item_uses);
CREATE INDEX IF NOT EXISTS idx_players_damage_gin ON players USING GIN (damage);
CREATE INDEX IF NOT EXISTS idx_players_purchase_time_gin ON players USING GIN (purchase_time);
CREATE INDEX IF NOT EXISTS idx_players_match_hero_side ON players (match_id, hero_id, is_radiant);
CREATE INDEX IF NOT EXISTS idx_players_match_account_hero ON players (match_id, account_id, hero_id);

-- ============================================================================
-- 3. PLAYER EVENT & TIME-SERIES TABLES
-- ============================================================================
CREATE TABLE IF NOT EXISTS player_minute_stats (
    match_id BIGINT,
    player_slot INT,
    minute INT,
    gold INT,
    last_hits INT,
    denies INT,
    xp INT,
    PRIMARY KEY (match_id, player_slot, minute)
);

CREATE TABLE IF NOT EXISTS player_time_series_arrays (
    match_id    BIGINT NOT NULL,
    player_slot INT    NOT NULL,
    gold_t      JSONB,
    xp_t        JSONB,
    PRIMARY KEY (match_id, player_slot)
);

CREATE TABLE IF NOT EXISTS player_abilities (
    match_id BIGINT,
    player_slot INT,
    ability_name VARCHAR,
    ability_uses INT,
    PRIMARY KEY (match_id, player_slot, ability_name)
);

CREATE TABLE IF NOT EXISTS player_ability_upgrades_log (
    match_id BIGINT,
    player_slot INT,
    upgrade_order INT,
    ability_id INT,
    PRIMARY KEY (match_id, player_slot, upgrade_order)
);

CREATE TABLE IF NOT EXISTS player_benchmarks (
    match_id BIGINT,
    player_slot INT,
    metric_name VARCHAR,
    raw_value FLOAT,
    pct FLOAT,
    PRIMARY KEY (match_id, player_slot, metric_name)
);

CREATE TABLE IF NOT EXISTS teamfights (
    match_id BIGINT,
    start_time INT,
    end_time INT,
    last_death INT,
    deaths INT,
    PRIMARY KEY (match_id, start_time),
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS teamfight_players (
    match_id BIGINT,
    start_time INT,
    player_slot INT,
    deaths INT,
    buybacks INT,
    damage INT,
    healing INT,
    gold_delta INT,
    xp_delta INT,
    xp_start INT,
    xp_end INT,
    ability_uses JSONB,
    item_uses JSONB,
    killed JSONB,
    deaths_pos JSONB,
    PRIMARY KEY (match_id, start_time, player_slot),
    FOREIGN KEY (match_id, start_time) REFERENCES teamfights(match_id, start_time) ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS player_kills_log (
    match_id BIGINT,
    player_slot INT,
    time INT,
    key VARCHAR,
    PRIMARY KEY (match_id, player_slot, time, key)
);

CREATE TABLE IF NOT EXISTS player_buyback_log (
    match_id BIGINT,
    player_slot INT,
    time INT,
    PRIMARY KEY (match_id, player_slot, time)
);

CREATE TABLE IF NOT EXISTS player_runes_log (
    match_id BIGINT,
    player_slot INT,
    time INT,
    key VARCHAR,
    seq BIGINT GENERATED ALWAYS AS IDENTITY,
    PRIMARY KEY (match_id, player_slot, time, key, seq)
);

CREATE TABLE IF NOT EXISTS player_purchase_log (
    match_id BIGINT,
    player_slot INT,
    time INT,
    key VARCHAR,
    charges INT,
    seq BIGINT GENERATED ALWAYS AS IDENTITY,
    PRIMARY KEY (match_id, player_slot, time, key, seq)
);

CREATE TABLE IF NOT EXISTS player_obs_log (
    match_id BIGINT,
    player_slot INT,
    time INT,
    key VARCHAR,
    x FLOAT, y FLOAT, z FLOAT,
    entityleft BOOLEAN,
    ehandle BIGINT,
    PRIMARY KEY (match_id, player_slot, time, key)
);

CREATE TABLE IF NOT EXISTS player_sen_log (
    match_id BIGINT,
    player_slot INT,
    time INT,
    key VARCHAR,
    x FLOAT, y FLOAT, z FLOAT,
    entityleft BOOLEAN,
    ehandle BIGINT,
    PRIMARY KEY (match_id, player_slot, time, key)
);

CREATE TABLE IF NOT EXISTS player_obs_left_log (
    match_id BIGINT,
    player_slot INT,
    time INT,
    key VARCHAR,
    attackername VARCHAR,
    x FLOAT, y FLOAT, z FLOAT,
    entityleft BOOLEAN,
    ehandle BIGINT,
    PRIMARY KEY (match_id, player_slot, time, key)
);

CREATE TABLE IF NOT EXISTS player_sen_left_log (
    match_id BIGINT,
    player_slot INT,
    time INT,
    key VARCHAR,
    attackername VARCHAR,
    x FLOAT, y FLOAT, z FLOAT,
    entityleft BOOLEAN,
    ehandle BIGINT,
    PRIMARY KEY (match_id, player_slot, time, key)
);

CREATE TABLE IF NOT EXISTS player_neutral_item_history (
    match_id BIGINT,
    player_slot INT,
    item_neutral VARCHAR,
    time INT,
    item_neutral_enhancement VARCHAR,
    PRIMARY KEY (match_id, player_slot, time, item_neutral)
);

CREATE TABLE IF NOT EXISTS player_permanent_buffs (
    match_id BIGINT,
    player_slot INT,
    permanent_buff INT,
    stack_count INT,
    grant_time INT,
    PRIMARY KEY (match_id, player_slot, permanent_buff, grant_time)
);

-- ============================================================================
-- 4. MATCH EVENTS
-- ============================================================================
CREATE TABLE IF NOT EXISTS objectives (
    match_id BIGINT,
    time INT,
    type VARCHAR,
    team INT,
    key VARCHAR,
    slot INT,
    player_slot INT,
    value INT,
    killer INT,
    PRIMARY KEY (match_id, time, type, team)
);
ALTER TABLE objectives
    DROP CONSTRAINT IF EXISTS objectives_match_id_fkey;
ALTER TABLE objectives
    ADD CONSTRAINT objectives_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE IF NOT EXISTS chat (
    match_id BIGINT,
    time INT,
    type VARCHAR,
    key VARCHAR,
    slot INT,
    player_slot INT,
    PRIMARY KEY (match_id, time, slot)
);
ALTER TABLE chat
    DROP CONSTRAINT IF EXISTS chat_match_id_fkey;
ALTER TABLE chat
    ADD CONSTRAINT chat_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE IF NOT EXISTS picks_bans (
    match_id BIGINT,
    is_pick BOOLEAN,
    hero_id INT,
    team INT,
    "order" INT,
    PRIMARY KEY (match_id, "order")
);
ALTER TABLE picks_bans
    DROP CONSTRAINT IF EXISTS picks_bans_match_id_fkey;
ALTER TABLE picks_bans
    ADD CONSTRAINT picks_bans_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

CREATE INDEX IF NOT EXISTS idx_picks_bans_match_order_pick_team ON picks_bans (match_id, "order", is_pick, team, hero_id);
CREATE INDEX IF NOT EXISTS idx_picks_bans_match_team_order_picks ON picks_bans (match_id, team, "order", hero_id) WHERE is_pick = TRUE;

CREATE TABLE IF NOT EXISTS match_gold_adv (
    match_id BIGINT,
    minute INT,
    radiant_gold_adv INT,
    PRIMARY KEY (match_id, minute)
);
ALTER TABLE match_gold_adv
    DROP CONSTRAINT IF EXISTS match_gold_adv_match_id_fkey;
ALTER TABLE match_gold_adv
    ADD CONSTRAINT match_gold_adv_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE IF NOT EXISTS match_xp_adv (
    match_id BIGINT,
    minute INT,
    radiant_xp_adv INT,
    PRIMARY KEY (match_id, minute)
);
ALTER TABLE match_xp_adv
    DROP CONSTRAINT IF EXISTS match_xp_adv_match_id_fkey;
ALTER TABLE match_xp_adv
    ADD CONSTRAINT match_xp_adv_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

-- ============================================================================
-- 5. INGESTION CHECKPOINTS
-- ============================================================================
CREATE TABLE IF NOT EXISTS public.ingestion_checkpoints (
    id INT PRIMARY KEY DEFAULT 1,
    last_fetched_datetime TIMESTAMP,
    last_parsed_match_id BIGINT DEFAULT 0,
    fetch_status VARCHAR(50) DEFAULT 'idle',
    parse_status VARCHAR(50) DEFAULT 'idle',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT single_row CHECK (id = 1)
);

INSERT INTO public.ingestion_checkpoints (id, last_fetched_datetime)
VALUES (1, NOW() - INTERVAL '7 days') ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- 6. PARTITION MANAGEMENT (Optimized from 006 fix)
-- ============================================================================
CREATE OR REPLACE FUNCTION public.create_player_partition(
    partition_name TEXT, from_val BIGINT, to_val BIGINT
) RETURNS TEXT AS $$
BEGIN
    EXECUTE format('CREATE TABLE %I PARTITION OF players FOR VALUES FROM (%L) TO (%L)', partition_name, from_val, to_val);
    RETURN format('Partition %s created ( %s → %s )', partition_name, from_val, to_val);
EXCEPTION
    WHEN duplicate_table THEN RETURN format('Partition %s already exists', partition_name);
    WHEN SQLSTATE '42P17' THEN RETURN format('Partition %s range overlaps an existing partition: %s', partition_name, SQLERRM);
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION public.ensure_player_partitions(
    max_match_id BIGINT DEFAULT 25000000000, partition_range BIGINT DEFAULT 5000000000
) RETURNS TABLE(partition_name TEXT, status TEXT) AS $$
DECLARE lower_bound BIGINT := 0; upper_bound BIGINT; name_suffix TEXT;
BEGIN
    WHILE lower_bound < max_match_id LOOP
        upper_bound := lower_bound + partition_range;
        name_suffix := format('p%s_to_%s', lower_bound, upper_bound);
        IF NOT EXISTS (
            SELECT 1 FROM pg_class c JOIN pg_inherits i ON c.oid = i.inhrelid JOIN pg_class p ON i.inhparent = p.oid
            WHERE p.relname = 'players' AND c.relispartition AND c.relname = format('players_%s', name_suffix)
        ) THEN
            partition_name := format('players_%s', name_suffix);
            status := public.create_player_partition(partition_name, lower_bound, upper_bound);
            RETURN NEXT;
        END IF;
        lower_bound := upper_bound;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

SELECT public.ensure_player_partitions(30000000000, 5000000000);
CREATE TABLE IF NOT EXISTS players_p_catchall PARTITION OF players DEFAULT;

-- ============================================================================
-- MIGRATIONS TRACKER
-- ============================================================================
CREATE TABLE IF NOT EXISTS _migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());
INSERT INTO _migrations (name) VALUES ('001__init.sql') ON CONFLICT DO NOTHING;
