-- 011_hero_draft_slot_agg.sql
-- Hero Draft Position Win Rate aggregate table.
--
-- Captures the win rate for each hero at each specific team-pick ordinal
-- (1st pick, 2nd pick, 3rd pick, 4th pick, 5th pick) for a given patch.
-- This captures patterns like:
--   - heroes strong as early foundational picks (high wr at ordinal 1-2)
--   - heroes that perform better as late counterpicks (high wr at ordinal 4-5)
--   - heroes that are overpicked too early (low wr at ordinal 1-2)
--
-- UNLOGGED for write speed during batch population (same as other ml.* tables).

CREATE SCHEMA IF NOT EXISTS ml;

CREATE UNLOGGED TABLE IF NOT EXISTS ml.hero_draft_slot_agg (
    patch_id           INT      NOT NULL,
    hero_id            INT      NOT NULL,
    team_pick_ordinal  SMALLINT NOT NULL, -- 1st/2nd/3rd/4th/5th pick for that team
    games              INT      NOT NULL DEFAULT 0,
    wins               INT      NOT NULL DEFAULT 0,
    win_rate           FLOAT    NOT NULL DEFAULT 0.5,
    PRIMARY KEY (patch_id, hero_id, team_pick_ordinal),
    CONSTRAINT chk_hero_draft_slot_ordinal
        CHECK (team_pick_ordinal BETWEEN 1 AND 5)
);

-- No separate lookup index needed — the PK already covers the join pattern
-- (patch_id, hero_id, team_pick_ordinal).
