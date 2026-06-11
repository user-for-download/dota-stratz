-- 012_fix_ml_indexes.sql
-- Post-review index cleanups.
--
-- 1. idx_matches_patch_hero is misnamed and redundant with
--    idx_matches_patch_start (see review issue #7).
-- 2. idx_players_account_hero has match_id as leading column but is used
--    for account_id/hero_id lookups — add a better complementary index.
-- 3. idx_hero_draft_slot_agg_lookup duplicates the PK — safe to drop.
-- 4. Add CHECK constraint on hero_draft_slot_agg.team_pick_ordinal.

-- ============================================================================
-- 1. Drop redundant index (duplicates idx_matches_patch_start coverage)
-- ============================================================================
DROP INDEX IF EXISTS idx_matches_patch_hero;

-- ============================================================================
-- 2. Add complementary index on (account_id, hero_id, match_id) for
--    player-hero aggregate lookups that filter by account+hero first.
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_players_account_hero_match
    ON players (account_id, hero_id, match_id);

-- ============================================================================
-- 3. Drop redundant index on hero_draft_slot_agg (PK covers the same columns)
-- ============================================================================
DROP INDEX IF EXISTS ml.idx_hero_draft_slot_agg_lookup;

-- ============================================================================
-- 4. Add CHECK constraint on team_pick_ordinal valid range (1-5)
-- ============================================================================
ALTER TABLE ml.hero_draft_slot_agg
    ADD CONSTRAINT chk_hero_draft_slot_ordinal
    CHECK (team_pick_ordinal BETWEEN 1 AND 5);
