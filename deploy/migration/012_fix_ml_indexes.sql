-- 012_fix_ml_indexes.sql
-- Post-review index cleanups.
--
-- 1. idx_matches_patch_hero is misnamed and redundant with
--    idx_matches_patch_start (see review issue #7).
-- 2. idx_hero_draft_slot_agg_lookup duplicates the PK — safe to drop.
-- 3. Add CHECK constraint on hero_draft_slot_agg.team_pick_ordinal
--    (guarded IF NOT EXISTS — 011 already creates it).

-- ============================================================================
-- 1. Drop redundant index (duplicates idx_matches_patch_start coverage)
-- ============================================================================
DROP INDEX IF EXISTS idx_matches_patch_hero;

-- ============================================================================
-- 2. Drop redundant index on hero_draft_slot_agg (PK covers the same columns)
-- ============================================================================
DROP INDEX IF EXISTS ml.idx_hero_draft_slot_agg_lookup;

-- ============================================================================
-- 3. Add CHECK constraint on team_pick_ordinal valid range (1-5)
--    Guarded by IF NOT EXISTS because 011_hero_draft_slot_agg.sql already
--    creates this constraint (migration 012 would fail if applied after 011
--    without this guard).
-- ============================================================================
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_hero_draft_slot_ordinal'
          AND conrelid = 'ml.hero_draft_slot_agg'::regclass
    ) THEN
        ALTER TABLE ml.hero_draft_slot_agg
            ADD CONSTRAINT chk_hero_draft_slot_ordinal
            CHECK (team_pick_ordinal BETWEEN 1 AND 5);
    END IF;
END $$;
