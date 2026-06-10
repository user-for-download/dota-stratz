-- 006_deferrable_fk.sql
--
-- Re-create foreign-key constraints on bulk-insert tables as
-- DEFERRABLE INITIALLY DEFERRED so that multi-row inserts do not need to
-- satisfy the FK at every row.
--
-- Why
-- ---
-- The parser's batch_writer.go inserts the `matches` parent row and its
-- 10 `players` rows in the SAME pgx.Batch. Without deferrable FKs, every
-- child row needs the parent to already exist within the same statement
-- (which is impossible across batched inserts that all execute against the
-- parent before the parent is committed). The conventional workaround is
-- to insert parent → child, but the parser writes everything in one batch
-- for throughput.
--
-- DEFERRABLE INITIALLY DEFERRED moves the FK check from row-time to
-- commit-time, allowing any order of insertion as long as the dependency
-- is satisfied by the time COMMIT runs. See Issue #28 in the audit notes
-- for the batch-Write FK-violation fallback that compensated while this
-- migration was not yet applied.
--
-- This migration is idempotent: if a constraint of the same name already
-- exists, the DROP is a no-op and the ADD replaces it.

-- players.match_id → matches.match_id
ALTER TABLE players
    DROP CONSTRAINT IF EXISTS players_match_id_fkey;
ALTER TABLE players
    ADD CONSTRAINT players_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

-- teamfights.match_id → matches.match_id
ALTER TABLE teamfights
    DROP CONSTRAINT IF EXISTS teamfights_match_id_fkey;
ALTER TABLE teamfights
    ADD CONSTRAINT teamfights_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

-- teamfight_players.(match_id, start_time) → teamfights.(match_id, start_time)
ALTER TABLE teamfight_players
    DROP CONSTRAINT IF EXISTS teamfight_players_match_id_start_time_fkey;
ALTER TABLE teamfight_players
    ADD CONSTRAINT teamfight_players_match_id_start_time_fkey
    FOREIGN KEY (match_id, start_time)
    REFERENCES teamfights(match_id, start_time) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

-- objectives.match_id → matches.match_id
ALTER TABLE objectives
    DROP CONSTRAINT IF EXISTS objectives_match_id_fkey;
ALTER TABLE objectives
    ADD CONSTRAINT objectives_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

-- chat.match_id → matches.match_id
ALTER TABLE chat
    DROP CONSTRAINT IF EXISTS chat_match_id_fkey;
ALTER TABLE chat
    ADD CONSTRAINT chat_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

-- picks_bans.match_id → matches.match_id
ALTER TABLE picks_bans
    DROP CONSTRAINT IF EXISTS picks_bans_match_id_fkey;
ALTER TABLE picks_bans
    ADD CONSTRAINT picks_bans_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

-- match_gold_adv.match_id → matches.match_id
ALTER TABLE match_gold_adv
    DROP CONSTRAINT IF EXISTS match_gold_adv_match_id_fkey;
ALTER TABLE match_gold_adv
    ADD CONSTRAINT match_gold_adv_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;

-- match_xp_adv.match_id → matches.match_id
ALTER TABLE match_xp_adv
    DROP CONSTRAINT IF EXISTS match_xp_adv_match_id_fkey;
ALTER TABLE match_xp_adv
    ADD CONSTRAINT match_xp_adv_match_id_fkey
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE
    DEFERRABLE INITIALLY DEFERRED;
