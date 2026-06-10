-- 005_partition_management.sql
--
-- Partition management for the `players` RANGE-partitioned table.
--
-- Note
-- ----
-- The functions create_player_partition() and ensure_player_partitions() are
-- defined in 001_init.sql along with the initial 0..30B partition set and
-- the players_p_catchall default partition. This migration is the canonical
-- place to document the partition scheme and re-verify it on a running
-- database, but performs no schema changes.
--
-- If a future change needs to extend the partition set, it should be
-- applied here (not in 001). For the moment, 001's CREATE OR REPLACE
-- definitions are the source of truth and 005 only asserts that the
-- expected partitions are present.

DO $$
DECLARE
    expected_partitions TEXT[] := ARRAY[
        'players_p0_to_5000000000',
        'players_p5000000000_to_10000000000',
        'players_p10000000000_to_15000000000',
        'players_p15000000000_to_20000000000',
        'players_p20000000000_to_25000000000',
        'players_p25000000000_to_30000000000'
    ];
    missing TEXT;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = 'players' AND relkind = 'p') THEN
        RAISE EXCEPTION 'players table is not a partitioned relation — 001_init.sql may be incomplete';
    END IF;

    FOREACH missing IN ARRAY expected_partitions LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = missing) THEN
            -- Best-effort: call 001's ensure_player_partitions() with a
            -- range up to 30B. If 001 is not present, fail loudly so the
            -- operator notices the inconsistency.
            PERFORM public.ensure_player_partitions(30000000000::bigint, 5000000000::bigint);
            -- Re-check; if still missing, raise.
            IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = missing) THEN
                RAISE EXCEPTION 'expected partition % is missing and could not be created', missing;
            END IF;
        END IF;
    END LOOP;

    IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = 'players_p_catchall') THEN
        CREATE TABLE players_p_catchall PARTITION OF players DEFAULT;
    END IF;
END
$$;
