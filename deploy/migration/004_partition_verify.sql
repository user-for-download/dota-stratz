-- 004_partition_verify.sql
-- Verifies that the `players` RANGE-partitioned table has the expected
-- partitions. The partitions themselves are created in 001_core.sql.
-- This file is purely a safety assertion for CI/testing — it performs no
-- schema changes on a healthy database.

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
        RAISE EXCEPTION 'players table is not a partitioned relation — 001_core.sql may be incomplete';
    END IF;

    FOREACH missing IN ARRAY expected_partitions LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = missing) THEN
            PERFORM public.ensure_player_partitions(30000000000::bigint, 5000000000::bigint);
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
