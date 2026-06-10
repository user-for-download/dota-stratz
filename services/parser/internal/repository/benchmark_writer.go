package repository

import (
	"encoding/json"
	"fmt"

	"github.com/dota-stratz/services/parser/internal/models"
	"github.com/jackc/pgx/v5"
)

func writeBenchmarks(batch *pgx.Batch, matchID int64, playerSlot int, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries map[string]models.Benchmark
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal benchmarks: %w", err)
	}
	for metricName, bench := range entries {
		batch.Queue(`
			INSERT INTO player_benchmarks (match_id, player_slot, metric_name, raw_value, pct)
			VALUES ($1,$2,$3,$4,$5)
			ON CONFLICT (match_id, player_slot, metric_name) DO NOTHING`,
			matchID, playerSlot, metricName, bench.Raw, bench.Pct,
		)
	}
	return nil
}

func writePermanentBuffs(batch *pgx.Batch, matchID int64, playerSlot int, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries []models.PermanentBuff
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal permanent_buffs: %w", err)
	}
	for _, pb := range entries {
		batch.Queue(`
			INSERT INTO player_permanent_buffs (match_id, player_slot, permanent_buff, stack_count, grant_time)
			VALUES ($1,$2,$3,$4,$5)
			ON CONFLICT (match_id, player_slot, permanent_buff, grant_time) DO NOTHING`,
			matchID, playerSlot, pb.PermanentBuff, pb.StackCount, pb.GrantTime,
		)
	}
	return nil
}

func writeNeutralItemHistory(batch *pgx.Batch, matchID int64, playerSlot int, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries []models.NeutralItemHistory
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal neutral_item_history: %w", err)
	}
	for _, nh := range entries {
		batch.Queue(`
			INSERT INTO player_neutral_item_history (match_id, player_slot, item_neutral, time, item_neutral_enhancement)
			VALUES ($1,$2,$3,$4,$5)
			ON CONFLICT (match_id, player_slot, time, item_neutral) DO NOTHING`,
			matchID, playerSlot, nh.ItemNeutral, nh.Time, nh.ItemNeutralEnhancement,
		)
	}
	return nil
}
