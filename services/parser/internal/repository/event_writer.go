package repository

import (
	"encoding/json"
	"fmt"

	"github.com/dota-stratz/services/parser/internal/models"
	"github.com/jackc/pgx/v5"
)

func writeKillsLog(batch *pgx.Batch, matchID int64, playerSlot int, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries []models.KillLog
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal kills_log: %w", err)
	}
	for _, k := range entries {
		batch.Queue(`
			INSERT INTO player_kills_log (match_id, player_slot, time, key)
			VALUES ($1,$2,$3,$4)
			ON CONFLICT (match_id, player_slot, time, key) DO NOTHING`,
			matchID, playerSlot, k.Time, string(k.Key),
		)
	}
	return nil
}

func writeBuybackLog(batch *pgx.Batch, matchID int64, playerSlot int, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries []models.BuybackLog
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal buyback_log: %w", err)
	}
	for _, b := range entries {
		batch.Queue(`
			INSERT INTO player_buyback_log (match_id, player_slot, time)
			VALUES ($1,$2,$3)
			ON CONFLICT (match_id, player_slot, time) DO NOTHING`,
			matchID, playerSlot, b.Time,
		)
	}
	return nil
}

func writeRunesLog(batch *pgx.Batch, matchID int64, playerSlot int, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries []models.RuneLog
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal runes_log: %w", err)
	}
	for idx, r := range entries {
		batch.Queue(`
			INSERT INTO player_runes_log (match_id, player_slot, time, key, seq)
			OVERRIDING SYSTEM VALUE
			VALUES ($1,$2,$3,$4,$5)
			ON CONFLICT (match_id, player_slot, time, key, seq) DO NOTHING`,
			matchID, playerSlot, r.Time, string(r.Key), idx,
		)
	}
	return nil
}

func writePurchaseLog(batch *pgx.Batch, matchID int64, playerSlot int, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries []models.PurchaseLog
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal purchase_log: %w", err)
	}
	for idx, pur := range entries {
		batch.Queue(`
			INSERT INTO player_purchase_log (match_id, player_slot, time, key, charges, seq)
			OVERRIDING SYSTEM VALUE
			VALUES ($1,$2,$3,$4,$5,$6)
			ON CONFLICT (match_id, player_slot, time, key, seq) DO NOTHING`,
			matchID, playerSlot, pur.Time, string(pur.Key), pur.Charges, idx,
		)
	}
	return nil
}

// tableAllowlist restricts which table names can be interpolated into SQL
// queries in writeObsLog and writeObsLeftLog, preventing SQL injection via
// uncontrolled table name input (Issue #34). When adding new event-log tables,
// add the table name here.
var tableAllowlist = map[string]bool{
	"player_obs_log":      true,
	"player_sen_log":      true,
	"player_obs_left_log": true,
	"player_sen_left_log": true,
}

func obsTableName(table string) (string, error) {
	if !tableAllowlist[table] {
		return "", fmt.Errorf("unrecognized obs table name %q", table)
	}
	return table, nil
}

func writeObsLog(batch *pgx.Batch, matchID int64, playerSlot int, raw json.RawMessage, table string) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	table, err := obsTableName(table)
	if err != nil {
		return err
	}
	var entries []models.ObsLog
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal %s: %w", table, err)
	}
	q := fmt.Sprintf(`
		INSERT INTO %s (match_id, player_slot, time, key, x, y, z, entityleft, ehandle)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
		ON CONFLICT (match_id, player_slot, time, key) DO NOTHING`, table)
	for _, o := range entries {
		batch.Queue(q, matchID, playerSlot, o.Time, string(o.Key), o.X, o.Y, o.Z, o.EntityLeft, o.EHandle)
	}
	return nil
}

func writeObsLeftLog(batch *pgx.Batch, matchID int64, playerSlot int, raw json.RawMessage, table string) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	table, err := obsTableName(table)
	if err != nil {
		return err
	}
	var entries []models.ObsLeftLog
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal %s: %w", table, err)
	}
	q := fmt.Sprintf(`
		INSERT INTO %s (match_id, player_slot, time, key, attackername, x, y, z, entityleft, ehandle)
		VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
		ON CONFLICT (match_id, player_slot, time, key) DO NOTHING`, table)
	for _, ol := range entries {
		batch.Queue(q, matchID, playerSlot, ol.Time, string(ol.Key), ol.AttackerName, ol.X, ol.Y, ol.Z, ol.EntityLeft, ol.EHandle)
	}
	return nil
}

// writeAbilityUpgrades writes player_ability_upgrades_log rows. OpenDota
// returns `ability_upgrades_arr` as a flat list of ability IDs in pick
// order (e.g. [5625, 5625, 5108]) — NOT as a list of {time, ability}
// objects (Issue #33). We decode as []int and use the array index as the
// upgrade order.
// writeTimeSeriesArrays queues an INSERT for the minute-by-minute gold/XP arrays.
// The JSONB arrays are stored in player_time_series_arrays (PK: match_id,
// player_slot) instead of player_minute_stats so they don't conflict with
// real minute-zero stat rows (issue #5 — the old schema used minute=0 as a
// sentinel which collided with real data).
func writeTimeSeriesArrays(batch *pgx.Batch, matchID int64, playerSlot int, goldT, xpT []float64) {
	if len(goldT) == 0 && len(xpT) == 0 {
		return
	}
	goldJSON, _ := json.Marshal(goldT)
	xpJSON, _ := json.Marshal(xpT)
	batch.Queue(`
		INSERT INTO player_time_series_arrays (match_id, player_slot, gold_t, xp_t)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (match_id, player_slot) DO UPDATE SET
			gold_t = EXCLUDED.gold_t,
			xp_t = EXCLUDED.xp_t`,
		matchID, playerSlot, goldJSON, xpJSON,
	)
}

func writeAbilityUpgrades(batch *pgx.Batch, matchID int64, playerSlot int, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var abilityIDs []int
	if err := json.Unmarshal(raw, &abilityIDs); err != nil {
		return fmt.Errorf("unmarshal ability_upgrades (expected []int): %w", err)
	}
	for order, abilityID := range abilityIDs {
		batch.Queue(`
			INSERT INTO player_ability_upgrades_log (match_id, player_slot, upgrade_order, ability_id)
			VALUES ($1,$2,$3,$4)
			ON CONFLICT (match_id, player_slot, upgrade_order) DO NOTHING`,
			matchID, playerSlot, order, abilityID,
		)
	}
	return nil
}
