package repository

import (
	"encoding/json"
	"fmt"

	"github.com/dota-stratz/services/parser/internal/models"
	"github.com/jackc/pgx/v5"
)

func writePicksBans(batch *pgx.Batch, matchID int64, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries []models.PickBan
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal picks_bans: %w", err)
	}
	for _, pb := range entries {
		batch.Queue(`
			INSERT INTO picks_bans (match_id, is_pick, hero_id, team, "order")
			VALUES ($1,$2,$3,$4,$5)
			ON CONFLICT (match_id, "order") DO NOTHING`,
			matchID, pb.IsPick, pb.HeroID, pb.Team, pb.Order,
		)
	}
	return nil
}

func writeObjectives(batch *pgx.Batch, matchID int64, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries []models.Objective
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal objectives: %w", err)
	}
	for _, obj := range entries {
		batch.Queue(`
			INSERT INTO objectives (match_id, time, type, team, key, slot, player_slot, value, killer)
			VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
			ON CONFLICT (match_id, time, type, team) DO NOTHING`,
			matchID, obj.Time, obj.Type, obj.Team, string(obj.Key), obj.Slot, obj.PlayerSlot, obj.Value, obj.Killer,
		)
	}
	return nil
}

func writeChat(batch *pgx.Batch, matchID int64, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries []models.ChatMessage
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal chat: %w", err)
	}
	for _, chat := range entries {
		batch.Queue(`
			INSERT INTO chat (match_id, time, type, key, slot, player_slot)
			VALUES ($1,$2,$3,$4,$5,$6)
			ON CONFLICT (match_id, time, slot) DO NOTHING`,
			matchID, chat.Time, chat.Type, string(chat.Key), chat.Slot, chat.PlayerSlot,
		)
	}
	return nil
}

func writeGoldAdv(batch *pgx.Batch, matchID int64, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries []int
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal gold_adv: %w", err)
	}
	for minute, adv := range entries {
		batch.Queue(`
			INSERT INTO match_gold_adv (match_id, minute, radiant_gold_adv)
			VALUES ($1,$2,$3)
			ON CONFLICT (match_id, minute) DO NOTHING`,
			matchID, minute, adv,
		)
	}
	return nil
}

func writeXPAdv(batch *pgx.Batch, matchID int64, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries []int
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal xp_adv: %w", err)
	}
	for minute, adv := range entries {
		batch.Queue(`
			INSERT INTO match_xp_adv (match_id, minute, radiant_xp_adv)
			VALUES ($1,$2,$3)
			ON CONFLICT (match_id, minute) DO NOTHING`,
			matchID, minute, adv,
		)
	}
	return nil
}
