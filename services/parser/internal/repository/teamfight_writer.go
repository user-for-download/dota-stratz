package repository

import (
	"encoding/json"
	"fmt"

	"github.com/dota-stratz/services/parser/internal/models"
	"github.com/jackc/pgx/v5"
)

// teamfightPlayerSlots maps a positional index in OpenDota's
// teamfights[].players[] array to the corresponding Dota player_slot.
// OpenDota returns the 10 teamfight stats blobs in canonical order:
// indices 0-4 = radiant (player_slot 0-4), indices 5-9 = dire
// (player_slot 128-132). Each blob has no player_slot field of its own;
// the slot is implicit in the index. The DB FK
// teamfight_players_match_id_player_slot_fkey requires a valid slot from
// the players table, so we look it up here.
var teamfightPlayerSlots = []int{0, 1, 2, 3, 4, 128, 129, 130, 131, 132}

func writeTeamfights(batch *pgx.Batch, matchID int64, raw json.RawMessage) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var entries []models.Teamfight
	if err := json.Unmarshal(raw, &entries); err != nil {
		return fmt.Errorf("unmarshal teamfights: %w", err)
	}
	for _, tf := range entries {
		batch.Queue(`
			INSERT INTO teamfights (match_id, start_time, end_time, last_death, deaths)
			VALUES ($1,$2,$3,$4,$5)
			ON CONFLICT (match_id, start_time) DO NOTHING`,
			matchID, tf.Start, tf.End, tf.LastDeath, tf.Deaths,
		)

		// OpenDota's teamfights[].players[] is a positional list of 10
		// stats blobs in the canonical Dota order — indices 0-4 are the
		// radiant team (player_slot 0-4) and indices 5-9 are the dire
		// team (player_slot 128-132). Each item has no player_slot
		// field; the slot is implicit in the array index. Map index → slot
		// explicitly so the FK to players(teamfight_players_match_id_player_slot_fkey)
		// is satisfied. Defensive cap at 10 in case the API ever returns
		// extra entries (e.g. coaches/spectators).
		for i, tfp := range tf.Players {
			if i >= len(teamfightPlayerSlots) {
				break
			}
			slot := teamfightPlayerSlots[i]
			batch.Queue(`
				INSERT INTO teamfight_players (
					match_id, start_time, player_slot, deaths, buybacks, damage, healing,
					gold_delta, xp_delta, xp_start, xp_end, ability_uses, item_uses, killed, deaths_pos
				) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
				ON CONFLICT (match_id, start_time, player_slot) DO NOTHING`,
				matchID, tf.Start, slot, tfp.Deaths, tfp.Buybacks, tfp.Damage, tfp.Healing,
				tfp.GoldDelta, tfp.XPDelta, tfp.XPStart, tfp.XPEnd,
				tfp.AbilityUses, tfp.ItemUses, tfp.Killed, tfp.DeathsPos,
			)
		}
	}
	return nil
}
