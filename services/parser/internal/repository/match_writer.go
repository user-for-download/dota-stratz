package repository

import (
	"github.com/dota-stratz/services/parser/internal/models"
	"github.com/jackc/pgx/v5"
)

// writeMatch queues an INSERT for the matches table. Idempotent via
// ON CONFLICT (match_id) DO NOTHING.
func writeMatch(batch *pgx.Batch, m models.OpenDotaMatch) {
	batch.Queue(`
		INSERT INTO matches (
			match_id, version, duration, start_time, series_id, series_type, cluster,
			replay_salt, radiant_win, pre_game_duration, match_seq_num,
			tower_status_radiant, tower_status_dire, barracks_status_radiant, barracks_status_dire,
			first_blood_time, lobby_type, human_players, game_mode, flags, engine,
			radiant_score, dire_score, radiant_team_id, radiant_name, radiant_logo, radiant_team_complete,
			dire_team_id, dire_name, dire_logo, dire_team_complete, radiant_captain, dire_captain,
			leagueid, patch, region, replay_url, throw, loss, metadata
		) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,$35,$36,$37,$38,$39,$40)
		ON CONFLICT (match_id) DO NOTHING`,
		m.MatchID, m.Version, m.Duration, m.StartTime, m.SeriesID, m.SeriesType, m.Cluster,
		m.ReplaySalt, m.RadiantWin, m.PreGameDuration, m.MatchSeqNum,
		m.TowerStatusRadiant, m.TowerStatusDire, m.BarracksStatusRadiant, m.BarracksStatusDire,
		m.FirstBloodTime, m.LobbyType, m.HumanPlayers, m.GameMode, m.Flags, m.Engine,
		m.RadiantScore, m.DireScore, m.RadiantTeamID, m.RadiantName, m.RadiantLogo, m.RadiantTeamComplete,
		m.DireTeamID, m.DireName, m.DireLogo, m.DireTeamComplete, m.RadiantCaptain, m.DireCaptain,
		m.LeagueID, m.Patch, m.Region, m.ReplayURL, m.Throw, m.Loss, m.Metadata,
	)
}

// writePlayer queues an INSERT for the players table. Idempotent via
// ON CONFLICT (match_id, player_slot) DO NOTHING.
func writePlayer(batch *pgx.Batch, m models.OpenDotaMatch, p models.Player) {
	batch.Queue(`
		INSERT INTO players (
			match_id, player_slot, account_id, hero_id, hero_variant, party_id, party_size,
			team_number, team_slot, is_radiant, radiant_win, win, lose, kills, deaths, assists,
			leaver_status, last_hits, denies, gold_per_min, xp_per_min, level, net_worth,
			gold, gold_spent, total_gold, total_xp, aghanims_scepter, aghanims_shard, moonshard,
			hero_damage, tower_damage, hero_healing, kills_per_min, kda, abandons,
			neutral_kills, tower_kills, courier_kills, lane_kills, hero_kills,
			observer_kills, sentry_kills, roshan_kills, necronomicon_kills, ancient_kills,
			buyback_count, observer_uses, sentry_uses, lane_efficiency, lane_efficiency_pct,
			lane, lane_role, is_roaming, actions_per_min, life_state_dead,
			obs_placed, sen_placed, creeps_stacked, camps_stacked, rune_pickups,
			firstblood_claimed, teamfight_participation, towers_killed, roshans_killed,
			observers_placed, stuns, item_0, item_1, item_2, item_3, item_4, item_5,
			backpack_0, backpack_1, backpack_2, item_neutral, item_neutral2,
			personaname, name, last_login, rank_tier, computed_mmr, is_subscriber,
			ability_targets, damage_targets, gold_reasons, xp_reasons, killed, item_uses,
			hero_hits, damage, damage_taken, damage_inflictor, runes, killed_by,
			kill_streaks, multi_kills, life_state, healing, damage_inflictor_received,
			lane_pos, obs, sen, actions, cosmetics, purchase_time, first_purchase_time,
			item_win, item_usage
		) VALUES (
			$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,
			$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,
			$35,$36,$37,$38,$39,$40,$41,$42,$43,$44,$45,$46,$47,$48,$49,$50,
			$51,$52,$53,$54,$55,$56,$57,$58,$59,$60,$61,$62,$63,$64,$65,$66,
			$67,$68,$69,$70,$71,$72,$73,$74,$75,$76,$77,$78,$79,$80,$81,$82,
			$83,$84,$85,$86,$87,$88,$89,$90,$91,$92,$93,$94,$95,$96,$97,$98,
			$99,$100,$101,$102,$103,$104,$105,$106,$107,$108,$109,$110
		) ON CONFLICT (match_id, player_slot) DO NOTHING`,
		m.MatchID, p.PlayerSlot, p.AccountID, p.HeroID, p.HeroVariant, p.PartyID, p.PartySize,
		p.TeamNumber, p.TeamSlot, p.IsRadiant, m.RadiantWin, p.Win, p.Lose, p.Kills, p.Deaths, p.Assists,
		p.LeaverStatus, p.LastHits, p.Denies, p.GoldPerMin, p.XpPerMin, p.Level, p.NetWorth,
		p.Gold, p.GoldSpent, p.TotalGold, p.TotalXP, p.AghanimsScepter, p.AghanimsShard, p.Moonshard,
		p.HeroDamage, p.TowerDamage, p.HeroHealing, p.KillsPerMin, p.KDA, p.Abandons,
		p.NeutralKills, p.TowerKills, p.CourierKills, p.LaneKills, p.HeroKills,
		p.ObserverKills, p.SentryKills, p.RoshanKills, p.NecronomiconKills, p.AncientKills,
		p.BuybackCount, p.ObserverUses, p.SentryUses, p.LaneEfficiency, p.LaneEfficiencyPct,
		p.Lane, p.LaneRole, p.IsRoaming, p.ActionsPerMin, p.LifeStateDead,
		p.ObsPlaced, p.SenPlaced, p.CreepsStacked, p.CampsStacked, p.RunePickups,
		p.FirstbloodClaimed, p.TeamfightParticipation, p.TowersKilled, p.RoshansKilled,
		p.ObserversPlaced, p.Stuns, p.Item0, p.Item1, p.Item2, p.Item3, p.Item4, p.Item5,
		p.Backpack0, p.Backpack1, p.Backpack2, p.ItemNeutral, p.ItemNeutral2,
		p.Personaname, p.Name, p.LastLogin, p.RankTier, p.ComputedMMR, p.IsSubscriber,
		p.AbilityTargets, p.DamageTargets, p.GoldReasons, p.XPReasons, p.Killed, p.ItemUses,
		p.HeroHits, p.Damage, p.DamageTaken, p.DamageInflictor, p.Runes, p.KilledBy,
		p.KillStreaks, p.MultiKills, p.LifeState, p.Healing, p.DamageInflictorReceived,
		p.LanePos, p.Obs, p.Sen, p.Actions, p.Cosmetics, p.PurchaseTime, p.FirstPurchaseTime,
		p.ItemWin, p.ItemUsage,
	)
}
