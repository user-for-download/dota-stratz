package models

import (
	"encoding/json"
	"math"
	"strconv"
	"time"
)

// RawMatchMessage is the envelope published by the detail-fetcher.
type RawMatchMessage struct {
	MatchID   int64           `json:"match_id"`
	RawJSON   json.RawMessage `json:"raw_json"`
	FetchedAt time.Time       `json:"fetched_at"`
}

// SaturatingInt64 unmarshals a JSON number into a *int64 (pointer to int64).
// If the number is too large to fit in int64 (OpenDota's `dire_logo` and
// `radiant_logo` fields can exceed 2^63-1 — observed 1.59e19), the value is
// clamped to math.MaxInt64 instead of failing the entire match parse.
//
// Exported so the custom UnmarshalJSON on OpenDotaMatch can use it.
//
// This is a defensive shim for the OpenDota API which returns oversized
// integers for some team logos. PostgreSQL's BIGINT is int64, so any value
// larger would overflow on insert anyway. Clamping preserves the rest of
// the match data and lets the parser continue (Issue #33).
func SaturatingInt64(data []byte) (*int64, error) {
	// Empty / null leaves the pointer nil
	if len(data) == 0 || string(data) == "null" {
		return nil, nil
	}
	var raw json.Number
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	if raw.String() == "" || raw.String() == "null" {
		return nil, nil
	}
	n, err := strconv.ParseInt(raw.String(), 10, 64)
	if err != nil {
		// Number overflows int64 — clamp to MaxInt64 so the rest of the
		// match still parses. The downstream insert will commit this as
		// the maximum representable value.
		if numErr, ok := err.(*strconv.NumError); ok && numErr.Err == strconv.ErrRange {
			max := int64(math.MaxInt64)
			return &max, nil
		}
		return nil, err
	}
	return &n, nil
}

// overflowableFields are the OpenDota fields that occasionally exceed int64
// (OpenDota bug). They are extracted from the raw JSON and decoded with
// SaturatingInt64 instead of letting the standard decoder fail.
var overflowableFields = [...]string{
	"radiant_team_id",
	"radiant_logo",
	"dire_team_id",
	"dire_logo",
	"radiant_captain",
	"dire_captain",
}

// UnmarshalJSON on OpenDotaMatch handles OpenDota's `dire_logo` /
// `radiant_logo` values that can exceed int64 (Issue #33). Standard json
// unmarshaling would fail the entire match for one oversize logo.
//
// Strategy:
//  1. Parse data into a map[string]json.RawMessage
//  2. Pull the six overflowable fields out, decode with SaturatingInt64
//  3. Re-marshal the map (with the overflowable fields removed) and
//     decode it into a *OpenDotaMatch via a type alias to avoid recursion
//  4. Assign the saturating values to the corresponding fields
func (m *OpenDotaMatch) UnmarshalJSON(data []byte) error {
	// Step 1: parse into map
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	// Step 2: saturating-decode overflowable fields; save the resulting
	// pointers in locals so they survive the second-pass copy below.
	var rTeamID, rLogo, dTeamID, dLogo, rCap, dCap *int64
	for _, key := range overflowableFields {
		v, ok := raw[key]
		if !ok {
			continue
		}
		ptr, err := SaturatingInt64(v)
		if err != nil {
			return err
		}
		switch key {
		case "radiant_team_id":
			rTeamID = ptr
		case "radiant_logo":
			rLogo = ptr
		case "dire_team_id":
			dTeamID = ptr
		case "dire_logo":
			dLogo = ptr
		case "radiant_captain":
			rCap = ptr
		case "dire_captain":
			dCap = ptr
		}
		// Remove the field from the map so the second-pass unmarshal
		// doesn't try to decode it again (and fail on overflow).
		delete(raw, key)
	}

	// Step 3: re-marshal and decode into a temporary via type alias to
	// avoid recursing into our own UnmarshalJSON.
	stripped, err := json.Marshal(raw)
	if err != nil {
		return err
	}
	type alias OpenDotaMatch // type alias hides methods
	var a alias
	if err := json.Unmarshal(stripped, &a); err != nil {
		return err
	}
	// Step 4: copy fields from alias back into m
	*m = OpenDotaMatch(a)
	// Step 5: restore the saturating values (which would otherwise be nil
	// because the JSON keys were stripped before the second-pass decode)
	m.RadiantTeamID = rTeamID
	m.RadiantLogo = rLogo
	m.DireTeamID = dTeamID
	m.DireLogo = dLogo
	m.RadiantCaptain = rCap
	m.DireCaptain = dCap
	return nil
}

// OpenDotaMatch maps the top-level fields for the matches table and its related tables.
type OpenDotaMatch struct {
	MatchID               int64 `json:"match_id"`
	Version               int   `json:"version"`
	Duration              int   `json:"duration"`
	StartTime             int64 `json:"start_time"`
	SeriesID              int64 `json:"series_id"`
	SeriesType            int   `json:"series_type"`
	Cluster               int   `json:"cluster"`
	ReplaySalt            int64 `json:"replay_salt"`
	RadiantWin            *bool `json:"radiant_win"`
	PreGameDuration       int   `json:"pre_game_duration"`
	MatchSeqNum           int64 `json:"match_seq_num"`
	TowerStatusRadiant    int   `json:"tower_status_radiant"`
	TowerStatusDire       int   `json:"tower_status_dire"`
	BarracksStatusRadiant int   `json:"barracks_status_radiant"`
	BarracksStatusDire    int   `json:"barracks_status_dire"`
	FirstBloodTime        int   `json:"first_blood_time"`
	LobbyType             int   `json:"lobby_type"`
	HumanPlayers          int   `json:"human_players"`
	GameMode              int   `json:"game_mode"`
	Flags                 int   `json:"flags"`
	Engine                int   `json:"engine"`
	RadiantScore          int   `json:"radiant_score"`
	DireScore             int   `json:"dire_score"`

	RadiantTeamID       *int64 `json:"radiant_team_id"`
	RadiantName         string `json:"radiant_name"`
	RadiantLogo         *int64 `json:"radiant_logo"`
	RadiantTeamComplete int    `json:"radiant_team_complete"`
	DireTeamID          *int64 `json:"dire_team_id"`
	DireName            string `json:"dire_name"`
	DireLogo            *int64 `json:"dire_logo"`
	DireTeamComplete    int    `json:"dire_team_complete"`

	RadiantCaptain *int64 `json:"radiant_captain"`
	DireCaptain    *int64 `json:"dire_captain"`

	LeagueID  int    `json:"leagueid"`
	Patch     int    `json:"patch"`
	Region    int    `json:"region"`
	ReplayURL string `json:"replay_url"`
	Throw     int    `json:"throw"`
	Loss      int    `json:"loss"`

	Metadata json.RawMessage `json:"metadata"`

	Players    []Player        `json:"players"`
	PicksBans  json.RawMessage `json:"picks_bans"`
	Objectives json.RawMessage `json:"objectives"`
	Chat       json.RawMessage `json:"chat"`
	Teamfights json.RawMessage `json:"teamfights"`
	GoldAdv    json.RawMessage `json:"radiant_gold_adv"`
	XPAdv      json.RawMessage `json:"radiant_xp_adv"`
}

// Player maps the per-player stats for the players table and its child tables.
type Player struct {
	PlayerSlot  int    `json:"player_slot"`
	AccountID   *int64 `json:"account_id"`
	HeroID      int    `json:"hero_id"`
	HeroVariant int    `json:"hero_variant"`

	PartyID    *int64 `json:"party_id"`
	PartySize  int    `json:"party_size"`
	TeamNumber int    `json:"team_number"`
	TeamSlot   int    `json:"team_slot"`
	IsRadiant  bool   `json:"isRadiant"`
	Win        int    `json:"win"`
	Lose       int    `json:"lose"`

	Kills        int       `json:"kills"`
	Deaths       int       `json:"deaths"`
	Assists      int       `json:"assists"`
	LeaverStatus int       `json:"leaver_status"`
	LastHits     int       `json:"last_hits"`
	Denies       int       `json:"denies"`
	GoldPerMin   int       `json:"gold_per_min"`
	XpPerMin     int       `json:"xp_per_min"`
	GoldT        []float64 `json:"gold_t"` // Minute-by-minute gold array
	XPT          []float64 `json:"xp_t"`   // Minute-by-minute XP array
	Level        int       `json:"level"`
	NetWorth     int       `json:"net_worth"`
	Gold         int       `json:"gold"`
	GoldSpent    int       `json:"gold_spent"`
	TotalGold    int       `json:"total_gold"`
	TotalXP      int       `json:"total_xp"`

	AghanimsScepter int `json:"aghanims_scepter"`
	AghanimsShard   int `json:"aghanims_shard"`
	Moonshard       int `json:"moonshard"`

	HeroDamage        int     `json:"hero_damage"`
	TowerDamage       int     `json:"tower_damage"`
	HeroHealing       int     `json:"hero_healing"`
	KillsPerMin       float64 `json:"kills_per_min"`
	KDA               float64 `json:"kda"`
	Abandons          int     `json:"abandons"`
	NeutralKills      int     `json:"neutral_kills"`
	TowerKills        int     `json:"tower_kills"`
	CourierKills      int     `json:"courier_kills"`
	LaneKills         int     `json:"lane_kills"`
	HeroKills         int     `json:"hero_kills"`
	ObserverKills     int     `json:"observer_kills"`
	SentryKills       int     `json:"sentry_kills"`
	RoshanKills       int     `json:"roshan_kills"`
	NecronomiconKills int     `json:"necronomicon_kills"`
	AncientKills      int     `json:"ancient_kills"`
	BuybackCount      int     `json:"buyback_count"`
	ObserverUses      int     `json:"observer_uses"`
	SentryUses        int     `json:"sentry_uses"`

	LaneEfficiency    float64 `json:"lane_efficiency"`
	LaneEfficiencyPct int     `json:"lane_efficiency_pct"`
	Lane              int     `json:"lane"`
	LaneRole          int     `json:"lane_role"`
	IsRoaming         bool    `json:"is_roaming"`
	ActionsPerMin     int     `json:"actions_per_min"`
	LifeStateDead     int     `json:"life_state_dead"`

	ObsPlaced              int     `json:"obs_placed"`
	SenPlaced              int     `json:"sen_placed"`
	CreepsStacked          int     `json:"creeps_stacked"`
	CampsStacked           int     `json:"camps_stacked"`
	RunePickups            int     `json:"rune_pickups"`
	FirstbloodClaimed      int     `json:"firstblood_claimed"`
	TeamfightParticipation float64 `json:"teamfight_participation"`
	TowersKilled           int     `json:"towers_killed"`
	RoshansKilled          int     `json:"roshans_killed"`
	ObserversPlaced        int     `json:"observers_placed"`
	Stuns                  float64 `json:"stuns"`

	Item0 int `json:"item_0"`
	Item1 int `json:"item_1"`
	Item2 int `json:"item_2"`
	Item3 int `json:"item_3"`
	Item4 int `json:"item_4"`
	Item5 int `json:"item_5"`

	Backpack0 int `json:"backpack_0"`
	Backpack1 int `json:"backpack_1"`
	Backpack2 int `json:"backpack_2"`

	ItemNeutral  int `json:"item_neutral"`
	ItemNeutral2 int `json:"item_neutral2"`

	Personaname  string  `json:"personaname"`
	Name         string  `json:"name"`
	LastLogin    string  `json:"last_login"`
	RankTier     int     `json:"rank_tier"`
	ComputedMMR  float64 `json:"computed_mmr"`
	IsSubscriber bool    `json:"is_subscriber"`

	// JSONB fields stored directly in Postgres (raw bytes, no Go unmarshal).
	AbilityTargets          json.RawMessage `json:"ability_targets"`
	DamageTargets           json.RawMessage `json:"damage_targets"`
	GoldReasons             json.RawMessage `json:"gold_reasons"`
	XPReasons               json.RawMessage `json:"xp_reasons"`
	Killed                  json.RawMessage `json:"killed"`
	ItemUses                json.RawMessage `json:"item_uses"`
	HeroHits                json.RawMessage `json:"hero_hits"`
	Damage                  json.RawMessage `json:"damage"`
	DamageTaken             json.RawMessage `json:"damage_taken"`
	DamageInflictor         json.RawMessage `json:"damage_inflictor"`
	Runes                   json.RawMessage `json:"runes"`
	KilledBy                json.RawMessage `json:"killed_by"`
	KillStreaks             json.RawMessage `json:"kill_streaks"`
	MultiKills              json.RawMessage `json:"multi_kills"`
	LifeState               json.RawMessage `json:"life_state"`
	Healing                 json.RawMessage `json:"healing"`
	DamageInflictorReceived json.RawMessage `json:"damage_inflictor_received"`
	LanePos                 json.RawMessage `json:"lane_pos"`
	Obs                     json.RawMessage `json:"obs"`
	Sen                     json.RawMessage `json:"sen"`
	Actions                 json.RawMessage `json:"actions"`
	Cosmetics               json.RawMessage `json:"cosmetics"`
	PurchaseTime            json.RawMessage `json:"purchase_time"`
	FirstPurchaseTime       json.RawMessage `json:"first_purchase_time"`
	ItemWin                 json.RawMessage `json:"item_win"`
	ItemUsage               json.RawMessage `json:"item_usage"`

	// Nested arrays for relational child tables.
	AbilityUpgradesArr json.RawMessage `json:"ability_upgrades_arr"`
	Benchmarks         json.RawMessage `json:"benchmarks"`
	KillsLog           json.RawMessage `json:"kills_log"`
	BuybackLog         json.RawMessage `json:"buyback_log"`
	RunesLog           json.RawMessage `json:"runes_log"`
	PurchaseLog        json.RawMessage `json:"purchase_log"`
	ObsLog             json.RawMessage `json:"obs_log"`
	SenLog             json.RawMessage `json:"sen_log"`
	ObsLeftLog         json.RawMessage `json:"obs_left_log"`
	SenLeftLog         json.RawMessage `json:"sen_left_log"`
	PermanentBuffs     json.RawMessage `json:"permanent_buffs"`
	NeutralItemHistory json.RawMessage `json:"neutral_item_history"`
}

// --- Nested Event Types (normalized into relational child tables) ---

type PickBan struct {
	IsPick bool `json:"is_pick"`
	HeroID int  `json:"hero_id"`
	Team   int  `json:"team"`
	Order  int  `json:"order"`
}

type Objective struct {
	Time       int        `json:"time"`
	Type       string     `json:"type"`
	Team       int        `json:"team"`
	Key        FlexString `json:"key"`
	Slot       int        `json:"slot"`
	PlayerSlot int        `json:"player_slot"`
	Value      int        `json:"value"`
	Killer     int        `json:"killer"`
}

type ChatMessage struct {
	Time       int        `json:"time"`
	Type       string     `json:"type"`
	Key        FlexString `json:"key"`
	Slot       int        `json:"slot"`
	PlayerSlot int        `json:"player_slot"`
}

type KillLog struct {
	Time int        `json:"time"`
	Key  FlexString `json:"key"`
}

type BuybackLog struct {
	Time int `json:"time"`
}

type RuneLog struct {
	Time int        `json:"time"`
	Key  FlexString `json:"key"`
}

type PurchaseLog struct {
	Time    int        `json:"time"`
	Key     FlexString `json:"key"`
	Charges int        `json:"charges"`
}

type ObsLog struct {
	Time       int        `json:"time"`
	Key        FlexString `json:"key"`
	X          float64    `json:"x"`
	Y          float64    `json:"y"`
	Z          float64    `json:"z"`
	EntityLeft bool       `json:"entityleft"`
	EHandle    int64      `json:"ehandle"`
}

type ObsLeftLog struct {
	Time         int        `json:"time"`
	Key          FlexString `json:"key"`
	AttackerName string     `json:"attackername"`
	X            float64    `json:"x"`
	Y            float64    `json:"y"`
	Z            float64    `json:"z"`
	EntityLeft   bool       `json:"entityleft"`
	EHandle      int64      `json:"ehandle"`
}

// FlexString handles OpenDota JSON fields that dynamically switch between
// strings and numbers (e.g. 'key' is a string for text chat, but an integer
// for chatwheel/runes). Without this custom type, Go's strict json.Unmarshal
// throws "cannot unmarshal number into Go struct field" when OpenDota sends
// a bare number for a string-typed field, sending the entire match to the DLQ.
type FlexString string

func (fs *FlexString) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		*fs = ""
		return nil
	}
	// If it's a JSON string, strip the quotes and handle escapes properly.
	if b[0] == '"' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		*fs = FlexString(s)
		return nil
	}
	// It's a raw number, boolean, etc. Cast the raw bytes directly to a string.
	*fs = FlexString(b)
	return nil
}

// NOTE: OpenDota returns `ability_upgrades_arr` as a flat list of integer
// ability IDs (e.g. [5625, 5625, 5108, ...]), not a list of objects.
// The batch writer decodes it as `[]int` and uses the array index as the
// upgrade_order. See writeAbilityUpgrades in batch_writer.go (Issue #33).

type Benchmark struct {
	Raw float64 `json:"raw"`
	Pct float64 `json:"pct"`
}

type PermanentBuff struct {
	PermanentBuff int `json:"permanent_buff"`
	StackCount    int `json:"stack_count"`
	GrantTime     int `json:"grant_time"`
}

type NeutralItemHistory struct {
	ItemNeutral            string `json:"item_neutral"`
	Time                   int    `json:"time"`
	ItemNeutralEnhancement string `json:"item_neutral_enhancement"`
}

type Teamfight struct {
	Start     int               `json:"start"`
	End       int               `json:"end"`
	LastDeath int               `json:"last_death"`
	Deaths    int               `json:"deaths"`
	Players   []TeamfightPlayer `json:"players"`
}

type TeamfightPlayer struct {
	Deaths      int             `json:"deaths"`
	Buybacks    int             `json:"buybacks"`
	Damage      int             `json:"damage"`
	Healing     int             `json:"healing"`
	GoldDelta   int             `json:"gold_delta"`
	XPDelta     int             `json:"xp_delta"`
	XPStart     int             `json:"xp_start"`
	XPEnd       int             `json:"xp_end"`
	AbilityUses json.RawMessage `json:"ability_uses"`
	ItemUses    json.RawMessage `json:"item_uses"`
	Killed      json.RawMessage `json:"killed"`
	DeathsPos   json.RawMessage `json:"deaths_pos"`
}
