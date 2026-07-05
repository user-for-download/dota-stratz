package models

import (
	"encoding/json"
	"testing"
)

// TestOpenDotaMatch_UnmarshalJSON_DireLogoOverflow verifies that an oversized
// `dire_logo` value (larger than int64) does not fail the entire match parse
// and instead is clamped to math.MaxInt64 (Issue #33).
func TestOpenDotaMatch_UnmarshalJSON_DireLogoOverflow(t *testing.T) {
	// OpenDota observed a dire_logo of 1.59e19, larger than int64 max
	// (9.22e18). The parser must clamp, not fail.
	const input = `{
		"match_id": 8830131096,
		"version": 22,
		"duration": 1234,
		"start_time": 1700000000,
		"radiant_win": true,
		"radiant_team_id": 9530716,
		"radiant_logo": 2417958726624961500,
		"dire_team_id": 10060977,
		"dire_logo": 15900751100275690000,
		"radiant_captain": 236214375,
		"dire_captain": 906511622
	}`

	var m OpenDotaMatch
	if err := json.Unmarshal([]byte(input), &m); err != nil {
		t.Fatalf("Unmarshal failed (oversized logo should clamp, not fail): %v", err)
	}
	if m.MatchID != 8830131096 {
		t.Errorf("MatchID = %d, want 8830131096", m.MatchID)
	}
	if m.DireLogo == nil || *m.DireLogo != 9223372036854775807 {
		t.Errorf("DireLogo = %v, want clamped to MaxInt64 (9223372036854775807)", m.DireLogo)
	}
	// Non-overflow fields should pass through unchanged
	if m.RadiantTeamID == nil || *m.RadiantTeamID != 9530716 {
		t.Errorf("RadiantTeamID = %v, want 9530716", m.RadiantTeamID)
	}
	if m.RadiantLogo == nil || *m.RadiantLogo != 2417958726624961500 {
		t.Errorf("RadiantLogo = %v, want 2417958726624961500 (fits in int64)", m.RadiantLogo)
	}
}

// TestOpenDotaMatch_UnmarshalJSON_NullFields verifies that null-valued
// pointer fields stay nil (not zero).
func TestOpenDotaMatch_UnmarshalJSON_NullFields(t *testing.T) {
	const input = `{
		"match_id": 8830131096,
		"duration": 1234,
		"start_time": 1700000000,
		"radiant_team_id": null,
		"radiant_logo": null,
		"dire_team_id": null,
		"dire_logo": null
	}`

	var m OpenDotaMatch
	if err := json.Unmarshal([]byte(input), &m); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if m.RadiantTeamID != nil {
		t.Errorf("RadiantTeamID = %v, want nil", m.RadiantTeamID)
	}
	if m.DireLogo != nil {
		t.Errorf("DireLogo = %v, want nil", m.DireLogo)
	}
}
