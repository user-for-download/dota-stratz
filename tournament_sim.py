#!/usr/bin/env python3
"""The International 2026 — Full Tournament Simulation with Drafts.

Simulates complete draft phases (bans + picks) for every match using
the /predict endpoint, then evaluates compositions with /predict-match.
Adds a ±10-15% confidence range to reflect model uncertainty.
"""

import json
import random
import time
import urllib.request

API = "http://localhost:8080"
PATCH = 60
CONFIDENCE_RANGE = 0.125  # ±12.5% uncertainty band

# Patch 60 draft pattern (normalized: 0=first_pick_team)
DRAFT_PATTERN = [
    (0, False), (0, False), (1, False), (1, False),  # B0 B0 B1 B1
    (0, False), (1, False), (1, False),                # B0 B1 B1
    (0, True), (1, True),                              # P0 P1
    (0, False), (0, False), (1, False),                # B0 B0 B1
    (1, True), (0, True), (0, True), (1, True), (1, True), (0, True),  # P1 P0 P0 P1 P1 P0
    (0, False), (1, False), (0, False), (1, False),    # B0 B1 B0 B1
    (0, True), (1, True),                               # P0 P1
]

TEAMS = [
    ("Aurora Gaming", "Invited"), ("BoomBoys", "Invited"),
    ("Team Falcons", "Invited"), ("Team Liquid", "Invited"),
    ("1w Team", "Invited"), ("Xtreme Gaming", "Invited"),
    ("Team Yandex", "Invited"), ("Team Spirit", "Invited"),
    ("TEAM VISION", "Qualified"), ("Nigma Galaxy", "Qualified"),
    ("HULIGANI", "Qualified"), ("Team Resilience", "Qualified"),
    ("Vici Gaming", "Qualified"), ("OG", "Qualified"),
    ("GamerLegion", "Qualified"), ("LGD Gaming", "Qualified"),
]

HERO_IDS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
            21,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
            41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
            61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,
            81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,
            101,102,103,104,105,106,107,108,109,110,111,112,113,114,119,120,121,123,126,128,129,131,135,136,137,138,145,155]

HERO_NAMES = {
    1:"Anti-Mage",2:"Axe",3:"Bane",4:"Bloodseeker",5:"Crystal Maiden",
    6:"Drow Ranger",7:"Earthshaker",8:"Juggernaut",9:"Mirana",10:"Morphling",
    11:"Shadow Fiend",12:"Phantom Lancer",13:"Puck",14:"Storm Spirit",
    15:"Sven",16:"Tiny",17:"Tinker",18:"Zeus",19:"Slardar",20:"Sniper",
    21:"Necrophos",22:"Warlock",23:"Faceless Void",25:"Shadow Shaman",
    26:"Venomancer",27:"Viper",28:"Spirit Breaker",29:"Weaver",30:"Nature's Prophet",
    31:"Lifestealer",32:"Dark Seer",33:"Clinkz",34:"Omniknight",
    35:"Enchantress",36:"Shadow Demon",37:"Dazzle",38:"Death Prophet",
    39:"Razor",40:"Sand King",41:"Windranger",42:"Phantom Assassin",
    43:"Outworld Destroyer",44:"Lich",45:"Lion",46:"Brewmaster",
    47:"Shadow Shaman",48:"Ursa",49:"Gyrocopter",50:"Alchemist",
    51:"Invoker",52:"Silencer",53:"Outworld Destroyer",54:"Lycan",
    55:"Brewmaster",56:"Dragon Knight",57:"Jakiro",58:"Batrider",
    59:"Chaos Knight",60:"Rubick",61:"Keeper of the Light",62:"Wisp",
    63:"Broodmother",64:"Queen of Pain",65:"Nyx Assassin",66:"Keeper of the Light",
    67:"Io",68:"Centaur Warrunner",69:"Visage",70:"Oracle",
    71:"Earth Spirit",72:"Terrorblade",73:"Phoenix",74:"Tusk",
    75:"Skywrath Mage",76:"Abaddon",77:"Elder Titan",78:"Legion Commander",
    79:"Ember Spirit",80:"Earth Spirit",81:"Techies",82:"Terrorblade",
    83:"Underlord",84:"Terrorblade",85:"Grimstroke",86:"Mars",
    87:"Hoodwink",88:"Void Spirit",89:"Snapfire",90:"Void Spirit",
    91:"Dawnbreaker",92:"Marci",93:"Primal Beast",94:"Pangolier",
    95:"Grimstroke",96:"Hoodwink",97:"Ringmaster",98:"Kez",
    99:"Beastmaster",100:"Axe",101:"Brewmaster",102:"Bloodseeker",
    103:"Crystal Maiden",104:"Shadow Fiend",105:"Nyx Assassin",
    106:"Doom",107:"Chaos Knight",108:"Shadow Demon",109:"Pangolier",
    110:"Ancient Apparition",111:"Outworld Destroyer",112:"Puck",
    113:"Templar Assassin",114:"Naga Siren",119:"Dark Willow",
    120:"Pangolier",121:"Grimstroke",123:"Hoodwink",126:"Void Spirit",
    128:"Primal Beast",129:"Marci",131:"Dawnbreaker",135:"Muerta",
    136:"Ringmaster",137:"Kez",138:"Lion",145:"Enchantress",155:"Lich",
}


def api_post(endpoint, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{API}{endpoint}", data=data,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def api_get(endpoint):
    req = urllib.request.Request(f"{API}{endpoint}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def hero_name(hid):
    return HERO_NAMES.get(hid, f"Hero {hid}")


def simulate_draft(team_a, team_b, first_pick_team):
    """Simulate a full 24-step draft for Bo1 or a single game of Bo3.
    Returns (draft_slots, radiant_heroes, dire_heroes, radiant_bans, dire_bans).
    """
    draft_slots = []
    taken = set()
    rad_picks, dire_picks = [], []
    rad_bans, dire_bans = [], []

    for step_idx, (team, is_pick) in enumerate(DRAFT_PATTERN):
        # Map normalized team to actual team
        recommending_team = team if first_pick_team == 0 else 1 - team

        # Build draft context so far
        draft_for_api = []
        for i, slot in enumerate(draft_slots):
            draft_for_api.append({
                "hero_id": slot[0], "is_pick": slot[1],
                "team": slot[2], "order": i + 1,
            })

        try:
            result = api_post("/predict", {
                "patch_id": PATCH,
                "draft": draft_for_api,
                "for_team": recommending_team,
                "first_pick_team": first_pick_team,
                "num_recommendations": 5,
            })
            recs = result.get("recommendations", [])
        except Exception:
            recs = []

        # Pick the top available hero, or random fallback
        chosen = None
        for r in recs:
            if r["hero_id"] not in taken:
                chosen = r["hero_id"]
                break
        if chosen is None:
            available = [h for h in HERO_IDS if h not in taken]
            chosen = random.choice(available) if available else 1

        taken.add(chosen)
        actual_team = team
        draft_slots.append((chosen, is_pick, actual_team))

        if is_pick:
            if actual_team == 0:
                rad_picks.append(chosen)
            else:
                dire_picks.append(chosen)
        else:
            if actual_team == 0:
                rad_bans.append(chosen)
            else:
                dire_bans.append(chosen)

    return draft_slots, rad_picks, dire_picks, rad_bans, dire_bans


def simulate_match(team_a, team_b, first_pick_team=0, game_num=1):
    """Simulate a full draft + predict-match for one game."""
    draft_slots, rad_picks, dire_picks, rad_bans, dire_bans = simulate_draft(
        team_a, team_b, first_pick_team
    )

    # Pad to 5 heroes if draft was incomplete
    while len(rad_picks) < 5:
        available = [h for h in HERO_IDS if h not in set(rad_picks + dire_picks + [s[0] for s in draft_slots])]
        if available:
            rad_picks.append(random.choice(available))
        else:
            break
    while len(dire_picks) < 5:
        available = [h for h in HERO_IDS if h not in set(rad_picks + dire_picks + [s[0] for s in draft_slots])]
        if available:
            dire_picks.append(random.choice(available))
        else:
            break

    if len(rad_picks) < 5 or len(dire_picks) < 5:
        # Fallback: random composition
        avail = [h for h in HERO_IDS if h not in [s[0] for s in draft_slots]]
        if len(avail) >= 10:
            chosen = random.sample(avail, 10)
            rad_picks = chosen[:5]
            dire_picks = chosen[5:]

    try:
        result = api_post("/predict-match", {
            "patch_id": PATCH,
            "radiant_heroes": rad_picks,
            "dire_heroes": dire_picks,
        })
        raw_prob = result["radiant_win_probability"]
    except Exception:
        raw_prob = 0.5

    # Apply confidence range (±12.5%)
    noise = random.uniform(-CONFIDENCE_RANGE, CONFIDENCE_RANGE)
    adjusted_prob = max(0.05, min(0.95, raw_prob + noise))

    winner = team_a if adjusted_prob >= 0.5 else team_b
    loser = team_b if winner == team_a else team_a

    return {
        "winner": winner,
        "loser": loser,
        "rad_prob": raw_prob,
        "adjusted_prob": adjusted_prob,
        "confidence_range": f"{max(0, (adjusted_prob - CONFIDENCE_RANGE))*100:.0f}-{min(1, (adjusted_prob + CONFIDENCE_RANGE))*100:.0f}%",
        "rad_picks": rad_picks,
        "dire_picks": dire_picks,
        "rad_bans": rad_bans,
        "dire_bans": dire_bans,
    }


def sim_bo1(team_a, team_b):
    result = simulate_match(team_a, team_b, first_pick_team=random.choice([0, 1]))
    return result["winner"], result["loser"], result


def sim_bo3(team_a, team_b):
    wins_a, wins_b = 0, 0
    games = []
    while wins_a < 2 and wins_b < 2:
        game_num = wins_a + wins_b + 1
        if wins_a == 0 and wins_b == 0:
            rad, dire = team_a, team_b
            fp = 0
        elif wins_a == 1 and wins_b == 0:
            rad, dire = team_b, team_a  # loser picks side
            fp = 1
        elif wins_a == 0 and wins_b == 1:
            rad, dire = team_a, team_b
            fp = 0
        else:  # game 3
            rad = team_a if random.random() < 0.5 else team_b
            dire = team_b if rad == team_a else team_a
            fp = 0 if rad == team_a else 1

        result = simulate_match(rad, dire, first_pick_team=fp, game_num=game_num)
        games.append(result)

        if result["winner"] == team_a:
            wins_a += 1
        else:
            wins_b += 1

    winner = team_a if wins_a > wins_b else team_b
    loser = team_b if winner == team_a else team_a
    return winner, loser, wins_a, wins_b, games


def print_draft(result):
    rad_str = ", ".join(hero_name(h) for h in result["rad_picks"][:5])
    dire_str = ", ".join(hero_name(h) for h in result["dire_picks"][:5])
    rad_ban_str = ", ".join(hero_name(h) for h in result["rad_bans"][:4]) or "none"
    dire_ban_str = ", ".join(hero_name(h) for h in result["dire_bans"][:4]) or "none"
    print(f"    Bans: {rad_ban_str} | {dire_ban_str}")
    print(f"    Picks: {rad_str} | {dire_str}")
    print(f"    Win Prob: {result['adjusted_prob']*100:.1f}% (±{CONFIDENCE_RANGE*100:.0f}%)")


# ============================================================
# SWISS SYSTEM
# ============================================================

def run_swiss(teams):
    records = {t: {"wins": 0, "losses": 0, "qualified": False, "eliminated": False}
               for t in teams}
    print(f"\n{'='*72}")
    print(f"  THE INTERNATIONAL 2026 — SWISS SYSTEM")
    print(f"  16 teams → 8 qualify (first to 3W or 3L)")
    print(f"{'='*72}\n")

    for rnd in range(1, 6):
        by_record = {}
        for t in teams:
            r = records[t]
            if r["qualified"] or r["eliminated"]:
                continue
            key = (r["wins"], r["losses"])
            by_record.setdefault(key, []).append(t)

        matchups = []
        for rec, pool in sorted(by_record.items()):
            random.shuffle(pool)
            for i in range(0, len(pool) - 1, 2):
                matchups.append((pool[i], pool[i + 1]))
            if len(pool) % 2 == 1:
                records[pool[-1]]["wins"] += 1
                if records[pool[-1]]["wins"] >= 3:
                    records[pool[-1]]["qualified"] = True
                print(f"  Round {rnd}: {pool[-1]} receives BYE → {records[pool[-1]]['wins']}-{records[pool[-1]]['losses']}")

        print(f"\n--- Round {rnd} ({len(matchups)} Bo1 matches) ---")
        for team_a, team_b in matchups:
            winner, loser, result = sim_bo1(team_a, team_b)
            records[winner]["wins"] += 1
            records[loser]["losses"] += 1
            if records[winner]["wins"] >= 3:
                records[winner]["qualified"] = True
            if records[loser]["losses"] >= 3:
                records[loser]["eliminated"] = True

            w_r = records[winner]
            l_r = records[loser]
            print(f"\n  {winner} {w_r['wins']}-{w_r['losses']} def. {loser} {l_r['wins']}-{l_r['losses']}")
            print_draft(result)

    print(f"\n{'='*72}")
    print(f"  SWISS RESULTS")
    print(f"{'='*72}")
    for t in sorted(teams, key=lambda x: (-records[x]["wins"], records[x]["losses"])):
        r = records[t]
        status = "✅ QUALIFIED" if r["qualified"] else ("❌ ELIMINATED" if r["eliminated"] else f"⚠️  {r['wins']}-{r['losses']}")
        print(f"  {t:>20s}  {r['wins']}-{r['losses']}  {status}")

    return [t for t in teams if records[t]["qualified"]][:8]


# ============================================================
# DOUBLE-ELIMINATION PLAYOFFS
# ============================================================

def run_playoffs(qualifiers):
    print(f"\n{'='*72}")
    print(f"  THE INTERNATIONAL 2026 — DOUBLE-ELIMINATION PLAYOFFS")
    print(f"{'='*72}\n")

    random.shuffle(qualifiers)
    seeds = {t: i + 1 for i, t in enumerate(qualifiers)}

    def p_match(label, t1, t2, is_bo3=True):
        if is_bo3:
            w, l, wa, wb, games = sim_bo3(t1, t2)
            print(f"\n  [{label}] {seeds[w]}. {w} {wa}-{wb} {seeds[l]}. {l}")
            for i, g in enumerate(games):
                print(f"    Game {i+1}:")
                print_draft(g)
            return w, l
        else:
            w, l, r = sim_bo1(t1, t2)
            print(f"\n  [{label}] {seeds[w]}. {w} def. {seeds[l]}. {l}")
            print_draft(r)
            return w, l

    # UB QF
    print("\n═══ UPPER BRACKET QUARTERFINALS ═══")
    ub_qf = []
    ubqf_losers = []
    for i in range(0, 8, 2):
        w, l = p_match("UB QF", qualifiers[i], qualifiers[i+1])
        ub_qf.append(w)
        ubqf_losers.append(l)

    # UB SF
    print("\n═══ UPPER BRACKET SEMIFINALS ═══")
    ub_sf = []
    ubsf_losers = []
    for i in range(0, 4, 2):
        w, l = p_match("UB SF", ub_qf[i], ub_qf[i+1])
        ub_sf.append(w)
        ubsf_losers.append(l)

    # UB Final
    print("\n═══ UPPER BRACKET FINAL ═══")
    ub_winner, ub_loser = p_match("UB Final", ub_sf[0], ub_sf[1])

    # LB R1
    print("\n═══ LOWER BRACKET ROUND 1 (UB QF losers) ═══")
    lb_r1 = []
    for i in range(0, 4, 2):
        w, l = p_match("LB R1", ubqf_losers[i], ubqf_losers[i+1])
        lb_r1.append(w)

    # LB R2
    print("\n═══ LOWER BRACKET ROUND 2 (vs UB SF losers) ═══")
    lb_r2 = []
    for i in range(2):
        w, l = p_match("LB R2", lb_r1[i], ubsf_losers[i])
        lb_r2.append(w)

    # LB SF
    print("\n═══ LOWER BRACKET SEMIFINAL ═══")
    lb_winner, lb_loser = p_match("LB SF", lb_r2[0], lb_r2[1])

    # Grand Final
    print("\n═══ GRAND FINAL ═══")
    print(f"  {seeds[ub_winner]}. {ub_winner} (Upper Bracket)  vs  {seeds[lb_winner]}. {lb_winner} (Lower Bracket)")
    g_winner, g_loser, ga, gb, g_games = sim_bo3(ub_winner, lb_winner)
    print(f"\n  Result: {g_winner} {ga}-{gb}")
    for i, g in enumerate(g_games):
        print(f"    Game {i+1}:")
        print_draft(g)

    if g_winner != ub_winner:
        print(f"\n═══ BRACKET RESET ═══")
        print(f"  {seeds[ub_winner]}. {ub_winner} must win twice!")
        g_winner2, g_loser2, ga2, gb2, g_games2 = sim_bo3(ub_winner, lb_winner)
        print(f"\n  Reset: {g_winner2} {ga2}-{gb2}")
        for i, g in enumerate(g_games2):
            print(f"    Game {i+1}:")
            print_draft(g)
        final = g_winner2
    else:
        final = g_winner

    print(f"\n{'='*72}")
    print(f"  🏆 THE INTERNATIONAL 2026 CHAMPION: {final}")
    print(f"{'='*72}\n")
    return final


if __name__ == "__main__":
    try:
        health = api_get("/health")
        print(f"API: {health['status']} | Models: {health['patch_models_loaded']}")
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)

    start = time.time()
    qualifiers = run_swiss([t[0] for t in TEAMS])
    champion = run_playoffs(qualifiers)
    elapsed = time.time() - start
    print(f"Simulation completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")
