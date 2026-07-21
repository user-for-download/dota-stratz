#!/usr/bin/env python3
"""Esports World Cup 2026 — Ultimate Multi-Threaded Simulation.

- True Async Logging (QueueListener) to prevent thread I/O blocking.
- Cleaned Hero Database (no duplicates).
- MCTS dynamically enabled for Picks (enforces Role & GPM budgets).
- Greedy enabled for Bans (maintains API speed).
"""

import argparse
import json
import random
import time
import urllib.request
import concurrent.futures
import logging
import logging.handlers
import queue
import sys
import os

# ============================================================
# 1. ASYNC LOGGING SETUP
# ============================================================
log_queue = queue.Queue(-1)
queue_handler = logging.handlers.QueueHandler(log_queue)

logger = logging.getLogger("EWC2026")
logger.setLevel(logging.INFO)
logger.addHandler(queue_handler)

console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(formatter)
listener = logging.handlers.QueueListener(log_queue, console_handler)
listener.start()

# ============================================================
# 2. CONSTANTS & DATABASE
# ============================================================
CONFIDENCE_RANGE = 0.125

# Patch 60 draft pattern — 24 actions, normalized so team 0 = first-pick team
# Source: services/api/api/draft_state.py DRAFT_PATTERNS[60]
# Raw: "B1 B1 B0 B0 B1 B0 B0 P1 P0 B1 B1 B0 P0 P1 P1 P0 P0 P1 B1 B0 B1 B0 P1 P0"
# After parse_draft_pattern normalization (first_team=1 → flip all teams):
DRAFT_PATTERN = [
    (0, False), (0, False), (1, False), (1, False),   # Phase 1: 4 bans
    (0, False), (1, False), (1, False),                # Phase 1: 3 bans
    (0, True), (1, True),                              # Phase 1: 2 picks
    (0, False), (0, False), (1, False),                # Phase 2: 3 bans
    (1, True), (0, True), (0, True), (1, True),        # Phase 2: 4 picks
    (1, True), (0, True),                              # Phase 2: 2 picks
    (0, False), (1, False), (0, False), (1, False),    # Phase 3: 4 bans
    (0, True), (1, True),                              # Phase 3: 2 picks
]
assert len(DRAFT_PATTERN) == 24, f"Draft pattern must have 24 actions, got {len(DRAFT_PATTERN)}"

TEAMS_DB = {
    "Team Spirit": 7119388, "1w": 10182357, "PVISION": 9824702,
    "Aurora Gaming": 9467224, "Xtreme Gaming": 8261500, "Team Yandex": 9823272,
    "Team Falcons": 9247354, "Team Liquid": 2163, "BoomEsports": 8255888,
    "MOUZ": 9338413, "PTime": 10182309, "OG": 2586976,
    "Virtus.pro": 9895392, "LGD Gaming": 10150538, "Rune Eaters": 9895247,
    "Level UP": 9256405, "Poor Rangers": 55, "L1 TEAM": 10182299,
    "Inner Circle": 10019843, "Vici Gaming": 726228, "REKONIX": 9828897,
    "GamerLegion": 9964962, "Team Nemesis": 9691969, "Nigma Galaxy": 10136357
}

REAL_GROUPS = {
    "A": ["Team Yandex", "1w", "LGD Gaming", "Virtus.pro", "OG", "Inner Circle"],
    "B": ["PVISION", "Team Spirit", "Vici Gaming", "MOUZ", "REKONIX", "Team Nemesis"],
    "C": ["Nigma Galaxy", "Aurora Gaming", "Team Liquid", "PTime", "Level UP", "L1 TEAM"],
    "D": ["Team Falcons", "BoomEsports", "Rune Eaters", "Xtreme Gaming", "GamerLegion", "Poor Rangers"],
}

HERO_NAMES = {
    1: "Anti-Mage", 2: "Axe", 3: "Bane", 4: "Bloodseeker", 5: "Crystal Maiden",
    6: "Drow Ranger", 7: "Earthshaker", 8: "Juggernaut", 9: "Mirana", 10: "Morphling",
    11: "Shadow Fiend", 12: "Phantom Lancer", 13: "Puck", 14: "Pudge", 15: "Razor",
    16: "Sand King", 17: "Storm Spirit", 18: "Sven", 19: "Tiny", 20: "Vengeful Spirit",
    21: "Windranger", 22: "Zeus", 23: "Kunkka", 25: "Lina", 26: "Lion",
    27: "Shadow Shaman", 28: "Slardar", 29: "Tidehunter", 30: "Witch Doctor", 31: "Lich",
    32: "Riki", 33: "Enigma", 34: "Tinker", 35: "Sniper", 36: "Necrophos",
    37: "Warlock", 38: "Beastmaster", 39: "Queen of Pain", 40: "Venomancer", 41: "Faceless Void",
    42: "Wraith King", 43: "Death Prophet", 44: "Phantom Assassin", 45: "Pugna", 46: "Templar Assassin",
    47: "Viper", 48: "Luna", 49: "Dragon Knight", 50: "Dazzle", 51: "Clockwerk",
    52: "Leshrac", 53: "Nature's Prophet", 54: "Lifestealer", 55: "Dark Seer", 56: "Clinkz",
    57: "Omniknight", 58: "Enchantress", 59: "Huskar", 60: "Night Stalker", 61: "Broodmother",
    62: "Bounty Hunter", 63: "Weaver", 64: "Jakiro", 65: "Batrider", 66: "Chen",
    67: "Spectre", 68: "Ancient Apparition", 69: "Doom", 70: "Ursa", 71: "Spirit Breaker",
    72: "Gyrocopter", 73: "Alchemist", 74: "Invoker", 75: "Silencer", 76: "Outworld Destroyer",
    77: "Lycan", 78: "Brewmaster", 79: "Shadow Demon", 80: "Lone Druid", 81: "Chaos Knight",
    82: "Meepo", 83: "Treant Protector", 84: "Ogre Magi", 85: "Undying", 86: "Rubick",
    87: "Disruptor", 88: "Nyx Assassin", 89: "Naga Siren", 90: "Keeper of the Light", 91: "Io",
    92: "Visage", 93: "Slark", 94: "Medusa", 95: "Troll Warlord", 96: "Centaur Warrunner",
    97: "Magnus", 98: "Timbersaw", 99: "Bristleback", 100: "Tusk", 101: "Skywrath Mage",
    102: "Abaddon", 103: "Elder Titan", 104: "Legion Commander", 105: "Techies", 106: "Ember Spirit",
    107: "Earth Spirit", 108: "Underlord", 109: "Terrorblade", 110: "Phoenix", 111: "Oracle",
    112: "Winter Wyvern", 113: "Arc Warden", 114: "Monkey King", 119: "Dark Willow", 120: "Pangolier",
    121: "Grimstroke", 123: "Hoodwink", 126: "Void Spirit", 128: "Snapfire", 129: "Mars",
    131: "Ringmaster", 135: "Dawnbreaker", 136: "Marci", 137: "Primal Beast",
    138: "Muerta", 145: "Kez", 155: "Largo"
}
HERO_IDS = list(HERO_NAMES.keys())

# ============================================================
# ROSTER FETCHING (Real player account_ids from OpenDota)
# ============================================================
def fetch_team_rosters():
    """Fetch real EWC team rosters from OpenDota API."""
    rosters = {}
    for team_name, team_id in TEAMS_DB.items():
        try:
            url = f"https://api.opendota.com/api/teams/{team_id}/players"
            req = urllib.request.Request(url, headers={"User-Agent": "EWC-Sim/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                players = json.loads(resp.read())
            active = sorted(players, key=lambda p: p.get("games_played", 0), reverse=True)[:5]
            rosters[team_name] = [p["account_id"] for p in active if p.get("account_id")]
        except Exception:
            rosters[team_name] = []
    return rosters

# ============================================================
# 3. CORE SIMULATION ENGINE
# ============================================================
def api_post(endpoint, body, api_base, retries=5):
    for attempt in range(retries):
        try:
            data = json.dumps(body).encode()
            req = urllib.request.Request(
                f"{api_base}{endpoint}",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except Exception as e:
            logger.warning(f"[API Timeout] {endpoint} failed (Attempt {attempt+1}/{retries}): {e}")
            if attempt == retries - 1:
                logger.error(f"[API ERROR] Max retries reached for {endpoint}.")
                raise e
            time.sleep(1.0)

def hero_name(hid):
    return HERO_NAMES.get(hid, f"Hero {hid}")

def format_draft_string(team_name, picks):
    return f"{team_name:<16} | " + ", ".join(hero_name(h) for h in picks)

def simulate_draft(team_a, team_b, first_pick_team, api_base, patch):
    draft_slots = []
    taken = set()
    rad_picks, dire_picks = [], []
    rad_bans, dire_bans = [], []

    rad_team_name = team_a if first_pick_team == 0 else team_b
    dire_team_name = team_b if first_pick_team == 0 else team_a

    for step_idx, (team, is_pick) in enumerate(DRAFT_PATTERN):
        recommending_team = team if first_pick_team == 0 else 1 - team
        draft_for_api = [{"hero_id": s[0], "is_pick": s[1], "team": s[2], "order": i + 1} for i, s in enumerate(draft_slots)]

        # Determine which player is picking (pick order -> role mapping)
        account_id = None
        if is_pick:
            my_picks = len(rad_picks) if recommending_team == 0 else len(dire_picks)
            if my_picks < 5:
                team_name = rad_team_name if recommending_team == 0 else dire_team_name
                roster = ROSTERS.get(team_name, [])
                if my_picks < len(roster):
                    account_id = roster[my_picks]  # Pick 0->Pos5, Pick 4->Pos1

        try:
            result = api_post("/predict", {
                "patch_id": patch,
                "draft": draft_for_api,
                "for_team": recommending_team,
                "first_pick_team": first_pick_team,
                "radiant_team_id": TEAMS_DB[rad_team_name],
                "dire_team_id": TEAMS_DB[dire_team_name],
                "account_id": account_id,
                "num_recommendations": 5,
                "run_mcts": is_pick,
            }, api_base)
            recs = result.get("recommendations", [])
        except Exception:
            recs = []

        valid_recs = [r for r in recs if r["hero_id"] not in taken]

        if valid_recs:
            if is_pick:
                chosen = random.choice(valid_recs[:2])["hero_id"]
            else:
                chosen = valid_recs[0]["hero_id"]
        else:
            available = [h for h in HERO_IDS if h not in taken]
            chosen = random.choice(available) if available else 1

        taken.add(chosen)
        draft_slots.append((chosen, is_pick, recommending_team))

        if is_pick:
            if recommending_team == 0: rad_picks.append(chosen)
            else: dire_picks.append(chosen)
        else:
            if recommending_team == 0: rad_bans.append(chosen)
            else: dire_bans.append(chosen)

    return draft_slots, rad_picks, dire_picks, rad_bans, dire_bans

def simulate_match(team_a, team_b, first_pick_team, api_base, patch, confidence_range):
    t_start = time.time()
    draft_slots, rad_picks, dire_picks, rad_bans, dire_bans = simulate_draft(
        team_a, team_b, first_pick_team, api_base, patch
    )
    rad_team_name = team_a if first_pick_team == 0 else team_b
    dire_team_name = team_b if first_pick_team == 0 else team_a

    try:
        result = api_post("/predict-match", {
            "patch_id": patch,
            "radiant_heroes": rad_picks,
            "dire_heroes": dire_picks,
            "radiant_team_id": TEAMS_DB[rad_team_name],
            "dire_team_id": TEAMS_DB[dire_team_name],
        }, api_base)
        raw_prob = result["radiant_win_probability"]
    except Exception:
        logger.error("API is unrecoverable. Terminating simulation...")
        sys.exit(1)

    noise = random.uniform(-confidence_range, confidence_range)
    adjusted_prob = max(0.05, min(0.95, raw_prob + noise))

    if first_pick_team == 0:
        winner = team_a if adjusted_prob >= 0.5 else team_b
    else:
        winner = team_b if adjusted_prob >= 0.5 else team_a

    loser = team_b if winner == team_a else team_a
    duration = time.time() - t_start

    return {
        "winner": winner, "loser": loser,
        "raw_prob": raw_prob, "adjusted_prob": adjusted_prob, "noise": noise,
        "rad_picks": rad_picks, "dire_picks": dire_picks,
        "radiant_team": rad_team_name, "dire_team": dire_team_name,
        "compute_time": duration
    }

def sim_series(team_a, team_b, num_games, label, api_base, patch, confidence_range):
    logger.info(f"[{label}] MATCH START: {team_a} vs {team_b} (Bo{num_games})")
    wins_a, wins_b = 0, 0
    games = []
    needed = (num_games // 2) + 1  # Bo2→2, Bo3→2, Bo5→3

    for i in range(num_games):
        if wins_a >= needed or wins_b >= needed:
            break
        fp = i % 2
        rad, dire = (team_a, team_b) if fp == 0 else (team_b, team_a)

        res = simulate_match(rad, dire, fp, api_base, patch, confidence_range)
        games.append(res)

        win_str = f"WIN: {res['winner']}"
        conf = f"{res['adjusted_prob']*100:.1f}% (Delta: {res['noise']*100:+.1f}%)"

        logger.debug(f"[{label}] Game {i+1} | {res['compute_time']:.1f}s compute")
        logger.debug(f"    Radiant: {format_draft_string(res['radiant_team'], res['rad_picks'])}")
        logger.debug(f"    Dire   : {format_draft_string(res['dire_team'], res['dire_picks'])}")
        logger.info(f"[{label}] Game {i+1} | {win_str:<25} | Confidence: {conf}")

        if res["winner"] == team_a: wins_a += 1
        else: wins_b += 1

    # Bo2: allow draw (1-1); Bo3/Bo5: always decisive
    if num_games == 2 and wins_a == 1 and wins_b == 1:
        winner = "Draw"
    else:
        winner = team_a if wins_a > wins_b else team_b if wins_b > wins_a else "Draw"
    logger.info(f"[{label}] SERIES END : {team_a} {wins_a}-{wins_b} {team_b} -> Advance: {winner}")
    return winner, wins_a, wins_b, games

# ============================================================
# MULTI-THREADED TOURNAMENT STAGES
# ============================================================

def run_group_stage(groups, executor, api_base, patch, confidence_range):
    logger.info(f"{'='*70}")
    logger.info(f" STAGE 1: REAL GROUP STAGE (Bo2 Round Robin)")
    logger.info(f"{'='*70}")

    futures = {g: [] for g in groups}
    for g_name, g_teams in groups.items():
        for i in range(len(g_teams)):
            for j in range(i + 1, len(g_teams)):
                fut = executor.submit(
                    sim_series, g_teams[i], g_teams[j], 2,
                    f"Group {g_name}", api_base, patch, confidence_range,
                )
                futures[g_name].append((g_teams[i], g_teams[j], fut))

    logger.info(f"--> Dispatched all 60 group matches to the AI Thread Pool...")

    standings = {}
    for g_name, g_teams in groups.items():
        points = {t: {"points": 0, "wins": 0, "draws": 0, "losses": 0} for t in g_teams}

        for t1, t2, fut in futures[g_name]:
            winner, w1, w2, games = fut.result()
            if w1 == 2:
                points[t1]["points"] += 3; points[t1]["wins"] += 1; points[t2]["losses"] += 1
            elif w2 == 2:
                points[t2]["points"] += 3; points[t2]["wins"] += 1; points[t1]["losses"] += 1
            else:
                points[t1]["points"] += 1; points[t2]["points"] += 1
                points[t1]["draws"] += 1; points[t2]["draws"] += 1

        sorted_standings = sorted(points.items(), key=lambda x: (x[1]["points"], random.random()), reverse=True)
        standings[g_name] = [t[0] for t in sorted_standings]

        logger.info(f"\n[ Group {g_name} Final Standings ]")
        for rank, (team, stats) in enumerate(sorted_standings, 1):
            status = "Playoffs" if rank == 1 else "Survival" if rank <= 4 else "Eliminated"
            logger.info(f"  {rank}. {team:<18} {stats['points']} pts ({stats['wins']}W-{stats['draws']}D-{stats['losses']}L) -> {status}")

    return standings

def run_survival_stage(standings, executor, api_base, patch, confidence_range):
    logger.info(f"\n{'='*70}")
    logger.info(f" STAGE 2: SURVIVAL STAGE (Bo3)")
    logger.info(f"{'='*70}")

    survival_winners = []
    r1_futures = {}

    for g_name, g_teams in standings.items():
        r1_futures[g_name] = executor.submit(
            sim_series, g_teams[2], g_teams[3], 3,
            f"LB R1 Grp {g_name}", api_base, patch, confidence_range,
        )

    r1_results = {}
    for g_name, fut in r1_futures.items():
        r1_results[g_name] = fut.result()

    r2_futures = {}
    for g_name, g_teams in standings.items():
        w1 = r1_results[g_name][0]
        r2_futures[g_name] = executor.submit(
            sim_series, g_teams[1], w1, 3,
            f"LB R2 Grp {g_name}", api_base, patch, confidence_range,
        )

    for g_name, g_teams in standings.items():
        w1, w_a, w_b, games = r1_results[g_name]
        w2, w_a2, w_b2, games2 = r2_futures[g_name].result()

        logger.info(f"--- Group {g_name} Survival Path ---")
        logger.info(f"  [Round 1] 3rd ({g_teams[2]}) vs 4th ({g_teams[3]}) -> {w1} advances")
        logger.info(f"  [Round 2] 2nd ({g_teams[1]}) vs R1 Winner ({w1}) -> {w2} QUALIFIES TO PLAYOFFS!")
        survival_winners.append({"group": g_name, "team": w2})

    return survival_winners

def run_playoffs(standings, survival_winners, executor, api_base, patch, confidence_range):
    logger.info(f"\n{'='*70}")
    logger.info(f" STAGE 3: PLAYOFFS (Single Elimination)")
    logger.info(f"{'='*70}")

    g1 = {g: standings[g][0] for g in ["A", "B", "C", "D"]}
    s_win = {sw["group"]: sw["team"] for sw in survival_winners}

    # Quarterfinals
    qf1_fut = executor.submit(sim_series, g1["A"], s_win["B"], 3, "Quarterfinal 1", api_base, patch, confidence_range)
    qf2_fut = executor.submit(sim_series, g1["C"], s_win["D"], 3, "Quarterfinal 2", api_base, patch, confidence_range)
    qf3_fut = executor.submit(sim_series, g1["B"], s_win["A"], 3, "Quarterfinal 3", api_base, patch, confidence_range)
    qf4_fut = executor.submit(sim_series, g1["D"], s_win["C"], 3, "Quarterfinal 4", api_base, patch, confidence_range)

    qf1 = qf1_fut.result()[0]
    qf2 = qf2_fut.result()[0]
    qf3 = qf3_fut.result()[0]
    qf4 = qf4_fut.result()[0]

    # Semifinals
    sf1_fut = executor.submit(sim_series, qf1, qf2, 3, "Semifinal 1", api_base, patch, confidence_range)
    sf2_fut = executor.submit(sim_series, qf3, qf4, 3, "Semifinal 2", api_base, patch, confidence_range)

    sf1 = sf1_fut.result()[0]
    sf2 = sf2_fut.result()[0]

    # Grand Final
    logger.info(f"\n{'*'*40}")
    logger.info(f"  GRAND FINAL (Bo5): {sf1} vs {sf2}")
    logger.info(f"{'*'*40}")

    gf_fut = executor.submit(sim_series, sf1, sf2, 5, "Grand Final", api_base, patch, confidence_range)
    champion, ga, gb, games = gf_fut.result()

    for i, res in enumerate(games, 1):
        logger.info(f"\n[CHAMPIONSHIP GAME {i}]")
        logger.info(f"    Radiant: {format_draft_string(res['radiant_team'], res['rad_picks'])}")
        logger.info(f"    Dire   : {format_draft_string(res['dire_team'], res['dire_picks'])}")
        logger.info(f"    Winner : {res['winner']} ({res['adjusted_prob']*100:.1f}%)")

    logger.info(f"\n{'*'*70}")
    logger.info(f"  ESPORTS WORLD CUP 2026 CHAMPION: {champion.upper()}")
    logger.info(f"{'*'*70}\n")

    return champion

def parse_args():
    p = argparse.ArgumentParser(description="EWC 2026 AI Draft Simulation")
    p.add_argument("--api", default="http://localhost:8080", help="API base URL (default: http://localhost:8080)")
    p.add_argument("--patch", type=int, default=60, help="Patch ID (default: 60)")
    p.add_argument("--workers", type=int, default=6, help="Thread pool size, max 8 (default: 6)")
    p.add_argument("--confidence", type=float, default=0.125, help="Noise range ± (default: 0.125)")
    p.add_argument("--output", "-o", default=None, help="Save JSON results to file")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging (draft details)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.seed is not None:
        random.seed(args.seed)

    # Wait for API to be ready
    try:
        urllib.request.urlopen(f"{args.api}/health", timeout=5)
    except Exception:
        logger.error("API is offline! Start it via `uvicorn api.main:app` first.")
        listener.stop()
        sys.exit(1)

    logger.info(f"Config: api={args.api} patch={args.patch} workers={args.workers} confidence=±{args.confidence}")

    # Fetch real team rosters from OpenDota
    logger.info("Fetching real team rosters from OpenDota...")
    ROSTERS = fetch_team_rosters()
    teams_with_rosters = sum(1 for v in ROSTERS.values() if v)
    logger.info(f"Loaded rosters for {teams_with_rosters}/{len(TEAMS_DB)} teams")

    t_start = time.time()
    results = {"config": vars(args), "groups": {}, "survival": [], "playoffs": {}}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, 8)) as executor:
        standings = run_group_stage(REAL_GROUPS, executor, args.api, args.patch, args.confidence)
        results["groups"] = {
            g: [{"team": t, "rank": i + 1} for i, t in enumerate(teams)]
            for g, teams in standings.items()
        }

        survival_advancers = run_survival_stage(standings, executor, args.api, args.patch, args.confidence)
        results["survival"] = survival_advancers

        champion = run_playoffs(standings, survival_advancers, executor, args.api, args.patch, args.confidence)
        results["champion"] = champion

    t_elapsed = time.time() - t_start
    results["elapsed_seconds"] = round(t_elapsed, 1)
    logger.info(f"EWC 2026 AI Simulation completed in {t_elapsed:.0f}s ({t_elapsed/60:.1f}min)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    listener.stop()
