#!/usr/bin/env python3
"""Esports World Cup 2026 — Ultimate Multi-Threaded Simulation.

- True Async Logging (QueueListener) to prevent thread I/O blocking.
- Cleaned Hero Database (no duplicates).
- MCTS dynamically enabled for Picks (enforces Role & GPM budgets).
- Greedy enabled for Bans (maintains API speed).
"""

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
# Create a thread-safe queue for logs
log_queue = queue.Queue(-1)
queue_handler = logging.handlers.QueueHandler(log_queue)

# Configure the root logger
logger = logging.getLogger("EWC2026")
logger.setLevel(logging.INFO)
logger.addHandler(queue_handler)

# Configure the listener (runs in a background thread to print logs)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s.%(msecs)03d | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(formatter)
listener = logging.handlers.QueueListener(log_queue, console_handler)
listener.start()

# ============================================================
# 2. CONSTANTS & DATABASE
# ============================================================
API = "http://localhost:8080"
PATCH = 60
CONFIDENCE_RANGE = 0.125
MAX_WORKERS = 6  # Must be ≤ API_POOL_MAX (8) to avoid connection exhaustion

DRAFT_PATTERN = [
    (0, False), (0, False), (1, False), (1, False),  # Bans
    (0, False), (1, False), (1, False),
    (0, True), (1, True),                            # Picks
    (0, False), (0, False), (1, False),              # Bans
    (1, True), (0, True), (0, True), (1, True), (1, True), (0, True), # Picks
    (0, False), (1, False), (0, False), (1, False),  # Bans
    (0, True), (1, True),                            # Picks
]

TEAMS_DB = {
    "Team Spirit": 7119388, "1w": 10182357, "PVISION": 9824702,
    "Aurora Gaming": 9467224, "Xtreme Gaming": 8261500, "Team Yandex": 9823272,
    "Team Falcons": 9247354, "Team Liquid": 2163, "BB Team": 9131584,
    "MOUZ": 9338413, "PTime": 10020555, "OG": 2586976,
    "Virtus.pro": 1883502, "LGD Gaming": 10150538, "Rune Eaters": 9758040,
    "Level UP": 8359797, "Poor Rangers": 55, "L1 TEAM": 9303383,
    "IC x Insanity": 8168562, "Vici Gaming": 726228, "REKONIX": 9828897,
    "GamerLegion": 9964962, "Team Nemesis": 9691969, "Nigma Galaxy": 7554697
}

REAL_GROUPS = {
    "A": ["BB Team", "Team Falcons", "Xtreme Gaming", "GamerLegion", "Rune Eaters", "Poor Rangers"],
    "B": ["Aurora Gaming", "Nigma Galaxy", "Team Liquid", "PTime", "L1 TEAM", "Level UP"],
    "C": ["PVISION", "MOUZ", "Team Spirit", "Vici Gaming", "REKONIX", "Team Nemesis"],
    "D": ["1w", "LGD Gaming", "Team Yandex", "OG", "Virtus.pro", "IC x Insanity"]
}

# Cleaned Hero Dictionary!
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
    135: "Dawnbreaker", 136: "Marci", 137: "Primal Beast", 138: "Muerta", 145: "Kez", 155: "Ringmaster"
}
HERO_IDS = list(HERO_NAMES.keys())


# ============================================================
# 3. CORE SIMULATION ENGINE
# ============================================================
def api_post(endpoint, body, retries=5):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{API}{endpoint}", data=data, headers={"Content-Type": "application/json"})

    for attempt in range(retries):
        try:
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

def simulate_draft(team_a, team_b, first_pick_team):
    draft_slots = []
    taken = set()
    rad_picks, dire_picks = [], []
    rad_bans, dire_bans = [], []

    rad_team_name = team_a if first_pick_team == 0 else team_b
    dire_team_name = team_b if first_pick_team == 0 else team_a

    for step_idx, (team, is_pick) in enumerate(DRAFT_PATTERN):
        recommending_team = team if first_pick_team == 0 else 1 - team
        draft_for_api = [{"hero_id": s[0], "is_pick": s[1], "team": s[2], "order": i + 1} for i, s in enumerate(draft_slots)]

        try:
            result = api_post("/predict", {
                "patch_id": PATCH,
                "draft": draft_for_api,
                "for_team": recommending_team,
                "first_pick_team": first_pick_team,
                "radiant_team_id": TEAMS_DB[rad_team_name],
                "dire_team_id": TEAMS_DB[dire_team_name],
                "num_recommendations": 10,
                "run_mcts": is_pick,  # <--- THE FIX: MCTS on for picks to enforce roles, off for bans to save speed!
            })
            recs = result.get("recommendations", [])
        except Exception:
            logger.error("API is unrecoverable. Terminating simulation...")
            os._exit(1)

        valid_recs = [r for r in recs if r["hero_id"] not in taken]

        if valid_recs:
            chosen = random.choice(valid_recs[:3])["hero_id"]
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

def simulate_match(team_a, team_b, first_pick_team=0):
    t_start = time.time()
    draft_slots, rad_picks, dire_picks, rad_bans, dire_bans = simulate_draft(team_a, team_b, first_pick_team)
    rad_team_name = team_a if first_pick_team == 0 else team_b
    dire_team_name = team_b if first_pick_team == 0 else team_a

    try:
        result = api_post("/predict-match", {
            "patch_id": PATCH,
            "radiant_heroes": rad_picks,
            "dire_heroes": dire_picks,
            "radiant_team_id": TEAMS_DB[rad_team_name],
            "dire_team_id": TEAMS_DB[dire_team_name],
        })
        raw_prob = result["radiant_win_probability"]
    except Exception:
        logger.error("API is unrecoverable. Terminating simulation...")
        os._exit(1)

    noise = random.uniform(-CONFIDENCE_RANGE, CONFIDENCE_RANGE)
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

def sim_series(team_a, team_b, num_games=2, label=""):
    logger.info(f"[{label}] MATCH START: {team_a} vs {team_b} (Bo{num_games})")
    wins_a, wins_b = 0, 0
    games = []
    required_wins = (num_games // 2) + 1 if num_games % 2 != 0 else num_games + 1

    for i in range(num_games):
        if wins_a == required_wins or wins_b == required_wins:
            break
        fp = i % 2
        rad, dire = (team_a, team_b) if fp == 0 else (team_b, team_a)

        res = simulate_match(rad, dire, first_pick_team=fp)
        games.append(res)

        # Build nice log formatting
        win_str = f"WIN: {res['winner']}"
        conf = f"{res['adjusted_prob']*100:.1f}% (Delta: {res['noise']*100:+.1f}%)"

        logger.debug(f"[{label}] Game {i+1} | {res['compute_time']:.1f}s compute")
        logger.debug(f"    Radiant: {format_draft_string(res['radiant_team'], res['rad_picks'])}")
        logger.debug(f"    Dire   : {format_draft_string(res['dire_team'], res['dire_picks'])}")
        logger.info(f"[{label}] Game {i+1} | {win_str:<25} | Confidence: {conf}")

        if res["winner"] == team_a: wins_a += 1
        else: wins_b += 1

    winner = team_a if wins_a > wins_b else team_b if wins_b > wins_a else "Draw"
    logger.info(f"[{label}] SERIES END : {team_a} {wins_a}-{wins_b} {team_b} -> Advance: {winner}")
    return winner, wins_a, wins_b, games

# ============================================================
# MULTI-THREADED TOURNAMENT STAGES
# ============================================================

def run_group_stage(groups, executor):
    logger.info(f"{'='*70}")
    logger.info(f" STAGE 1: REAL GROUP STAGE (Bo2 Round Robin)")
    logger.info(f"{'='*70}")

    futures = {g: [] for g in groups}
    for g_name, g_teams in groups.items():
        for i in range(len(g_teams)):
            for j in range(i + 1, len(g_teams)):
                fut = executor.submit(sim_series, g_teams[i], g_teams[j], 2, f"Group {g_name}")
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

def run_survival_stage(standings, executor):
    logger.info(f"\n{'='*70}")
    logger.info(f" STAGE 2: SURVIVAL STAGE (Bo3)")
    logger.info(f"{'='*70}")

    survival_winners = []
    r1_futures = {}

    for g_name, g_teams in standings.items():
        r1_futures[g_name] = executor.submit(sim_series, g_teams[2], g_teams[3], 3, f"LB R1 Grp {g_name}")

    r1_results = {}
    for g_name, fut in r1_futures.items():
        r1_results[g_name] = fut.result()

    r2_futures = {}
    for g_name, g_teams in standings.items():
        w1 = r1_results[g_name][0]
        r2_futures[g_name] = executor.submit(sim_series, g_teams[1], w1, 3, f"LB R2 Grp {g_name}")

    for g_name, g_teams in standings.items():
        w1, w_a, w_b, games = r1_results[g_name]
        w2, w_a2, w_b2, games2 = r2_futures[g_name].result()

        logger.info(f"--- Group {g_name} Survival Path ---")
        logger.info(f"  [Round 1] 3rd ({g_teams[2]}) vs 4th ({g_teams[3]}) -> {w1} advances")
        logger.info(f"  [Round 2] 2nd ({g_teams[1]}) vs R1 Winner ({w1}) -> {w2} QUALIFIES TO PLAYOFFS!")
        survival_winners.append({"group": g_name, "team": w2})

    return survival_winners

def run_playoffs(standings, survival_winners, executor):
    logger.info(f"\n{'='*70}")
    logger.info(f" STAGE 3: PLAYOFFS (Single Elimination)")
    logger.info(f"{'='*70}")

    g1 = {g: standings[g][0] for g in ["A", "B", "C", "D"]}
    s_win = {sw["group"]: sw["team"] for sw in survival_winners}

    # Quarterfinals
    qf1_fut = executor.submit(sim_series, g1["A"], s_win["B"], 3, "Quarterfinal 1")
    qf2_fut = executor.submit(sim_series, g1["C"], s_win["D"], 3, "Quarterfinal 2")
    qf3_fut = executor.submit(sim_series, g1["B"], s_win["A"], 3, "Quarterfinal 3")
    qf4_fut = executor.submit(sim_series, g1["D"], s_win["C"], 3, "Quarterfinal 4")

    qf1 = qf1_fut.result()[0]
    qf2 = qf2_fut.result()[0]
    qf3 = qf3_fut.result()[0]
    qf4 = qf4_fut.result()[0]

    # Semifinals
    sf1_fut = executor.submit(sim_series, qf1, qf2, 3, "Semifinal 1")
    sf2_fut = executor.submit(sim_series, qf3, qf4, 3, "Semifinal 2")

    sf1 = sf1_fut.result()[0]
    sf2 = sf2_fut.result()[0]

    # Grand Final
    logger.info(f"\n{'*'*40}")
    logger.info(f"  GRAND FINAL (Bo5): {sf1} vs {sf2}")
    logger.info(f"{'*'*40}")

    gf_fut = executor.submit(sim_series, sf1, sf2, 5, "Grand Final")
    champion, ga, gb, games = gf_fut.result()

    # Print out the final majestic drafts of the Grand Final
    for i, res in enumerate(games, 1):
        logger.info(f"\n[CHAMPIONSHIP GAME {i}]")
        logger.info(f"    Radiant: {format_draft_string(res['radiant_team'], res['rad_picks'])}")
        logger.info(f"    Dire   : {format_draft_string(res['dire_team'], res['dire_picks'])}")
        logger.info(f"    Winner : {res['winner']} ({res['adjusted_prob']*100:.1f}%)")

    logger.info(f"\n{'*'*70}")
    logger.info(f"  🏆 ESPORTS WORLD CUP 2026 CHAMPION: {champion.upper()} 🏆")
    logger.info(f"{'*'*70}\n")

if __name__ == "__main__":
    # Wait for API to be ready
    try:
        urllib.request.urlopen(f"{API}/health", timeout=5)
    except:
        logger.error("API is offline! Start it via `uvicorn api.main:app` first.")
        listener.stop()
        sys.exit(1)

    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        standings = run_group_stage(REAL_GROUPS, executor)
        survival_advancers = run_survival_stage(standings, executor)
        run_playoffs(standings, survival_advancers, executor)

    t_elapsed = time.time() - t_start
    logger.info(f"EWC 2026 AI Simulation completed in {t_elapsed:.0f}s ({t_elapsed/60:.1f}min)")

    # Gracefully shutdown async logger
    listener.stop()
