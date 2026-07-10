#!/usr/bin/env python3
"""The International 2026 — Full Tournament Simulation with Drafts.

Format:
  - Group Stage (August 13 - 16): 16-team Swiss (All Bo3).
    - Top 3 advance directly to Playoffs.
    - 4th-13th advance to the Elimination Round.
    - Bottom 3 are eliminated.
  - Elimination Round: 10 teams play Bo3s. 5 advance to Playoffs.
  - Main Event (August 20 - 23): 8-Team Double Elimination.
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
log_queue = queue.Queue(-1)
queue_handler = logging.handlers.QueueHandler(log_queue)

logger = logging.getLogger("TI2026")
logger.setLevel(logging.INFO)
logger.addHandler(queue_handler)

console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s.%(msecs)03d | %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(formatter)
listener = logging.handlers.QueueListener(log_queue, console_handler)
listener.start()

# ============================================================
# 2. CONSTANTS & DATABASE
# ============================================================
API = "http://localhost:8080"
PATCH = 60
CONFIDENCE_RANGE = 0.125
MAX_WORKERS = 6

DRAFT_PATTERN = [
    (0, False), (0, False), (1, False), (1, False),
    (0, False), (1, False), (1, False),
    (0, True), (1, True),
    (0, False), (0, False), (1, False),
    (1, True), (0, True), (0, True), (1, True), (1, True), (0, True),
    (0, False), (1, False), (0, False), (1, False),
    (0, True), (1, True),
]

TEAMS = [
    "Aurora Gaming", "BoomBoys", "Team Falcons", "Team Liquid",
    "1w Team", "Xtreme Gaming", "Team Yandex", "Team Spirit",
    "TEAM VISION", "Nigma Galaxy", "HULIGANI", "Team Resilience",
    "Vici Gaming", "OG", "GamerLegion", "LGD Gaming"
]

TEAMS_DB = {
    "Aurora Gaming": 9467224, "BoomBoys": 8255888, "Team Falcons": 9247354,
    "Team Liquid": 2163, "1w Team": 10182357, "Xtreme Gaming": 8261500,
    "Team Yandex": 9823272, "Team Spirit": 7119388, "TEAM VISION": 9572001,
    "Nigma Galaxy": 7554697, "HULIGANI": 10149530, "Team Resilience": 5017210,
    "Vici Gaming": 726228, "OG": 2586976, "GamerLegion": 9964962, "LGD Gaming": 10150538,
}

# 7.41d Client-Matched Hero Grid (125 Heroes)
HERO_NAMES = {
    1: "Anti-Mage", 2: "Axe", 3: "Bane", 4: "Bloodseeker", 5: "Crystal Maiden",
    6: "Drow Ranger", 7: "Earthshaker", 8: "Juggernaut", 9: "Mirana", 10: "Morphling",
    11: "Shadow Fiend", 12: "Phantom Lancer", 13: "Puck", 14: "Pudge", 15: "Razor",
    16: "Sand King", 17: "Storm Spirit", 18: "Sven", 19: "Tiny", 20: "Vengeful Spirit",
    21: "Windranger", 22: "Zeus", 23: "Kunkka", 25: "Lina", 26: "Lion", 27: "Shadow Shaman",
    28: "Slardar", 29: "Tidehunter", 30: "Witch Doctor", 31: "Lich", 32: "Riki",
    33: "Enigma", 34: "Tinker", 35: "Sniper", 36: "Necrophos", 37: "Warlock",
    38: "Beastmaster", 39: "Queen of Pain", 40: "Venomancer", 41: "Faceless Void",
    42: "Wraith King", 43: "Death Prophet", 44: "Phantom Assassin", 45: "Pugna",
    46: "Templar Assassin", 47: "Viper", 48: "Luna", 49: "Dragon Knight", 50: "Dazzle",
    51: "Clockwerk", 52: "Leshrac", 53: "Nature's Prophet", 54: "Lifestealer",
    55: "Dark Seer", 56: "Clinkz", 57: "Omniknight", 58: "Enchantress", 59: "Huskar",
    60: "Night Stalker", 61: "Broodmother", 62: "Bounty Hunter", 63: "Weaver",
    64: "Jakiro", 65: "Batrider", 66: "Chen", 67: "Spectre", 68: "Ancient Apparition",
    69: "Doom", 70: "Ursa", 71: "Spirit Breaker", 72: "Gyrocopter", 73: "Alchemist",
    74: "Invoker", 75: "Silencer", 76: "Outworld Destroyer", 77: "Lycan", 78: "Brewmaster",
    79: "Shadow Demon", 80: "Lone Druid", 81: "Chaos Knight", 82: "Meepo",
    83: "Treant Protector", 84: "Ogre Magi", 85: "Undying", 86: "Rubick",
    87: "Disruptor", 88: "Nyx Assassin", 89: "Naga Siren", 90: "Keeper of the Light",
    91: "Io", 92: "Visage", 93: "Slark", 94: "Medusa", 95: "Troll Warlord",
    96: "Centaur Warrunner", 97: "Magnus", 98: "Timbersaw", 99: "Bristleback",
    100: "Tusk", 101: "Skywrath Mage", 102: "Abaddon", 103: "Elder Titan",
    104: "Legion Commander", 105: "Techies", 106: "Ember Spirit", 107: "Earth Spirit",
    108: "Underlord", 109: "Terrorblade", 110: "Phoenix", 111: "Oracle",
    112: "Winter Wyvern", 113: "Arc Warden", 114: "Monkey King", 119: "Dark Willow",
    120: "Pangolier", 121: "Grimstroke", 123: "Hoodwink", 126: "Void Spirit",
    128: "Snapfire", 129: "Mars", 131: "Ringmaster", 135: "Dawnbreaker", 136: "Marci",
    137: "Primal Beast", 138: "Muerta", 145: "Kez", 155: "Largo"
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
                "num_recommendations": 5,
                "run_mcts": False,
            })
            recs = result.get("recommendations", [])
        except Exception:
            recs = []

        valid_recs = [r for r in recs if r["hero_id"] not in taken]

        if valid_recs:
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
        raw_prob = 0.5

    noise = random.uniform(-CONFIDENCE_RANGE, CONFIDENCE_RANGE)
    adjusted_prob = max(0.05, min(0.95, raw_prob + noise))

    winner = team_a if (first_pick_team == 0 and adjusted_prob >= 0.5) or (first_pick_team == 1 and adjusted_prob < 0.5) else team_b
    loser = team_b if winner == team_a else team_a
    duration = time.time() - t_start

    return {
        "winner": winner, "loser": loser,
        "raw_prob": raw_prob, "adjusted_prob": adjusted_prob,
        "rad_picks": rad_picks, "dire_picks": dire_picks,
        "rad_bans": rad_bans, "dire_bans": dire_bans,
        "radiant_team": rad_team_name, "dire_team": dire_team_name,
        "compute_time": duration
    }

def print_draft(res, label):
    rad_str = format_draft_string(res['radiant_team'], res['rad_picks'])
    dire_str = format_draft_string(res['dire_team'], res['dire_picks'])
    logger.info(f"[{label}]   Radiant: {rad_str}")
    logger.info(f"[{label}]   Dire   : {dire_str}")

def sim_series(team_a, team_b, label="", best_of=3):
    logger.info(f"[{label}] MATCH START: {team_a} vs {team_b} (Bo{best_of})")
    wins_a, wins_b = 0, 0
    games = []
    req_wins = (best_of // 2) + 1

    for i in range(best_of):
        if wins_a == req_wins or wins_b == req_wins:
            break
        fp = i % 2
        rad, dire = (team_a, team_b) if fp == 0 else (team_b, team_a)

        res = simulate_match(rad, dire, first_pick_team=fp)
        games.append(res)

        logger.info(f"[{label}] Game {i+1} | WIN: {res['winner']} | Prob: {res['adjusted_prob']*100:.1f}%")
        print_draft(res, label)

        if res["winner"] == team_a: wins_a += 1
        else: wins_b += 1

    winner = team_a if wins_a > wins_b else team_b
    loser = team_b if winner == team_a else team_a
    logger.info(f"[{label}] SERIES END: {team_a} {wins_a}-{wins_b} {team_b} -> Advance: {winner}")
    return winner, loser, wins_a, wins_b, games

def run_concurrent_matches(matchups, executor, best_of=3, label_prefix=""):
    futures = []
    for i, (t1, t2) in enumerate(matchups):
        lbl = f"{label_prefix} {i+1}"
        futures.append(executor.submit(sim_series, t1, t2, lbl, best_of))
    return [fut.result() for fut in futures]


# ============================================================
# STAGE 1: SWISS SYSTEM
# ============================================================
def run_swiss(teams, executor):
    records = {t: {"wins": 0, "losses": 0, "game_wins": 0, "game_losses": 0, "qualified": False, "eliminated": False} for t in teams}
    logger.info(f"\n{'='*72}")
    logger.info(f"  THE INTERNATIONAL 2026 — GROUP STAGE (Aug 13-16)")
    logger.info(f"  Swiss System: All matches Bo3.")
    logger.info(f"{'='*72}\n")

    for rnd in range(1, 6):
        active_teams = [t for t in teams if not records[t]["qualified"] and not records[t]["eliminated"]]

        # Sort by wins, then inverse losses, to naturally pair equivalent records
        active_teams.sort(key=lambda t: (-records[t]["wins"], records[t]["losses"], random.random()))

        if not active_teams:
            break

        matchups = []
        while len(active_teams) >= 2:
            matchups.append((active_teams.pop(0), active_teams.pop(0)))

        logger.info(f"\n--- Round {rnd} ({len(matchups)} Bo3 matches) ---")
        results = run_concurrent_matches(matchups, executor, best_of=3, label_prefix=f"Swiss R{rnd}")

        for winner, loser, wins_a, wins_b, _ in results:
            records[winner]["wins"] += 1
            records[loser]["losses"] += 1
            records[winner]["game_wins"] += max(wins_a, wins_b)
            records[winner]["game_losses"] += min(wins_a, wins_b)
            records[loser]["game_wins"] += min(wins_a, wins_b)
            records[loser]["game_losses"] += max(wins_a, wins_b)

            if records[winner]["wins"] >= 3: records[winner]["qualified"] = True
            if records[loser]["losses"] >= 3: records[loser]["eliminated"] = True

    # Rank the teams
    def swiss_sort_key(t):
        r = records[t]
        gw, gl = r["game_wins"], r["game_losses"]
        g_rate = gw / (gw + gl) if (gw + gl) > 0 else 0
        return (r["wins"], -r["losses"], g_rate, random.random())

    ranked_teams = sorted(teams, key=swiss_sort_key, reverse=True)

    logger.info(f"\n{'='*72}\n  SWISS FINAL STANDINGS\n{'='*72}")
    for rank, t in enumerate(ranked_teams, 1):
        r = records[t]
        status = "✅ PLAYOFFS" if rank <= 3 else ("🟨 ELIMINATION ROUND" if rank <= 13 else "❌ ELIMINATED")
        logger.info(f"  {rank:>2}. {t:>18s}  {r['wins']}W-{r['losses']}L (GW: {r['game_wins']}-{r['game_losses']})  {status}")

    return {
        "playoffs_direct": ranked_teams[:3],
        "elimination": ranked_teams[3:13],
        "full_records": records
    }


# ============================================================
# STAGE 2: ELIMINATION ROUND
# ============================================================
def run_elimination_round(elim_teams, records, executor):
    logger.info(f"\n{'='*72}")
    logger.info(f"  THE INTERNATIONAL 2026 — ELIMINATION ROUND")
    logger.info(f"  10 Teams (4th-13th). 5 advance to Playoffs.")
    logger.info(f"{'='*72}\n")

    # Matchups: 4th vs 13th, 5th vs 12th, etc.
    matchups = [
        (elim_teams[0], elim_teams[9]),
        (elim_teams[1], elim_teams[8]),
        (elim_teams[2], elim_teams[7]),
        (elim_teams[3], elim_teams[6]),
        (elim_teams[4], elim_teams[5]),
    ]

    results = run_concurrent_matches(matchups, executor, best_of=3, label_prefix="Elim Round")

    elim_advancers = []
    for winner, loser, wins_a, wins_b, _ in results:
        elim_advancers.append(winner)
        # Update records purely for playoff seeding purposes
        records[winner]["game_wins"] += max(wins_a, wins_b)
        records[winner]["game_losses"] += min(wins_a, wins_b)

    return elim_advancers


# ============================================================
# STAGE 3: MAIN EVENT (PLAYOFFS)
# ============================================================
def run_playoffs(playoffs_direct, elim_advancers, records, executor):
    logger.info(f"\n{'='*72}")
    logger.info(f"  THE INTERNATIONAL 2026 — MAIN EVENT (Aug 20-23)")
    logger.info(f"  8-Team Double Elimination.")
    logger.info(f"{'='*72}\n")

    # Seed the 8 teams by their overall Swiss + Elim stats
    def seed_sort_key(t):
        r = records[t]
        gw, gl = r["game_wins"], r["game_losses"]
        g_rate = gw / (gw + gl) if (gw + gl) > 0 else 0
        return (r["wins"], -r["losses"], g_rate, random.random())

    all_playoff_teams = playoffs_direct + elim_advancers
    seeded = sorted(all_playoff_teams, key=seed_sort_key, reverse=True)

    logger.info("--- PLAYOFF SEEDS ---")
    for i, t in enumerate(seeded, 1):
        logger.info(f"  Seed {i}: {t}")

    # UB QF (1v8, 4v5, 2v7, 3v6)
    logger.info("\n═══ UPPER BRACKET QUARTERFINALS ═══")
    qf_matchups = [(seeded[0], seeded[7]), (seeded[3], seeded[4]),
                   (seeded[1], seeded[6]), (seeded[2], seeded[5])]
    qf_res = run_concurrent_matches(qf_matchups, executor, best_of=3, label_prefix="UB QF")
    ub_sf_teams = [r[0] for r in qf_res]
    lb_r1_teams = [r[1] for r in qf_res]

    # LB R1
    logger.info("\n═══ LOWER BRACKET ROUND 1 ═══")
    lb1_matchups = [(lb_r1_teams[0], lb_r1_teams[1]), (lb_r1_teams[2], lb_r1_teams[3])]
    lb1_res = run_concurrent_matches(lb1_matchups, executor, best_of=3, label_prefix="LB R1")
    lb_r2_wait = [r[0] for r in lb1_res]

    # UB SF
    logger.info("\n═══ UPPER BRACKET SEMIFINALS ═══")
    sf_matchups = [(ub_sf_teams[0], ub_sf_teams[1]), (ub_sf_teams[2], ub_sf_teams[3])]
    sf_res = run_concurrent_matches(sf_matchups, executor, best_of=3, label_prefix="UB SF")
    ub_fin_teams = [r[0] for r in sf_res]
    lb_r2_drop = [r[1] for r in sf_res]

    # LB R2
    logger.info("\n═══ LOWER BRACKET ROUND 2 ═══")
    lb2_matchups = [(lb_r2_wait[0], lb_r2_drop[0]), (lb_r2_wait[1], lb_r2_drop[1])]
    lb2_res = run_concurrent_matches(lb2_matchups, executor, best_of=3, label_prefix="LB R2")
    lb_r3_teams = [r[0] for r in lb2_res]

    # LB SF (R3)
    logger.info("\n═══ LOWER BRACKET SEMIFINAL ═══")
    lb3_res = run_concurrent_matches([(lb_r3_teams[0], lb_r3_teams[1])], executor, best_of=3, label_prefix="LB SF")
    lb_fin_wait = lb3_res[0][0]

    # UB Final
    logger.info("\n═══ UPPER BRACKET FINAL ═══")
    ubf_res = run_concurrent_matches([(ub_fin_teams[0], ub_fin_teams[1])], executor, best_of=3, label_prefix="UB Fin")
    grand_finalist_ub = ubf_res[0][0]
    lb_fin_drop = ubf_res[0][1]

    # LB Final
    logger.info("\n═══ LOWER BRACKET FINAL ═══")
    lbf_res = run_concurrent_matches([(lb_fin_wait, lb_fin_drop)], executor, best_of=3, label_prefix="LB Fin")
    grand_finalist_lb = lbf_res[0][0]

    # Grand Final
    logger.info(f"\n{'*'*50}")
    logger.info(f"  GRAND FINAL (Bo5): {grand_finalist_ub} vs {grand_finalist_lb}")
    logger.info(f"{'*'*50}")

    gf_res = run_concurrent_matches([(grand_finalist_ub, grand_finalist_lb)], executor, best_of=5, label_prefix="GF")
    champion = gf_res[0][0]

    logger.info(f"\n{'='*72}")
    logger.info(f"  🏆 THE INTERNATIONAL 2026 CHAMPION: {champion.upper()} 🏆")
    logger.info(f"{'*'*72}\n")


if __name__ == "__main__":
    try:
        urllib.request.urlopen(f"{API}/health", timeout=5)
    except:
        logger.error("API is offline! Start it via `uvicorn api.app:app` first.")
        listener.stop()
        sys.exit(1)

    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        swiss_results = run_swiss(TEAMS, executor)
        elim_advancers = run_elimination_round(swiss_results["elimination"], swiss_results["full_records"], executor)
        run_playoffs(swiss_results["playoffs_direct"], elim_advancers, swiss_results["full_records"], executor)

    t_elapsed = time.time() - t_start
    logger.info(f"TI 2026 AI Simulation completed in {t_elapsed:.0f}s ({t_elapsed/60:.1f}min)")

    listener.stop()
