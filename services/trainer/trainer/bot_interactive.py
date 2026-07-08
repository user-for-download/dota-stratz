"""Interactive Draft Bot — simulate a full draft with AI recommendations.

Usage:
    python -m trainer.bot_interactive [--patch 60] [--mcts] [--iterations 200]
"""

import sys
import time
import logging
import argparse

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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
    48:"Ursa",49:"Gyrocopter",50:"Alchemist",
    51:"Invoker",52:"Silencer",54:"Lycan",
    56:"Dragon Knight",57:"Jakiro",58:"Batrider",
    59:"Chaos Knight",60:"Rubick",61:"Keeper of the Light",62:"Wisp",
    63:"Broodmother",64:"Queen of Pain",65:"Nyx Assassin",
    67:"Io",68:"Centaur Warrunner",69:"Visage",70:"Oracle",
    71:"Earth Spirit",72:"Terrorblade",73:"Phoenix",74:"Tusk",
    75:"Skywrath Mage",76:"Abaddon",77:"Elder Titan",78:"Legion Commander",
    79:"Ember Spirit",81:"Techies",83:"Underlord",85:"Grimstroke",
    86:"Mars",87:"Hoodwink",88:"Void Spirit",89:"Snapfire",
    91:"Dawnbreaker",92:"Marci",93:"Primal Beast",94:"Pangolier",
    97:"Ringmaster",98:"Kez",99:"Beastmaster",
    102:"Bloodseeker",103:"Crystal Maiden",104:"Shadow Fiend",105:"Nyx Assassin",
    106:"Doom",107:"Chaos Knight",108:"Shadow Demon",109:"Pangolier",
    110:"Ancient Apparition",111:"Outworld Destroyer",112:"Puck",
    113:"Templar Assassin",114:"Naga Siren",119:"Dark Willow",
    120:"Pangolier",121:"Grimstroke",123:"Hoodwink",126:"Void Spirit",
    128:"Primal Beast",129:"Marci",131:"Dawnbreaker",135:"Muerta",
    136:"Ringmaster",137:"Kez",138:"Lion",145:"Enchantress",155:"Lich",
}

CM_FORMAT = [
    ("ban", 0), ("ban", 0), ("ban", 1), ("ban", 1), ("ban", 0), ("ban", 1), ("ban", 1),
    ("pick", 0), ("pick", 1),
    ("ban", 0), ("ban", 0), ("ban", 1),
    ("pick", 1), ("pick", 0), ("pick", 0), ("pick", 1), ("pick", 1), ("pick", 0),
    ("ban", 0), ("ban", 1), ("ban", 0), ("ban", 1),
    ("pick", 0), ("pick", 1),
]


def hero_name(hid):
    return HERO_NAMES.get(hid, f"Hero {hid}")


def run_interactive(patch_id, use_mcts, iterations):
    from trainer.db import make_engine
    from trainer.config import TrainerConfig
    from trainer.inference_cache import InferenceCache
    from trainer.draft_state import DraftStateBuilder

    cfg = TrainerConfig()
    cfg.patch_id = patch_id
    engine = make_engine(cfg)

    logger.info("Loading inference cache for patch %d...", patch_id)
    cache = InferenceCache(engine, patch_id)
    builder = DraftStateBuilder(cache)

    # Load model
    from pathlib import Path
    model_path = Path(cfg.model_dir) / f"draftbert_compiled_{patch_id}.pt"
    if not model_path.exists():
        model_path = Path(cfg.model_dir) / f"draftbert_weights_{patch_id}.pt"

    if model_path.exists():
        logger.info("Loading model from %s", model_path)
        if 'compiled' in str(model_path):
            model = torch.jit.load(str(model_path), map_location="cpu")
            model.eval()
        else:
            from trainer.model_pt import MultiModalDraftBERT
            model = MultiModalDraftBERT(
                vocab_size=cfg.max_hero_id + 5, d_model=cfg.d_model,
                nhead=cfg.nhead, num_layers=cfg.num_layers,
                num_continuous_features=builder.num_features,
                max_seq_len=cfg.max_seq_len, dropout=cfg.dropout,
                transformer_dropout=cfg.transformer_dropout,
                fusion_hidden=cfg.fusion_hidden,
            )
            state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
    else:
        logger.warning("No model found, using random weights")
        from trainer.model_pt import MultiModalDraftBERT
        model = MultiModalDraftBERT(
            vocab_size=cfg.max_hero_id + 5, d_model=cfg.d_model,
            nhead=cfg.nhead, num_layers=cfg.num_layers,
            num_continuous_features=builder.num_features,
            max_seq_len=cfg.max_seq_len, dropout=0.0,
            transformer_dropout=0.0, fusion_hidden=cfg.fusion_hidden,
        )
        model.eval()

    # Draft state
    taken = set()
    draft_history = []
    is_pick_phase = True
    turn = 0
    team = 0  # 0=Radiant, 1=Dire

    print("\n" + "=" * 60)
    print("  DOTA 2 DRAFT BOT — Interactive Mode")
    print(f"  Patch: {patch_id} | Heroes: {len(cache.valid_hero_ids)}")
    print(f"  Mode: {'MCTS' if use_mcts else 'Greedy'} ({iterations} iterations)")
    print("=" * 60)
    print("\nCommands: 'pick <id>', 'ban <id>', 'suggest', 'status', 'undo', 'quit'")
    print("Format: CM draft (Ban/Ban/Ban/Ban/Ban/Ban/Ban, Pick/Pick, ...)\n")

    while turn < len(CM_FORMAT):
        action_type, side = CM_FORMAT[turn]
        team_name = "Radiant" if side == 0 else "Dire"
        print(f"\n--- Turn {turn+1}: {team_name} {action_type.upper()} ---")
        print(f"Taken: {[hero_name(h) for h in sorted(taken)] or 'none'}")

        # Get suggestion
        try:
            from trainer.bot_greedy import GreedyDraftBot
            bot = GreedyDraftBot(model, builder)
            suggestions = bot.suggest_next_action(
                current_heroes=[h for h, _, _ in draft_history],
                current_actions=[a for _, a, _ in draft_history],
                is_radiant_turn=(side == 0),
                is_pick=(action_type == "pick"),
                radiant_picks=[h for h, a, s in draft_history if a == "pick" and s == 0],
                dire_picks=[h for h, a, s in draft_history if a == "pick" and s == 1],
                top_k=5,
            )
            print("Suggested:", ", ".join(f"{hero_name(s['hero_id'])} ({s['win_probability']*100:.1f}%)" for s in suggestions[:5]))
        except Exception as e:
            logger.warning("Suggestion failed: %s", e)

        # User input
        while True:
            cmd = input(f"  {team_name} {action_type} > ").strip().lower()
            if cmd == "quit":
                print("\nDraft abandoned.")
                return
            elif cmd == "status":
                print(f"  Turn {turn+1}, {team_name} {action_type}")
                print(f"  Radiant picks: {[hero_name(h) for h, a, s in draft_history if a == 'pick' and s == 0]}")
                print(f"  Dire picks: {[hero_name(h) for h, a, s in draft_history if a == 'pick' and s == 1]}")
                print(f"  Radiant bans: {[hero_name(h) for h, a, s in draft_history if a == 'ban' and s == 0]}")
                print(f"  Dire bans: {[hero_name(h) for h, a, s in draft_history if a == 'ban' and s == 1]}")
                continue
            elif cmd == "undo":
                if draft_history:
                    h, a, s = draft_history.pop()
                    taken.discard(h)
                    turn -= 1
                    print(f"  Undid {a} {hero_name(h)}")
                else:
                    print("  Nothing to undo")
                continue
            elif cmd.startswith("pick ") or cmd.startswith("ban "):
                parts = cmd.split()
                try:
                    hid = int(parts[1])
                    if hid in taken:
                        print(f"  {hero_name(hid)} already taken!")
                        continue
                    if hid not in cache.valid_hero_ids:
                        print(f"  {hero_name(hid)} not in valid hero pool")
                        continue
                    taken.add(hid)
                    draft_history.append((hid, action_type, side))
                    turn += 1
                    print(f"  {team_name} {action_type}: {hero_name(hid)}")
                    break
                except (ValueError, IndexError):
                    print("  Usage: pick <hero_id> or ban <hero_id>")
                    continue
            elif cmd == "suggest":
                continue
            else:
                print("  Unknown command. Use: pick/ban <id>, suggest, status, undo, quit")

    # Draft complete
    print("\n" + "=" * 60)
    print("  DRAFT COMPLETE")
    print("=" * 60)
    rad_picks = [h for h, a, s in draft_history if a == "pick" and s == 0]
    dire_picks = [h for h, a, s in draft_history if a == "pick" and s == 1]
    rad_bans = [h for h, a, s in draft_history if a == "ban" and s == 0]
    dire_bans = [h for h, a, s in draft_history if a == "ban" and s == 1]

    print(f"\nRadiant picks: {', '.join(hero_name(h) for h in rad_picks)}")
    print(f"Dire picks:    {', '.join(hero_name(h) for h in dire_picks)}")
    print(f"Radiant bans:  {', '.join(hero_name(h) for h in rad_bans)}")
    print(f"Dire bans:     {', '.join(hero_name(h) for h in dire_bans)}")

    # Evaluate composition
    if len(rad_picks) == 5 and len(dire_picks) == 5:
        try:
            from trainer.inference_cache import InferenceCache
            # Use model to evaluate
            print("\nEvaluating composition...")
            # Build feature vector for Radiant
            feat = builder.build_tabular_features(
                hypothetical_hero_id=rad_picks[-1],
                is_radiant_turn=True, is_pick=True,
                radiant_picks=rad_picks, dire_picks=dire_picks,
            )
            with torch.no_grad():
                logit = model(
                    torch.tensor([rad_picks + [0] * (cfg.max_seq_len - len(rad_picks))], dtype=torch.long),
                    torch.tensor([[3] * len(rad_picks) + [0] * (cfg.max_seq_len - len(rad_picks))], dtype=torch.long),
                    torch.tensor([feat], dtype=torch.float32),
                )
                prob = torch.sigmoid(logit).item()
            print(f"  Radiant win probability: {prob*100:.1f}%")
            print(f"  Dire win probability: {(1-prob)*100:.1f}%")
        except Exception as e:
            logger.warning("Evaluation failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Interactive Draft Bot")
    parser.add_argument("--patch", type=int, default=60, help="Patch ID")
    parser.add_argument("--mcts", action="store_true", help="Use MCTS bot")
    parser.add_argument("--iterations", type=int, default=200, help="MCTS iterations")
    args = parser.parse_args()

    run_interactive(args.patch, args.mcts, args.iterations)


if __name__ == "__main__":
    main()
