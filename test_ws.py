#!/usr/bin/env python3
"""Test WebSocket live prediction stream with 45-feature LiveDraftBERT."""
import asyncio
import json
import websockets


async def test_live_stream(match_id: int):
    uri = "ws://localhost:8080/ws/live"
    print(f"Connecting to {uri} for match {match_id}...")

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"match_id": match_id, "interval": 5}))

        while True:
            response = await websocket.recv()
            data = json.loads(response)

            if data.get("type") == "error":
                print(f"Error: {data['detail']}")
                break

            minute = data.get("minute")
            win_prob = data.get("radiant_win_probability", 0) * 100
            feats = data.get("features", {})

            print(f"\n--- Minute {minute} | Radiant Win Prob: {win_prob:.1f}% ---")
            print(f"  Gold Adv:       {feats.get('radiant_gold_adv', 0):+.0f}")
            print(f"  Deep Ward Diff: {feats.get('deep_ward_diff', 0):+.0f}")
            print(f"  Save Item Diff: {feats.get('save_item_diff', 0):+.0f}")
            print(f"  Scaling Threats:{feats.get('scaling_threat_diff', 0):+.0f}")
            print(f"  CC Effectiveness:{feats.get('cc_effectiveness_diff', 0):+.1f}")
            print(f"  T1 Tower Diff:  {feats.get('t1_tower_diff', 0):+.0f}")
            print(f"  T3 Tower Diff:  {feats.get('t3_tower_diff', 0):+.0f}")
            print(f"  Tower Dmg Diff: {feats.get('tower_damage_diff', 0):+.0f}")
            print(f"  Neutral Tier:   {feats.get('neutral_tier_diff', 0):+.0f}")
            print("-" * 45)


if __name__ == "__main__":
    asyncio.run(test_live_stream(8824218541))
