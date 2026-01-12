#!/usr/bin/env python3
"""Validate configuration for MLA-HFT system."""
import json
import sys
from pathlib import Path

root = Path(__file__).parent.parent
config_file = root / "config" / "params.json"

with open(config_file) as f:
    p = json.load(f)

print("=" * 60)
print("CONFIG VALIDATION")
print("=" * 60)

print("\n[Trading Config]")
trading = p.get("trading", {})
print(f"  signal_threshold:      {trading.get('signal_threshold', 'MISSING')}")
print(f"  min_q_diff:            {trading.get('min_q_diff', 'MISSING')}")
print(f"  min_take_profit_pct:   {trading.get('min_take_profit_pct', 'MISSING')}")
print(f"  min_stop_loss_pct:     {trading.get('min_stop_loss_pct', 'MISSING')}")
print(f"  min_position_size_pct: {trading.get('min_position_size_pct', 'MISSING')}")

print("\n[Reward Config]")
r = p.get('ml_model', {}).get('reward', {})
print(f"  trade_bonus:           {r.get('trade_bonus', 'MISSING')}")
print(f"  positive_bias:         {r.get('positive_bias', 'MISSING')}")
print(f"  kelly_weight:          {r.get('kelly_weight', 'MISSING')}")
print(f"  early_bonus_amount:    {r.get('early_bonus_amount', 'MISSING')}")
print(f"  hold_penalty:          {r.get('hold_penalty', 'MISSING')}")
print(f"  min_trade_reward:      {r.get('min_trade_reward', 'MISSING')}")
print(f"  max_trade_reward:      {r.get('max_trade_reward', 'MISSING')}")

print("\n[ML Model Config]")
ml = p.get('ml_model', {})
print(f"  data_days:             {ml.get('data_days', 'MISSING')}")
print(f"  use_fgsm:              {ml.get('use_fgsm', 'MISSING')}")

print("\n=" * 60)

# Validate all keys exist
missing = []
if trading.get('signal_threshold') is None:
    missing.append("trading.signal_threshold")
if r.get('kelly_weight') is None:
    missing.append("ml_model.reward.kelly_weight")
if ml.get('data_days') is None:
    missing.append("ml_model.data_days")

if missing:
    print(f"MISSING KEYS: {missing}")
    sys.exit(1)
else:
    print("ALL REQUIRED KEYS PRESENT ✓")
    sys.exit(0)
