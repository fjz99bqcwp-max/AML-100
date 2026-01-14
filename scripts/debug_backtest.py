#!/usr/bin/env python3
"""Debug BacktestEnvironment to verify trade calculations."""
import pandas as pd
import numpy as np
import joblib
import sys
sys.path.insert(0, 'src')
from ml_model import BacktestEnvironment

data = joblib.load('data/cache/training_60a77a544d6a.joblib')
df = data['df']

print(f"Data shape: {df.shape}")

# Create environment
env = BacktestEnvironment(
    df=df,
    initial_capital=1000,
    transaction_cost=0.0005,
    slippage=0.0005,
    take_profit_pct=99.0,
    stop_loss_pct=99.0,
    max_hold_bars=60
)

print(f"Initial capital: {env.capital}")
print(f"Initial position: {env.position}")

# Run through data with RSI signals - track capital at each step
seq_len = 60
capital_history = [1000]

for i in range(seq_len, len(df) - 1):
    rsi = df.iloc[i].get('rsi', 50)
    action = 0
    if rsi < 20:
        action = 1
    elif rsi > 80:
        action = 2
    
    # Capture capital before step
    cap_before = env.capital
    pos_before = env.position
    
    # Pass the correct data index
    reward, done = env.step(action, 0.5, data_idx=i)
    
    # Track capital after each trade closes
    if len(env.trades) > len(capital_history) - 1:
        capital_history.append(env.capital)
        # Print trade info with more details
        trade = env.trades[-1]
        notional = trade['size'] * trade['entry_price']
        print(f"Trade {len(env.trades)}: cap_before={cap_before:.2f}, cap_after={env.capital:.2f}, pnl$={trade['pnl']:.2f}, notional={notional:.2f}, pos_before={pos_before:.4f}")
    
    if done:
        break

# Force close any remaining position
if env.position > 0:
    final_price = df.iloc[len(df)-1]['close']
    env._close_position(final_price, "end_of_data")
    print(f"Trade {len(env.trades)}: FORCED CLOSE at end of data")

print(f"\nTotal trades: {len(env.trades)}")
print(f"Final capital: {env.capital:.2f}")
print(f"Total return: {((env.capital - 1000) / 1000) * 100:.2f}%")

# Calculate win rate and profit factor
wins = [t for t in env.trades if t['pnl'] > 0]
losses = [t for t in env.trades if t['pnl'] <= 0]
print(f"Win rate: {len(wins) / len(env.trades) * 100:.1f}% ({len(wins)}/{len(env.trades)})")
total_wins = sum(t['pnl'] for t in wins)
total_losses = abs(sum(t['pnl'] for t in losses))
print(f"Profit factor: {total_wins / total_losses if total_losses > 0 else 'inf':.2f}")
