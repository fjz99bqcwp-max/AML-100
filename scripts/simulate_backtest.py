#!/usr/bin/env python3
"""Simulate exact backtest logic to verify RSI mean reversion edge."""
import pandas as pd
import numpy as np
import joblib

data = joblib.load('data/cache/training_60a77a544d6a.joblib')
df = data['df']

print(f"Data rows: {len(df)}")
print(f"RSI column present: {'rsi' in df.columns}")

# Show sample RSI values
print(f"\nSample RSI values (first 20):")
for i in range(20):
    print(f"  {i}: RSI={df.iloc[i]['rsi']:.2f}, Close={df.iloc[i]['close']:.4f}")

# Check for NaN RSI values
nan_count = df['rsi'].isna().sum()
print(f"\nNaN RSI values: {nan_count} ({nan_count/len(df)*100:.1f}%)")

# Count RSI extreme occurrences
rsi_low = df['rsi'] < 20
rsi_high = df['rsi'] > 80
print(f"RSI < 20 bars: {rsi_low.sum()}")
print(f"RSI > 80 bars: {rsi_high.sum()}")

# Detailed simulation with trace
print("\n=== DETAILED TRADE TRACE (first 5 trades) ===")
hold_bars = 60
capital = 1000
position = 0
entry_idx = 0
entry_price = 0
trades = []
costs = 0.002  # 0.2% round-trip
position_size = 0.5  # 50%
trace_count = 0

for i in range(len(df) - hold_bars - 1):
    rsi = df.iloc[i]['rsi']
    close = df.iloc[i]['close']
    
    # Check exit first
    if position != 0:
        bars_held = i - entry_idx
        if bars_held >= hold_bars:
            # Exit at time
            exit_price = df.iloc[i]['close']
            if position == 1:  # Long
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:  # Short
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            
            raw_pnl = pnl_pct
            pnl_pct -= costs * 100  # 0.2% costs
            capital_change = capital * position_size * (pnl_pct / 100)
            capital += capital_change
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': i,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'side': 'long' if position == 1 else 'short',
                'raw_pnl': raw_pnl,
                'net_pnl': pnl_pct, 
                'capital_change': capital_change
            })
            
            if trace_count < 5:
                print(f"\nTrade {len(trades)}:")
                print(f"  Entry: idx={entry_idx}, price={entry_price:.4f}, side={'LONG' if position==1 else 'SHORT'}")
                print(f"  Exit: idx={i}, price={exit_price:.4f}")
                print(f"  Raw PnL: {raw_pnl:+.4f}%, Net PnL: {pnl_pct:+.4f}%")
                print(f"  Capital: {capital:.2f}")
                trace_count += 1
            
            position = 0
    
    # Entry if flat
    if position == 0:
        if not pd.isna(rsi):
            if rsi < 20:
                position = 1
                entry_idx = i
                entry_price = close
            elif rsi > 80:
                position = -1
                entry_idx = i
                entry_price = close

# Calculate results
wins = [t for t in trades if t['net_pnl'] > 0]
losses = [t for t in trades if t['net_pnl'] <= 0]

print(f"\n=== SUMMARY ===")
print(f"Trades: {len(trades)} (Wins: {len(wins)}, Losses: {len(losses)})")
print(f"Win rate: {len(wins)/len(trades)*100:.1f}%" if trades else "No trades")
print(f"Avg win: +{np.mean([t['net_pnl'] for t in wins]):.3f}%" if wins else "No wins")
print(f"Avg loss: {np.mean([t['net_pnl'] for t in losses]):.3f}%" if losses else "No losses")
print(f"Total return: {(capital/1000-1)*100:+.2f}%")
print(f"Profit factor: {sum(t['net_pnl'] for t in wins) / abs(sum(t['net_pnl'] for t in losses)):.2f}" if losses else "N/A")

# Show long vs short performance
longs = [t for t in trades if t['side'] == 'long']
shorts = [t for t in trades if t['side'] == 'short']
print(f"\nLong trades: {len(longs)}, Avg PnL: {np.mean([t['net_pnl'] for t in longs]):+.3f}%" if longs else "No longs")
print(f"Short trades: {len(shorts)}, Avg PnL: {np.mean([t['net_pnl'] for t in shorts]):+.3f}%" if shorts else "No shorts")
