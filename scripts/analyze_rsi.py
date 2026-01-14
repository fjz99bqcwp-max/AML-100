#!/usr/bin/env python3
"""Analyze RSI mean reversion edge at different holding periods."""
import pandas as pd
import numpy as np
import joblib

# Load cached data
data = joblib.load("data/cache/training_60a77a544d6a.joblib")

# Handle dict format
if isinstance(data, dict) and 'df' in data:
    df = data['df']
elif isinstance(data, pd.DataFrame):
    df = data
else:
    raise ValueError(f"Unknown data format: {type(data)}")

print(f"Data shape: {df.shape}")
print(f"Data columns: {df.columns.tolist()[:10]}...")

# Calculate RSI if not present
if 'rsi' not in df.columns:
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

# Find RSI extreme signals
rsi_low = df['rsi'] < 20
rsi_high = df['rsi'] > 80

print(f"Total bars: {len(df)}")
print(f"RSI < 20: {rsi_low.sum()} ({rsi_low.sum()/len(df)*100:.2f}%)")
print(f"RSI > 80: {rsi_high.sum()} ({rsi_high.sum()/len(df)*100:.2f}%)")

# Analyze consecutive RSI extremes (clustering)
rsi_low_groups = (rsi_low != rsi_low.shift()).cumsum()[rsi_low]
rsi_high_groups = (rsi_high != rsi_high.shift()).cumsum()[rsi_high]

print(f"\nRSI clustering analysis:")
print(f"Number of RSI<20 clusters: {rsi_low_groups.nunique()}")
print(f"Number of RSI>80 clusters: {rsi_high_groups.nunique()}")
print(f"Avg bars per RSI<20 cluster: {rsi_low.sum() / max(rsi_low_groups.nunique(),1):.1f}")
print(f"Avg bars per RSI>80 cluster: {rsi_high.sum() / max(rsi_high_groups.nunique(),1):.1f}")

# Analyze different holding periods for RSI<20 (BUY signal)
print("\n=== RSI<20 BUY returns by holding period ===")
for hold in [5, 10, 15, 20, 30, 45, 60]:
    returns = []
    for i in df[df['rsi'] < 20].index:
        idx = df.index.get_loc(i)
        if idx + hold < len(df):
            entry = df.iloc[idx]['close']
            exit_price = df.iloc[idx + hold]['close']
            ret = (exit_price - entry) / entry * 100
            returns.append(ret)
    if returns:
        avg = np.mean(returns)
        wins = sum(1 for r in returns if r > 0.1)  # > 0.1% to cover costs
        win_rate = wins / len(returns) * 100
        print(f"  {hold:2d} bars: {avg:+.4f}% avg, {win_rate:.1f}% win rate (n={len(returns)})")

# Analyze different holding periods for RSI>80 (SELL signal)
print("\n=== RSI>80 SELL returns by holding period ===")
for hold in [5, 10, 15, 20, 30, 45, 60]:
    returns = []
    for i in df[df['rsi'] > 80].index:
        idx = df.index.get_loc(i)
        if idx + hold < len(df):
            entry = df.iloc[idx]['close']
            exit_price = df.iloc[idx + hold]['close']
            ret = (entry - exit_price) / entry * 100  # Short position
            returns.append(ret)
    if returns:
        avg = np.mean(returns)
        wins = sum(1 for r in returns if r > 0.1)  # > 0.1% to cover costs
        win_rate = wins / len(returns) * 100
        print(f"  {hold:2d} bars: {avg:+.4f}% avg, {win_rate:.1f}% win rate (n={len(returns)})")

# Calculate expected profitability accounting for costs
print("\n=== Expected NET returns (after 0.2% round-trip costs) ===")
costs = 0.2  # 0.1% each way
for hold in [15, 30, 45, 60]:
    buy_returns = []
    sell_returns = []
    
    for i in df[df['rsi'] < 20].index:
        idx = df.index.get_loc(i)
        if idx + hold < len(df):
            entry = df.iloc[idx]['close']
            exit_price = df.iloc[idx + hold]['close']
            ret = (exit_price - entry) / entry * 100 - costs
            buy_returns.append(ret)
    
    for i in df[df['rsi'] > 80].index:
        idx = df.index.get_loc(i)
        if idx + hold < len(df):
            entry = df.iloc[idx]['close']
            exit_price = df.iloc[idx + hold]['close']
            ret = (entry - exit_price) / entry * 100 - costs
            sell_returns.append(ret)
    
    all_returns = buy_returns + sell_returns
    if all_returns:
        avg = np.mean(all_returns)
        total = sum(all_returns)
        wins = sum(1 for r in all_returns if r > 0)
        win_rate = wins / len(all_returns) * 100
        print(f"  {hold:2d} bars: {avg:+.4f}% avg, {win_rate:.1f}% win rate, {total:+.2f}% total (n={len(all_returns)})")
