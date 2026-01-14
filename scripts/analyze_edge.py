#!/usr/bin/env python3
"""Analyze RSI edge in cached vs live data."""
import joblib
import pandas as pd
import numpy as np

# Load the cached data that worked well
data = joblib.load('data/cache/training_60a77a544d6a.joblib')
df = data['df']

print('=== CACHED DATA (where RSI edge was found) ===')

# RSI distribution
rsi = df['rsi']
print(f'RSI < 20 count: {(rsi < 20).sum()}')
print(f'RSI > 80 count: {(rsi > 80).sum()}')

# Check returns after RSI extremes
results = []
for i in range(60, len(df)-60):
    if rsi.iloc[i] < 20:
        entry = df.iloc[i]['close']
        exit_price = df.iloc[i+60]['close']
        ret = (exit_price - entry) / entry * 100
        results.append(('buy', ret))
    elif rsi.iloc[i] > 80:
        entry = df.iloc[i]['close']
        exit_price = df.iloc[i+60]['close']
        ret = (entry - exit_price) / entry * 100  # Short = entry - exit
        results.append(('sell', ret))

buys = [r[1] for r in results if r[0] == 'buy']
sells = [r[1] for r in results if r[0] == 'sell']

print(f'\nBuy signals (RSI<20): {len(buys)}')
if buys:
    print(f'  Avg 60-bar return: {np.mean(buys):.3f}%')
    print(f'  Win rate: {sum(1 for b in buys if b > 0.2) / len(buys) * 100:.1f}% (after 0.2% costs)')
    
print(f'\nSell signals (RSI>80): {len(sells)}')
if sells:
    print(f'  Avg 60-bar return: {np.mean(sells):.3f}%')
    print(f'  Win rate: {sum(1 for s in sells if s > 0.2) / len(sells) * 100:.1f}% (after 0.2% costs)')
