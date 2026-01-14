#!/usr/bin/env python3
"""Analyze market data for regime detection"""
import pandas as pd
import json
import sys
sys.path.insert(0, '.')
from src.data_fetcher import DataFetcher

# Load training data 
df = pd.read_parquet('data/historical/xyz:XYZ100_1m_20260114.parquet')
print(f'Raw data: {len(df)} bars')

# Create DataFetcher to compute indicators
with open('config/params.json') as f:
    params = json.load(f)
fetcher = DataFetcher(params)
df = fetcher._add_technical_features(df.copy())

# Check ADX distribution
if 'adx' in df.columns:
    adx = df['adx'].dropna()
    print(f'\nADX Analysis:')
    print(f'  ADX < 25 (ranging): {(adx < 25).sum()} bars ({100*(adx < 25).mean():.1f}%)')
    print(f'  ADX >= 25 (trending): {(adx >= 25).sum()} bars ({100*(adx >= 25).mean():.1f}%)')
    print(f'  ADX mean: {adx.mean():.1f}')

# Check RSI distribution
if 'rsi' in df.columns:
    rsi = df['rsi'].dropna()
    print(f'\nRSI Analysis:')
    print(f'  RSI < 30 (oversold): {(rsi < 30).sum()} bars')
    print(f'  RSI > 70 (overbought): {(rsi > 70).sum()} bars')

# Check price
print(f'\nPrice Analysis:')
print(f'  Start: ${df.iloc[0]["close"]:.2f}')
print(f'  End: ${df.iloc[-1]["close"]:.2f}')
print(f'  Change: {100*(df.iloc[-1]["close"]-df.iloc[0]["close"])/df.iloc[0]["close"]:.2f}%')
