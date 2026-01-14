#!/usr/bin/env python3
"""Analyze market regime and price trend"""
import pandas as pd
from pathlib import Path

hist_file = Path('data/historical/xyz:XYZ100_1m_20260114.parquet')
if hist_file.exists():
    df = pd.read_parquet(hist_file)
    print(f'Total bars: {len(df)}')
    
    # Check ADX distribution
    if 'adx' in df.columns:
        adx = df['adx'].dropna()
        print(f'\nADX Statistics:')
        print(f'  Mean: {adx.mean():.1f}')
        print(f'  Median: {adx.median():.1f}')
        print(f'  ADX < 25 (ranging): {(adx < 25).sum()} bars ({100*(adx < 25).mean():.1f}%)')
        print(f'  ADX >= 25 (trending): {(adx >= 25).sum()} bars ({100*(adx >= 25).mean():.1f}%)')
    
    # Check RSI distribution  
    if 'rsi' in df.columns:
        rsi = df['rsi'].dropna()
        print(f'\nRSI Statistics:')
        print(f'  RSI < 20 (oversold): {(rsi < 20).sum()} bars')
        print(f'  RSI > 80 (overbought): {(rsi > 80).sum()} bars')
        print(f'  RSI < 30: {(rsi < 30).sum()} bars')
        print(f'  RSI > 70: {(rsi > 70).sum()} bars')
    
    # Check price trend
    if 'close' in df.columns:
        start_price = df['close'].iloc[0]
        end_price = df['close'].iloc[-1]
        print(f'\nPrice trend:')
        print(f'  Start: ${start_price:.2f}')
        print(f'  End: ${end_price:.2f}')
        print(f'  Change: {100*(end_price-start_price)/start_price:.1f}%')
else:
    print('No historical data found')
