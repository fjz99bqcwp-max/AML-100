#!/usr/bin/env python3
"""Debug script for backtest environment"""
import sys
sys.path.insert(0, '/Users/nheosdisplay/VSC/AML/AML-100')

from src.ml_model import BacktestEnvironment
import pandas as pd
import numpy as np

# Create simple test data with known movement
np.random.seed(42)
n = 100
prices = 100 + np.cumsum(np.random.randn(n) * 0.1)

df = pd.DataFrame({
    'open': prices,
    'high': prices * 1.002,
    'low': prices * 0.998,
    'close': prices,
    'volume': 1000,
    'timestamp': range(n)
})

env = BacktestEnvironment(
    df=df,
    initial_capital=1000.0,
    transaction_cost=0.0005,
    slippage=0.0001,
    take_profit_pct=0.5,
    stop_loss_pct=0.25,
    max_hold_bars=15
)

# Simulate trades: buy, wait, let TP/SL trigger
for i in range(50):
    action = 1 if i % 10 == 0 else 0  # Buy every 10 bars
    reward, done = env.step(action, 0.4)
    if done:
        print(f"Done at step {i}")
        break

print(f'Trades: {len(env.trades)}')
print(f'Final capital: {env.capital:.2f}')
print(f'Initial: {env.initial_capital:.2f}')
print(f'Return: {((env.capital - env.initial_capital) / env.initial_capital) * 100:.2f}%')

if env.trades:
    print("\nFirst 5 trades:")
    for t in env.trades[:5]:
        print(f"  {t['side']}: entry={t['entry_price']:.4f}, exit={t['exit_price']:.4f}, pnl={t['pnl']:.4f}, pnl_pct={t['pnl_pct']:.4f}%, reason={t['reason']}")
