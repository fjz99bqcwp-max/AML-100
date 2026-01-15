import os
import sys
import numpy as np
os.chdir('/Users/nheosdisplay/VSC/AML/AML-100')
sys.path.insert(0, '/Users/nheosdisplay/VSC/AML/AML-100')

from src.ml_model import MLModel
from src.data_fetcher import DataFetcher
from src.hyperliquid_api import HyperliquidAPI
import asyncio

async def fetch_data():
    """Fetch market data asynchronously"""
    api = HyperliquidAPI()
    fetcher = DataFetcher(api=api)
    df = await fetcher.fetch_spx_data(days=30)
    return df

def main():
    # Load model
    model = MLModel()
    model_path = 'best_model.pt'
    
    try:
        success = model.load_model(model_path)
        print(f'Load success: {success}')
        print(f'Model initialized: {model.model is not None}')
        print(f'Epsilon: {model.epsilon:.4f}')
    except Exception as e:
        print(f'Error loading model: {e}')
        return
    
    # Get real data for realistic features
    print('\n--- Fetching real market data ---')
    df = asyncio.run(fetch_data())
    print(f'Fetched {len(df)} data points')
    
    # Prepare features using the model's feature preparation
    print('\n--- Preparing features ---')
    features, feature_names = model.prepare_features(df)
    print(f'Feature shape: {features.shape}')
    print(f'Feature names: {feature_names}')
    
    # Create feature sequences
    seq_len = 30
    n_samples = min(200, len(features) - seq_len)
    
    feature_batch = np.array([features[i:i+seq_len] for i in range(n_samples)])
    print(f'Batch shape: {feature_batch.shape}')
    
    # Get predictions
    print('\n--- Getting predictions ---')
    actions, confs, q_vals = model.predict_batch(feature_batch)
    
    print(f'\nQ-Values distribution:')
    print(f'  Q-Hold (action 0): mean={q_vals[:,0].mean():.4f}, std={q_vals[:,0].std():.4f}')
    print(f'  Q-Buy  (action 1): mean={q_vals[:,1].mean():.4f}, std={q_vals[:,1].std():.4f}')
    print(f'  Q-Sell (action 2): mean={q_vals[:,2].mean():.4f}, std={q_vals[:,2].std():.4f}')
    
    print(f'\nAction distribution:')
    unique, counts = np.unique(actions, return_counts=True)
    for a, c in zip(unique, counts):
        action_name = ['HOLD', 'BUY', 'SELL'][a]
        print(f'  {action_name}: {c} ({c/n_samples*100:.1f}%)')
    
    print(f'\nQ-value differences (max-min):')
    q_diffs = q_vals.max(axis=1) - q_vals.min(axis=1)
    print(f'  Mean diff: {q_diffs.mean():.4f}')
    print(f'  Min diff:  {q_diffs.min():.4f}')
    print(f'  Max diff:  {q_diffs.max():.4f}')
    
    # Check min_q_diff threshold
    min_q_diff = 0.003
    filtered_actions = actions.copy()
    filtered_actions[q_diffs < min_q_diff] = 0  # HOLD
    
    print(f'\nAfter min_q_diff filter ({min_q_diff}):')
    unique, counts = np.unique(filtered_actions, return_counts=True)
    for a, c in zip(unique, counts):
        action_name = ['HOLD', 'BUY', 'SELL'][a]
        print(f'  {action_name}: {c} ({c/n_samples*100:.1f}%)')
    
    # Analyze direction accuracy
    print('\n--- Direction accuracy analysis ---')
    # Get actual price changes
    prices = df['close'].values
    
    correct = 0
    wrong = 0
    
    for i in range(n_samples):
        idx = i + seq_len
        if idx + 1 >= len(prices):
            break
        
        action = filtered_actions[i]
        price_change = prices[idx + 1] - prices[idx]
        
        if action == 0:  # HOLD
            continue
        elif action == 1:  # BUY
            if price_change > 0:
                correct += 1
            else:
                wrong += 1
        elif action == 2:  # SELL
            if price_change < 0:
                correct += 1
            else:
                wrong += 1
    
    total_trades = correct + wrong
    if total_trades > 0:
        print(f'  Total trades: {total_trades}')
        print(f'  Correct direction: {correct} ({correct/total_trades*100:.1f}%)')
        print(f'  Wrong direction: {wrong} ({wrong/total_trades*100:.1f}%)')
    else:
        print('  No trades made')
    
    # Show first 10 predictions with price context
    print('\n--- First 10 predictions with context ---')
    for i in range(min(10, n_samples)):
        idx = i + seq_len
        if idx + 1 >= len(prices):
            break
        
        action = actions[i]
        action_name = ['HOLD', 'BUY', 'SELL'][action]
        q_diff = q_diffs[i]
        current_price = prices[idx]
        next_price = prices[idx + 1]
        pct_change = (next_price - current_price) / current_price * 100
        
        direction = 'UP' if pct_change > 0 else 'DOWN'
        correct_symbol = '✓' if (action == 1 and pct_change > 0) or (action == 2 and pct_change < 0) else '✗'
        if action == 0:
            correct_symbol = '-'
        
        print(f'  [{i:3d}] Action={action_name:4s} Q-diff={q_diff:.4f} | Price: {current_price:.2f} -> {next_price:.2f} ({pct_change:+.3f}% {direction}) {correct_symbol}')


if __name__ == '__main__':
    main()
