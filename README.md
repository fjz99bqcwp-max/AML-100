# AML-100: Machine Learning Autonomous HFT for XYZ100-USDC on Hyperliquid

A fully autonomous machine learning-based high-frequency trading system for **XYZ100-USDC equity perpetual futures** on Hyperliquid mainnet (HIP-3/XYZ deployment). Optimized for Apple Silicon M4 with Metal Performance Shaders (MPS) acceleration.

## Trading Objectives

| Metric | Target |
|--------|--------|
| Monthly Return | 5-25% |
| Profit Factor | >= 1.1 |
| Sharpe Ratio | >= 1.5 |
| Max Drawdown | <= 5% |
| Auto-Halt Threshold | 4% drawdown |
| Latency Target | < 1ms inference |

## XYZ100-USDC Specific Features

This system is specifically optimized for XYZ100 equity perpetuals:

- **Equity Volatility Features**: Enhanced feature engineering for equity-linked perps
- **Implied Volatility Proxy**: Estimates IV from price range data
- **Gap Detection**: Identifies and trades session gaps
- **Mean Reversion Signals**: Tuned for equity market dynamics
- **24/7 Oracle Support**: Leverages Hyperliquid's continuous pricing model
- **BTC Fallback**: Automatic fallback to BTC data when XYZ100 history is insufficient

### XYZ100 vs BTC: Key Differences

| Feature | XYZ100-USDC | BTC-USDC |
|---------|-------------|----------|
| Volatility | Higher (equity) | Moderate (crypto) |
| Session Gaps | Possible | Rare |
| Mean Reversion | Stronger | Weaker |
| TP/SL Settings | 3.5%/1.2% | 2.3%/0.8% |
| Position Size | 40% | 40% |

## Architecture Overview

### Core Components

- **Hybrid LSTM+DQN Model**: Enhanced with equity-specific features for temporal pattern recognition
- **Bayesian Optimization**: Auto-tunes trading parameters with wider ranges for equity volatility
- **Kelly Criterion Risk Management**: Optimal position sizing with volatility scaling
- **Async WebSocket Data**: Real-time market data with sub-millisecond processing
- **BTC Fallback System**: Automatic fallback when XYZ100 history < 5000 records

## Quick Start

### Prerequisites

- Python 3.12+
- macOS with Apple Silicon M4 (recommended) or any Unix-like system
- Hyperliquid account with API keys
- Access to XYZ100-USDC on HIP-3 deployment

### Installation

1. **Navigate to AML-100 directory**
   ```bash
   cd /Users/nheosdisplay/VSC/AML/AML-100
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   Create a `.env` file in the project root:
   ```
   HYPERLIQUID_API_KEY=your_api_key_here
   HYPERLIQUID_API_SECRET=your_api_secret_here
   HYPERLIQUID_WALLET_ADDRESS=your_wallet_address_here
   ```

5. **Configure initial capital** (optional)
   ```bash
   export AML_INITIAL_CAPITAL=10000  # Default: $10,000
   ```

### Launch

```bash
# Navigate to project directory
cd /Users/nheosdisplay/VSC/AML/AML-100

# Activate virtual environment and source environment
source .venv/bin/activate && source .env

# Run with caffeinate to prevent sleep (full autonomous mode)
caffeinate python scripts/launch.py

# Or with specific options
python scripts/launch.py --mode autonomous --capital 10000

# Backtest only mode
python scripts/launch.py --mode backtest

# Training only mode
python scripts/launch.py --mode train
```

## Project Structure

```
AML-100/
├── config/
│   ├── objectives.json      # Trading objectives and thresholds
│   ├── params.json          # XYZ100-optimized parameters
│   └── api.json             # API configuration for XYZ100
├── src/
│   ├── hyperliquid_api.py   # Async API wrapper
│   ├── data_fetcher.py      # XYZ100 data + BTC fallback
│   ├── risk_manager.py      # Risk management with equity vol
│   ├── ml_model.py          # Hybrid LSTM+DQN with equity features
│   ├── optimizer.py         # Bayesian optimization
│   └── main.py              # Core orchestration
├── monitoring/
│   ├── monitor_live.py      # Live terminal dashboard
│   ├── monitor_cycles.py    # Cycle status monitor
│   ├── dashboard_metrics.py # Performance metrics
│   └── dashboard_health.py  # System health monitor
├── scripts/
│   ├── setup_env.sh         # Environment setup
│   ├── launch.py            # Main entry point
│   └── backup.py            # Backup management
├── tests/                   # Test suite
├── data/                    # Cached market data (auto-created)
├── models/                  # Saved ML models (auto-created)
├── logs/                    # System logs (auto-created)
├── requirements.txt         # Python dependencies
└── README.md
```

## Configuration

### Trading Parameters (config/params.json)

Optimized for XYZ100-USDC equity perpetual volatility:

- take_profit_pct: 3.5%
- stop_loss_pct: 1.2%
- position_size_pct: 40%
- max_positions: 1
- leverage: 1
- tp_mult: 2.5
- sl_mult: 0.5
- learning_rate: 0.0005
- epsilon_end: 0.05
- backtest_days: 90
- slippage_pct: 0.02%
- latency_ms: 1-10ms

### Objectives (config/objectives.json)

- monthly_performance_min: 5%
- monthly_performance_max: 25%
- drawdown_max: 5%
- auto_stop_drawdown: 4%
- sharpe_min: 1.5
- profit_factor_min: 1.1

## BTC Fallback System

When XYZ100-USDC has insufficient historical data (< 5000 records or < 30 days), the system automatically:

1. **Logs Warning**: XYZ100-USDC insufficient history - Using BTC fallback
2. **Uses BTC Data**: For training and backtesting
3. **Continues XYZ100 Live**: Live trading still uses XYZ100-USDC orderbook
4. **Auto-Checks Hourly**: Periodically checks if XYZ100 has enough data
5. **Auto-Switches**: When XYZ100 reaches 5000+ records, automatically switches

## Autonomous Behavior

The system operates in a continuous loop:

1. **Hourly Backtest** (every 3600 seconds)
   - Runs vectorized backtest on 90 days of data
   - Uses 0.02% slippage, 1-10ms latency simulation
   - Evaluates strategy performance

2. **Performance Assessment**
   - Determines status: CRITICAL | POOR | MODERATE | GOOD
   - Triggers parameter adjustments based on status

3. **Parameter Optimization**
   - Bayesian optimization with XYZ100-tuned ranges
   - TP: 1-5%, SL: 0.4-2% for equity volatility

4. **Model Training**
   - Continuous learning with equity-specific features
   - ~18 epochs for critical retrains with live append

5. **Trading Cycles** (every 300 seconds)
   - Real-time signal generation < 1ms
   - Position management
   - Risk enforcement with equity vol adjustment

### Status-Based Actions

| Status | Position Size | Action |
|--------|--------------|--------|
| CRITICAL | 50% reduction | Halt new trades, run optimization (~18 epochs) |
| POOR | 25% reduction | Reduce exposure, quick adjust |
| MODERATE | Normal | Maintain strategy |
| GOOD | 10% increase | Optimize for growth |

## Monitoring

### Live Terminal Dashboard

```bash
python monitoring/monitor_live.py
```

Displays real-time:
- Current XYZ100 position and P/L
- Signal confidence
- Risk metrics
- Fallback status

### Cycle Monitor

```bash
python monitoring/monitor_cycles.py
```

Shows:
- Backtest results
- Training progress
- Optimization history
- XYZ100 data availability

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_ml_model.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Risk Management

### Automatic Protections

- **Drawdown Halt**: Trading pauses at 4% drawdown
- **Position Limits**: Maximum 1 position at a time
- **Dynamic Stop-Loss**: Adjusts based on equity volatility (1.2x ATR)
- **Kelly Criterion**: Optimal position sizing with 0.25 fraction

### Equity Volatility Adjustment

The risk manager applies a 1.2x volatility adjustment for XYZ100.

### Manual Override

```bash
# Emergency stop
touch /tmp/mla_emergency_stop

# Resume trading
rm /tmp/mla_emergency_stop
```

## Advanced Configuration

### MPS/GPU Optimization

The system automatically detects and uses Apple Metal Performance Shaders:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

### XYZ100 Equity Features

Custom features for equity perpetuals in src/data_fetcher.py:

- equity_vol_5: 5-period annualized volatility
- equity_vol_20: 20-period annualized volatility
- implied_vol_proxy: IV estimate from price range
- mean_revert_signal: Mean reversion strength
- gap_pct: Gap detection
- gap_filled: Gap fill indicator
- trend_persistence: Trend strength
- volume_spike: Volume anomaly detection

## Performance Metrics

The system tracks and optimizes for:

- **Sharpe Ratio**: Risk-adjusted returns (target >= 1.5)
- **Profit Factor**: Gross profit / Gross loss (target >= 1.1)
- **Max Drawdown**: Peak-to-trough decline (limit <= 5%)
- **Win Rate**: Percentage of winning trades
- **Average R:R**: Average risk-reward ratio
- **Monthly Projection**: Extrapolated monthly return

## Security

- API keys stored in environment variables only
- No credentials in code or config files
- Encrypted WebSocket connections
- Rate limiting (< 15 API calls/min)

## Logging

Logs are stored in logs/:
- mla_hft_*.log: Main system events
- Trade executions with XYZ100 symbol
- Fallback status changes
- Model training progress

## Backup and Recovery

### Automatic Backups

```bash
# Create backup
python scripts/backup.py --create

# List backups
python scripts/backup.py --list

# Restore from backup
python scripts/backup.py --restore backup_20240101_120000.tar.gz
```

## Disclaimer

**This software is for educational and research purposes only.**

Trading equity perpetuals involves substantial risk of loss. XYZ100-USDC is a new instrument with limited history. Past performance does not guarantee future results.

- Always test with small amounts first
- Never invest more than you can afford to lose
- Monitor the system regularly
- Be aware of fallback mode limitations

## Troubleshooting

### Rate Limit Issues

If you see RATE LIMIT HIT:
1. System will auto-backoff with exponential delay
2. Check config/api.json rate limit settings
3. Circuit breaker will pause after 3 consecutive failures

### Insufficient XYZ100 History

If you see XYZ100-USDC insufficient history:
1. System automatically uses BTC fallback
2. Live trading continues on XYZ100
3. System rechecks hourly for XYZ100 data
4. Auto-switches when 5000+ records available

### Low Data Quality

If backtest shows poor results:
1. Check if using fallback (BTC data)
2. Increase data_days in params.json
3. Consider waiting for more XYZ100 history

---

**Built for autonomous trading of XYZ100-USDC on Hyperliquid HIP-3 mainnet**
