# AML-100: Autonomous ML Trading System for XYZ100-USDC

A fully autonomous machine learning-based trading system for **XYZ100-USDC equity perpetual futures** on Hyperliquid mainnet. Optimized for Apple Silicon M4 with Metal Performance Shaders (MPS) acceleration.

## Features

- **Hybrid LSTM+DQN Model**: Deep reinforcement learning with temporal pattern recognition
- **torch.compile() Optimization**: Sub-millisecond inference on M4 chips
- **Bayesian Optimization**: Auto-tunes trading parameters
- **Kelly Criterion Risk Management**: Optimal position sizing
- **Async WebSocket Data**: Real-time market data processing
- **Automatic Fallback**: Uses synthetic/SPX data when XYZ100 history is insufficient

## Trading Objectives

| Metric | Target |
|--------|--------|
| Monthly Return | 10-30% |
| Profit Factor | ≥ 1.2 |
| Sharpe Ratio | ≥ 1.5 |
| Max Drawdown | ≤ 5% |
| Auto-Halt | 4% drawdown |
| Inference Latency | < 1ms |

## Quick Start

### Prerequisites

- Python 3.9+
- macOS with Apple Silicon (M1/M2/M3/M4) recommended
- Hyperliquid account with API credentials

### Installation

```bash
# Clone and navigate to project
cd /path/to/AML-100

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Hyperliquid credentials
```

### Usage

```bash
# Activate environment
source .venv/bin/activate

# Full autonomous mode (prevents sleep on macOS)
caffeinate python AML-100.py --hft

# Training only
python AML-100.py --train --epochs 100 --days 30

# Backtest with hybrid data (70% real + 30% synthetic)
python AML-100.py --backtest --data hybrid --days 60

# Backtest with synthetic data
python AML-100.py --backtest --data synthetic --days 90

# Optimization only
python AML-100.py --optimize --trials 50

# Live monitoring dashboard
streamlit run monitoring/dashboard_streamlit.py
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--train` | Run model training only |
| `--backtest` | Run backtest only |
| `--optimize` | Run parameter optimization only |
| `--data` | Data source: `xyz100`, `hybrid`, `synthetic`, `btc` |
| `--days` | Historical data days (default: 30) |
| `--epochs` | Training epochs (default: 100) |
| `--trials` | Optimization trials (default: 50) |
| `--hft` | Enable HFT mode (tighter TP/SL) |
| `--log-level` | Logging level: DEBUG, INFO, WARNING, ERROR |

## Project Structure

```
AML-100/
├── AML-100.py              # Main entry point
├── config/
│   ├── api.json            # API configuration
│   ├── objectives.json     # Trading objectives
│   ├── params.json         # Trading parameters
│   └── params_ultra_hft.json  # Ultra-HFT variant
├── src/
│   ├── main.py             # Core orchestration
│   ├── ml_model.py         # Hybrid LSTM+DQN model
│   ├── data_fetcher.py     # Data acquisition
│   ├── hyperliquid_api.py  # API wrapper
│   ├── risk_manager.py     # Risk management
│   └── optimizer.py        # Bayesian optimization
├── monitoring/
│   ├── dashboard_streamlit.py  # Web dashboard
│   ├── monitor_live.py     # Terminal dashboard
│   └── monitor_cycles.py   # Cycle monitor
├── scripts/
│   ├── backup.py           # Backup management
│   ├── check_wallet.py     # Wallet checker
│   └── validate_config.py  # Config validation
├── tests/                  # Test suite (77 tests)
├── data/                   # Market data (auto-created)
├── models/                 # Saved models (auto-created)
├── logs/                   # System logs (auto-created)
└── requirements.txt        # Dependencies
```

## Configuration

### Key Parameters (config/params.json)

```json
{
  "trading": {
    "take_profit_pct": 0.5,
    "stop_loss_pct": 0.25,
    "position_size_pct": 0.4,
    "min_q_diff": 0.50,
    "max_hold_seconds": 60
  },
  "ml_model": {
    "epsilon_decay": 0.997,
    "early_stop_patience": 30,
    "enable_torch_compile": true
  },
  "backtest": {
    "enable_multiprocessing": true,
    "max_workers": 8,
    "chunk_size": 5000
  }
}
```

**Performance Tip**: Multiprocessing is enabled by default for faster backtests (2-4x speedup on multi-core systems). Set `enable_multiprocessing: false` to disable if needed.

### Objectives (config/objectives.json)

- `monthly_performance_min`: 15%
- `drawdown_max`: 5%
- `auto_stop_drawdown`: 4%
- `sharpe_ratio_min`: 1.5
- `profit_factor_min`: 1.3

## Autonomous Behavior

When running without flags, the system operates in a continuous loop:

1. **Hourly Backtest** - Evaluates strategy performance
2. **Status Assessment** - CRITICAL → POOR → MODERATE → GOOD
3. **Parameter Optimization** - Bayesian optimization if needed
4. **Model Training** - Continuous learning
5. **Trading Cycles** - Real-time signal generation and execution

### Status-Based Actions

| Status | Action |
|--------|--------|
| CRITICAL | Halt trading, run optimization |
| POOR | Reduce position size 25% |
| MODERATE | Maintain strategy |
| GOOD | Increase position size 10% |

## Risk Management

- **Drawdown Halt**: Pauses at 4% drawdown
- **Position Limits**: Maximum 1 position
- **Dynamic Stop-Loss**: ATR-based adjustment
- **Kelly Criterion**: Optimal position sizing

### Emergency Stop

```bash
# Create emergency stop file
touch /tmp/mla_emergency_stop

# Resume trading
rm /tmp/mla_emergency_stop
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Performance

Recent backtest results (60-day hybrid data):

| Metric | Value |
|--------|-------|
| Return | ~0% (flat) |
| Max Drawdown | 2.25% |
| Win Rate | 47.6% |
| Profit Factor | 0.97 |
| Total Trades | 2,152 |

## Monitoring

### Streamlit Dashboard

```bash
streamlit run monitoring/dashboard_streamlit.py
```

Features:
- Real-time wallet PnL
- Trade density heatmap
- Position tracking
- Alert system

### Terminal Dashboard

```bash
python monitoring/monitor_live.py
```

## Troubleshooting

### XYZ100 API Returns 500 Errors

The system automatically falls back to synthetic data when XYZ100 API is unavailable:

```
Using synthetic_spx fallback data (129600 rows)
```

### Rate Limiting

The system includes automatic exponential backoff and circuit breaker:

```
Circuit breaker OPEN: 3 consecutive failures. Pausing for 60s
```

### Model Not Training

Check training parameters:
- Increase `trade_bonus` if model produces too many HOLD actions
- Adjust `min_q_diff` threshold for trade filtering

## Security

- API keys stored in `.env` (gitignored)
- No credentials in code or config
- Encrypted WebSocket connections
- Rate limiting (< 15 calls/min)

## Disclaimer

**This software is for educational and research purposes only.**

Trading perpetual futures involves substantial risk of loss. Past performance does not guarantee future results.

- Always test with small amounts first
- Never invest more than you can afford to lose
- Monitor the system regularly

---

**Built for Hyperliquid HIP-3 mainnet**
