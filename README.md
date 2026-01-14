# AML-100: Autonomous ML Trading System for XYZ100-USDC

A fully autonomous machine learning trading system for **XYZ100-USDC equity perpetual futures** on HyperLiquid mainnet. Optimized for Apple M4 (24GB RAM) with Metal Performance Shaders (MPS) acceleration.

**Timezone**: Zurich CET | **Last Cleaned**: 2026-01-14

## Model Architecture

| Component | Specification |
|-----------|--------------|
| **LSTM Encoder** | 2 layers, hidden_size=128 |
| **DQN Head** | output_size=3 (HOLD/BUY/SELL) |
| **Features** | OHLCV, volume, returns, time-based only |
| **Learning Rate** | 0.0001 |
| **Epsilon Decay** | 0.997 |
| **Leverage** | 1x (default) |

## Trading Objectives

| Metric | Target |
|--------|--------|
| Monthly Return | ≥15% |
| Sharpe Ratio | ≥1.5 |
| Max Drawdown | ≤5% |
| Take Profit | 0.5% |
| Stop Loss | 0.25% |
| Position Size | 40% |

## Project Structure (Cleaned)

```
AML-100/
├── scripts/
│   ├── launch.py           # Main entry point (autonomous/backtest/train)
│   ├── run_hft.sh          # Quick-start shell script
│   └── setup_env.sh        # Environment setup
├── config/
│   ├── api.json            # API credentials
│   ├── params.json         # Trading/ML parameters (SPEC defaults)
│   └── objectives.json     # Performance objectives
├── src/
│   ├── main.py             # Core trading logic
│   ├── ml_model.py         # LSTM+DQN hybrid model
│   ├── data_fetcher.py     # Market data (SDK + synthetic)
│   ├── hyperliquid_api.py  # HyperLiquid SDK integration
│   ├── risk_manager.py     # Position/risk management
│   └── optimizer.py        # Bayesian optimization
├── monitoring/
│   └── monitor_live.py     # Live trading dashboard
├── tests/
│   ├── test_ml_model.py    # Model tests
│   ├── test_optimizer.py   # Optimizer tests
│   └── test_risk_manager.py # Risk tests
├── models/
│   └── best_model.pt       # Trained model checkpoint
├── data/
│   ├── historical/         # Historical price data
│   └── trading/            # Trade logs
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
├── pytest.ini              # Test configuration
└── README.md               # This file
```

## Quick Start

```bash
# 1. Setup environment
./scripts/setup_env.sh
source .venv/bin/activate

# 2. Configure credentials
cp .env.example .env
# Edit .env with HYPERLIQUID_WALLET_ADDRESS and HYPERLIQUID_API_SECRET

# 3. Run autonomous mode (prevents macOS sleep)
caffeinate python scripts/launch.py --mode autonomous --asset XYZ100 \
    --wallet 0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584

# Or use the shell script
./scripts/run_hft.sh autonomous
```

## Execution Modes

| Mode | Command | Description |
|------|---------|-------------|
| **Autonomous** | `--mode autonomous` | Full loop: backtest→optimize→train→live |
| **Backtest** | `--mode backtest --days 180` | Historical simulation |
| **Train** | `--mode train --epochs 180` | Model training only |
| **Optimize** | `--mode optimize --trials 50` | Bayesian parameter tuning |
| **Live** | `--mode live` | Live trading (requires objectives met) |

## CLI Reference

```bash
python scripts/launch.py \
    --mode autonomous \          # Required: autonomous|backtest|train|optimize|live
    --asset XYZ100 \             # Trading asset (default: XYZ100)
    --wallet 0x... \             # Wallet address for live trading
    --days 180 \                 # Historical data days
    --epochs 180 \               # Training epochs max
    --trials 50 \                # Optimization trials
    --cycle-hours 1 \            # Autonomous cycle interval
    --reset-defaults \           # Reset to SPEC defaults
    --log-level INFO             # DEBUG|INFO|WARNING|ERROR
```

## Backtesting Recommendations

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Data Source | 90% real SPX + 10% synthetic | Robust training with augmentation |
| Duration | 180 days | Sufficient for regime coverage |
| Slippage | 0.05% | Realistic for limit orders |
| Latency | 5-15ms | HyperLiquid execution range |
| Commission | 0.05% | HyperLiquid maker fee |
| Workers | 6 | M4 optimization |

## Training Recommendations

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 180 max | Prevent overfitting |
| Early Stop | patience=30 | Avoid wasted compute |
| Learning Rate | 0.0001 | Stable convergence |
| Batch Size | 32 | M4 memory optimal |
| Epsilon Decay | 0.997 | Gradual exploration reduction |
| num_workers | 6 | M4 MPS optimization |
| Rewards | Vol-scaled | Adaptive to market regime |

## Autonomous Mode Steps

1. **Load configs** - Reset to SPEC defaults on first run
2. **Backtest (Phase 1)** - 180-day hybrid data simulation
3. **Check objectives** - Sharpe ≥1.5, DD ≤5%, return ≥15%
4. **Optimize (Phase 2)** - Bayesian tuning if objectives not met
5. **Train (Phase 3)** - Retrain model with optimized params
6. **Validate (Phase 4)** - 30-day validation backtest
7. **Live (Phase 5)** - Execute trades if objectives met
8. **Loop** - Repeat hourly with parameter adjustments

## SDK Integration

Uses [HyperLiquid Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk):

```python
from hyperliquid.info import Info
from hyperliquid.utils import constants

info = Info(constants.MAINNET_API_URL, skip_ws=True)
user_state = info.user_state("0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584")
```

## License

MIT License - See LICENSE file for details.
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
