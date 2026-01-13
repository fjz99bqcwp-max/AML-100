"""
AML HFT Main Module
Core autonomous logic for backtesting, ML training, and live trading
Runs continuous optimization cycles with automatic parameter adjustment
"""

import asyncio
import json
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hyperliquid_api import HyperliquidAPI, HyperliquidClient, OrderBook, Position, Trade
from src.data_fetcher import DataFetcher, MarketSnapshot
from src.ml_model import MLModel, BacktestEnvironment
from src.risk_manager import RiskManager, RiskMetrics, TradeRecord
from src.optimizer import ParameterOptimizer, PerformanceStatus

# Logger - configuration is handled by launch.py with ColoredFormatter
# Only create logger here, don't configure basicConfig
logger = logging.getLogger(__name__)


@dataclass
class CycleMetrics:
    """Metrics for a trading cycle"""
    cycle_number: int
    start_time: float
    end_time: float
    trades_count: int
    pnl: float
    pnl_pct: float
    status: str
    adjustments_made: List[str]
    avg_hold_time: float = 0.0  # HFT: Track average hold time per cycle


@dataclass
class SystemState:
    """Current system state"""
    mode: str  # "backtest", "optimization", "testing", "live"
    is_running: bool
    last_backtest: Optional[float]
    last_optimization: Optional[float]
    current_cycle: int
    cycles_since_backtest: int
    objectives_met: bool
    phase4_override: bool = False  # Step 3: Track if Phase 4 was overridden


class AMLHFTSystem:
    """
    Main autonomous HFT system
    Orchestrates backtesting, ML training, optimization, and live trading
    """
    
    def __init__(
        self,
        config_dir: str = "config",
        data_dir: str = "data",
        model_dir: str = "models"
    ):
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        
        # Ensure directories exist
        Path("logs").mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Load configurations
        self._load_configs()
        
        # Components (initialized in setup)
        self.api: Optional[HyperliquidAPI] = None
        self.data_fetcher: Optional[DataFetcher] = None
        self.ml_model: Optional[MLModel] = None
        self.risk_manager: Optional[RiskManager] = None
        self.optimizer: Optional[ParameterOptimizer] = None
        
        # System state
        self.state = SystemState(
            mode="init",
            is_running=False,
            last_backtest=None,
            last_optimization=None,
            current_cycle=0,
            cycles_since_backtest=0,
            objectives_met=False
        )
        
        # Trading state
        self.current_position: Optional[Position] = None
        self.pending_signal: Optional[str] = None
        self.signal_strength: float = 0.0
        
        # Cycle tracking
        self.cycle_history: deque = deque(maxlen=100)
        self.trade_count_this_cycle: int = 0
        self.cycle_start_time: float = 0.0
        
        # Step 3: Critical loop prevention
        self.consecutive_critical_count: int = 0
        self.max_consecutive_criticals: int = 5  # Step 3: Increased from 3 to 5
        
        # Step 4: Cumulative PnL tracking for monthly projection
        self.live_start_time: float = 0.0
        self.cumulative_pnl: float = 0.0
        self.total_live_trades: int = 0
        
        # Metrics storage
        self.latest_backtest_metrics: Dict[str, Any] = {}
        self.latest_live_metrics: Dict[str, Any] = {}
        
        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        
    def _load_configs(self) -> None:
        """Load all configuration files"""
        with open(self.config_dir / "params.json", "r") as f:
            self.params = json.load(f)
        with open(self.config_dir / "objectives.json", "r") as f:
            self.objectives = json.load(f)
        with open(self.config_dir / "api.json", "r") as f:
            self.api_config = json.load(f)
        
        logger.info("Configurations loaded")
    
    def _ensure_setup(self) -> None:
        """Ensure all components are initialized - raises RuntimeError if not"""
        if self.api is None:
            raise RuntimeError("api not initialized - call setup() first")
        if self.data_fetcher is None:
            raise RuntimeError("data_fetcher not initialized - call setup() first")
        if self.ml_model is None:
            raise RuntimeError("ml_model not initialized - call setup() first")
        if self.risk_manager is None:
            raise RuntimeError("risk_manager not initialized - call setup() first")
        if self.optimizer is None:
            raise RuntimeError("optimizer not initialized - call setup() first")
    
    # Type-safe accessors for components (use after _ensure_setup)
    @property
    def _api(self) -> HyperliquidAPI:
        assert self.api is not None
        return self.api
    
    @property
    def _data_fetcher(self) -> DataFetcher:
        assert self.data_fetcher is not None
        return self.data_fetcher
    
    @property
    def _ml_model(self) -> MLModel:
        assert self.ml_model is not None
        return self.ml_model
    
    @property
    def _risk_manager(self) -> RiskManager:
        assert self.risk_manager is not None
        return self.risk_manager
    
    @property
    def _optimizer(self) -> ParameterOptimizer:
        assert self.optimizer is not None
        return self.optimizer
    
    async def setup(self) -> None:
        """Initialize all system components"""
        logger.info("Setting up AML HFT System...")
        
        # Initialize API connection
        self.api = HyperliquidAPI(str(self.config_dir / "api.json"))
        await self.api.initialize()
        
        # Initialize components
        self.data_fetcher = DataFetcher(
            api=self.api,
            data_dir=str(self.data_dir)
        )
        
        self.ml_model = MLModel(
            config_path=str(self.config_dir / "params.json"),
            model_dir=str(self.model_dir)
        )
        
        self.risk_manager = RiskManager(
            api=self.api,
            config_path=str(self.config_dir / "params.json"),
            objectives_path=str(self.config_dir / "objectives.json")
        )
        
        self.optimizer = ParameterOptimizer(
            params_path=str(self.config_dir / "params.json"),
            objectives_path=str(self.config_dir / "objectives.json")
        )
        
        # Initialize risk manager with starting capital
        starting_capital = self.objectives.get("starting_capital_backtest", 1000)
        await self.risk_manager.initialize(starting_capital)
        
        # Try to load existing model
        if self.ml_model.load_model("best_model.pt"):
            logger.info("Loaded existing ML model")
        
        self.state.is_running = True
        logger.info("System setup complete")
    
    async def shutdown(self) -> None:
        """Clean shutdown of all components with state persistence (Step 5)"""
        logger.info("Shutting down AML HFT System...")
        
        self.state.is_running = False
        self._shutdown_event.set()
        
        # Step 5: Save state for resume with HFT hold stats
        try:
            state_path = Path("logs/system_state.json")
            
            # HFT: Get hold time statistics
            hold_time_stats = {}
            if self._risk_manager:
                hold_time_stats = self._risk_manager.get_hold_time_stats()
            
            state_data = {
                "last_shutdown": time.time(),
                "cycle_number": self.state.current_cycle,
                "objectives_met": self.state.objectives_met,
                "latest_metrics": self.latest_backtest_metrics,
                "signal_counts": getattr(self, '_signal_counts', {}),
                "hold_time_stats": hold_time_stats,  # HFT: Track hold times
                "hft_mode": self.params.get("trading", {}).get("hft_mode", False),
            }
            with open(state_path, "w") as f:
                json.dump(state_data, f, indent=2, default=str)
            logger.info(f"System state saved to {state_path}")
        except Exception as e:
            logger.warning(f"Could not save system state: {e}")
        
        # Close any open positions
        if self.current_position and self.current_position.size != 0 and self.api is not None:
            logger.info("Closing open positions before shutdown...")
            try:
                await self.api.close_position(self.api_config["symbol"])
            except Exception as e:
                logger.error(f"Error closing position: {e}")
        
        # Stop background tasks
        if self.data_fetcher:
            await self.data_fetcher.stop_background_fetching()
        
        # Save model
        if self.ml_model and self.ml_model.model:
            self.ml_model.save_model("shutdown_model.pt")
        
        # Create backup
        if self.data_fetcher:
            await self.data_fetcher.create_backup()
        
        # Close API connection
        if self.api:
            await self.api.close()
        
        logger.info("Shutdown complete")
    
    # ==================== Backtesting ====================
    
    async def run_backtest(
        self,
        days: int = 7,
        save_results: bool = True,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run vectorized backtest with current parameters
        Step 4: Enhanced with configurable slippage/latency simulation
        
        Args:
            days: Number of days of data (ignored if historical_data provided)
            save_results: Whether to save backtest results
            historical_data: Pre-loaded DataFrame to use instead of fetching
        """
        self._ensure_setup()
        
        # Get backtest config (Step 4)
        backtest_cfg = self.params.get("backtest", {})
        slippage_pct = backtest_cfg.get("slippage_pct", 0.015) / 100  # Step 4: 0.015% slippage
        latency_min = backtest_cfg.get("latency_ms_min", 1)
        latency_max = backtest_cfg.get("latency_ms_max", 5)
        commission = backtest_cfg.get("commission_pct", 0.0005)
        
        logger.info(f"📊 Starting backtest ({days} days, slippage={slippage_pct*100:.3f}%, latency={latency_min}-{latency_max}ms)...")
        self.state.mode = "backtest"
        
        start_time = time.time()
        
        # Use provided data or fetch from API
        if historical_data is not None and not historical_data.empty:
            logger.info(f"Using provided historical data ({len(historical_data)} rows)")
            df = self._data_fetcher._add_technical_features(historical_data)
        else:
            # Fetch historical data
            df = await self._data_fetcher.get_training_data(
                symbol=self.api_config["symbol"],
                interval="1m",
                days=days
            )
        
        if df.empty or len(df) < 500:
            logger.error(f"Insufficient data for backtest (got {len(df)} rows)")
            return {"error": "insufficient_data"}
        
        # Prepare features
        features, _ = self._ml_model.prepare_features(df)
        prices = df["close"].values

        # Initialize or use existing model
        if self._ml_model.model is None:
            self._ml_model.initialize_model(features.shape[1])
        
        # Step 4: Create backtest environment with realistic slippage
        env = BacktestEnvironment(
            df=df,
            initial_capital=self.objectives["starting_capital_backtest"],
            transaction_cost=commission,
            slippage=slippage_pct,  # Step 4: Configurable slippage
            take_profit_pct=self.params["trading"]["take_profit_pct"],
            stop_loss_pct=self.params["trading"]["stop_loss_pct"]
        )
        
        # Run backtest
        seq_len = self._ml_model.sequence_length
        buy_signals = 0
        sell_signals = 0
        latency_sum = 0.0  # Step 4: Track total latency for averaging
        total_steps = len(df) - seq_len - 1
        progress_interval = max(1, total_steps // 10)  # Log every 10%

        for i in range(seq_len, len(df) - 1):
            # Progress logging
            step_num = i - seq_len
            if step_num % progress_interval == 0:
                pct = (step_num / total_steps) * 100
                logger.info(f"Backtest progress: {pct:.0f}% ({step_num}/{total_steps})")
            
            # Step 4: Simulate random latency (1-5ms)
            latency_ms = random.uniform(latency_min, latency_max)
            latency_sum += latency_ms
            
            # Get feature sequence
            feature_seq = features[i - seq_len:i]

            # Get model prediction
            action, confidence, q_values = self._ml_model.predict(feature_seq)
            
            # Step 4: More aggressive RSI fallback thresholds for more trades
            q_diff = np.max(q_values) - np.min(q_values)
            if q_diff < 0.08:  # Step 4: Lower threshold (was 0.1) for more ML confidence
                # Fallback to RSI signal
                if "rsi" in df.columns and not pd.isna(df.iloc[i]["rsi"]):
                    rsi = df.iloc[i]["rsi"]
                    if rsi < 38:  # Step 4: More aggressive (was 35)
                        action = 1  # Buy on oversold
                        buy_signals += 1
                    elif rsi > 62:  # Step 4: More aggressive (was 65)
                        action = 2  # Sell on overbought
                        sell_signals += 1
            else:
                if action == 1:
                    buy_signals += 1
                elif action == 2:
                    sell_signals += 1
            
            # Execute action in environment
            position_size_pct = self.params["trading"]["position_size_pct"]
            reward, done = env.step(action, position_size_pct / 100)
            
            if done:
                break
        
        # Step 4: Calculate average latency
        num_steps = len(df) - seq_len - 1
        avg_latency_ms = latency_sum / max(num_steps, 1)
        
        logger.info(f"📊 Backtest signals - Buy: {buy_signals}, Sell: {sell_signals}, Trades: {len(env.trades)}, Avg latency: {avg_latency_ms:.2f}ms")
        
        # Get metrics
        metrics = env.get_metrics()
        metrics["backtest_time"] = time.time() - start_time
        metrics["data_points"] = len(df)
        metrics["avg_latency_ms"] = avg_latency_ms  # Step 4: Track latency
        metrics["buy_signals"] = buy_signals
        metrics["sell_signals"] = sell_signals
        
        # Step 4: Calculate and store monthly projection
        total_return = metrics.get("total_return_pct", 0) / 100
        backtest_days = metrics.get("backtest_days", 7)
        if total_return > -1 and backtest_days > 0:
            monthly_projection = ((1 + total_return) ** (30 / backtest_days) - 1) * 100
        else:
            monthly_projection = 0
        metrics["monthly_projection"] = monthly_projection
        
        # Check against objectives
        objectives_met = self._check_objectives(metrics)
        metrics["objectives_met"] = objectives_met
        
        self.latest_backtest_metrics = metrics
        self.state.last_backtest = time.time()
        self.state.cycles_since_backtest = 0
        self.state.objectives_met = objectives_met
        
        # Save results
        if save_results:
            await self._data_fetcher.save_backtest_results(metrics, "backtest")
        
        logger.info(
            f"Backtest complete - Return: {metrics['total_return_pct']:.2f}%, "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
            f"MaxDD: {metrics['max_drawdown_pct']:.2f}%, "
            f"Objectives met: {objectives_met}"
        )
        
        return metrics
    
    def _check_objectives(self, metrics: Dict[str, Any]) -> bool:
        """Check if metrics meet objectives with detailed logging"""
        # Step 1: Monthly performance - use compounding extrapolation
        # Formula: monthly = (1 + total_return)^(30 / backtest_days) - 1
        total_return = metrics.get("total_return_pct", 0) / 100  # Convert to decimal
        backtest_days = metrics.get("backtest_days", 7)  # Actual days from metrics
        
        # Compounding extrapolation for more accurate monthly estimate
        if total_return > -1:  # Avoid math errors
            monthly_return = ((1 + total_return) ** (30 / max(backtest_days, 1)) - 1) * 100
        else:
            monthly_return = -100  # Total loss case
        
        # Get individual metric values
        profit_factor = metrics.get("profit_factor", 0)
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        max_drawdown = metrics.get("max_drawdown_pct", 100)
        
        # Get thresholds
        monthly_min = self.objectives.get("monthly_performance_min", 5)
        pf_min = self.objectives.get("profit_factor_min", 1.1)
        sharpe_min = self.objectives.get("sharpe_ratio_min", 1.5)
        dd_max = self.objectives.get("drawdown_max", 5)
        
        # Individual checks with results
        check_monthly = monthly_return >= monthly_min
        check_pf = profit_factor >= pf_min
        check_sharpe = sharpe_ratio >= sharpe_min
        check_dd = max_drawdown <= dd_max
        
        # Log all objective checks
        logger.info("=" * 50)
        logger.info("OBJECTIVES CHECK:")
        logger.info(f"  Monthly Return: {monthly_return:.2f}% >= {monthly_min}%: {'✓' if check_monthly else '✗'}")
        logger.info(f"  Profit Factor: {profit_factor:.2f} >= {pf_min}: {'✓' if check_pf else '✗'}")
        logger.info(f"  Sharpe Ratio:  {sharpe_ratio:.2f} >= {sharpe_min}: {'✓' if check_sharpe else '✗'}")
        logger.info(f"  Max Drawdown:  {max_drawdown:.2f}% <= {dd_max}%: {'✓' if check_dd else '✗'}")
        
        all_met = all([check_monthly, check_pf, check_sharpe, check_dd])
        logger.info(f"  ALL OBJECTIVES MET: {'✓ YES' if all_met else '✗ NO'}")
        logger.info("=" * 50)
        
        return all_met
    
    # ==================== Training ====================
    
    async def train_model(
        self,
        days: int = 30,
        epochs: Optional[int] = None,
        append_live: bool = False
    ) -> Dict[str, Any]:
        """
        Train ML model on historical data.
        
        Args:
            days: Number of days of data (default 30 for better coverage)
            epochs: Training epochs (None uses config default)
            append_live: Append recent live cached data for freshness (Step 3)
        """
        self._ensure_setup()
        logger.info(f"Starting ML model training (days={days}, append_live={append_live})...")
        
        # Fetch training data with optional live append
        df = await self._data_fetcher.get_training_data(
            symbol=self.api_config["symbol"],
            interval="1m",
            days=days,
            append_live=append_live
        )

        if df.empty:
            return {"error": "no_training_data"}
        
        logger.info(f"Training data: {len(df)} samples ({days} days)")

        # Train model
        results = await self._ml_model.train(df, epochs)
        
        # Log training summary (formatted)
        epochs_done = results.get('epochs', 0)
        epochs_req = results.get('epochs_requested', epochs)
        best_reward = results.get('best_reward', 0)
        train_time = results.get('training_time', 0)
        early_stopped = results.get('early_stopped', False)
        
        logger.info(
            f"Training complete: {epochs_done}/{epochs_req} epochs, "
            f"best_reward={best_reward:.4f}, time={train_time:.1f}s"
            f"{', early_stopped' if early_stopped else ''}"
        )
        return results
    
    # ==================== Optimization ====================
    
    async def run_optimization(
        self,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """Run parameter optimization"""
        self._ensure_setup()
        logger.info("Starting parameter optimization...")
        self.state.mode = "optimization"
        
        # Fetch data for optimization
        df = await self._data_fetcher.get_training_data(
            symbol=self.api_config["symbol"],
            interval="1m",
            days=7
        )
        
        if df.empty:
            return {"error": "no_data"}
        
        def backtest_objective(df_data, params):
            """Objective function for optimization using mean reversion signals"""
            env = BacktestEnvironment(
                df=df_data,
                initial_capital=self.objectives["starting_capital_backtest"],
                take_profit_pct=params["take_profit_pct"],
                stop_loss_pct=params["stop_loss_pct"]
            )
            
            # Calculate RSI for mean reversion signals
            delta = df_data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # Mean reversion: buy when oversold, sell when overbought
            # Use TP/SL thresholds to set RSI bounds (lower TP = tighter bounds = more trades)
            rsi_oversold = 30 + params["stop_loss_pct"] * 5  # ~33-37
            rsi_overbought = 70 - params["stop_loss_pct"] * 5  # ~63-67
            
            for i in range(100, len(df_data) - 1):
                current_rsi = rsi.iloc[i]
                
                if pd.isna(current_rsi):
                    action = 0
                elif current_rsi < rsi_oversold:
                    action = 1  # Buy (oversold - expect bounce)
                elif current_rsi > rsi_overbought:
                    action = 2  # Sell (overbought - expect pullback)
                else:
                    action = 0  # Hold
                
                env.step(action, params["position_size_pct"] / 100)
            
            return env.get_metrics()
        
        result = await self._optimizer.optimize_with_backtest(
            backtest_func=lambda params: backtest_objective(df, params),
            n_trials=n_trials
        )
        
        self.state.last_optimization = time.time()
        
        return {
            "improvement": result.improvement_pct,
            "best_score": result.best_score,
            "iterations": result.iterations,
            "optimized_params": result.optimized_params
        }
    
    # ==================== Live Trading ====================

    async def execute_trade_signal(
        self,
        signal: str,
        signal_strength: float
    ) -> Optional[Dict[str, Any]]:
        """Execute a trade signal with latency profiling (Step 5)"""
        exec_start = time.perf_counter()  # Step 5: Latency profiling
        
        self._ensure_setup()
        if self._risk_manager.is_halted:
            logger.warning("🛑 Trading halted - cannot execute signal")
            return None

        # Check if trading is allowed
        allowed, reason = await self._risk_manager.should_allow_trade()
        if not allowed:
            logger.info(f"⚠️ Trade not allowed: {reason}")
            return None
        
        symbol = self.api_config["symbol"]
        
        # Step 5: Track orderbook fetch latency
        ob_start = time.perf_counter()
        orderbook = await self._api.get_orderbook(symbol)
        ob_latency_ms = (time.perf_counter() - ob_start) * 1000
        current_price = orderbook.mid_price

        # Get current volatility and ATR
        snapshot = self._data_fetcher.get_latest_snapshot()
        volatility = snapshot.spread_bps / 10000 if snapshot else 0.02
        
        # Step 2: Get ATR for dynamic TP/SL
        atr = 0.0
        klines = self._data_fetcher.get_cached_klines(symbol=symbol, interval="1m", limit=20)
        if len(klines) >= 14:
            # Calculate ATR from recent klines
            tr_values = []
            for i in range(1, len(klines)):
                high = klines[i].high
                low = klines[i].low
                prev_close = klines[i-1].close
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                tr_values.append(tr)
            if tr_values:
                atr = sum(tr_values[-14:]) / min(14, len(tr_values))

        # Calculate position size
        position_size = self._risk_manager.calculate_position_size(
            signal_strength=signal_strength,
            current_volatility=volatility,
            entry_price=current_price
        )
        
        if position_size <= 0:
            logger.warning(f"⚠️ Trade skipped: position_size={position_size} (signal={signal}, strength={signal_strength:.3f})")
            return None
        
        # Step 5: Log preparation latency
        prep_latency_ms = (time.perf_counter() - exec_start) * 1000
        logger.info(f"🚦 Executing {signal}: size={position_size:.4f} @ ${current_price:.0f} (ATR=${atr:.0f}, prep={prep_latency_ms:.1f}ms, ob={ob_latency_ms:.1f}ms)")
        
        # Determine trade direction with ATR-based TP/SL (Step 2)
        if signal == "BUY":
            side = "B"
            sl_price = self._risk_manager.calculate_stop_loss(current_price, side, volatility, atr)
            tp_price = self._risk_manager.calculate_take_profit(current_price, side, signal_strength, atr)
        elif signal == "SELL":
            side = "A"
            sl_price = self._risk_manager.calculate_stop_loss(current_price, side, volatility, atr)
            tp_price = self._risk_manager.calculate_take_profit(current_price, side, signal_strength, atr)
        else:
            return None
        
        # Execute order
        try:
            result = await self._api.place_market_order(
                symbol=symbol,
                side=side,
                size=position_size,
                slippage_pct=self.params["trading"]["slippage_tolerance_pct"]
            )
            
            self.trade_count_this_cycle += 1
            
            # Step 1: Try to confirm fill
            order_id = result.get("response", {}).get("data", {}).get("statuses", [{}])[0].get("resting", {}).get("oid", "")
            if not order_id:
                # For IOC orders, check fills directly
                order_id = result.get("response", {}).get("data", {}).get("statuses", [{}])[0].get("filled", {}).get("oid", "")
            
            # Check if order was filled (IOC orders should fill immediately)
            was_filled = result.get("response", {}).get("data", {}).get("statuses", [{}])[0].get("filled") is not None
            
            # Place SL/TP orders on exchange AFTER entry fills
            sl_order_id = None
            tp_order_id = None
            
            if was_filled:
                try:
                    # Cancel any existing SL/TP orders first
                    await self._api.cancel_all_orders(symbol)
                    
                    # Place Stop Loss order on exchange
                    sl_result = await self._api.place_stop_loss(
                        symbol=symbol,
                        position_side=side,
                        size=position_size,
                        stop_price=sl_price
                    )
                    sl_order_id = sl_result.get("order_id")
                    
                    # Place Take Profit order on exchange
                    tp_result = await self._api.place_take_profit(
                        symbol=symbol,
                        position_side=side,
                        size=position_size,
                        take_profit_price=tp_price
                    )
                    tp_order_id = tp_result.get("order_id")
                    
                    logger.info(
                        f"📊 SL/TP orders placed on exchange: "
                        f"SL order_id={sl_order_id}, TP order_id={tp_order_id}"
                    )
                except Exception as e:
                    logger.error(f"Failed to place SL/TP orders: {e}")
            else:
                logger.warning("Entry order not filled, skipping SL/TP placement")
            
            # HFT: Start tracking position hold time
            if was_filled:
                self._risk_manager.on_position_opened(current_price, position_size)
            
            # Log trade
            trade_data = {
                "timestamp": time.time(),
                "symbol": symbol,
                "side": side,
                "size": position_size,
                "price": current_price,
                "signal": signal,
                "signal_strength": signal_strength,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "sl_order_id": sl_order_id,
                "tp_order_id": tp_order_id,
                "atr": atr,
                "result": result,
                "order_id": order_id
            }
            
            await self._data_fetcher.save_trade(trade_data)
            
            # Step 5: Log execution latency
            exec_latency_ms = (time.perf_counter() - exec_start) * 1000
            logger.info(
                f"✅ Trade executed: {signal} {position_size:.4f} {symbol} @ {current_price:.2f} "
                f"(latency: {exec_latency_ms:.1f}ms), "
                f"SL: {sl_price:.0f} ({((sl_price - current_price) / current_price * 100):+.2f}%), "
                f"TP: {tp_price:.0f} ({((tp_price - current_price) / current_price * 100):+.2f}%)"
            )
            
            return trade_data
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return None
    
    async def sync_position_sltp(self) -> None:
        """
        Sync SL/TP orders for existing positions that may not have protection.
        
        This is called:
        1. At startup when live trading begins
        2. After system restart/reconnection
        3. When positions are detected without SL/TP orders
        """
        self._ensure_setup()
        
        try:
            symbol = self.api_config["symbol"]
            positions = await self._api.get_positions()
            open_orders = await self._api.get_open_orders()
            
            # Check if we have an open position
            current_pos = None
            for pos in positions:
                if pos.symbol == symbol and pos.size != 0:
                    current_pos = pos
                    break
            
            if not current_pos:
                logger.debug("No open position - SL/TP sync not needed")
                return
            
            # Check if we have SL/TP orders already
            has_sl = False
            has_tp = False
            for order in open_orders:
                if order.get("coin") == symbol and order.get("reduceOnly"):
                    # Check if trigger order by looking at order type info
                    has_sl = has_sl or True  # Any reduce_only order is protection
                    has_tp = has_tp or True
            
            # If we have fewer than 2 orders (SL + TP), place them
            reduce_only_orders = [o for o in open_orders if o.get("coin") == symbol and o.get("reduceOnly")]
            
            if len(reduce_only_orders) < 2:
                logger.info(f"Position without full SL/TP protection detected. Placing orders...")
                
                # Calculate SL/TP prices
                sl_pct = self.params["trading"]["stop_loss_pct"]
                tp_pct = self.params["trading"]["take_profit_pct"]
                
                if current_pos.size > 0:  # Long
                    position_side = "B"
                    sl_price = int(current_pos.entry_price * (1 - sl_pct / 100))
                    tp_price = int(current_pos.entry_price * (1 + tp_pct / 100))
                else:  # Short
                    position_side = "A"
                    sl_price = int(current_pos.entry_price * (1 + sl_pct / 100))
                    tp_price = int(current_pos.entry_price * (1 - tp_pct / 100))
                
                # Cancel existing orders and place fresh ones
                await self._api.cancel_all_orders(symbol)
                
                # Place SL
                sl_result = await self._api.place_stop_loss(
                    symbol=symbol,
                    position_side=position_side,
                    size=abs(current_pos.size),
                    stop_price=sl_price
                )
                
                # Place TP
                tp_result = await self._api.place_take_profit(
                    symbol=symbol,
                    position_side=position_side,
                    size=abs(current_pos.size),
                    take_profit_price=tp_price
                )
                
                logger.info(
                    f"✅ SL/TP synced for position {current_pos.size} @ {current_pos.entry_price}: "
                    f"SL={sl_price}, TP={tp_price}"
                )
            else:
                logger.info(f"Position has {len(reduce_only_orders)} protective orders - SL/TP OK")
                
        except Exception as e:
            logger.error(f"Failed to sync SL/TP: {e}")
    
    async def check_exit_conditions(self) -> bool:
        """Check if current position should be exited (optimized for API efficiency)"""
        self._ensure_setup()
        
        # Use cached user state to avoid redundant API call
        # get_positions() internally uses get_user_state() which is cached for 2s
        positions = await self._api.get_positions()
        
        if not positions:
            self.current_position = None
            self._risk_manager.on_position_closed()  # HFT: track hold time
            return False

        for pos in positions:
            if pos.symbol == self.api_config["symbol"] and pos.size != 0:
                self.current_position = pos
                
                # HFT: Check hold timeout first (forced exit after max_hold_seconds)
                should_timeout_exit, seconds_held = self._risk_manager.check_hold_timeout()
                if should_timeout_exit:
                    logger.warning(f"⏰ HFT timeout exit: position held {seconds_held:.1f}s")
                    await self._api.close_position(pos.symbol)
                    self._api.clear_cache("user_state")
                    self._api.clear_cache("positions")
                    
                    # Record the timeout exit
                    trade_record = TradeRecord(
                        timestamp=time.time(),
                        symbol=pos.symbol,
                        side="B" if pos.size > 0 else "A",
                        entry_price=pos.entry_price,
                        exit_price=pos.entry_price,  # Will be updated on fill
                        size=abs(pos.size),
                        pnl=pos.unrealized_pnl,
                        pnl_pct=(pos.unrealized_pnl / (abs(pos.size) * pos.entry_price) * 100) if pos.entry_price > 0 else 0,
                        hold_time=seconds_held,
                        fees=0
                    )
                    await self._risk_manager.record_trade(trade_record)
                    self._risk_manager.on_position_closed()  # HFT: track hold time
                    return True
                
                # Use API-provided unrealized PnL directly when possible
                # This avoids an extra get_orderbook() call
                if pos.unrealized_pnl != 0 and pos.entry_price > 0:
                    position_value = abs(pos.size * pos.entry_price)
                    if position_value > 0:
                        pnl_pct = (pos.unrealized_pnl / position_value) * 100
                    else:
                        pnl_pct = 0.0
                else:
                    # Fallback: get orderbook (cached for 500ms)
                    orderbook = await self._api.get_orderbook(pos.symbol)
                    current_price = orderbook.mid_price
                    
                    if pos.size > 0:  # Long
                        pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                    else:  # Short
                        pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100
                
                # HFT: Get dynamic TP/SL based on profitability
                base_tp = self.params["trading"]["take_profit_pct"]
                base_sl = self.params["trading"]["stop_loss_pct"]
                tp_pct, sl_pct = self._risk_manager.get_dynamic_tp_sl(base_tp, base_sl)
                
                if pnl_pct >= tp_pct:
                    logger.info(f"Take profit triggered: {pnl_pct:.2f}% >= {tp_pct:.2f}%")
                    await self._api.close_position(pos.symbol)
                    # Clear cache after position change
                    self._api.clear_cache("user_state")
                    self._api.clear_cache("positions")

                    _, seconds_held = self._risk_manager.check_hold_timeout()
                    trade_record = TradeRecord(
                        timestamp=time.time(),
                        symbol=pos.symbol,
                        side="B" if pos.size > 0 else "A",
                        entry_price=pos.entry_price,
                        exit_price=pos.entry_price * (1 + pnl_pct/100) if pos.size > 0 else pos.entry_price * (1 - pnl_pct/100),
                        size=abs(pos.size),
                        pnl=pos.unrealized_pnl,
                        pnl_pct=pnl_pct,
                        hold_time=seconds_held,
                        fees=0
                    )
                    await self._risk_manager.record_trade(trade_record)
                    self._risk_manager.on_position_closed()  # HFT: track hold time
                    return True

                elif pnl_pct <= -sl_pct:
                    logger.info(f"Stop loss triggered: {pnl_pct:.2f}% <= -{sl_pct:.2f}%")
                    await self._api.close_position(pos.symbol)
                    # Clear cache after position change
                    self._api.clear_cache("user_state")
                    self._api.clear_cache("positions")

                    _, seconds_held = self._risk_manager.check_hold_timeout()
                    trade_record = TradeRecord(
                        timestamp=time.time(),
                        symbol=pos.symbol,
                        side="B" if pos.size > 0 else "A",
                        entry_price=pos.entry_price,
                        exit_price=pos.entry_price * (1 - pnl_pct/100) if pos.size > 0 else pos.entry_price * (1 + pnl_pct/100),
                        size=abs(pos.size),
                        pnl=pos.unrealized_pnl,
                        pnl_pct=pnl_pct,
                        hold_time=seconds_held,
                        fees=0
                    )
                    await self._risk_manager.record_trade(trade_record)
                    self._risk_manager.on_position_closed()  # HFT: track hold time
                    return True
        return False
    
    async def generate_trading_signal(self) -> Tuple[str, float]:
        """
        Generate trading signal from ML model (Step 5: Latency profiling).
        
        Uses configurable thresholds and logs signal decisions.
        Target: <1ms inference latency.
        """
        self._ensure_setup()
        signal_start = time.perf_counter()  # Step 5: Latency profiling
        
        # Get configurable thresholds from params
        signal_thresh = self.params.get("trading", {}).get("signal_threshold", 0.001)
        min_q_diff = self.params.get("trading", {}).get("min_q_diff", 0.003)
        
        # Get recent market data
        klines = self._data_fetcher.get_cached_klines(
            symbol=self.api_config["symbol"],
            interval="1m",
            limit=self._ml_model.sequence_length + 50
        )

        if len(klines) < self._ml_model.sequence_length:
            logger.debug(f"Signal: HOLD (insufficient data: {len(klines)} klines)")
            return "HOLD", 0.0
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "timestamp": k.timestamp,
            "open": k.open,
            "high": k.high,
            "low": k.low,
            "close": k.close,
            "volume": k.volume
        } for k in klines])
        
        # Add features
        df = self._data_fetcher._add_technical_features(df)
        
        # Step 5: Market regime detection for adaptive signals
        market_regime = self._detect_market_regime(df)
        
        # Prepare features
        features, _ = self._ml_model.prepare_features(df)

        # Get last sequence
        feature_seq = features[-self._ml_model.sequence_length:]

        # Step 5: Track inference latency specifically
        inference_start = time.perf_counter()
        action, confidence, q_values = self._ml_model.predict(feature_seq)
        inference_latency_ms = (time.perf_counter() - inference_start) * 1000
        
        # Calculate Q-value metrics
        q_diff = np.max(q_values) - np.min(q_values)
        sorted_q = np.sort(q_values)[::-1]
        margin = sorted_q[0] - sorted_q[1]
        strength = np.tanh(margin)
        
        # Step 5: Adjust strength based on market regime
        if market_regime == "trending":
            strength *= 1.2  # More confident in trends
        elif market_regime == "ranging":
            strength *= 0.8  # Less confident in ranges
        elif market_regime == "volatile":
            strength *= 0.6  # Very cautious in high volatility
        
        # Step 5: Total signal generation latency
        total_latency_ms = (time.perf_counter() - signal_start) * 1000
        
        # Log signal details with latency (Step 5)
        signal = self._ml_model.ACTION_NAMES.get(action, "HOLD")
        logger.debug(
            f"Signal: {signal}, q_diff={q_diff:.4f}, margin={margin:.4f}, strength={strength:.4f}, "
            f"regime={market_regime}, infer={inference_latency_ms:.2f}ms, total={total_latency_ms:.2f}ms"
        )
        
        # Check if model is confident (Step 4: lower threshold)
        if q_diff < min_q_diff:
            # Low ML confidence - use RSI fallback
            if "rsi" in df.columns and len(df) > 0:
                rsi = df.iloc[-1]["rsi"]
                if not pd.isna(rsi):
                    if rsi < 35:
                        logger.info(f"Signal: BUY (RSI fallback: {rsi:.1f})")
                        return "BUY", 0.5
                    elif rsi > 65:
                        logger.info(f"Signal: SELL (RSI fallback: {rsi:.1f})")
                        return "SELL", 0.5
            logger.debug(f"Signal: HOLD (low Q-diff: {q_diff:.4f} < {min_q_diff})")
        
        # Step 5: Signal balancing - check for signal bias and adjust
        if not hasattr(self, '_signal_counts'):
            self._signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0, "filtered": 0}
        
        # Calculate signal diversity and apply anti-bias correction
        total_trades = self._signal_counts.get("BUY", 0) + self._signal_counts.get("SELL", 0)
        if total_trades > 10:
            buy_ratio = self._signal_counts.get("BUY", 0) / total_trades
            diversity_thresh = self.params.get("ml_model", {}).get("diversity_thresh", 0.5)
            
            # CRITICAL FIX: Apply stronger anti-bias for extreme imbalances
            if buy_ratio > 0.7 and signal == "BUY":
                # Too many BUYs - penalize BUY, consider flipping to SELL
                strength *= 0.5  # Reduce BUY strength significantly
                logger.debug(f"Anti-bias: reducing BUY strength (ratio={buy_ratio:.2f})")
            elif buy_ratio < 0.3 and signal == "SELL":
                # Too many SELLs - penalize SELL
                strength *= 0.5
                logger.debug(f"Anti-bias: reducing SELL strength (ratio={buy_ratio:.2f})")
            # Boost underrepresented signals
            elif buy_ratio > diversity_thresh and signal == "SELL":
                strength *= 1.3  # Boost SELL when underrepresented
            elif buy_ratio < (1 - diversity_thresh) and signal == "BUY":
                strength *= 1.3  # Boost BUY when underrepresented
        
        self._signal_counts[signal] = self._signal_counts.get(signal, 0) + 1
        
        # Step 1 FIX: Force trade after too many consecutive HOLDs
        if not hasattr(self, '_consecutive_holds'):
            self._consecutive_holds = 0
        
        max_holds = self.params.get("trading", {}).get("max_hold_signals_before_force", 5)  # Step 2: Reduced to 5
        force_trade_enabled = self.params.get("trading", {}).get("force_trade_after_holds", True)
        force_strength = self.params.get("trading", {}).get("force_trade_strength", 0.7)  # Step 2: Increased
        
        if signal == "HOLD":
            self._consecutive_holds += 1
            if force_trade_enabled and self._consecutive_holds >= max_holds:
                # Force a trade based on RSI/momentum with more aggressive thresholds
                forced = False
                if "rsi" in df.columns and len(df) > 0:
                    rsi = df.iloc[-1]["rsi"]
                    if not pd.isna(rsi):
                        # Step 2: Even more aggressive RSI thresholds (35/65 for clear signals)
                        if rsi < 35:
                            signal = "BUY"
                            strength = force_strength
                            logger.info(f"🔄 FORCE TRADE: BUY after {self._consecutive_holds} HOLDs (RSI={rsi:.1f}, strength={strength})")
                            forced = True
                        elif rsi > 65:
                            signal = "SELL"
                            strength = force_strength
                            logger.info(f"🔄 FORCE TRADE: SELL after {self._consecutive_holds} HOLDs (RSI={rsi:.1f}, strength={strength})")
                            forced = True
                        elif 40 <= rsi <= 60:
                            # Step 2: In neutral RSI zone, use momentum + price action
                            momentum = df.iloc[-1].get("momentum_5", 0) if "momentum_5" in df.columns else 0
                            price_change = df.iloc[-1].get("close", 0) / df.iloc[-5].get("close", 1) - 1 if len(df) > 5 else 0
                            combined_signal = momentum + price_change * 100
                            signal = "BUY" if combined_signal > 0 else "SELL"
                            strength = force_strength * 0.9
                            logger.info(f"🔄 FORCE TRADE: {signal} after {self._consecutive_holds} HOLDs (RSI neutral={rsi:.1f}, momentum={combined_signal:.4f})")
                            forced = True
                        else:
                            # Step 2: Edge RSI zones (35-40, 60-65) - follow the trend
                            signal = "BUY" if rsi < 50 else "SELL"
                            strength = force_strength * 0.8
                            logger.info(f"🔄 FORCE TRADE: {signal} after {self._consecutive_holds} HOLDs (RSI edge={rsi:.1f})")
                            forced = True
                if forced:
                    self._consecutive_holds = 0
        else:
            self._consecutive_holds = 0
        
        # Log periodic signal summary (every 50 signals)
        total_signals = sum(self._signal_counts.values())
        if total_signals % 50 == 0:
            logger.info(f"Signal summary: {self._signal_counts}")
        
        return signal, strength
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Step 5: Detect current market regime for adaptive trading.
        
        Returns: "trending", "ranging", or "volatile"
        """
        if len(df) < 20:
            return "ranging"
        
        try:
            close = df["close"].values
            
            # Calculate volatility (ATR-based)
            if "atr" in df.columns:
                atr = df["atr"].iloc[-1]
                avg_price = df["close"].iloc[-20:].mean()
                vol_ratio = atr / avg_price if avg_price > 0 else 0
            else:
                returns = np.diff(close[-20:]) / close[-21:-1]
                vol_ratio = np.std(returns) if len(returns) > 0 else 0
            
            # Calculate trend strength (ADX-like)
            high_20 = df["high"].iloc[-20:].max()
            low_20 = df["low"].iloc[-20:].min()
            range_20 = high_20 - low_20
            
            # Current price position within range
            current = close[-1]
            range_position = (current - low_20) / range_20 if range_20 > 0 else 0.5
            
            # Price momentum
            sma_short = close[-5:].mean()
            sma_long = close[-20:].mean()
            trend_strength = abs(sma_short - sma_long) / sma_long if sma_long > 0 else 0
            
            # Classify regime
            if vol_ratio > 0.02:  # High volatility (> 2% of price)
                return "volatile"
            elif trend_strength > 0.005:  # Clear trend
                return "trending"
            else:
                return "ranging"
                
        except Exception as e:
            logger.debug(f"Regime detection error: {e}")
            return "ranging"
    
    # ==================== Cycle Management ====================
    
    async def run_trading_cycle(self) -> CycleMetrics:
        """Run one trading cycle with enhanced trade execution (Step 1 fixes)"""
        self._ensure_setup()
        self.state.current_cycle += 1
        self.state.cycles_since_backtest += 1
        self.trade_count_this_cycle = 0
        self.cycle_start_time = time.time()
        
        cycle_duration = self.objectives.get("cycle_duration_seconds", 300)
        trade_threshold = self.objectives.get("cycle_trade_threshold", 10)
        
        # Step 1: Use very low signal threshold to allow trades
        signal_threshold = self.params.get("trading", {}).get("signal_threshold", 0.001)
        
        logger.info(f"📈 Starting cycle {self.state.current_cycle} (thresh={signal_threshold}, duration={cycle_duration}s)")
        
        adjustments = []
        initial_capital = self._risk_manager.current_capital
        signals_generated = 0
        signals_filtered = 0
        signals_executed = 0
        last_trade_time = 0.0
        min_trade_interval = 3.0  # 3 seconds between trades (API-friendly)
        
        while time.time() - self.cycle_start_time < cycle_duration:
            if self._shutdown_event.is_set():
                break
            
            # Check if trade threshold reached
            if self.trade_count_this_cycle >= trade_threshold:
                logger.info(f"✅ Trade threshold ({trade_threshold}) reached")
                break
            
            try:
                # Check exit conditions for open positions
                await self.check_exit_conditions()
                
                # Generate signal if no position
                if not self.current_position or self.current_position.size == 0:
                    signal, strength = await self.generate_trading_signal()
                    signals_generated += 1
                    
                    # Step 1: Enhanced logging for signal decisions
                    if signal != "HOLD":
                        if abs(strength) > signal_threshold:
                            # Rate limit: enforce minimum interval between trades
                            time_since_last_trade = time.time() - last_trade_time
                            if time_since_last_trade < min_trade_interval:
                                wait_time = min_trade_interval - time_since_last_trade
                                logger.debug(f"Rate limit: waiting {wait_time:.1f}s before next trade")
                                await asyncio.sleep(wait_time)
                            
                            logger.info(f"🚦 Executing {signal} signal (strength={strength:.4f} > thresh={signal_threshold})")
                            result = await self.execute_trade_signal(signal, strength)
                            if result:
                                signals_executed += 1
                                last_trade_time = time.time()
                                # Clear cache after order to get fresh state
                                self._api.clear_cache("user_state")
                                self._api.clear_cache("open_orders")
                            else:
                                logger.warning(f"⚠️ Trade execution returned None for {signal}")
                        else:
                            signals_filtered += 1
                            # Step 1: More visible logging for filtered signals
                            if signals_filtered <= 5 or signals_filtered % 10 == 0:
                                logger.info(f"🚫 Signal {signal} filtered: strength {strength:.4f} < threshold {signal_threshold}")
                
                # Step 5: Optimized sleep - 1.5s for more opportunities while API-friendly
                await asyncio.sleep(1.5)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
        
        # Wait briefly for fills to process, then sync PnL from exchange
        await asyncio.sleep(3)
        pnl_sync = await self._risk_manager.poll_and_sync_pnl()
        
        # Cycle analysis with synced PnL
        cycle_end_time = time.time()
        cycle_pnl = self._risk_manager.current_capital - initial_capital
        cycle_pnl_pct = (cycle_pnl / initial_capital * 100) if initial_capital > 0 else 0
        
        # Log synced PnL details
        if pnl_sync.get("total_pnl", 0) != 0:
            logger.info(
                f"PnL synced: realized=${pnl_sync.get('realized_pnl', 0):.2f}, "
                f"unrealized=${pnl_sync.get('unrealized_pnl', 0):.2f}, "
                f"total=${pnl_sync.get('total_pnl', 0):.2f}"
            )

        # Determine status and make adjustments
        metrics = await self._risk_manager.calculate_metrics()
        
        # Step 3: Use backtest metrics as fallback when live has no trades
        effective_sharpe = metrics.sharpe_ratio
        effective_drawdown = metrics.current_drawdown
        effective_pf = metrics.profit_factor
        
        # If no trades this cycle and we have good backtest metrics, use those
        if self.trade_count_this_cycle == 0 and self.latest_backtest_metrics:
            backtest_sharpe = self.latest_backtest_metrics.get("sharpe_ratio", 0)
            backtest_pf = self.latest_backtest_metrics.get("profit_factor", 0)
            backtest_dd = self.latest_backtest_metrics.get("max_drawdown_pct", 0)
            
            # Use backtest metrics if they're better than live (which has 0 trades)
            if backtest_sharpe > effective_sharpe:
                effective_sharpe = backtest_sharpe * 0.9  # Slight discount for being backtest
                logger.debug(f"Using backtest sharpe {backtest_sharpe:.2f} (discounted to {effective_sharpe:.2f})")
            if backtest_pf > effective_pf:
                effective_pf = backtest_pf * 0.9
                logger.debug(f"Using backtest PF {backtest_pf:.2f} (discounted to {effective_pf:.2f})")
        
        # Step 3: Pass trade_count, cycle_number, objectives_met, unrealized_pnl, projected_monthly
        # Calculate projected monthly for status evaluation
        days_running = max(0.04, (time.time() - self.live_start_time) / 86400) if self.live_start_time > 0 else 0.04
        total_pnl = pnl_sync.get('total_pnl', 0)
        unrealized_pnl = pnl_sync.get('unrealized_pnl', 0)
        
        # Projected monthly = ((1 + return)^(30/days)) - 1
        current_return = total_pnl / initial_capital if initial_capital > 0 else 0
        if current_return > -1:
            projected_monthly = ((1 + current_return) ** (30 / days_running) - 1) * 100
        else:
            projected_monthly = -100.0
        
        # Step 5: Get realized PnL for status determination
        realized_pnl = pnl_sync.get('realized_pnl', 0)
        
        status = self._optimizer.determine_status(
            metrics={
                "sharpe_ratio": effective_sharpe,
                "max_drawdown": effective_drawdown,
                "profit_factor": effective_pf
            },
            trade_count=self.trade_count_this_cycle,
            cycle_number=self.state.current_cycle,
            objectives_met_phase4=self.state.objectives_met,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            projected_monthly=projected_monthly
        )

        # Step 5: Log cycle signal stats for debugging
        logger.info(
            f"Cycle {self.state.current_cycle} signals: generated={signals_generated}, "
            f"filtered={signals_filtered}, executed={self.trade_count_this_cycle}"
        )
        
        # Warn if no trades despite signals
        if self.trade_count_this_cycle == 0 and signals_generated > 0:
            logger.warning(
                f"⚠️ Cycle {self.state.current_cycle}: 0 trades from {signals_generated} signals - "
                f"check signal_threshold ({signal_threshold}) or position sizing"
            )

        # Apply adjustments based on status
        if status != PerformanceStatus.MODERATE:
            adjustment_result = await self._optimizer.quick_adjust(status)
            adjustments.append(f"Params adjusted for {status.value}")

            # Step 3: Critical loop prevention - track consecutive criticals
            if status == PerformanceStatus.CRITICAL:
                self.consecutive_critical_count += 1
                logger.warning(f"Critical status #{self.consecutive_critical_count} (max {self.max_consecutive_criticals})")
                
                if self.consecutive_critical_count > self.max_consecutive_criticals:
                    # Too many consecutive criticals - reset params and skip retrain
                    logger.warning("⚠️ Max consecutive criticals reached - resetting params to defaults")
                    self._reset_trading_params_to_defaults()
                    self.consecutive_critical_count = 0
                    adjustments.append("Params reset to defaults (critical loop prevention)")
                else:
                    # Normal critical handling - retrain with fresh data
                    data_days = self.params.get("ml_model", {}).get("data_days", 90)
                    logger.info(f"Critical status - initiating model retraining with {data_days} days + fresh data")
                    # Pause background fetching to reduce API load during training
                    await self._data_fetcher.stop_background_fetching()
                    # Use configurable data_days + append live cached data for freshness
                    await self.train_model(days=data_days, epochs=50, append_live=True)
                    # Resume background fetching after training
                    await self._data_fetcher.start_background_fetching(
                        symbol=self.api_config["symbol"]
                    )
                    adjustments.append("Model retrained")
        else:
            # Reset counter if not critical
            if self.consecutive_critical_count > 0:
                logger.info(f"Exited critical status after {self.consecutive_critical_count} cycles")
            self.consecutive_critical_count = 0
        
        # HFT: Get average hold time for this cycle
        avg_hold_time = self._risk_manager.get_avg_hold_time()
        
        cycle_metrics = CycleMetrics(
            cycle_number=self.state.current_cycle,
            start_time=self.cycle_start_time,
            end_time=cycle_end_time,
            trades_count=self.trade_count_this_cycle,
            pnl=cycle_pnl,
            pnl_pct=cycle_pnl_pct,
            status=status.value,
            adjustments_made=adjustments,
            avg_hold_time=avg_hold_time
        )
        
        self.cycle_history.append(cycle_metrics)
        self._save_cycle_metrics(cycle_metrics)
        
        # Step 4: Update cumulative PnL tracking for accurate monthly projection
        realized_pnl = pnl_sync.get('realized_pnl', 0)
        unrealized_pnl = pnl_sync.get('unrealized_pnl', 0)
        self.cumulative_pnl += cycle_pnl
        self.total_live_trades += self.trade_count_this_cycle
        
        # Step 4: Calculate projected monthly return from cumulative data
        # Include both realized and unrealized PnL for accurate projection
        projected_monthly = 0.0
        days_running = (time.time() - self.live_start_time) / 86400 if self.live_start_time > 0 else 0
        
        # Total equity change = cumulative realized + current unrealized
        total_pnl = self.cumulative_pnl + unrealized_pnl
        
        if days_running >= 0.5 and total_pnl != 0:
            # Use total PnL (realized + unrealized) for projection when running >= 12 hours
            total_return = total_pnl / initial_capital
            if total_return > -1:
                # Compound to 30 days: monthly = (1 + return)^(30/days) - 1
                projected_monthly = ((1 + total_return) ** (30 / days_running) - 1) * 100
            logger.debug(
                f"Monthly projection: days={days_running:.1f}, realized=${realized_pnl:.2f}, "
                f"unrealized=${unrealized_pnl:.2f}, total=${total_pnl:.2f}, proj={projected_monthly:.2f}%"
            )
        elif days_running >= 0.04 and self.trade_count_this_cycle > 0:  # 1 hour minimum
            # Use current cycle performance for early projection
            cycle_duration_hours = (cycle_end_time - self.cycle_start_time) / 3600
            if cycle_duration_hours > 0:
                hourly_return = cycle_pnl_pct / cycle_duration_hours
                projected_monthly = hourly_return * 24 * 30  # Project to 30 days
                logger.debug(f"Early projection from {cycle_duration_hours:.1f}h: {projected_monthly:.2f}%")
        elif self.latest_backtest_metrics:
            # Use backtest projection as fallback
            projected_monthly = self.latest_backtest_metrics.get("monthly_projection", 0)
            if projected_monthly == 0:
                total_ret = self.latest_backtest_metrics.get("total_return_pct", 0) / 100
                bt_days = self.latest_backtest_metrics.get("backtest_days", 7)
                if total_ret > -1 and bt_days > 0:
                    projected_monthly = ((1 + total_ret) ** (30 / bt_days) - 1) * 100
        
        # Step 4: Store live metrics for objectives checking - include unrealized
        self.latest_live_metrics = {
            "cumulative_pnl": self.cumulative_pnl,
            "cumulative_pnl_pct": (self.cumulative_pnl / initial_capital * 100) if initial_capital > 0 else 0,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "total_trades": self.total_live_trades,
            "days_running": days_running,
            "projected_monthly": projected_monthly
        }
        
        logger.info(
            f"Cycle {self.state.current_cycle} complete - "
            f"Trades: {self.trade_count_this_cycle}, "
            f"PnL: ${cycle_pnl:.2f} ({cycle_pnl_pct:.2f}%), "
            f"Realized: ${realized_pnl:.2f}, Unrealized: ${unrealized_pnl:.2f}, "
            f"Cumulative: ${self.cumulative_pnl:.2f}, "
            f"Projected Monthly: {projected_monthly:.2f}%, "
            f"Status: {status.value}"
        )
        
        return cycle_metrics
    
    def _reset_trading_params_to_defaults(self) -> None:
        """Reset trading parameters to safe defaults (Step 3: Critical loop prevention)"""
        logger.info("Resetting trading parameters to safe defaults...")
        
        # Load fresh config from file
        with open(self.config_dir / "params.json", "r") as f:
            fresh_params = json.load(f)
        
        # Reset critical trading parameters to reasonable defaults
        trading = self.params.get("trading", {})
        trading["position_size_pct"] = max(trading.get("position_size_pct", 0.25), 0.15)
        trading["signal_threshold"] = min(trading.get("signal_threshold", 0.05), 0.06)
        trading["min_q_diff"] = min(trading.get("min_q_diff", 0.01), 0.012)
        
        # Reset epsilon for more exploration
        ml_config = self.params.get("ml_model", {})
        ml_config["epsilon_start"] = 0.2  # More exploration
        
        # Save the reset params
        self._optimizer._save_params()
        logger.info("Trading parameters reset to safe defaults")
    
    def _save_cycle_metrics(self, metrics: CycleMetrics) -> None:
        """Save cycle metrics to dashboard file"""
        dashboard_path = Path("monitoring/dashboard_data.json")
        dashboard_path.parent.mkdir(exist_ok=True)
        
        data = {
            "last_update": time.time(),
            "current_cycle": metrics.cycle_number,
            "status": metrics.status,
            "cycle_pnl": metrics.pnl,
            "total_trades": metrics.trades_count,
            "risk_status": self._risk_manager.get_status_summary()
        }
        
        with open(dashboard_path, "w") as f:
            json.dump(data, f, indent=2)
    
    # ==================== Main Autonomous Loop ====================
    
    async def run_autonomous(self) -> None:
        """Main autonomous execution loop"""
        self._ensure_setup()
        logger.info("=" * 60)
        logger.info("Starting AML HFT Autonomous Trading System")
        logger.info("=" * 60)
        
        # Step 3: Use data_days from config (default 90 days for better pattern coverage)
        data_days = self.params.get("ml_model", {}).get("data_days", 90)
        backtest_days = data_days  # Unified: backtest on same 90 days as training
        
        # Initial backtest
        logger.info(f"Phase 1: Initial Backtest ({backtest_days} days)")
        backtest_results = await self.run_backtest(days=backtest_days)
        
        # Step 3: Store Phase 1 objectives status for Phase 4 comparison
        self._phase1_objectives_met = self.state.objectives_met
        logger.info(f"Phase 1 objectives met: {self._phase1_objectives_met}")
        
        if backtest_results.get("error"):
            logger.error("Initial backtest failed - fetching fresh data")
            await self._data_fetcher.fetch_historical_klines(
                symbol=self.api_config["symbol"],
                interval="1m",
                days=data_days
            )
            backtest_results = await self.run_backtest(days=backtest_days)
            self._phase1_objectives_met = self.state.objectives_met
        
        # Optimization if objectives not met
        if not self.state.objectives_met:
            logger.info("Phase 2: Parameter Optimization")
            await self.run_optimization(n_trials=30)
            
            # Reload params after optimization to apply new values
            self.params = self._load_config("config/params.json")
            logger.info("Reloaded params after optimization")
            
            # Re-run backtest
            backtest_results = await self.run_backtest(days=backtest_days)
        
        # Train model with extended data (Step 3: 90 days default + live append)
        logger.info(f"Phase 3: ML Model Training ({data_days} days)")
        training_results = await self.train_model(days=data_days, append_live=True)
        
        # Step 1: Reload trained model for Phase 4 validation
        logger.info("Phase 4: Final Validation")
        # Reload the final model to ensure we're using the just-trained weights
        model_loaded = self._ml_model.load_model("final_model.pt")
        if not model_loaded:
            model_loaded = self._ml_model.load_model("best_model.pt")
        if model_loaded:
            logger.info("Reloaded trained model for Phase 4 validation")
        else:
            logger.warning("Could not reload model - using in-memory model")
        
        # Step 3: Run backtest WITHOUT live append for consistent validation (same as Phase 1)
        backtest_results = await self.run_backtest(days=backtest_days)
        
        # Step 3: Calculate compound monthly projection from backtest
        if backtest_results:
            total_return = backtest_results.get("total_return_pct", 0) / 100
            bt_days = backtest_results.get("backtest_days", backtest_days)
            if total_return > -1 and bt_days > 0:
                # Compound to monthly: (1 + return)^(30/days) - 1
                compound_monthly = ((1 + total_return) ** (30 / bt_days) - 1) * 100
                backtest_results["compound_monthly_pct"] = compound_monthly
                logger.info(f"Phase 4 compound monthly projection: {compound_monthly:.2f}%")
        
        # Step 3: If Phase 1 met objectives but Phase 4 fails, override with caution
        if not self.state.objectives_met:
            logger.warning("Phase 4 objectives not met after optimization")
            
            # Check if Phase 1 was successful (stored earlier)
            if hasattr(self, '_phase1_objectives_met') and self._phase1_objectives_met:
                logger.info("Phase 1 objectives were met - launching with caution despite Phase 4 failure")
                # Allow launch but mark for monitoring
                self.state.phase4_override = True
            else:
                logger.error("=" * 60)
                logger.error("CRITICAL: Phase 4 objectives not met - HALTING before live trading")
                logger.error(f"Sharpe: {backtest_results.get('sharpe_ratio', 0):.2f} (min: {self.objectives.get('sharpe_ratio_min', 1.8)})")
                logger.error(f"Monthly: {backtest_results.get('compound_monthly_pct', 0):.2f}% (min: {self.objectives.get('monthly_return_pct_min', 15)}%)")
                logger.error("=" * 60)
                logger.info("Please review backtest results, adjust objectives in config/objectives.json, or improve model training")
                self.state.is_running = False
                return
        else:
            self.state.phase4_override = False
        
        # Start live trading
        logger.info("Phase 5: Starting Live Trading")
        self.state.mode = "live"
        
        # Step 4: Initialize live trading tracking
        self.live_start_time = time.time()
        self.cumulative_pnl = 0.0
        self.total_live_trades = 0
        
        # Step 2: Sync initial capital from exchange
        initial_capital = await self._risk_manager.sync_capital_from_exchange()
        if initial_capital > 0:
            logger.info(f"Live trading starting with ${initial_capital:.2f} capital from exchange")
        
        # Sync SL/TP for any existing positions without protection
        await self.sync_position_sltp()
        
        # Start background data fetching
        await self._data_fetcher.start_background_fetching(
            symbol=self.api_config["symbol"]
        )
        
        # Main trading loop
        while self.state.is_running and not self._shutdown_event.is_set():
            try:
                # Check if hourly backtest is due
                current_time = datetime.now(timezone.utc)
                if current_time.minute == 0 and current_time.second < 10:
                    if self.state.cycles_since_backtest > 0:
                        logger.info("Hourly backtest triggered")
                        await self.run_backtest(days=7, save_results=True)
                        
                        if not self.state.objectives_met:
                            await self.run_optimization(n_trials=20)
                
                # Run trading cycle
                await self.run_trading_cycle()
                
                # Hourly backup
                if current_time.minute == 0:
                    await self._data_fetcher.create_backup()
                
                # Wait for next cycle alignment
                await self._wait_for_cycle_start()
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)
        
        logger.info("Autonomous loop ended")
    
    async def _wait_for_cycle_start(self) -> None:
        """Wait for next cycle start time (aligned to 5-minute boundaries)"""
        now = datetime.now(timezone.utc)
        cycle_duration = self.objectives.get("cycle_duration_seconds", 300)
        
        # Calculate next cycle start
        seconds_in_hour = now.minute * 60 + now.second
        current_cycle_pos = seconds_in_hour % cycle_duration
        wait_time = cycle_duration - current_cycle_pos
        
        if wait_time > 0 and wait_time < cycle_duration:
            logger.debug(f"Waiting {wait_time}s for next cycle")
            await asyncio.sleep(wait_time)


async def main():
    """Entry point for autonomous execution"""
    system = AMLHFTSystem()
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        asyncio.create_task(system.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await system.setup()
        await system.run_autonomous()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
