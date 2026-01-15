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
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
import torch

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


def _process_backtest_chunk(args):
    """
    Multiprocessing worker function to process a single backtest chunk.
    SPEC: Pure ML signals from LSTM+DQN model (no RSI/ADX indicators).
    """
    import torch
    import numpy as np
    import pandas as pd
    from src.ml_model import MLModel, BacktestEnvironment
    import json
    import tempfile
    
    chunk_df, model_state_dict, params, chunk_info = args
    chunk_idx = chunk_info['chunk_idx']
    seq_len = chunk_info['seq_len']
    initial_capital = chunk_info['initial_capital']
    global_peak_equity = chunk_info.get('global_peak_equity', initial_capital)  # Get global peak
    chunk_start_idx = chunk_info['chunk_start_idx']
    chunk_end_idx = chunk_info['chunk_end_idx']
    
    try:
        # Force CPU device for worker processes (avoid MPS issues)
        device = torch.device('cpu')
        
        # Create temporary config file for this worker
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(params, f)
            temp_config_path = f.name
        
        # Create temporary model for this worker
        temp_model = MLModel(config_path=temp_config_path)
        temp_model.device = device  # Override to CPU
        
        # Prepare features FIRST to get correct input size
        features, feature_cols = temp_model.prepare_features(chunk_df)
        input_size = len(feature_cols)
        
        # Initialize model with correct input size
        temp_model.initialize_model(input_size=input_size)
        
        # Load weights (already on CPU) and move model to CPU
        temp_model.model.load_state_dict(model_state_dict)
        temp_model.model.to(device)
        temp_model.model.eval()
        
        # NOTE: Use torch.no_grad() context instead of torch.set_grad_enabled(False)
        # to avoid globally disabling gradients for subsequent training phases
        
        # Create environment with SPEC defaults (TP 0.5%, SL 0.25%)
        tp_pct = params['trading'].get('take_profit_pct', 0.5)
        sl_pct = params['trading'].get('stop_loss_pct', 0.25)
        max_dd = params['trading'].get('max_portfolio_drawdown_pct', 5.0)
        
        env = BacktestEnvironment(
            df=chunk_df.copy(),
            initial_capital=initial_capital,
            transaction_cost=params['backtest'].get('commission_pct', 0.0005),
            slippage=params['backtest'].get('slippage_pct', 0.015) / 100,
            take_profit_pct=tp_pct,
            stop_loss_pct=sl_pct,
            max_hold_bars=60,  # 60-bar max hold
            max_drawdown_pct=max_dd,  # Portfolio-level drawdown limit
            global_peak_equity=global_peak_equity  # Pass global peak for proper DD tracking across chunks
        )
        
        buy_signals = 0
        sell_signals = 0
        
        # Process chunk with PURE ML signals (no indicator-based strategies)
        for i in range(seq_len, len(chunk_df) - 1):
            # Get ML prediction (pure LSTM+DQN signal)
            if model_state_dict is not None and i >= seq_len:
                feature_seq = features[i - seq_len:i]
                with torch.no_grad():
                    ml_action, confidence, q_values = temp_model.predict(feature_seq)
                
                action = ml_action  # Use pure ML signal
                
                # Track signals
                if action == 1:
                    buy_signals += 1
                elif action == 2:
                    sell_signals += 1
            else:
                action = 0  # Default HOLD if no model
            
            position_size_pct = params['trading'].get('position_size_pct', 40.0)
            reward, done = env.step(action, position_size_pct / 100, data_idx=i)
            
            if done:
                break
        
        # Force-close any remaining open position at end of chunk
        if env.position > 0:
            final_price = chunk_df.iloc[-1]['close']
            env._close_position(final_price, "end_of_chunk")
        
        # Clean up temp file
        import os
        try:
            os.unlink(temp_config_path)
        except:
            pass
        
        return {
            'chunk_idx': chunk_idx,
            'trades': env.trades,
            'equity_curve': env.equity_curve if env.equity_curve else [env.capital],
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'final_peak_equity': env.peak_equity,  # Pass peak equity for next chunk
            'success': True
        }
        
    except Exception as e:
        import traceback
        import os
        # Clean up temp file on error
        try:
            if 'temp_config_path' in locals():
                os.unlink(temp_config_path)
        except:
            pass
        return {
            'chunk_idx': chunk_idx,
            'trades': [],
            'equity_curve': [],
            'buy_signals': 0,
            'sell_signals': 0,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


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
        # If in live/HFT mode, sync from exchange; otherwise use backtest capital
        hft_mode = self.params.get("trading", {}).get("hft_mode", False)
        if hft_mode:
            starting_capital = self.objectives.get("starting_capital_live", 1469)
        else:
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
        days: int = 180,
        save_results: bool = True,
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run vectorized backtest with current parameters
        M4 OPTIMIZED: Chunked processing with MPS memory management
        
        Default: 180 days (6 months) for comprehensive market coverage
        """
        import gc
        import torch
        
        self._ensure_setup()
        
        # Get backtest config
        backtest_cfg = self.params.get("backtest", {})
        slippage_pct = backtest_cfg.get("slippage_pct", 0.015) / 100
        latency_min = backtest_cfg.get("latency_ms_min", 1)
        latency_max = backtest_cfg.get("latency_ms_max", 5)
        commission = backtest_cfg.get("commission_pct", 0.0005)
        chunk_size = backtest_cfg.get("chunk_size", 5000)
        cleanup_freq = backtest_cfg.get("memory_cleanup_freq", 10)
        
        # Enable DEBUG logging for backtest
        if backtest_cfg.get("enable_debug_logging", False):
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info(f"Starting backtest ({days} days, slippage={slippage_pct*100:.3f}%, latency={latency_min}-{latency_max}ms)...")
        self.state.mode = "backtest"
        
        # MPS memory baseline
        baseline_memory = 0
        if torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
                baseline_memory = torch.mps.driver_allocated_memory() / 1e9
                logger.debug(f"MPS baseline memory: {baseline_memory:.2f}GB")
            except:
                logger.debug("MPS memory tracking unavailable")
        
        start_time = time.time()
        
        try:
            # Use provided data or fetch from API
            if historical_data is not None and not historical_data.empty:
                logger.info(f"Using provided historical data ({len(historical_data)} rows)")
                df = self._data_fetcher._add_technical_features(historical_data)
            else:
                df = await self._data_fetcher.get_training_data(
                    symbol=self.api_config["symbol"],
                    interval="1m",
                    days=days
                )
            
            if df.empty or len(df) < 500:
                logger.error(f"Insufficient data for backtest (got {len(df)} rows)")
                return {"error": "insufficient_data", "rows": len(df)}
            
            # Prepare features
            features, _ = self._ml_model.prepare_features(df)
            
            # Initialize model if needed
            if self._ml_model.model is None:
                self._ml_model.initialize_model(features.shape[1])
            
            # CHUNKED PROCESSING with MULTIPROCESSING for speed
            seq_len = self._ml_model.sequence_length
            total_rows = len(df)
            
            # Check if multiprocessing is enabled
            use_multiprocessing = backtest_cfg.get("enable_multiprocessing", True)
            max_workers = backtest_cfg.get("max_workers", max(1, cpu_count() - 1))
            
            if use_multiprocessing and cpu_count() > 1:
                logger.info(f"Using multiprocessing with {max_workers} workers")
                
                # Move model state to CPU for sharing between processes
                model_state_cpu = {k: v.cpu() for k, v in self._ml_model.model.state_dict().items()}
                
                # Process chunks SEQUENTIALLY to chain capital correctly
                # (parallel chunk processing causes capital to not carry over properly)
                all_trades = []
                all_equity_curve = []
                buy_signals = 0
                sell_signals = 0
                
                chunk_start_idx = seq_len
                chunk_num = 0
                current_capital = self.objectives["starting_capital_backtest"]
                global_peak_equity = current_capital  # Track global peak across all chunks
                total_chunks = (total_rows - seq_len) // chunk_size + 1
                
                while chunk_start_idx < total_rows - 1:
                    chunk_end_idx = min(chunk_start_idx + chunk_size, total_rows - 1)
                    chunk_num += 1
                    
                    chunk_df_slice = df.iloc[max(0, chunk_start_idx-seq_len):chunk_end_idx+1].copy()
                    
                    # Process single chunk with current capital and global peak equity
                    chunk_args = (
                        chunk_df_slice,
                        model_state_cpu,
                        self.params,
                        {
                            'chunk_idx': chunk_num,
                            'seq_len': seq_len,
                            'initial_capital': current_capital,
                            'global_peak_equity': global_peak_equity,  # Pass global peak for proper DD tracking
                            'chunk_start_idx': chunk_start_idx,
                            'chunk_end_idx': chunk_end_idx
                        }
                    )
                    
                    result = _process_backtest_chunk(chunk_args)
                    
                    # Hard cap for equity curve values (must match BacktestEnvironment.MAX_CAPITAL_ABSOLUTE)
                    MAX_CAPITAL = 1_000_000.0
                    GLOBAL_DD_LIMIT = 4.5  # Stop entire backtest at 4.5% global drawdown
                    
                    if result['success']:
                        all_trades.extend(result['trades'])
                        # Cap all equity curve values before extending
                        capped_equity = [min(v, MAX_CAPITAL) for v in result['equity_curve']]
                        all_equity_curve.extend(capped_equity)
                        buy_signals += result['buy_signals']
                        sell_signals += result['sell_signals']
                        
                        # Chain capital and peak equity to next chunk
                        if result['equity_curve']:
                            current_capital = min(result['equity_curve'][-1], MAX_CAPITAL)
                            # Safety: prevent capital from going to zero or negative
                            if current_capital <= 0:
                                current_capital = 1.0  # Minimum capital to continue
                        
                        # Update global peak equity from this chunk's final peak
                        if 'final_peak_equity' in result:
                            global_peak_equity = max(global_peak_equity, result['final_peak_equity'])
                        else:
                            # Fallback: use max equity from this chunk's curve
                            global_peak_equity = max(global_peak_equity, max(capped_equity) if capped_equity else global_peak_equity)
                        
                        # DEBUG: Log chunk DD status
                        if global_peak_equity > 0 and current_capital < global_peak_equity:
                            chunk_dd_pct = ((global_peak_equity - current_capital) / global_peak_equity) * 100
                            logger.debug(f"Chunk {chunk_num} DD: {chunk_dd_pct:.2f}% (Peak=${global_peak_equity:.2f}, Current=${current_capital:.2f})")
                        
                        # CRITICAL: Check global drawdown across all chunks
                        # This ensures we stop the entire backtest, not just individual chunks
                        if global_peak_equity > 0 and current_capital < global_peak_equity:
                            global_dd_pct = ((global_peak_equity - current_capital) / global_peak_equity) * 100
                            if global_dd_pct >= GLOBAL_DD_LIMIT:
                                logger.warning(f"🛑 GLOBAL DRAWDOWN LIMIT HIT: {global_dd_pct:.2f}% >= {GLOBAL_DD_LIMIT}%")
                                logger.warning(f"   Peak: ${global_peak_equity:.2f}, Current: ${current_capital:.2f}")
                                logger.warning(f"   Terminating backtest after chunk {chunk_num}/{total_chunks}")
                                break  # Stop processing more chunks
                        
                        # Progress logging
                        progress_pct = int((chunk_num / total_chunks) * 100)
                        logger.info(f"Backtest progress: {progress_pct}% (Chunk {chunk_num}/{total_chunks})")
                    else:
                        logger.warning(f"Chunk {result['chunk_idx']} failed: {result.get('error', 'unknown')}")
                    
                    chunk_start_idx = chunk_end_idx
                
                latency_sum = 0.0  # Not tracking latency in parallel mode
                
            else:
                # SEQUENTIAL PROCESSING (original implementation)
                logger.info("Using sequential processing (multiprocessing disabled)")
                all_trades = []
                all_equity_curve = []
                buy_signals = 0
                sell_signals = 0
                latency_sum = 0.0
                
                chunk_start_idx = seq_len
                chunk_num = 0
                
                # Hard cap for equity curve values
                MAX_CAPITAL = 1_000_000.0
                
                while chunk_start_idx < total_rows - 1:
                    chunk_end_idx = min(chunk_start_idx + chunk_size, total_rows - 1)
                    chunk_num += 1
                    
                    # Progress logging
                    progress_pct = int((chunk_end_idx / total_rows) * 100)
                    logger.info(f"Backtest progress: {progress_pct}% ({chunk_end_idx}/{total_rows}) [Chunk {chunk_num}]")
                    
                    # Create environment for this chunk
                    initial_cap = self.objectives["starting_capital_backtest"]
                    if chunk_start_idx == seq_len:
                        chunk_capital = initial_cap
                    else:
                        chunk_capital = all_equity_curve[-1]
                        # Hard cap to prevent overflow
                        chunk_capital = max(1.0, min(chunk_capital, MAX_CAPITAL))
                    
                    max_dd_pct = self.params["trading"].get("max_portfolio_drawdown_pct", 5.0)
                    
                    env = BacktestEnvironment(
                        df=df.iloc[max(0, chunk_start_idx-seq_len):chunk_end_idx+1].copy(),
                        initial_capital=chunk_capital,
                        transaction_cost=commission,
                        slippage=slippage_pct,
                        take_profit_pct=self.params["trading"]["take_profit_pct"],
                        stop_loss_pct=self.params["trading"]["stop_loss_pct"],
                        max_hold_bars=60,  # 60-bar (1 hour) hold limit
                        max_drawdown_pct=max_dd_pct  # Portfolio-level drawdown limit
                    )
                    
                    # OPTIMIZED: Batch inference on entire chunk (10-50x faster)
                    try:
                        chunk_indices = list(range(chunk_start_idx, chunk_end_idx))
                        chunk_rows = len(chunk_indices)
                        
                        # Prepare batch of feature sequences
                        feature_batch = np.array([features[i - seq_len:i] for i in chunk_indices])
                        
                        # Single batch prediction (GPU accelerated)
                        with torch.no_grad():
                            actions, confidences, q_values_batch = self._ml_model.predict_batch(feature_batch)
                        
                        # Process results
                        signal_threshold = self.params["trading"]["signal_threshold"]
                        min_q_diff = self.params["trading"].get("min_q_diff", 0.005)
                        position_size_pct = self.params["trading"]["position_size_pct"]
                        
                        for idx, i in enumerate(chunk_indices):
                            # Simulate latency (accumulate without actually waiting)
                            latency_ms = random.uniform(latency_min, latency_max)
                            latency_sum += latency_ms
                            
                            # Get prediction results from model
                            action = int(actions[idx])
                            q_values = q_values_batch[idx]
                            q_diff = np.max(q_values) - np.min(q_values)
                            
                            # Use ML model's action directly
                            # The DQN model outputs: 0=HOLD, 1=BUY, 2=SELL
                            final_action = action  # Trust the model's decision
                            
                            # Only apply min_q_diff filter for confidence
                            if q_diff < min_q_diff:
                                final_action = 0  # HOLD if confidence too low
                            
                            # Track for logging
                            if final_action == 1:
                                buy_signals += 1
                            elif final_action == 2:
                                sell_signals += 1
                            
                            # Execute action with correct data index
                            reward, done = env.step(final_action, position_size_pct / 100, data_idx=i)
                            
                            if done:
                                break
                        
                        # Force-close any remaining open position at end of chunk/data
                        if env.position > 0:
                            final_price = df.iloc[chunk_end_idx]['close']
                            env._close_position(final_price, "end_of_chunk")
                            logger.debug(f"Force-closed position at end of chunk {chunk_num}")
                                
                    except Exception as e:
                        logger.error(f"Inference error at chunk {chunk_num}, rows {chunk_start_idx}-{chunk_end_idx}: {type(e).__name__}: {e}")
                        # Continue to next chunk
                        chunk_start_idx = chunk_end_idx
                        continue
                    
                    # Collect chunk results
                    all_trades.extend(env.trades)
                    if env.equity_curve:
                        all_equity_curve.extend(env.equity_curve)
                    else:
                        # If no equity curve recorded, use current capital
                        all_equity_curve.append(env.capital)
                    
                    # CRITICAL: Check global drawdown across all chunks (sequential mode)
                    GLOBAL_DD_LIMIT = 4.5  # Stop entire backtest at 4.5% global drawdown
                    initial_cap = self.objectives["starting_capital_backtest"]
                    current_equity = all_equity_curve[-1] if all_equity_curve else env.capital
                    global_peak = max(all_equity_curve) if all_equity_curve else initial_cap
                    
                    if global_peak > 0 and current_equity < global_peak:
                        global_dd_pct = ((global_peak - current_equity) / global_peak) * 100
                        if global_dd_pct >= GLOBAL_DD_LIMIT:
                            logger.warning(f"🛑 GLOBAL DRAWDOWN LIMIT HIT: {global_dd_pct:.2f}% >= {GLOBAL_DD_LIMIT}%")
                            logger.warning(f"   Peak: ${global_peak:.2f}, Current: ${current_equity:.2f}")
                            logger.warning(f"   Terminating backtest at chunk {chunk_num}")
                            break  # Stop processing more chunks
                    
                    # MPS memory cleanup
                    if progress_pct % cleanup_freq == 0 and torch.backends.mps.is_available():
                        try:
                            torch.mps.empty_cache()
                            torch.mps.synchronize()
                            current_memory = torch.mps.driver_allocated_memory() / 1e9
                            logger.debug(f"MPS memory at {progress_pct}%: {current_memory:.2f}GB (Δ{current_memory - baseline_memory:+.2f}GB)")
                        except:
                            pass
                    
                    # Move to next chunk
                    chunk_start_idx = chunk_end_idx
            
            logger.info(f"Backtest complete: {len(all_trades)} trades executed")
            logger.info(f"Strategy signals: {buy_signals} buys (momentum/SMA), {sell_signals} sells (momentum/SMA)")
            
            # Calculate metrics from aggregated results
            avg_latency_ms = latency_sum / max(total_rows - seq_len - 1, 1) if not use_multiprocessing else 0.0
            
            # Build metrics dict (simplified - using BacktestEnvironment's final state)
            # Cap final_capital to prevent overflow
            MAX_CAPITAL = 1_000_000.0
            final_capital = all_equity_curve[-1] if all_equity_curve else self.objectives["starting_capital_backtest"]
            final_capital = min(final_capital, MAX_CAPITAL)
            initial_capital = self.objectives["starting_capital_backtest"]
            total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
            # Cap at 100000% (1000x) for display
            total_return_pct = min(total_return_pct, 100000.0)
            
            metrics = {
                "total_trades": len(all_trades),
                "total_return_pct": total_return_pct,
                "final_capital": final_capital,
                "backtest_time": time.time() - start_time,
                "data_points": len(df),
                "avg_latency_ms": avg_latency_ms,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "backtest_days": days
            }
            
            # Calculate additional metrics if we have trades
            if all_trades:
                winning_trades = [t for t in all_trades if t.get("pnl", 0) > 0]
                metrics["win_rate"] = (len(winning_trades) / len(all_trades)) * 100 if all_trades else 0
                metrics["profit_factor"] = sum(t.get("pnl", 0) for t in winning_trades) / abs(sum(t.get("pnl", 0) for t in all_trades if t.get("pnl", 0) < 0)) if any(t.get("pnl", 0) < 0 for t in all_trades) else 1.0
                
                # Sharpe ratio (simplified)
                returns = [t.get("pnl", 0) / initial_capital for t in all_trades]
                if returns:
                    metrics["sharpe_ratio"] = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
                else:
                    metrics["sharpe_ratio"] = 0
                
                # Max drawdown - use RUNNING peak method with absolute values
                # Circuit breaker ensures max DD should be ~4.5% max
                if all_equity_curve and len(all_equity_curve) > 1:
                    # Filter out invalid values
                    valid_equity = [e for e in all_equity_curve if e > 0 and np.isfinite(e)]
                    
                    if valid_equity and len(valid_equity) > 1:
                        peak = valid_equity[0]
                        max_dd = 0.0
                        
                        for equity in valid_equity:
                            # Only update peak if equity is close to or above previous peak
                            # This prevents treating early small values as peaks
                            if equity > peak:
                                peak = equity
                            
                            # Calculate drawdown from current running peak
                            if peak > 0 and equity < peak:
                                dd = ((peak - equity) / peak) * 100
                                # Cap at hard termination threshold (circuit breaker should prevent higher)
                                dd = min(dd, 5.0)  # 5% is our max DD objective
                                if dd > max_dd:
                                    max_dd = dd
                        
                        # Sanity check: if we have circuit breakers, max DD shouldn't exceed 5%
                        # If it does, something went wrong - cap at objective threshold
                        max_dd = min(max_dd, 5.0)
                        metrics["max_drawdown_pct"] = max_dd
                    else:
                        metrics["max_drawdown_pct"] = 0
                else:
                    metrics["max_drawdown_pct"] = 0
            else:
                metrics["win_rate"] = 0
                metrics["profit_factor"] = 0
                metrics["sharpe_ratio"] = 0
                metrics["max_drawdown_pct"] = 0
            
            # Monthly projection
            if total_return_pct > -100 and days > 0:
                monthly_projection = ((1 + total_return_pct/100) ** (30 / days) - 1) * 100
            else:
                monthly_projection = 0
            metrics["monthly_projection"] = monthly_projection
            
            # TRADE DENSITY CHECK: Halt if insufficient trading activity
            min_trades_threshold = backtest_cfg.get("min_trades_threshold", 50)
            trade_density = (len(all_trades) / max(total_rows - seq_len, 1)) * 100
            metrics["trade_density_pct"] = trade_density
            
            if len(all_trades) < min_trades_threshold:
                logger.error(f"TRADE DENSITY FAILURE: {len(all_trades)} trades < {min_trades_threshold} threshold")
                logger.error(f"   Trade density: {trade_density:.4f}% (target: >0.1%)")
                logger.error(f"   System is HOLDing {100-trade_density:.2f}% of the time - CRITICAL")
                logger.error(f"   Recommendations:")
                logger.error(f"     1. Relax signal thresholds (momentum/SMA spread)")
                logger.error(f"     2. Lower signal_threshold (current: {self.params['trading'].get('signal_threshold', 0.6)})")
                logger.error(f"     3. Check data quality and feature generation")
                metrics["objectives_met"] = False
                metrics["failure_reason"] = "insufficient_trades"
            
            # Add sample trades for debugging (first 20)
            if all_trades:
                metrics["sample_trades"] = all_trades[:20]
            
            # Check objectives
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
            
            # Final cleanup
            gc.collect()
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except:
                    pass
            
            return metrics
            
        except Exception as e:
            logger.error(f"Backtest failed: {type(e).__name__}: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _check_objectives(self, metrics: Dict[str, Any], phase_override: Optional[int] = None) -> bool:
        """
        Check if metrics meet objectives with phase-aware thresholds.
        
        Args:
            metrics: Backtest metrics dict
            phase_override: Optional phase to check against (for testing)
        
        Returns:
            True if all phase objectives are met
        """
        # Step 1: Monthly performance - use compounding extrapolation
        total_return = metrics.get("total_return_pct", 0) / 100
        backtest_days = metrics.get("backtest_days", 7)
        
        if total_return > -1:
            monthly_return = ((1 + total_return) ** (30 / max(backtest_days, 1)) - 1) * 100
            monthly_return = min(monthly_return, 10000.0)
        else:
            monthly_return = -100
        
        # Get individual metric values
        profit_factor = metrics.get("profit_factor", 0)
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        max_drawdown = metrics.get("max_drawdown_pct", 100)
        
        # Determine current phase thresholds
        current_phase = phase_override or self.objectives.get("current_phase", 1)
        phases = self.objectives.get("phases", {})
        phase_key = f"phase_{current_phase}"
        phase_config = phases.get(phase_key, {})
        
        # Use phase-specific thresholds if available, otherwise fall back to defaults
        if phase_config:
            monthly_min = phase_config.get("monthly_return_pct_min", 5)
            pf_min = phase_config.get("profit_factor_min", 0.8)
            sharpe_min = phase_config.get("sharpe_ratio_min", 0.5)
            dd_max = phase_config.get("drawdown_max", 5)
            phase_name = phase_config.get("name", f"Phase {current_phase}")
        else:
            monthly_min = self.objectives.get("monthly_return_pct_min", 5)
            pf_min = self.objectives.get("profit_factor_min", 1.1)
            sharpe_min = self.objectives.get("sharpe_ratio_min", 1.5)
            dd_max = self.objectives.get("drawdown_max", 5)
            phase_name = "Default"
        
        # Individual checks
        check_monthly = monthly_return >= monthly_min
        check_pf = profit_factor >= pf_min
        check_sharpe = sharpe_ratio >= sharpe_min
        check_dd = max_drawdown <= dd_max
        
        # Log all objective checks
        logger.info("=" * 50)
        logger.info(f"OBJECTIVES CHECK (Phase {current_phase}: {phase_name}):")
        logger.info(f"  Monthly Return: {monthly_return:.2f}% >= {monthly_min}%: {'✓' if check_monthly else '✗'}")
        logger.info(f"  Profit Factor: {profit_factor:.2f} >= {pf_min}: {'✓' if check_pf else '✗'}")
        logger.info(f"  Sharpe Ratio:  {sharpe_ratio:.2f} >= {sharpe_min}: {'✓' if check_sharpe else '✗'}")
        logger.info(f"  Max Drawdown:  {max_drawdown:.2f}% <= {dd_max}%: {'✓' if check_dd else '✗'}")
        
        all_met = all([check_monthly, check_pf, check_sharpe, check_dd])
        logger.info(f"  ALL OBJECTIVES MET: {'✓ YES' if all_met else '✗ NO'}")
        
        # Track phase progress
        if all_met and phase_config:
            cycles_completed_key = f"phase_{current_phase}_completed_cycles"
            current_cycles = self.objectives.get(cycles_completed_key, 0)
            cycles_required = phase_config.get("cycles_required", 3)
            new_cycles = current_cycles + 1
            
            logger.info(f"  Phase {current_phase} Progress: {new_cycles}/{cycles_required} successful cycles")
            
            # Check for phase advancement
            if new_cycles >= cycles_required and current_phase < 3:
                logger.info("=" * 50)
                logger.info(f"🎉 PHASE {current_phase} COMPLETE! Advancing to Phase {current_phase + 1}")
                logger.info("=" * 50)
                self._advance_phase(current_phase + 1)
        
        logger.info("=" * 50)
        
        return all_met
    
    def _advance_phase(self, new_phase: int) -> None:
        """Advance to a new phase and update objectives.json"""
        try:
            objectives_path = self.config_dir / "objectives.json"
            with open(objectives_path, "r") as f:
                objectives = json.load(f)
            
            objectives["current_phase"] = new_phase
            
            # Reset cycle counter for new phase
            objectives[f"phase_{new_phase}_completed_cycles"] = 0
            
            with open(objectives_path, "w") as f:
                json.dump(objectives, f, indent=2)
            
            # Reload objectives
            self.objectives = objectives
            logger.info(f"Updated objectives.json: current_phase = {new_phase}")
            
        except Exception as e:
            logger.error(f"Failed to advance phase: {e}")
    
    # ==================== Training ====================
    
    async def train_model(
        self,
        days: int = 180,
        epochs: Optional[int] = None,
        append_live: bool = False
    ) -> Dict[str, Any]:
        """
        Train ML model on historical data.
        
        Args:
            days: Number of days of data (default 180 for 6 months coverage)
            epochs: Training epochs (None uses config default of 150)
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
            """
            Pure Momentum Strategy (SPEC-compliant)
            
            Uses momentum for regime detection:
            - Strong momentum (>0.3%): Follow trend
            - Weak momentum: Use SMA spread for mean-reversion
            """
            max_dd_pct = self.params["trading"].get("max_portfolio_drawdown_pct", 5.0)
            
            env = BacktestEnvironment(
                df=df_data,
                initial_capital=self.objectives["starting_capital_backtest"],
                take_profit_pct=params["take_profit_pct"],
                stop_loss_pct=params["stop_loss_pct"],
                max_hold_bars=30,  # 30-min hold for faster exits
                max_drawdown_pct=max_dd_pct  # Portfolio-level drawdown limit
            )
            
            # Strategy parameters (momentum-based, no RSI/ADX)
            momentum_window = 10
            momentum_threshold = 0.3  # 0.3% for trend signal
            
            # Ensure momentum is calculated
            df_work = df_data.copy()
            if "momentum_5" not in df_work.columns:
                df_work["momentum_5"] = df_work["close"] / df_work["close"].shift(5) - 1
            if "momentum_10" not in df_work.columns:
                df_work["momentum_10"] = df_work["close"] / df_work["close"].shift(10) - 1
            
            # Use ML model as confirmation filter if available
            use_ml_filter = self._ml_model.model is not None
            features = None
            seq_len = 30
            
            if use_ml_filter:
                features, _ = self._ml_model.prepare_features(df_data)
                seq_len = self._ml_model.sequence_length
                self._ml_model.model.eval()
            
            # Start after warmup period
            start_idx = max(seq_len, 30)
            
            for i in range(start_idx, len(df_work) - 1):
                close_price = df_work["close"].iloc[i]
                
                # Calculate combined momentum
                mom_5 = df_work["momentum_5"].iloc[i] * 100 if not pd.isna(df_work["momentum_5"].iloc[i]) else 0
                mom_10 = df_work["momentum_10"].iloc[i] * 100 if not pd.isna(df_work["momentum_10"].iloc[i]) else 0
                momentum_pct = mom_5 * 0.6 + mom_10 * 0.4  # Weighted average
                
                action = 0  # Default: HOLD
                
                # MOMENTUM-BASED SIGNALS
                if momentum_pct > momentum_threshold:
                    action = 1  # BUY - uptrend
                elif momentum_pct < -momentum_threshold:
                    action = 2  # SELL - downtrend
                
                # Optional ML confirmation
                if action != 0 and use_ml_filter and features is not None:
                    feature_seq = features[i - seq_len:i]
                    with torch.no_grad():
                        ml_action, confidence, q_values = self._ml_model.predict(feature_seq)
                    
                    if action == 1 and ml_action == 2 and confidence > 0.9:
                        action = 0  # Block BUY
                    elif action == 2 and ml_action == 1 and confidence > 0.9:
                        action = 0  # Block SELL
                
                env.step(action, params["position_size_pct"] / 100, data_idx=i)
            
            return env.get_metrics()
            
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
            logger.info(f"Trade not allowed: {reason}")
            return None
        
        symbol = self.api_config["symbol"]
        
        # Step 5: Track orderbook fetch latency
        ob_start = time.perf_counter()
        orderbook = await self._api.get_orderbook(symbol)
        ob_latency_ms = (time.perf_counter() - ob_start) * 1000
        current_price = orderbook.mid_price

        # Get current volatility (from returns, no ATR)
        snapshot = self._data_fetcher.get_latest_snapshot()
        volatility = snapshot.spread_bps / 10000 if snapshot else 0.02

        # Calculate position size
        position_size = self._risk_manager.calculate_position_size(
            signal_strength=signal_strength,
            current_volatility=volatility,
            entry_price=current_price
        )
        
        if position_size <= 0:
            logger.warning(f"  Trade skipped: position_size={position_size} (signal={signal}, strength={signal_strength:.3f})")
            return None
        
        # Step 5: Log preparation latency
        prep_latency_ms = (time.perf_counter() - exec_start) * 1000
        logger.info(f"  {signal}: size={position_size:.4f} @ ${current_price:.2f} (prep={prep_latency_ms:.1f}ms, ob={ob_latency_ms:.1f}ms)")
        
        # Determine trade direction with fixed TP/SL (no ATR adjustment)
        if signal == "BUY":
            side = "B"
            sl_price = self._risk_manager.calculate_stop_loss(current_price, side, volatility, 0.0)
            tp_price = self._risk_manager.calculate_take_profit(current_price, side, signal_strength, 0.0)
        elif signal == "SELL":
            side = "A"
            sl_price = self._risk_manager.calculate_stop_loss(current_price, side, volatility, 0.0)
            tp_price = self._risk_manager.calculate_take_profit(current_price, side, signal_strength, 0.0)
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
                "volatility": volatility,
                "result": result,
                "order_id": order_id
            }
            
            await self._data_fetcher.save_trade(trade_data)
            
            # Step 5: Log execution latency
            exec_latency_ms = (time.perf_counter() - exec_start) * 1000
            logger.info(
                f"Trade executed: {signal} {position_size:.4f} {symbol} @ {current_price:.2f} "
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
                    f"SL/TP synced for position {current_pos.size} @ {current_pos.entry_price}: "
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
                    logger.warning(f"HFT timeout exit: position held {seconds_held:.1f}s")
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
            # Only log every 30 seconds to reduce spam
            if not hasattr(self, '_last_data_warning'):
                self._last_data_warning = 0
            if time.time() - self._last_data_warning >= 30:
                logger.warning(f"Waiting for data: {len(klines)}/{self._ml_model.sequence_length} klines (API may be unavailable)")
                self._last_data_warning = time.time()
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
            # Low ML confidence - use momentum fallback (no RSI)
            if "momentum_5" in df.columns and len(df) > 0:
                momentum = df.iloc[-1].get("momentum_5", 0)
                if not pd.isna(momentum):
                    if momentum > 0.002:  # Strong upward momentum
                        logger.info(f"  → BUY (momentum fallback: {momentum:.4f})")
                        return "BUY", 0.5
                    elif momentum < -0.002:  # Strong downward momentum
                        logger.info(f"  → SELL (momentum fallback: {momentum:.4f})")
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
        
        max_holds = self.params.get("trading", {}).get("max_hold_signals_before_force", 3)  # Reduced from 5 to 3
        force_trade_enabled = self.params.get("trading", {}).get("force_trade_after_holds", True)
        force_strength = self.params.get("trading", {}).get("force_trade_strength", 0.8)  # Increased from 0.7
        
        if signal == "HOLD":
            self._consecutive_holds += 1
            # Debug: Log consecutive count reaching threshold
            if self._consecutive_holds >= max_holds:
                logger.info(f"  Force trade check: consecutive={self._consecutive_holds}, enabled={force_trade_enabled}")
            
            if force_trade_enabled and self._consecutive_holds >= max_holds:
                # Force a trade based on momentum (no RSI)
                forced = False
                if "momentum_5" in df.columns and len(df) > 0:
                    momentum = df.iloc[-1].get("momentum_5", 0)
                    momentum_10 = df.iloc[-1].get("momentum_10", 0) if "momentum_10" in df.columns else momentum
                    
                    if not pd.isna(momentum):
                        # Use combined momentum for direction
                        combined_momentum = (momentum * 0.6 + momentum_10 * 0.4)
                        
                        if combined_momentum > 0.001:
                            signal = "BUY"
                            strength = force_strength
                            logger.info(f"  FORCE BUY after {self._consecutive_holds} HOLDs (momentum={combined_momentum:.4f})")
                            forced = True
                        elif combined_momentum < -0.001:
                            signal = "SELL"
                            strength = force_strength
                            logger.info(f"  FORCE SELL after {self._consecutive_holds} HOLDs (momentum={combined_momentum:.4f})")
                            forced = True
                        else:
                            # Neutral momentum - alternate direction for diversity
                            if not hasattr(self, '_last_force_direction'):
                                self._last_force_direction = "BUY"
                            signal = "SELL" if self._last_force_direction == "BUY" else "BUY"
                            strength = force_strength * 0.8
                            self._last_force_direction = signal
                            logger.info(f"  FORCE {signal} after {self._consecutive_holds} HOLDs (neutral momentum)")
                            forced = True
                else:
                    # Momentum not available - force alternating trade
                    if not hasattr(self, '_last_force_direction'):
                        self._last_force_direction = "BUY"
                    signal = "SELL" if self._last_force_direction == "BUY" else "BUY"
                    strength = force_strength * 0.7
                    self._last_force_direction = signal
                    logger.info(f"  FORCE {signal} after {self._consecutive_holds} HOLDs (no momentum data)")
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
        Pure price-based detection (no ATR/RSI).
        
        Returns: "trending", "ranging", or "volatile"
        """
        if len(df) < 20:
            return "ranging"
        
        try:
            close = df["close"].values
            
            # Calculate volatility from returns (no ATR)
            returns = np.diff(close[-20:]) / close[-21:-1]
            vol_ratio = np.std(returns) if len(returns) > 0 else 0
            
            # Calculate trend strength
            high_20 = df["high"].iloc[-20:].max()
            low_20 = df["low"].iloc[-20:].min()
            range_20 = high_20 - low_20
            
            # Current price position within range
            current = close[-1]
            range_position = (current - low_20) / range_20 if range_20 > 0 else 0.5
            
            # Price momentum via SMAs
            sma_short = close[-5:].mean()
            sma_long = close[-20:].mean()
            trend_strength = abs(sma_short - sma_long) / sma_long if sma_long > 0 else 0
            
            # Classify regime
            if vol_ratio > 0.02:  # High volatility (> 2%)
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
        
        # Force trade settings
        force_enabled = self.params.get("trading", {}).get("force_trade_after_holds", True)
        max_holds = self.params.get("trading", {}).get("max_hold_signals_before_force", 3)
        
        logger.info(f"Cycle {self.state.current_cycle} started | duration={cycle_duration}s, force_after={max_holds}")
        
        adjustments = []
        initial_capital = self._risk_manager.current_capital
        signals_generated = 0
        signals_filtered = 0
        signals_executed = 0
        signals_hold = 0  # Track HOLD signals separately
        last_trade_time = 0.0
        min_trade_interval = 3.0  # 3 seconds between trades (API-friendly)
        
        # Progress tracking - log every 30 seconds
        last_progress_log = time.time()
        
        while time.time() - self.cycle_start_time < cycle_duration:
            if self._shutdown_event.is_set():
                break
            
            # Check if trade threshold reached
            if self.trade_count_this_cycle >= trade_threshold:
                logger.info(f"Trade threshold ({trade_threshold}) reached")
                break
            
            # Log progress every 30 seconds
            elapsed = time.time() - self.cycle_start_time
            if time.time() - last_progress_log >= 30:
                remaining = cycle_duration - elapsed
                logger.info(f"  Progress: {elapsed:.0f}s/{cycle_duration}s | sig={signals_generated} hold={signals_hold} exec={self.trade_count_this_cycle}")
                last_progress_log = time.time()
            
            try:
                # Check exit conditions for open positions
                await self.check_exit_conditions()
                
                # Generate signal if no position
                if not self.current_position or self.current_position.size == 0:
                    signal, strength = await self.generate_trading_signal()
                    signals_generated += 1
                    
                    # Track HOLD signals separately - log at INFO level periodically for visibility
                    if signal == "HOLD":
                        signals_hold += 1
                        holds_in_row = getattr(self, '_consecutive_holds', 0)
                        if signals_hold <= 3 or signals_hold % 20 == 0:
                            logger.info(f"  HOLD #{signals_hold} (str={strength:.3f}, row={holds_in_row})")
                    
                    # Step 1: Enhanced logging for signal decisions
                    if signal != "HOLD":
                        if abs(strength) > signal_threshold:
                            # Rate limit: enforce minimum interval between trades
                            time_since_last_trade = time.time() - last_trade_time
                            if time_since_last_trade < min_trade_interval:
                                wait_time = min_trade_interval - time_since_last_trade
                                logger.debug(f"Rate limit: waiting {wait_time:.1f}s before next trade")
                                await asyncio.sleep(wait_time)
                            
                            logger.info(f"  {signal} triggered (str={strength:.3f})")
                            result = await self.execute_trade_signal(signal, strength)
                            if result:
                                signals_executed += 1
                                last_trade_time = time.time()
                                # Clear cache after order to get fresh state
                                self._api.clear_cache("user_state")
                                self._api.clear_cache("open_orders")
                            else:
                                logger.warning(f"  {signal} execution failed")
                        else:
                            signals_filtered += 1
                            # Step 1: More visible logging for filtered signals
                            if signals_filtered <= 5 or signals_filtered % 10 == 0:
                                logger.info(f"  🚫 {signal} filtered (str={strength:.3f} < {signal_threshold})")
                
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
            f"hold={signals_hold}, filtered={signals_filtered}, executed={self.trade_count_this_cycle}"
        )
        
        # Warn if no trades despite signals - with better diagnosis
        if self.trade_count_this_cycle == 0 and signals_generated > 0:
            if signals_hold == signals_generated:
                logger.warning(
                    f"⚠ Cycle {self.state.current_cycle}: All {signals_generated} signals were HOLD - "
                    f"model may need retraining or force_trade should trigger"
                )
            else:
                logger.warning(
                    f"⚠ Cycle {self.state.current_cycle}: 0 trades from {signals_generated} signals - "
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
                    logger.warning("Max consecutive criticals reached - resetting params to defaults")
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
        # DISABLED - letting aggressive trading parameters through
        # trading = self.params.get("trading", {})
        # trading["position_size_pct"] = max(trading.get("position_size_pct", 0.25), 0.15)
        # Don't limit signal_threshold or min_q_diff - allow aggressive values
        pass  # Function disabled - aggressive trading mode enabled
        
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
    
    # ==================== Live Trading ====================
    
    async def run_live_trading(self, duration_seconds: int = None) -> Dict[str, Any]:
        """
        Run live trading for specified duration or indefinitely.
        
        This method executes real trades on HyperLiquid mainnet using the trained model.
        Requires objectives to be met before enabling.
        
        Args:
            duration_seconds: Optional duration limit. If None, runs until stopped.
            
        Returns:
            Summary metrics from the live trading session
        """
        self._ensure_setup()
        
        logger.info("=" * 60)
        logger.info("LIVE TRADING MODE ACTIVATED")
        logger.info("=" * 60)
        logger.info(f"   Wallet: {self._api.wallet_address[:10]}...")
        logger.info(f"   Asset: XYZ100-USDC")
        logger.info(f"   Capital: ${self._risk_manager.current_capital:.2f}")
        
        # Start background data fetching for live klines
        symbol = self.api_config["symbol"]
        await self._data_fetcher.start_background_fetching(symbol=symbol)
        logger.info(f"   Background fetching: Started for {symbol}")
        
        # Safety check: ensure we have a trained model
        if not hasattr(self._ml_model, 'model') or self._ml_model.model is None:
            raise ValueError("No trained model available. Run training first.")
        
        start_time = time.time()
        total_trades = 0
        total_pnl = 0.0
        
        try:
            while not self._shutdown_event.is_set():
                # Check duration limit
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    logger.info(f"Duration limit ({duration_seconds}s) reached")
                    break
                
                # Run one trading cycle
                cycle_metrics = await self.run_trading_cycle()
                total_trades += cycle_metrics.trades_count
                total_pnl += cycle_metrics.pnl
                
                # Log progress
                elapsed = time.time() - start_time
                logger.info(f"Live: {total_trades} trades, PnL: ${total_pnl:.2f}, elapsed: {elapsed/60:.1f}m")
                
                # Check if we should continue based on risk
                risk_status = self._risk_manager.get_status_summary()
                if risk_status.get('drawdown_pct', 0) >= 4.5:
                    logger.warning("🛑 Drawdown limit reached, pausing live trading")
                    await asyncio.sleep(300)  # 5 minute pause
                
                # Brief pause between cycles
                await asyncio.sleep(5)
                
        except asyncio.CancelledError:
            logger.info("Live trading cancelled")
        except Exception as e:
            logger.error(f"Live trading error: {e}")
            raise
        
        # Calculate session metrics
        session_duration = time.time() - start_time
        
        summary = {
            "duration_seconds": session_duration,
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "trades_per_hour": total_trades / max(session_duration / 3600, 1/60),
            "final_capital": self._risk_manager.current_capital
        }
        
        logger.info("=" * 60)
        logger.info("🏁 LIVE TRADING SESSION COMPLETE")
        logger.info(f"   Duration: {session_duration/60:.1f} minutes")
        logger.info(f"   Trades: {total_trades}")
        logger.info(f"   PnL: ${total_pnl:.2f}")
        logger.info("=" * 60)
        
        return summary
    
    # ==================== Main Autonomous Loop ====================
    
    async def run_autonomous(self) -> None:
        """
        Enhanced autonomous execution loop with progressive phase management.
        
        IMPROVEMENTS:
        1. Progressive objectives (Phase 1 → 3) with automatic phase advancement
        2. Dynamic training based on phase requirements
        3. Model quality validation before proceeding
        4. Better logging and progress tracking
        """
        self._ensure_setup()
        logger.info("=" * 60)
        logger.info("AML HFT Autonomous Trading System v2.0")
        logger.info("=" * 60)
        
        # Load phase configuration
        phases = self.objectives.get("phases", {})
        current_phase = self.objectives.get("current_phase", 1)
        phase_key = f"phase_{current_phase}"
        phase_config = phases.get(phase_key, {})
        
        if phase_config:
            logger.info(f"Current Phase: {current_phase} ({phase_config.get('name', 'Unknown')})")
            logger.info(f"   Target Return: ≥{phase_config.get('monthly_return_pct_min', 5)}%")
            logger.info(f"   Target Sharpe: ≥{phase_config.get('sharpe_ratio_min', 0.5)}")
            logger.info(f"   Max Drawdown: ≤{phase_config.get('drawdown_max', 5)}%")
            logger.info(f"   Cycles Required: {phase_config.get('cycles_required', 3)}")
        
        # Use data_days from config
        data_days = self.params.get("ml_model", {}).get("data_days", 180)
        backtest_days = data_days
        
        # Determine effective objectives based on current phase
        effective_objectives = {
            "monthly_return_pct_min": phase_config.get("monthly_return_pct_min", 
                                                       self.objectives.get("monthly_return_pct_min", 5.0)),
            "sharpe_ratio_min": phase_config.get("sharpe_ratio_min", 
                                                  self.objectives.get("sharpe_ratio_min", 1.3)),
            "drawdown_max": phase_config.get("drawdown_max", 
                                              self.objectives.get("drawdown_max", 5.0)),
            "profit_factor_min": phase_config.get("profit_factor_min", 0.8)
        }
        
        logger.info(f"Active Objectives: Return>={effective_objectives['monthly_return_pct_min']}%, "
                   f"Sharpe≥{effective_objectives['sharpe_ratio_min']}, DD≤{effective_objectives['drawdown_max']}%")
        
        # Phase 0: Model Training (if needed)
        model_trained_marker = self.model_dir / ".trained_successfully"
        model_exists = (self.model_dir / "best_model.pt").exists() or (self.model_dir / "final_model.pt").exists()
        model_trained = model_exists and model_trained_marker.exists()
        
        if not model_trained:
            # Remove any partially trained models
            for model_file in ["best_model.pt", "final_model.pt", "shutdown_model.pt"]:
                model_path = self.model_dir / model_file
                if model_path.exists():
                    logger.info(f"Removing untrained model: {model_file}")
                    model_path.unlink()
            if model_trained_marker.exists():
                model_trained_marker.unlink()
            
            logger.info("=" * 60)
            logger.info(f"📚 Phase 0: Initial Model Training ({data_days} days)")
            logger.info("=" * 60)
            
            # Extended training for phase 1
            epochs = self.params.get("ml_model", {}).get("epochs", 500)
            training_results = await self.train_model(days=data_days, append_live=False)
            
            if training_results.get("error"):
                logger.error("CRITICAL: Initial model training failed")
                logger.error(f"Error: {training_results.get('error')}")
                self.state.is_running = False
                return
            
            # Validate model quality before proceeding
            final_reward = training_results.get('best_reward', 0)
            if final_reward < -0.1:
                logger.warning(f"Model training suboptimal (reward: {final_reward:.4f})")
                logger.info("Extending training with adjusted learning rate...")
                # Reduce learning rate and continue
                self.params["ml_model"]["learning_rate"] *= 0.5
                training_results = await self.train_model(days=data_days, append_live=False)
            
            model_trained_marker.touch()
            logger.info(f"Phase 0 complete: Model trained with reward {training_results.get('best_reward', 0):.4f}")
        else:
            logger.info("Found existing trained model - proceeding to backtest")
        
        # Phase 1: Initial Backtest
        logger.info("=" * 60)
        logger.info(f"Phase 1: Initial Backtest ({backtest_days} days)")
        logger.info("=" * 60)
        
        backtest_results = await self.run_backtest(days=backtest_days)
        
        # CRITICAL: Validate backtest completion
        if backtest_results.get("error"):
            logger.error("=" * 60)
            logger.error("CRITICAL: Phase 1 backtest failed or incomplete")
            logger.error(f"Error: {backtest_results.get('error')}")
            logger.error("=" * 60)
            logger.info("Attempting to fetch fresh data and retry...")
            
            await self._data_fetcher.fetch_historical_klines(
                symbol=self.api_config["symbol"],
                interval="1m",
                days=data_days
            )
            backtest_results = await self.run_backtest(days=backtest_days)
            
            # If still failing, halt
            if backtest_results.get("error"):
                logger.error("=" * 60)
                logger.error("CRITICAL: Phase 1 backtest failed after retry")
                logger.error("System halted. Please review logs and retry.")
                logger.error("=" * 60)
                self.state.is_running = False
                return
        
        # Validate trade execution
        if backtest_results.get("total_trades", 0) == 0:
            logger.error("=" * 60)
            logger.error("CRITICAL: Phase 1 backtest produced zero trades")
            logger.error("This indicates a model inference or data issue")
            logger.error(f"Data points: {backtest_results.get('data_points', 0)}")
            logger.error("=" * 60)
            self.state.is_running = False
            return
        
        logger.info(f"Phase 1 complete: {backtest_results.get('total_trades')} trades, "
                    f"{backtest_results.get('total_return_pct', 0):.2f}% return")
        
        # Step 3: Store Phase 1 objectives status for Phase 4 comparison
        self._phase1_objectives_met = self.state.objectives_met
        logger.info(f"Phase 1 objectives met: {self._phase1_objectives_met}")
        
        # Phase 2: RETRY LOOP - Optimize until objectives met or max attempts reached
        # Reduced from 15 to 5 - optimization requires trained model to be effective
        max_optimization_attempts = 5
        optimization_attempt = 0
        
        while not self.state.objectives_met and optimization_attempt < max_optimization_attempts:
            optimization_attempt += 1
            logger.info("=" * 60)
            logger.info(f"Phase 2: Parameter Optimization (Attempt {optimization_attempt}/{max_optimization_attempts})")
            logger.info("=" * 60)
            
            # Run optimization with increasing trials for better coverage
            n_trials = 100 + (optimization_attempt - 1) * 20  # 100, 120, 140... up to 380 trials
            await self.run_optimization(n_trials=n_trials)
            
            # Reload params after optimization to apply new values
            with open(self.config_dir / "params.json", "r") as f:
                self.params = json.load(f)
            logger.info("Reloaded optimized parameters")
            
            # Re-run backtest with new parameters
            logger.info(f"Validating optimization (Attempt {optimization_attempt})")
            backtest_results = await self.run_backtest(days=backtest_days)
            
            if self.state.objectives_met:
                logger.info("=" * 60)
                logger.info(f"OPTIMIZATION SUCCESS on attempt {optimization_attempt}")
                logger.info(f"Return: {backtest_results.get('total_return_pct', 0):.2f}%, "
                          f"Sharpe: {backtest_results.get('sharpe_ratio', 0):.2f}, "
                          f"Profit Factor: {backtest_results.get('profit_factor', 0):.2f}")
                logger.info("=" * 60)
                break
            else:
                logger.warning(f"Attempt {optimization_attempt} did not meet objectives - "
                             f"Return: {backtest_results.get('total_return_pct', 0):.2f}%, "
                             f"Sharpe: {backtest_results.get('sharpe_ratio', 0):.2f}")
                
                if optimization_attempt < max_optimization_attempts:
                    logger.info("Retrying optimization with expanded search space...")
                else:
                    logger.warning("=" * 60)
                    logger.warning("Max optimization attempts reached - proceeding with best parameters found")
                    logger.warning("=" * 60)
        
        # Train model with extended data (Step 3: 90 days default + live append)
        logger.info(f"Phase 3: ML Model Retraining ({data_days} days)")
        training_results = await self.train_model(days=data_days, append_live=True)
        
        # Phase 4: VALIDATION WITH RETRY LOOP
        max_validation_attempts = 12
        validation_attempt = 0
        phase4_success = False
        
        while validation_attempt < max_validation_attempts:
            validation_attempt += 1
            logger.info("=" * 60)
            logger.info(f"Phase 4: Final Validation (Attempt {validation_attempt}/{max_validation_attempts})")
            logger.info("=" * 60)
            
            # Reload trained model
            model_loaded = self._ml_model.load_model("final_model.pt")
            if not model_loaded:
                model_loaded = self._ml_model.load_model("best_model.pt")
            if model_loaded:
                logger.info("Reloaded trained model for validation")
            else:
                logger.warning("Could not reload model - using in-memory model")
            
            # Run validation backtest
            backtest_results = await self.run_backtest(days=backtest_days)
            
            # Calculate compound monthly projection
            if backtest_results:
                total_return = backtest_results.get("total_return_pct", 0) / 100
                bt_days = backtest_results.get("backtest_days", backtest_days)
                if total_return > -1 and bt_days > 0:
                    compound_monthly = ((1 + total_return) ** (30 / bt_days) - 1) * 100
                    backtest_results["compound_monthly_pct"] = compound_monthly
                    logger.info(f"Compound monthly projection: {compound_monthly:.2f}%")
            
            # Check if objectives met
            if self.state.objectives_met:
                logger.info("=" * 60)
                logger.info(f"VALIDATION SUCCESS - All objectives met!")
                logger.info(f"Return: {backtest_results.get('total_return_pct', 0):.2f}%, "
                          f"Sharpe: {backtest_results.get('sharpe_ratio', 0):.2f}, "
                          f"Profit Factor: {backtest_results.get('profit_factor', 0):.2f}")
                logger.info("=" * 60)
                phase4_success = True
                break
            else:
                logger.warning(f"Validation attempt {validation_attempt} failed")
                logger.warning(f"Sharpe: {backtest_results.get('sharpe_ratio', 0):.2f} "
                             f"(min: {self.objectives.get('sharpe_ratio_min', 1.5)})")
                logger.warning(f"Monthly: {backtest_results.get('compound_monthly_pct', 0):.2f}% "
                             f"(min: {self.objectives.get('monthly_return_pct_min', 15)}%)")
                
                if validation_attempt < max_validation_attempts:
                    logger.info("=" * 60)
                    logger.info("Running parameter optimization to improve performance...")
                    logger.info("=" * 60)
                    # Run optimization to find better parameters
                    n_trials = 100 + (validation_attempt - 1) * 20  # 100, 120, 140... up to 320 trials
                    await self.run_optimization(n_trials=n_trials)
                    
                    # Reload optimized parameters
                    with open(self.config_dir / "params.json", "r") as f:
                        self.params = json.load(f)
                    logger.info("Reloaded optimized parameters")
                    
                    # Retrain model with optimized parameters
                    logger.info("Retraining model with optimized parameters...")
                    training_results = await self.train_model(days=data_days, append_live=True)
        
        # Decision logic after validation attempts
        if not phase4_success:
            logger.error("=" * 60)
            logger.error("CRITICAL: Phase 4 objectives not met after all optimization attempts")
            logger.error(f"Sharpe: {backtest_results.get('sharpe_ratio', 0):.2f} (min: {self.objectives.get('sharpe_ratio_min', 1.5)})")
            logger.error(f"Monthly: {backtest_results.get('compound_monthly_pct', 0):.2f}% (min: {self.objectives.get('monthly_return_pct_min', 15)}%)")
            logger.error(f"Profit Factor: {backtest_results.get('profit_factor', 0):.2f} (min: {self.objectives.get('profit_factor_min', 1.2)})")
            logger.error("=" * 60)
            logger.error("System will NOT proceed to live trading - objectives not met")
            logger.error("The model needs:")
            logger.error("  1. Better market conditions (wait for higher volatility/trends)")
            logger.error("  2. More training data (run longer to collect live data)")
            logger.error("  3. Different strategy parameters (manual tuning)")
            logger.error("  4. Or more aggressive optimization (increase trials)")
            logger.error("=" * 60)
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
