"""
Risk Manager Module
Advanced risk management for autonomous HFT system
Implements dynamic position sizing, drawdown protection, and Kelly criterion
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import numpy as np
import pandas as pd

from src.hyperliquid_api import HyperliquidAPI, Position

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Current risk metrics snapshot"""
    timestamp: float
    current_drawdown: float
    max_drawdown: float
    daily_pnl: float
    daily_pnl_pct: float
    volatility: float
    var_95: float
    expected_shortfall: float
    position_heat: float  # % of capital at risk
    win_rate: float
    profit_factor: float
    sharpe_ratio: float


@dataclass
class TradeRecord:
    """Record of a completed trade"""
    timestamp: float
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    hold_time: float
    fees: float


class RiskManager:
    """
    Comprehensive risk management system for HFT
    Handles position sizing, drawdown protection, and risk limits
    """
    
    def __init__(
        self,
        api: HyperliquidAPI,
        config_path: str = "config/params.json",
        objectives_path: str = "config/objectives.json"
    ):
        self.api = api
        
        # Load configurations
        with open(config_path, "r") as f:
            self.params = json.load(f)
        with open(objectives_path, "r") as f:
            self.objectives = json.load(f)
        
        self.risk_config = self.params.get("risk", {})
        
        # Trading state
        self.starting_capital: float = 0.0
        self.current_capital: float = 0.0
        self.peak_capital: float = 0.0
        self.daily_start_capital: float = 0.0
        
        # Step 1: Enhanced PnL tracking
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.session_start_capital: float = 0.0  # Capital at session start
        self._last_fill_time: float = 0.0
        self._pending_fill_pnl: float = 0.0  # PnL from fills not yet synced
        
        # Trade history
        self.trade_history: deque = deque(maxlen=1000)
        self.pnl_history: deque = deque(maxlen=10000)
        
        # Risk state
        self.is_halted: bool = False
        self.halt_reason: str = ""
        self.risk_level: str = "normal"  # normal, elevated, critical
        
        # Metrics cache
        self._last_metrics: Optional[RiskMetrics] = None
        self._metrics_update_time: float = 0.0
        
    async def initialize(self, starting_capital: float) -> None:
        """Initialize risk manager with starting capital"""
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.peak_capital = starting_capital
        self.daily_start_capital = starting_capital
        self.session_start_capital = starting_capital
        
        # Step 4: Compounding tracking
        self.profits_to_reinvest = 0.0
        self.reinvestment_rate = self.risk_config.get("reinvestment_rate", 0.5)  # 50% of profits
        
        # Step 1: Register fill callback with API for real-time PnL updates
        if hasattr(self.api, 'register_fill_callback'):
            self.api.register_fill_callback(self._on_fill_event)
            logger.info("Registered fill callback for real-time PnL tracking")
        
        logger.info(f"Risk manager initialized with ${starting_capital:,.2f}")
    
    async def _on_fill_event(self, fill_data: Dict[str, Any]) -> None:
        """Step 1: Handle fill event from WebSocket for real-time PnL"""
        try:
            closed_pnl = float(fill_data.get("closedPnl", 0))
            fee = float(fill_data.get("fee", 0))
            
            if closed_pnl != 0:
                self.realized_pnl += closed_pnl
                self._pending_fill_pnl += closed_pnl
                self._last_fill_time = time.time()
                logger.info(f"💰 Realized PnL updated: ${closed_pnl:+.2f} (total: ${self.realized_pnl:+.2f})")
            
            self.total_fees += fee
            
        except Exception as e:
            logger.error(f"Error processing fill for PnL: {e}")
    
    async def update_capital(self, new_capital: float) -> None:
        """Update current capital and track metrics"""
        old_capital = self.current_capital
        self.current_capital = new_capital
        
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
            
            # Step 4: Track profits for reinvestment in GOOD status
            if self.risk_level == "normal":
                profit = new_capital - old_capital
                if profit > 0:
                    self.profits_to_reinvest += profit * self.reinvestment_rate
        
        # Record PnL point
        self.pnl_history.append({
            "timestamp": time.time(),
            "capital": new_capital,
            "pnl": new_capital - self.starting_capital
        })
        
        # Check for auto-stop conditions
        await self._check_risk_limits()
    
    async def record_trade(self, trade: TradeRecord) -> None:
        """Record completed trade"""
        self.trade_history.append(trade)
        
        # Update capital
        new_capital = self.current_capital + trade.pnl - trade.fees
        await self.update_capital(new_capital)
        
        logger.info(
            f"Trade recorded: {trade.side} {trade.size} @ {trade.entry_price} -> {trade.exit_price}, "
            f"PnL: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)"
        )
    
    async def _check_risk_limits(self) -> None:
        """Check and enforce risk limits"""
        metrics = await self.calculate_metrics()
        
        # Auto-stop if drawdown exceeds limit
        auto_stop_dd = self.objectives.get("auto_stop_drawdown", 4.0)
        if metrics.current_drawdown >= auto_stop_dd:
            self.is_halted = True
            self.halt_reason = f"Drawdown limit exceeded: {metrics.current_drawdown:.2f}%"
            self.risk_level = "critical"
            logger.critical(self.halt_reason)
            return
        
        # Check daily loss limit
        max_daily_loss = self.risk_config.get("max_daily_loss_pct", 2.0)
        if metrics.daily_pnl_pct <= -max_daily_loss:
            self.is_halted = True
            self.halt_reason = f"Daily loss limit exceeded: {metrics.daily_pnl_pct:.2f}%"
            self.risk_level = "critical"
            logger.critical(self.halt_reason)
            return
        
        # Update risk level
        if metrics.current_drawdown >= auto_stop_dd * 0.75:
            self.risk_level = "critical"
        elif metrics.current_drawdown >= auto_stop_dd * 0.5:
            self.risk_level = "elevated"
        else:
            self.risk_level = "normal"
    
    async def calculate_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics with extended caching for low latency"""
        now = time.time()
        
        # Step 5: Use cached metrics if recent enough (< 3 seconds for faster trade checks)
        if self._last_metrics and now - self._metrics_update_time < 3.0:
            return self._last_metrics
        
        # Calculate drawdown
        current_drawdown = 0.0
        if self.peak_capital > 0:
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
        
        # Calculate max drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Daily PnL
        daily_pnl = self.current_capital - self.daily_start_capital
        daily_pnl_pct = (daily_pnl / self.daily_start_capital * 100) if self.daily_start_capital > 0 else 0
        
        # Trade statistics
        win_rate = self._calculate_win_rate()
        profit_factor = self._calculate_profit_factor()
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Volatility and VaR
        volatility = self._calculate_volatility()
        var_95, es = self._calculate_var_es()
        
        # Position heat
        position_heat = await self._calculate_position_heat()
        
        metrics = RiskMetrics(
            timestamp=now,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            volatility=volatility,
            var_95=var_95,
            expected_shortfall=es,
            position_heat=position_heat,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio
        )
        
        self._last_metrics = metrics
        self._metrics_update_time = now
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from PnL history"""
        if len(self.pnl_history) < 2:
            return 0.0
        
        capitals = [p["capital"] for p in self.pnl_history]
        peak = capitals[0]
        max_dd = 0.0
        
        for capital in capitals:
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        if not self.trade_history:
            return 0.0
        
        wins = sum(1 for t in self.trade_history if t.pnl > 0)
        return wins / len(self.trade_history) * 100
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not self.trade_history:
            return 1.0  # Neutral when no trades yet (avoids triggering CRITICAL)
        
        gross_profit = sum(t.pnl for t in self.trade_history if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trade_history if t.pnl < 0))
        
        if gross_loss == 0:
            return 2.0 if gross_profit > 0 else 1.0  # Good or neutral
        
        return gross_profit / gross_loss
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(self.pnl_history) < 10:
            return 0.0
        
        capitals = [p["capital"] for p in self.pnl_history]
        returns = [(capitals[i] - capitals[i-1]) / capitals[i-1] 
                   for i in range(1, len(capitals)) if capitals[i-1] > 0]
        
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming minute data, 525600 minutes per year)
        annualized_return = mean_return * 525600
        annualized_std = std_return * np.sqrt(525600)
        
        return (annualized_return - risk_free_rate) / annualized_std
    
    def _calculate_volatility(self) -> float:
        """Calculate annualized volatility"""
        if len(self.pnl_history) < 20:
            return 0.0
        
        capitals = [p["capital"] for p in self.pnl_history]
        returns = [(capitals[i] - capitals[i-1]) / capitals[i-1] 
                   for i in range(1, len(capitals)) if capitals[i-1] > 0]
        
        if not returns:
            return 0.0
        
        return np.std(returns) * np.sqrt(525600) * 100
    
    def _calculate_var_es(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Expected Shortfall"""
        if len(self.pnl_history) < 100:
            return 0.0, 0.0
        
        capitals = [p["capital"] for p in self.pnl_history]
        returns = [(capitals[i] - capitals[i-1]) / capitals[i-1] 
                   for i in range(1, len(capitals)) if capitals[i-1] > 0]
        
        if not returns:
            return 0.0, 0.0
        
        returns = np.array(returns)
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        var_95 = -var_threshold * self.current_capital
        
        # Expected shortfall (average of losses beyond VaR)
        tail_losses = returns[returns <= var_threshold]
        if len(tail_losses) > 0:
            es = float(-np.mean(tail_losses) * self.current_capital)
        else:
            es = var_95
        
        return float(var_95), float(es)
    
    async def _calculate_position_heat(self) -> float:
        """Calculate percentage of capital at risk in positions"""
        try:
            positions = await self.api.get_positions()
            
            total_risk = 0.0
            for pos in positions:
                if pos.size != 0:
                    # Risk = position value * estimated max loss (SL)
                    sl_pct = self.params["trading"]["stop_loss_pct"] / 100
                    position_value = abs(pos.size * pos.entry_price)
                    total_risk += position_value * sl_pct
            
            return (total_risk / self.current_capital * 100) if self.current_capital > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating position heat: {e}")
            return 0.0
    
    def calculate_position_size(
        self,
        signal_strength: float,
        current_volatility: float,
        entry_price: float
    ) -> float:
        """
        Calculate optimal position size using modified Kelly criterion
        Accounts for signal strength, volatility, and risk limits
        XYZ100-optimized with leverage cap and high vol reduction
        """
        if self.is_halted:
            return 0.0
        
        # Base position size from params
        base_size_pct = self.params["trading"]["position_size_pct"]
        
        # XYZ100 leverage cap (default 20x max)
        xyz100_leverage_cap = self.risk_config.get("xyz100_leverage_cap", 20)
        max_leverage = self.params["trading"].get("max_leverage", xyz100_leverage_cap)
        current_leverage = self.params["trading"].get("leverage", 1)
        if current_leverage > max_leverage:
            logger.warning(f"Leverage {current_leverage}x exceeds XYZ100 cap {max_leverage}x, reducing")
            current_leverage = max_leverage
        
        # Kelly fraction (reduced for safety)
        kelly_fraction = self.risk_config.get("kelly_fraction", 0.25)
        
        # Win rate and win/loss ratio from history
        win_rate = self._calculate_win_rate() / 100
        if win_rate == 0:
            win_rate = 0.5  # Default assumption
        
        # Calculate average win/loss
        wins = [t.pnl_pct for t in self.trade_history if t.pnl > 0]
        losses = [abs(t.pnl_pct) for t in self.trade_history if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else self.params["trading"]["take_profit_pct"]
        avg_loss = np.mean(losses) if losses else self.params["trading"]["stop_loss_pct"]
        
        # Kelly criterion: f = (p * b - q) / b
        # where p = win_rate, q = 1 - p, b = avg_win / avg_loss
        # Step 4: Apply growth_factor for optimistic sizing in good conditions
        growth_factor = self.risk_config.get("growth_factor", 1.2) if self.risk_level == "normal" else 1.0
        
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly_pct = ((win_rate * b) - (1 - win_rate)) / b
            kelly_pct = max(0, min(kelly_pct, 1)) * kelly_fraction * growth_factor
        else:
            kelly_pct = base_size_pct * growth_factor
        
        # Adjust for volatility (reduce size in high vol) - Step 2: Less aggressive reduction
        # XYZ100: High vol threshold triggers significant position reduction
        high_vol_threshold = self.risk_config.get("high_vol_threshold", 0.03)
        high_vol_reduction = self.risk_config.get("high_vol_position_reduction", 0.5)
        
        vol_lookback = self.risk_config.get("volatility_lookback", 20)
        if current_volatility > 0:
            # XYZ100: More aggressive reduction in high vol (equity-like behavior)
            if current_volatility > high_vol_threshold:
                vol_adjustment = high_vol_reduction
                logger.info(f"⚠️ High vol detected ({current_volatility:.4f} > {high_vol_threshold}), reducing position to {high_vol_reduction*100:.0f}%")
            else:
                vol_adjustment = 0.02 / current_volatility  # Target 2% vol
                vol_adjustment = max(0.6, min(vol_adjustment, 1.5))  # Step 2: Higher floor (0.6 vs 0.5)
        else:
            vol_adjustment = 1.0
        
        # Adjust for signal strength (scale with confidence) - Step 2: Higher floor
        signal_adjustment = max(0.4, min(abs(signal_strength) + 0.2, 1.2))  # Step 2: +0.2 boost
        
        # Step 2: More aggressive risk adjustments for trade generation
        risk_adjustments = {
            "normal": 1.5,    # Step 2: More aggressive in normal mode (was 1.2)
            "elevated": 0.8,  # Step 2: Less conservative (was 0.6)
            "critical": 0.5   # Step 2: Still trade in critical (was 0.4)
        }
        risk_adjustment = risk_adjustments.get(self.risk_level, 1.0)
        
        # Calculate final size
        size_pct = kelly_pct * vol_adjustment * signal_adjustment * risk_adjustment
        
        # Step 1: Enforce minimum position size to enable trades even in critical mode
        min_size_pct = self.params["trading"].get("min_position_size_pct", 0.05)
        size_pct = max(size_pct, min_size_pct)
        
        # Step 2: Allow exceeding base_size in good conditions (up to 75% more)
        max_size = base_size_pct * 1.75 if self.risk_level == "normal" else base_size_pct * 1.2
        size_pct = min(size_pct, max_size)
        
        # Convert to actual size
        position_value = self.current_capital * (size_pct / 100)
        position_size = position_value / entry_price
        
        # Enforce exchange minimum order size (0.001 BTC for Hyperliquid)
        min_order_size = 0.001
        if position_size < min_order_size:
            # Step 2: Scale up to minimum if we have enough capital (higher threshold: 15%)
            min_position_value = min_order_size * entry_price
            if min_position_value <= self.current_capital * 0.15:  # Step 2: Max 15% per trade (was 10%)
                position_size = min_order_size
                logger.info(f"📈 Position scaled to minimum: {position_size:.4f} BTC (${min_position_value:.2f})")
            else:
                logger.warning(f"⚠️ Insufficient capital for min trade: need ${min_position_value:.2f}, have ${self.current_capital:.2f}")
                return 0.0
        
        # Round to appropriate decimals
        position_size = round(position_size, 4)
        
        logger.info(
            f"📊 Position sizing: kelly={kelly_pct:.4f}, vol_adj={vol_adjustment:.2f}, "
            f"signal_adj={signal_adjustment:.2f}, risk_adj={risk_adjustment:.2f}, "
            f"final_size={position_size:.4f} BTC"
        )
        
        return position_size
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        current_volatility: float,
        atr: float = 0.0
    ) -> float:
        """
        Step 2: Calculate dynamic stop loss based on ATR and volatility.
        Uses tighter SL for faster realizations while respecting volatility.
        """
        base_sl_pct = self.params["trading"]["stop_loss_pct"]
        sl_mult = self.params["trading"].get("sl_mult", 1.0)  # Step 2: ATR multiplier
        
        # Step 2: Use ATR-based SL if available for tighter stops
        if atr > 0 and entry_price > 0:
            # ATR-based SL: entry ± (ATR * multiplier)
            atr_sl_distance = atr * sl_mult
            atr_sl_pct = (atr_sl_distance / entry_price) * 100
            
            # Use tighter of ATR-based or percentage-based
            sl_pct = min(atr_sl_pct, base_sl_pct)
            logger.debug(f"ATR-based SL: {atr_sl_pct:.3f}% (ATR={atr:.2f}, mult={sl_mult})")
        else:
            # Fallback to volatility-adjusted percentage
            if current_volatility > 0:
                vol_factor = max(0.5, min(current_volatility / 0.02, 1.5))
                sl_pct = base_sl_pct * vol_factor
            else:
                sl_pct = base_sl_pct
        
        # Step 2: Enforce tighter bounds for faster realizations
        min_sl_pct = self.params["trading"].get("min_stop_loss_pct", 0.15)
        max_sl_pct = 1.5  # Tighter max for HFT
        sl_pct = max(min_sl_pct, min(sl_pct, max_sl_pct))
        
        if side == "B":  # Long position
            sl_price = entry_price * (1 - sl_pct / 100)
        else:  # Short position
            sl_price = entry_price * (1 + sl_pct / 100)
        
        # Round to whole dollars for BTC tick size
        return int(round(sl_price, 0))
    
    def calculate_take_profit(
        self,
        entry_price: float,
        side: str,
        signal_strength: float,
        atr: float = 0.0
    ) -> float:
        """
        Step 2: Calculate dynamic take profit based on ATR and signal strength.
        Uses tighter TP for faster realizations.
        """
        base_tp_pct = self.params["trading"]["take_profit_pct"]
        tp_mult = self.params["trading"].get("tp_mult", 1.5)  # Step 2: ATR multiplier
        
        # Step 2: Use ATR-based TP if available
        if atr > 0 and entry_price > 0:
            # ATR-based TP: entry ± (ATR * multiplier)
            atr_tp_distance = atr * tp_mult
            atr_tp_pct = (atr_tp_distance / entry_price) * 100
            
            # Use the tighter of ATR or signal-adjusted percentage
            signal_adj_tp = base_tp_pct * (1 + abs(signal_strength) * 0.5)
            tp_pct = min(atr_tp_pct, signal_adj_tp)
            logger.debug(f"ATR-based TP: {atr_tp_pct:.3f}% (ATR={atr:.2f}, mult={tp_mult})")
        else:
            # Fallback to signal strength adjustment
            tp_pct = base_tp_pct * (1 + abs(signal_strength) * 0.3)
        
        # Step 2: Enforce tighter bounds for faster realizations
        min_tp_pct = self.params["trading"].get("min_take_profit_pct", 0.2)
        max_tp_pct = 2.5  # Tighter max for HFT
        tp_pct = max(min_tp_pct, min(tp_pct, max_tp_pct))
        
        if side == "B":  # Long position
            tp_price = entry_price * (1 + tp_pct / 100)
        else:  # Short position
            tp_price = entry_price * (1 - tp_pct / 100)
        
        # Round to whole dollars for BTC tick size
        return int(round(tp_price, 0))
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        side: str,
        highest_price: float = 0.0,
        lowest_price: float = float('inf')
    ) -> Tuple[float, bool]:
        """
        Step 4: Calculate trailing stop loss for position protection.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            side: "B" for long, "A" for short
            highest_price: Highest price since entry (for longs)
            lowest_price: Lowest price since entry (for shorts)
        
        Returns:
            (trailing_sl_price, should_close): New SL price and whether to close position
        """
        trailing_pct = self.params["trading"].get("trailing_sl_pct", 0.15)
        trigger_pct = self.params["trading"].get("trailing_sl_trigger_pct", 0.2)
        
        if side == "B":  # Long position
            # Calculate profit percentage
            profit_pct = (current_price - entry_price) / entry_price * 100
            
            # Update highest price
            if current_price > highest_price or highest_price == 0:
                highest_price = current_price
            
            # Only trail if profit exceeds trigger
            if profit_pct >= trigger_pct:
                # Trailing stop at trigger_pct below highest
                trail_sl = highest_price * (1 - trailing_pct / 100)
                
                # Check if current price breaks trailing stop
                if current_price <= trail_sl:
                    logger.info(
                        f"🔻 TRAILING STOP triggered: price ${current_price:.2f} <= trail ${trail_sl:.2f}"
                    )
                    return trail_sl, True
                
                return trail_sl, False
            
            # Not yet in profit trigger zone
            return entry_price * (1 - self.params["trading"]["stop_loss_pct"] / 100), False
            
        else:  # Short position
            profit_pct = (entry_price - current_price) / entry_price * 100
            
            if current_price < lowest_price or lowest_price == float('inf'):
                lowest_price = current_price
            
            if profit_pct >= trigger_pct:
                trail_sl = lowest_price * (1 + trailing_pct / 100)
                
                if current_price >= trail_sl:
                    logger.info(
                        f"🔺 TRAILING STOP triggered: price ${current_price:.2f} >= trail ${trail_sl:.2f}"
                    )
                    return trail_sl, True
                
                return trail_sl, False
            
            return entry_price * (1 + self.params["trading"]["stop_loss_pct"] / 100), False
    
    async def should_allow_trade(self) -> Tuple[bool, str]:
        """Check if new trades should be allowed"""
        if self.is_halted:
            return False, self.halt_reason
        
        metrics = await self.calculate_metrics()
        
        # Check position heat limit
        max_heat = 50.0  # Max 50% of capital at risk
        if metrics.position_heat >= max_heat:
            return False, f"Position heat limit: {metrics.position_heat:.1f}%"
        
        # Allow trades in critical mode with reduced size (handled in calculate_position_size)
        # Don't block completely - this was causing 0 trades issue
        if self.risk_level == "critical":
            logger.info("Trading in critical mode with reduced position size (25%)")
        
        return True, "OK"
    
    async def reset_daily_metrics(self) -> None:
        """Reset daily tracking (call at start of new day)"""
        self.daily_start_capital = self.current_capital
        logger.info(f"Daily metrics reset. Starting capital: ${self.current_capital:,.2f}")
    
    async def resume_trading(self) -> bool:
        """Attempt to resume trading after halt"""
        if not self.is_halted:
            return True
        
        # Recalculate metrics
        metrics = await self.calculate_metrics()
        
        # Only resume if drawdown has recovered
        auto_stop_dd = self.objectives.get("auto_stop_drawdown", 4.0)
        resume_threshold = auto_stop_dd * 0.5
        
        if metrics.current_drawdown < resume_threshold:
            self.is_halted = False
            self.halt_reason = ""
            self.risk_level = "elevated"  # Start cautious
            logger.info("Trading resumed - operating in elevated risk mode")
            return True
        
        return False
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get current risk status summary"""
        metrics = self._last_metrics
        
        return {
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "risk_level": self.risk_level,
            "current_capital": self.current_capital,
            "starting_capital": self.starting_capital,
            "total_pnl": self.current_capital - self.starting_capital,
            "total_pnl_pct": (self.current_capital - self.starting_capital) / self.starting_capital * 100 if self.starting_capital > 0 else 0,
            "current_drawdown": metrics.current_drawdown if metrics else 0,
            "max_drawdown": metrics.max_drawdown if metrics else 0,
            "win_rate": metrics.win_rate if metrics else 0,
            "profit_factor": metrics.profit_factor if metrics else 0,
            "sharpe_ratio": metrics.sharpe_ratio if metrics else 0,
            "trade_count": len(self.trade_history)
        }
    
    # ==================== Step 2: PnL Sync Methods ====================
    
    async def sync_capital_from_exchange(self) -> float:
        """
        Step 2: Sync capital from exchange account state.
        Pulls actual wallet balance to update PnL tracking.
        """
        try:
            user_state = await self.api.get_user_state()
            
            # Get account value (cash + unrealized PnL)
            margin_summary = user_state.get("marginSummary", {})
            account_value = float(margin_summary.get("accountValue", 0))
            
            if account_value > 0:
                old_capital = self.current_capital
                pnl_change = account_value - old_capital
                
                if abs(pnl_change) > 0.01:  # Only log significant changes
                    logger.info(
                        f"Capital synced from exchange: ${old_capital:.2f} -> ${account_value:.2f} "
                        f"(PnL: ${pnl_change:+.2f})"
                    )
                
                await self.update_capital(account_value)
            
            return account_value
        except Exception as e:
            logger.warning(f"Failed to sync capital from exchange: {e}")
            return self.current_capital
    
    async def process_fill_event(self, fill_data: Dict[str, Any]) -> None:
        """
        Step 2: Process a trade fill event from WebSocket.
        Updates PnL and records the trade.
        """
        try:
            side = fill_data.get("side", "")
            size = float(fill_data.get("sz", 0))
            price = float(fill_data.get("px", 0))
            fee = float(fill_data.get("fee", 0))
            closed_pnl = float(fill_data.get("closedPnl", 0))
            
            if closed_pnl != 0:
                # This fill closed a position - record the trade
                trade_record = TradeRecord(
                    timestamp=time.time(),
                    symbol=fill_data.get("coin", "BTC"),
                    side=side,
                    entry_price=0,  # Will be updated from position history
                    exit_price=price,
                    size=size,
                    pnl=closed_pnl,
                    pnl_pct=(closed_pnl / (size * price)) * 100 if size * price > 0 else 0,
                    hold_time=0,
                    fees=fee
                )
                await self.record_trade(trade_record)
                logger.info(f"Fill processed: {side} {size} @ {price}, PnL: ${closed_pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing fill event: {e}")
    
    async def poll_and_sync_pnl(self) -> Dict[str, Any]:
        """
        Optimized PnL sync with minimal API calls.
        Uses caching and batched requests to reduce rate limit hits.
        """
        try:
            # SINGLE API call: get_user_state() (cached for 2s)
            # This call provides positions AND account value
            state = await self.api.get_user_state()
            margin = state.get('marginSummary', {})
            account_value = float(margin.get('accountValue', self.current_capital))
            
            # Update capital from account value
            if account_value > 0:
                old_capital = self.current_capital
                if abs(account_value - old_capital) > 0.01:
                    logger.debug(f"Capital synced: ${old_capital:.2f} -> ${account_value:.2f}")
                await self.update_capital(account_value)
            
            # Extract positions from same state response (no extra API call)
            positions = []
            for pos in state.get("assetPositions", []):
                position_data = pos.get("position", {})
                if float(position_data.get("szi", 0)) != 0:
                    positions.append({
                        "symbol": position_data.get("coin", ""),
                        "size": float(position_data.get("szi", 0)),
                        "entry_price": float(position_data.get("entryPx", 0)),
                        "unrealized_pnl": float(position_data.get("unrealizedPnl", 0)),
                    })
            
            # Calculate unrealized from positions (use API-provided, avoid extra orderbook calls)
            self.unrealized_pnl = sum(p["unrealized_pnl"] for p in positions)
            
            # SECOND API call: get_user_fills (cached for 5s)
            fills = await self.api.get_user_fills(limit=50)
            session_realized = 0.0
            for fill in fills:
                if hasattr(fill, 'closed_pnl') and fill.closed_pnl != 0:
                    session_realized += fill.closed_pnl
            
            if session_realized != 0:
                self.realized_pnl = session_realized
            
            # Get realized from API tracker if available
            api_realized = self.api.get_total_realized_pnl() if hasattr(self.api, 'get_total_realized_pnl') else 0
            if api_realized != 0:
                self.realized_pnl = api_realized
            
            total_pnl = self.realized_pnl + self.unrealized_pnl
            account_pnl = account_value - self.starting_capital
            
            # Log every 30 seconds to reduce log spam
            if not hasattr(self, '_last_pnl_log') or time.time() - self._last_pnl_log > 30:
                logger.info(
                    f"PnL: realized=${self.realized_pnl:+.2f}, "
                    f"unrealized=${self.unrealized_pnl:+.2f}, "
                    f"account=${account_pnl:+.2f}"
                )
                self._last_pnl_log = time.time()
            
            return {
                "capital": account_value,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "total_pnl": total_pnl,
                "account_pnl": account_pnl,
                "fees": self.total_fees,
                "open_positions": len(positions)
            }
        except Exception as e:
            logger.error(f"Error polling PnL: {e}")
            return {"capital": self.current_capital, "total_pnl": 0, "realized_pnl": self.realized_pnl}
