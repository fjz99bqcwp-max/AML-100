"""
Optimizer Module
Autonomous parameter optimization using Bayesian optimization
Adjusts trading parameters based on performance status
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import optuna
from optuna.samplers import TPESampler

# Suppress Optuna's verbose trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


class PerformanceStatus(Enum):
    """Trading performance status levels"""
    CRITICAL = "critical"    # Very poor - prioritize risk reduction
    POOR = "poor"            # Below objectives - minor adjustments
    MODERATE = "moderate"    # Aligned with objectives - no changes
    GOOD = "good"            # Above expectations - slight risk increase


@dataclass
class OptimizationResult:
    """Result of an optimization run"""
    timestamp: float
    status: PerformanceStatus
    original_params: Dict[str, Any]
    optimized_params: Dict[str, Any]
    improvement_pct: float
    iterations: int
    best_score: float


class ParameterOptimizer:
    """
    Autonomous parameter optimizer for HFT system
    Uses Bayesian optimization with status-based adjustments
    """
    
    def __init__(
        self,
        params_path: str = "config/params.json",
        objectives_path: str = "config/objectives.json"
    ):
        self.params_path = Path(params_path)
        self.objectives_path = Path(objectives_path)
        
        # Load configurations
        self._load_configs()
        
        # Optimization state
        self.optimization_history: List[OptimizationResult] = []
        self.current_study: Optional[optuna.Study] = None
        
    def _load_configs(self) -> None:
        """Load configuration files"""
        with open(self.params_path, "r") as f:
            self.params = json.load(f)
        with open(self.objectives_path, "r") as f:
            self.objectives = json.load(f)
        
        self.opt_config = self.params.get("optimization", {})
        self.param_bounds = self.opt_config.get("param_bounds", {})
    
    def _save_params(self) -> None:
        """Save updated parameters"""
        with open(self.params_path, "w") as f:
            json.dump(self.params, f, indent=4)
        logger.info("Parameters saved to config/params.json")
    
    def determine_status(
        self,
        metrics: Dict[str, float],
        trade_count: int = 0,
        cycle_number: int = 0,
        objectives_met_phase4: bool = False,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        projected_monthly: float = 0.0
    ) -> PerformanceStatus:
        """
        Determine current performance status based on metrics.
        
        Step 5 Enhanced: Include realized + unrealized PnL in status determination.
        - Consider trade_count to avoid critical on PnL=0 with active trades
        - Add warm-up period (first 5 cycles) with more lenient thresholds
        - If Phase 4 objectives met, use MODERATE even if live metrics are poor
        - Include unrealized/realized PnL and projected monthly in decision
        
        Args:
            metrics: Performance metrics (sharpe_ratio, max_drawdown, profit_factor)
            trade_count: Number of trades in the current cycle
            cycle_number: Current cycle number (for warm-up detection)
            objectives_met_phase4: Whether Phase 4 validation passed
            unrealized_pnl: Current unrealized PnL (Step 3)
            realized_pnl: Current realized PnL (Step 5)
            projected_monthly: Current projected monthly return (Step 3)
        """
        thresholds = self.objectives.get("status_thresholds", {})
        status_pnl_thresh = self.objectives.get("status_pnl_thresh", 0.0)
        
        sharpe = metrics.get("sharpe_ratio", 0)
        drawdown = metrics.get("max_drawdown", 100)
        profit_factor = metrics.get("profit_factor", 0)
        
        # Step 5: Calculate total PnL for status
        total_pnl = realized_pnl + unrealized_pnl
        
        # Step 3: Warm-up period - first 5 cycles get MODERATE status
        min_cycles_for_status = self.objectives.get("min_cycles_for_status", 5)
        if cycle_number > 0 and cycle_number <= min_cycles_for_status:
            logger.info(f"Warm-up period (cycle {cycle_number}/{min_cycles_for_status}) - using MODERATE status")
            return PerformanceStatus.MODERATE
        
        # Step 5: If total PnL is positive with trades, use GOOD status
        if total_pnl > status_pnl_thresh and trade_count >= 5:
            logger.info(f"Positive total PnL ${total_pnl:.2f} with {trade_count} trades - using GOOD")
            return PerformanceStatus.GOOD
        
        # Step 3: If Phase 4 objectives were met, be lenient on live metrics
        if objectives_met_phase4 and drawdown < 2.0:
            if trade_count > 0:
                logger.debug(f"Phase 4 objectives met, {trade_count} trades - using MODERATE")
                return PerformanceStatus.MODERATE
        
        # Step 3: If trades executed but PnL=0, assume fills pending - use MODERATE
        min_trades_for_status = self.objectives.get("min_trades_for_status", 3)
        if trade_count >= min_trades_for_status:
            if profit_factor == 1.0 and sharpe == 0:
                logger.info(f"{trade_count} trades executed, PnL pending - using MODERATE (not critical)")
                return PerformanceStatus.MODERATE
            # Step 3: With trades, only go critical if drawdown is severe
            if drawdown < 3.0:
                logger.debug(f"{trade_count} trades with manageable drawdown {drawdown:.2f}% - avoiding critical")
                if profit_factor >= 0.9:
                    return PerformanceStatus.MODERATE
                else:
                    return PerformanceStatus.POOR
        
        # Step 3: Check unrealized PnL - if positive, not critical
        if unrealized_pnl > 0 and trade_count > 0:
            logger.debug(f"Positive unrealized PnL ${unrealized_pnl:.2f} - using MODERATE")
            return PerformanceStatus.MODERATE
        
        # Step 3: Check projected monthly - if meeting targets, use GOOD
        if projected_monthly >= 5.0 and drawdown < 2.0:
            logger.info(f"Projected monthly {projected_monthly:.2f}% meets target - using GOOD")
            return PerformanceStatus.GOOD
        elif projected_monthly >= 3.0 and drawdown < 3.0:
            return PerformanceStatus.MODERATE
        
        # Check for CRITICAL status - with stricter thresholds
        critical = thresholds.get("critical", {})
        if ((sharpe < critical.get("sharpe_below", 0.3) and trade_count >= 10) or
            drawdown > critical.get("drawdown_above", 3.5) or
            (profit_factor < critical.get("profit_factor_below", 0.7) and trade_count >= 10)):
            # Step 3: Log reason for critical
            logger.warning(
                f"CRITICAL status: sharpe={sharpe:.2f}, drawdown={drawdown:.2f}%, "
                f"pf={profit_factor:.2f}, trades={trade_count}"
            )
            return PerformanceStatus.CRITICAL
        
        # Check for POOR status
        poor = thresholds.get("poor", {})
        if (sharpe < poor.get("sharpe_below", 1.0) or
            drawdown > poor.get("drawdown_above", 2.5) or
            profit_factor < poor.get("profit_factor_below", 1.0)):
            return PerformanceStatus.POOR
        
        # Check for GOOD status
        good = thresholds.get("good", {})
        if (sharpe > good.get("sharpe_above", 2.0) and
            drawdown < good.get("drawdown_below", 1.0) and
            profit_factor > good.get("profit_factor_above", 1.5)):
            return PerformanceStatus.GOOD
        
        # Default to MODERATE
        return PerformanceStatus.MODERATE
    
    def get_adjustment_factors(
        self,
        status: PerformanceStatus
    ) -> Dict[str, float]:
        """
        Get parameter adjustment factors based on status.
        Step 3: More aggressive adjustments for targets.
        """
        factors = {
            PerformanceStatus.CRITICAL: {
                "position_size_mult": 0.5,    # Halve position size
                "tp_mult": 0.8,               # Tighter TP
                "sl_mult": 0.7,               # Tighter SL
                "learning_rate_mult": 1.5,    # Faster adaptation
                "risk_tolerance": 0.5
            },
            PerformanceStatus.POOR: {
                "position_size_mult": 0.8,    # Step 3: Less reduction (was 0.75)
                "tp_mult": 0.95,              # Step 3: Wider TP
                "sl_mult": 0.8,               # Step 3: Tighter SL
                "learning_rate_mult": 1.3,
                "risk_tolerance": 0.7
            },
            PerformanceStatus.MODERATE: {
                "position_size_mult": 1.1,    # Step 3: Slight boost in moderate
                "tp_mult": 1.0,
                "sl_mult": 1.0,
                "learning_rate_mult": 1.0,
                "risk_tolerance": 1.0
            },
            PerformanceStatus.GOOD: {
                "position_size_mult": 1.3,    # Step 3: Increased to 30% boost for growth
                "tp_mult": 1.2,               # Step 3: Wider TP for larger gains
                "sl_mult": 0.9,               # Step 3: Tighter SL for protection
                "learning_rate_mult": 0.85,   # Slower changes
                "risk_tolerance": 1.3
            }
        }
        
        return factors.get(status, factors[PerformanceStatus.MODERATE])
    
    async def quick_adjust(
        self,
        status: PerformanceStatus
    ) -> Dict[str, Any]:
        """
        Quick parameter adjustment based on status (Step 4: enforce minimums).
        Used during live trading cycles.
        
        Improvements:
        - Use config-based minimums to prevent over-reduction
        - Log detailed adjustment info
        - Protect against 0-trade scenarios
        """
        original_params = {
            "take_profit_pct": self.params["trading"]["take_profit_pct"],
            "stop_loss_pct": self.params["trading"]["stop_loss_pct"],
            "position_size_pct": self.params["trading"]["position_size_pct"]
        }
        
        factors = self.get_adjustment_factors(status)
        
        # Apply adjustments
        new_tp = original_params["take_profit_pct"] * factors["tp_mult"]
        new_sl = original_params["stop_loss_pct"] * factors["sl_mult"]
        new_size = original_params["position_size_pct"] * factors["position_size_mult"]
        
        # Step 4: Get minimums from config or use safe defaults
        trading_config = self.params.get("trading", {})
        min_tp = trading_config.get("min_take_profit_pct", 0.3)
        min_sl = trading_config.get("min_stop_loss_pct", 0.2)
        min_size = trading_config.get("min_position_size_pct", 0.05)
        
        # Enforce bounds with config-based minimums
        bounds = self.param_bounds
        self.params["trading"]["take_profit_pct"] = np.clip(
            new_tp,
            max(min_tp, bounds.get("take_profit_pct", [0.1, 2.0])[0]),
            bounds.get("take_profit_pct", [0.1, 2.0])[1]
        )
        self.params["trading"]["stop_loss_pct"] = np.clip(
            new_sl,
            max(min_sl, bounds.get("stop_loss_pct", [0.1, 1.5])[0]),
            bounds.get("stop_loss_pct", [0.1, 1.5])[1]
        )
        self.params["trading"]["position_size_pct"] = np.clip(
            new_size,
            max(min_size, bounds.get("position_size_pct", [0.05, 0.5])[0]),
            bounds.get("position_size_pct", [0.05, 0.5])[1]
        )
        
        # Save changes
        self._save_params()
        
        new_params = {
            "take_profit_pct": self.params["trading"]["take_profit_pct"],
            "stop_loss_pct": self.params["trading"]["stop_loss_pct"],
            "position_size_pct": self.params["trading"]["position_size_pct"]
        }
        
        logger.info(
            f"Quick adjustment for {status.value}: "
            f"TP {original_params['take_profit_pct']:.2f}% -> {new_params['take_profit_pct']:.2f}% (min={min_tp}%), "
            f"SL {original_params['stop_loss_pct']:.2f}% -> {new_params['stop_loss_pct']:.2f}% (min={min_sl}%), "
            f"Size {original_params['position_size_pct']:.2f}% -> {new_params['position_size_pct']:.2f}% (min={min_size}%)"
        )
        
        return {
            "status": status.value,
            "original": original_params,
            "adjusted": new_params,
            "minimums_enforced": {"tp": min_tp, "sl": min_sl, "size": min_size}
        }
    
    async def optimize_parameters(
        self,
        objective_func: Callable,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None
    ) -> OptimizationResult:
        """
        Full Bayesian optimization of parameters
        Uses Optuna with TPE sampler
        """
        if n_trials is None:
            n_trials = self.opt_config.get("max_iterations", 50)
        
        original_params = {
            "take_profit_pct": self.params["trading"]["take_profit_pct"],
            "stop_loss_pct": self.params["trading"]["stop_loss_pct"],
            "position_size_pct": self.params["trading"]["position_size_pct"],
            "learning_rate": self.params["ml_model"]["learning_rate"]
        }
        
        # Create Optuna study
        sampler = TPESampler(seed=42)
        self.current_study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )
        
        # Define objective wrapper
        def optuna_objective(trial: optuna.Trial) -> float:
            # Sample parameters
            params = {
                "take_profit_pct": trial.suggest_float(
                    "take_profit_pct",
                    *self.param_bounds.get("take_profit_pct", [0.1, 2.0])
                ),
                "stop_loss_pct": trial.suggest_float(
                    "stop_loss_pct",
                    *self.param_bounds.get("stop_loss_pct", [0.1, 1.5])
                ),
                "position_size_pct": trial.suggest_float(
                    "position_size_pct",
                    *self.param_bounds.get("position_size_pct", [0.05, 0.5])
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    *self.param_bounds.get("learning_rate", [0.0001, 0.01]),
                    log=True
                )
            }
            
            # Run objective function (backtest)
            try:
                score = objective_func(params)
                return score
            except Exception as e:
                logger.error(f"Optimization trial failed: {e}")
                return float("-inf")
        
        # Run optimization
        logger.info(f"Starting Bayesian optimization with {n_trials} trials")
        start_time = time.time()
        
        self.current_study.optimize(
            optuna_objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False
        )
        
        optimization_time = time.time() - start_time
        
        # Get best parameters
        best_params = self.current_study.best_params
        best_score = self.current_study.best_value
        
        # Calculate improvement
        original_score = objective_func(original_params)
        improvement = ((best_score - original_score) / abs(original_score) * 100
                      if original_score != 0 else 0)
        
        # Update parameters if improvement is significant
        if improvement > 0:
            self.params["trading"]["take_profit_pct"] = best_params["take_profit_pct"]
            self.params["trading"]["stop_loss_pct"] = best_params["stop_loss_pct"]
            self.params["trading"]["position_size_pct"] = best_params["position_size_pct"]
            self.params["ml_model"]["learning_rate"] = best_params["learning_rate"]
            self._save_params()
            logger.info(f"Parameters updated with {improvement:.2f}% improvement")
        else:
            logger.info("No improvement found, keeping original parameters")
        
        result = OptimizationResult(
            timestamp=time.time(),
            status=PerformanceStatus.MODERATE,
            original_params=original_params,
            optimized_params=best_params,
            improvement_pct=improvement,
            iterations=n_trials or 0,
            best_score=best_score
        )
        
        self.optimization_history.append(result)
        
        return result
    
    async def optimize_with_backtest(
        self,
        backtest_func: Callable,
        df=None,  # df is no longer used - backtest_func should already have it bound
        n_trials: int = 50
    ) -> OptimizationResult:
        """
        Optimize parameters using backtest results
        backtest_func should be a callable that takes params dict and returns metrics dict
        """
        def objective(params: Dict[str, Any]) -> float:
            # Run backtest with given parameters
            try:
                metrics = backtest_func(params)  # Just pass params, df should be bound
                
                # Multi-objective scoring
                sharpe = metrics.get("sharpe_ratio", 0)
                profit_factor = metrics.get("profit_factor", 0)
                max_dd = metrics.get("max_drawdown_pct", 100)
                returns = metrics.get("total_return_pct", 0)
                
                # Penalty for exceeding drawdown limit
                dd_limit = self.objectives.get("drawdown_max", 5)
                dd_penalty = max(0, max_dd - dd_limit) * 0.5
                
                # Combined score (maximize)
                score = (
                    sharpe * 0.3 +
                    min(profit_factor, 3) * 0.2 +  # Cap PF contribution
                    returns * 0.3 +
                    (100 - max_dd) * 0.2 -
                    dd_penalty
                )
                
                return score
                
            except Exception as e:
                logger.error(f"Backtest failed: {e}")
                return float("-inf")
        
        return await self.optimize_parameters(objective, n_trials=n_trials)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization history"""
        if not self.optimization_history:
            return {"total_optimizations": 0}
        
        improvements = [r.improvement_pct for r in self.optimization_history]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "avg_improvement": np.mean(improvements),
            "best_improvement": max(improvements),
            "worst_improvement": min(improvements),
            "last_optimization": self.optimization_history[-1].timestamp,
            "current_params": {
                "take_profit_pct": self.params["trading"]["take_profit_pct"],
                "stop_loss_pct": self.params["trading"]["stop_loss_pct"],
                "position_size_pct": self.params["trading"]["position_size_pct"]
            }
        }
    
    def suggest_ml_hyperparameters(
        self,
        status: PerformanceStatus
    ) -> Dict[str, Any]:
        """
        Suggest ML hyperparameter adjustments based on status
        """
        current = self.params["ml_model"]
        factors = self.get_adjustment_factors(status)
        
        suggestions = {
            "learning_rate": current["learning_rate"] * factors["learning_rate_mult"],
            "epochs": current["epochs"],
            "batch_size": current["batch_size"]
        }
        
        # Adjust epochs based on status
        if status == PerformanceStatus.CRITICAL:
            suggestions["epochs"] = min(current["epochs"] * 2, 200)
        elif status == PerformanceStatus.POOR:
            suggestions["epochs"] = min(current["epochs"] + 20, 150)
        
        # Ensure bounds
        suggestions["learning_rate"] = np.clip(
            suggestions["learning_rate"],
            self.param_bounds.get("learning_rate", [0.0001, 0.01])[0],
            self.param_bounds.get("learning_rate", [0.0001, 0.01])[1]
        )
        
        return suggestions
    
    def reset_to_defaults(self) -> None:
        """Reset parameters to default values"""
        self.params["trading"] = {
            "take_profit_pct": 0.5,
            "stop_loss_pct": 0.3,
            "position_size_pct": 0.1,
            "max_positions": 1,
            "leverage": 1,
            "order_type": "limit",
            "slippage_tolerance_pct": 0.05
        }
        
        self.params["ml_model"]["learning_rate"] = 0.001
        self.params["ml_model"]["epochs"] = 100
        
        self._save_params()
        logger.info("Parameters reset to defaults")
