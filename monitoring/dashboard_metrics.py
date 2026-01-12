"""
Dashboard Metrics
Writes key trading metrics to JSON after each trade
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class DashboardMetrics:
    """Manages dashboard metrics output"""
    
    def __init__(self, output_path: str = "monitoring/metrics.json"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            "last_update": 0,
            "session_start": time.time(),
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "trade_count": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "avg_hold_time": 0.0,
            "current_position": None,
            "last_trade": None,
            "daily_pnl": 0.0,
            "weekly_pnl": 0.0,
            "monthly_pnl": 0.0
        }
    
    def update_trade(
        self,
        pnl: float,
        pnl_pct: float,
        hold_time: float,
        side: str,
        size: float,
        entry_price: float,
        exit_price: float
    ) -> None:
        """Update metrics after a trade"""
        self.metrics["trade_count"] += 1
        self.metrics["realized_pnl"] += pnl
        
        if pnl > 0:
            self.metrics["winning_trades"] += 1
            self.metrics["largest_win"] = max(self.metrics["largest_win"], pnl)
        else:
            self.metrics["losing_trades"] += 1
            self.metrics["largest_loss"] = min(self.metrics["largest_loss"], pnl)
        
        # Calculate win rate
        if self.metrics["trade_count"] > 0:
            self.metrics["win_rate"] = (
                self.metrics["winning_trades"] / self.metrics["trade_count"] * 100
            )
        
        # Track last trade
        self.metrics["last_trade"] = {
            "timestamp": time.time(),
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "hold_time": hold_time
        }
        
        self._save()
    
    def update_position(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        unrealized_pnl: float,
        leverage: int
    ) -> None:
        """Update current position info"""
        if size != 0:
            self.metrics["current_position"] = {
                "symbol": symbol,
                "size": size,
                "side": "LONG" if size > 0 else "SHORT",
                "entry_price": entry_price,
                "unrealized_pnl": unrealized_pnl,
                "leverage": leverage
            }
        else:
            self.metrics["current_position"] = None
        
        self.metrics["unrealized_pnl"] = unrealized_pnl
        self._save()
    
    def update_risk_metrics(
        self,
        sharpe_ratio: float,
        sortino_ratio: float,
        max_drawdown: float,
        current_drawdown: float,
        profit_factor: float
    ) -> None:
        """Update risk metrics"""
        self.metrics["sharpe_ratio"] = sharpe_ratio
        self.metrics["sortino_ratio"] = sortino_ratio
        self.metrics["max_drawdown"] = max_drawdown
        self.metrics["current_drawdown"] = current_drawdown
        self.metrics["profit_factor"] = profit_factor
        self._save()
    
    def update_capital(
        self,
        current_capital: float,
        starting_capital: float
    ) -> None:
        """Update capital metrics"""
        self.metrics["total_pnl"] = current_capital - starting_capital
        self.metrics["total_pnl_pct"] = (
            (current_capital - starting_capital) / starting_capital * 100
            if starting_capital > 0 else 0
        )
        self._save()
    
    def _save(self) -> None:
        """Save metrics to JSON file"""
        self.metrics["last_update"] = time.time()
        self.metrics["last_update_str"] = datetime.now(timezone.utc).isoformat()
        
        with open(self.output_path, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()


# Standalone runner for testing
if __name__ == "__main__":
    metrics = DashboardMetrics()
    
    # Simulate some trades
    metrics.update_capital(1050, 1000)
    metrics.update_trade(
        pnl=25.50,
        pnl_pct=2.5,
        hold_time=120,
        side="LONG",
        size=0.01,
        entry_price=45000,
        exit_price=46000
    )
    
    print("Metrics saved to monitoring/metrics.json")
    print(json.dumps(metrics.get_metrics(), indent=2))
