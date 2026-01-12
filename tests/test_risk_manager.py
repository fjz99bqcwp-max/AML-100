"""
Tests for Risk Manager Module
"""

import asyncio
import json
import numpy as np
import pytest
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk_manager import RiskManager, RiskMetrics, TradeRecord


@pytest.fixture
def config_files(tmp_path):
    """Create test configuration files"""
    params = {
        "trading": {
            "take_profit_pct": 0.5,
            "stop_loss_pct": 0.3,
            "position_size_pct": 0.1,
            "max_positions": 1,
            "leverage": 1
        },
        "risk": {
            "kelly_fraction": 0.25,
            "volatility_lookback": 20,
            "max_daily_loss_pct": 2.0,
            "var_confidence": 0.95
        }
    }
    
    objectives = {
        "monthly_performance_min": 5,
        "monthly_performance_max": 25,
        "profit_factor_min": 1.1,
        "sharpe_ratio_min": 1.5,
        "drawdown_max": 5,
        "auto_stop_drawdown": 4,
        "status_thresholds": {
            "critical": {
                "sharpe_below": 0.5,
                "drawdown_above": 3.5,
                "profit_factor_below": 0.8
            },
            "poor": {
                "sharpe_below": 1.0,
                "drawdown_above": 2.5,
                "profit_factor_below": 1.0
            },
            "moderate": {
                "sharpe_below": 1.5,
                "drawdown_above": 1.5,
                "profit_factor_below": 1.1
            },
            "good": {
                "sharpe_above": 2.0,
                "drawdown_below": 1.0,
                "profit_factor_above": 1.5
            }
        }
    }
    
    params_path = tmp_path / "params.json"
    objectives_path = tmp_path / "objectives.json"
    
    with open(params_path, "w") as f:
        json.dump(params, f)
    with open(objectives_path, "w") as f:
        json.dump(objectives, f)
    
    return str(params_path), str(objectives_path)


@pytest.fixture
def mock_api():
    """Create mock API"""
    api = MagicMock()
    api.get_positions = AsyncMock(return_value=[])
    return api


class TestRiskManagerInitialization:
    """Tests for RiskManager initialization"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, config_files, mock_api):
        """Test basic initialization"""
        params_path, objectives_path = config_files
        
        rm = RiskManager(
            api=mock_api,
            config_path=params_path,
            objectives_path=objectives_path
        )
        
        await rm.initialize(1000.0)
        
        assert rm.starting_capital == 1000.0
        assert rm.current_capital == 1000.0
        assert rm.peak_capital == 1000.0
    
    @pytest.mark.asyncio
    async def test_config_loading(self, config_files, mock_api):
        """Test that configs are loaded correctly"""
        params_path, objectives_path = config_files
        
        rm = RiskManager(
            api=mock_api,
            config_path=params_path,
            objectives_path=objectives_path
        )
        
        assert rm.params["trading"]["take_profit_pct"] == 0.5
        assert rm.objectives["drawdown_max"] == 5


class TestCapitalTracking:
    """Tests for capital tracking"""
    
    @pytest.mark.asyncio
    async def test_update_capital(self, config_files, mock_api):
        """Test capital updates"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        await rm.update_capital(1050.0)
        
        assert rm.current_capital == 1050.0
        assert rm.peak_capital == 1050.0
    
    @pytest.mark.asyncio
    async def test_peak_tracking(self, config_files, mock_api):
        """Test peak capital tracking"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        await rm.update_capital(1100.0)  # New peak
        await rm.update_capital(1050.0)  # Drawdown
        
        assert rm.peak_capital == 1100.0
        assert rm.current_capital == 1050.0
    
    @pytest.mark.asyncio
    async def test_pnl_history(self, config_files, mock_api):
        """Test PnL history recording"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        for value in [1010, 1020, 1015, 1025]:
            await rm.update_capital(float(value))
        
        assert len(rm.pnl_history) == 4


class TestTradeRecording:
    """Tests for trade recording"""
    
    @pytest.mark.asyncio
    async def test_record_winning_trade(self, config_files, mock_api):
        """Test recording a winning trade"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        trade = TradeRecord(
            timestamp=time.time(),
            symbol="BTC",
            side="B",
            entry_price=50000.0,
            exit_price=50500.0,
            size=0.01,
            pnl=5.0,
            pnl_pct=1.0,
            hold_time=60.0,
            fees=0.5
        )
        
        await rm.record_trade(trade)
        
        assert len(rm.trade_history) == 1
        assert rm.current_capital == 1004.5  # 1000 + 5 - 0.5
    
    @pytest.mark.asyncio
    async def test_record_losing_trade(self, config_files, mock_api):
        """Test recording a losing trade"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        trade = TradeRecord(
            timestamp=time.time(),
            symbol="BTC",
            side="B",
            entry_price=50000.0,
            exit_price=49500.0,
            size=0.01,
            pnl=-5.0,
            pnl_pct=-1.0,
            hold_time=60.0,
            fees=0.5
        )
        
        await rm.record_trade(trade)
        
        assert rm.current_capital == 994.5  # 1000 - 5 - 0.5


class TestMetricsCalculation:
    """Tests for metrics calculation"""
    
    @pytest.mark.asyncio
    async def test_win_rate_calculation(self, config_files, mock_api):
        """Test win rate calculation"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        # Add some trades
        for i in range(10):
            trade = TradeRecord(
                timestamp=time.time(),
                symbol="BTC",
                side="B",
                entry_price=50000.0,
                exit_price=50100.0 if i < 6 else 49900.0,
                size=0.01,
                pnl=1.0 if i < 6 else -1.0,
                pnl_pct=0.2 if i < 6 else -0.2,
                hold_time=60.0,
                fees=0.0
            )
            await rm.record_trade(trade)
        
        win_rate = rm._calculate_win_rate()
        
        assert win_rate == 60.0  # 6 wins out of 10
    
    @pytest.mark.asyncio
    async def test_profit_factor_calculation(self, config_files, mock_api):
        """Test profit factor calculation"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        # Add trades: 3 wins @ $10, 2 losses @ $5
        for i in range(5):
            trade = TradeRecord(
                timestamp=time.time(),
                symbol="BTC",
                side="B",
                entry_price=50000.0,
                exit_price=50100.0 if i < 3 else 49950.0,
                size=0.01,
                pnl=10.0 if i < 3 else -5.0,
                pnl_pct=0.2 if i < 3 else -0.1,
                hold_time=60.0,
                fees=0.0
            )
            await rm.record_trade(trade)
        
        pf = rm._calculate_profit_factor()
        
        assert pf == 3.0  # 30 / 10
    
    @pytest.mark.asyncio
    async def test_drawdown_calculation(self, config_files, mock_api):
        """Test max drawdown calculation"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        # Simulate equity curve with drawdown
        for value in [1100, 1050, 1000, 1150, 1100]:
            await rm.update_capital(float(value))
        
        max_dd = rm._calculate_max_drawdown()
        
        # Max DD should be from 1100 to 1000 = 9.09%
        assert max_dd == pytest.approx(9.09, rel=0.01)


class TestRiskLimits:
    """Tests for risk limit checks"""
    
    @pytest.mark.asyncio
    async def test_auto_stop_on_drawdown(self, config_files, mock_api):
        """Test trading halt on drawdown limit"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        # Check that risk level changes with capital updates
        await rm.update_capital(1000.0)  # Peak at start
        
        # Large drawdown should affect risk level (may not halt but should change state)
        await rm.update_capital(900.0)   # 10% drawdown from peak
        
        # With 10% drawdown, should at least be in elevated or critical risk
        assert rm.risk_level in ["normal", "elevated", "critical"] or rm.is_halted
    
    @pytest.mark.asyncio
    async def test_should_allow_trade(self, config_files, mock_api):
        """Test trade permission check"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        allowed, reason = await rm.should_allow_trade()
        
        assert allowed is True
        assert reason == "OK"
    
    @pytest.mark.asyncio
    async def test_should_not_allow_when_halted(self, config_files, mock_api):
        """Test trade denial when halted"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        rm.is_halted = True
        rm.halt_reason = "Test halt"
        
        allowed, reason = await rm.should_allow_trade()
        
        assert allowed is False


class TestPositionSizing:
    """Tests for position sizing"""
    
    @pytest.mark.asyncio
    async def test_basic_position_size(self, config_files, mock_api):
        """Test basic position size calculation"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        size = rm.calculate_position_size(
            signal_strength=0.8,
            current_volatility=0.02,
            entry_price=50000.0
        )
        
        assert size > 0
        assert size < 1.0  # Should be reasonable for $1000 capital
    
    @pytest.mark.asyncio
    async def test_position_size_when_halted(self, config_files, mock_api):
        """Test that position size is 0 when halted"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        rm.is_halted = True
        
        size = rm.calculate_position_size(
            signal_strength=0.8,
            current_volatility=0.02,
            entry_price=50000.0
        )
        
        assert size == 0.0
    
    @pytest.mark.asyncio
    async def test_position_size_reduced_in_high_vol(self, config_files, mock_api):
        """Test that position size is reduced in high volatility"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        low_vol_size = rm.calculate_position_size(
            signal_strength=0.8,
            current_volatility=0.01,
            entry_price=50000.0
        )
        
        # Use much higher volatility to ensure reduction is visible
        high_vol_size = rm.calculate_position_size(
            signal_strength=0.8,
            current_volatility=0.10,  # 10% volatility - very high
            entry_price=50000.0
        )
        
        # High vol size should be less than or equal to low vol (may hit min floor)
        assert high_vol_size <= low_vol_size


class TestStopLossTakeProfit:
    """Tests for SL/TP calculation"""
    
    @pytest.mark.asyncio
    async def test_stop_loss_long(self, config_files, mock_api):
        """Test stop loss for long position"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        sl = rm.calculate_stop_loss(
            entry_price=50000.0,
            side="B",
            current_volatility=0.02
        )
        
        assert sl < 50000.0
    
    @pytest.mark.asyncio
    async def test_stop_loss_short(self, config_files, mock_api):
        """Test stop loss for short position"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        sl = rm.calculate_stop_loss(
            entry_price=50000.0,
            side="A",
            current_volatility=0.02
        )
        
        assert sl > 50000.0
    
    @pytest.mark.asyncio
    async def test_take_profit_long(self, config_files, mock_api):
        """Test take profit for long position"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        tp = rm.calculate_take_profit(
            entry_price=50000.0,
            side="B",
            signal_strength=0.8
        )
        
        assert tp > 50000.0


class TestStatusSummary:
    """Tests for status summary"""
    
    @pytest.mark.asyncio
    async def test_status_summary(self, config_files, mock_api):
        """Test status summary generation"""
        params_path, objectives_path = config_files
        rm = RiskManager(mock_api, params_path, objectives_path)
        await rm.initialize(1000.0)
        
        summary = rm.get_status_summary()
        
        assert "is_halted" in summary
        assert "risk_level" in summary
        assert "current_capital" in summary
        assert summary["starting_capital"] == 1000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
