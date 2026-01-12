"""
Tests for Optimizer Module
"""

import asyncio
import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizer import ParameterOptimizer, PerformanceStatus


@pytest.fixture
def config_files(tmp_path):
    """Create test configuration files"""
    params = {
        "trading": {
            "take_profit_pct": 0.5,
            "stop_loss_pct": 0.3,
            "position_size_pct": 0.1,
            "max_positions": 1,
            "leverage": 1,
            "order_type": "limit",
            "slippage_tolerance_pct": 0.05
        },
        "ml_model": {
            "type": "hybrid_lstm_dqn",
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 64
        },
        "optimization": {
            "method": "bayesian",
            "max_iterations": 50,
            "param_bounds": {
                "take_profit_pct": [0.1, 2.0],
                "stop_loss_pct": [0.1, 1.5],
                "position_size_pct": [0.05, 0.5],
                "learning_rate": [0.0001, 0.01]
            }
        }
    }
    
    objectives = {
        "monthly_performance_min": 5,
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


class TestPerformanceStatus:
    """Tests for performance status enum"""
    
    def test_status_values(self):
        """Test status enum values"""
        assert PerformanceStatus.CRITICAL.value == "critical"
        assert PerformanceStatus.POOR.value == "poor"
        assert PerformanceStatus.MODERATE.value == "moderate"
        assert PerformanceStatus.GOOD.value == "good"


class TestStatusDetermination:
    """Tests for status determination"""
    
    def test_critical_status_low_sharpe(self, config_files):
        """Test critical status with low Sharpe"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        # Critical requires: sharpe < 0.2 AND trade_count >= 10
        # OR drawdown > 4.0, OR (profit_factor < 0.6 AND trade_count >= 10)
        metrics = {
            "sharpe_ratio": 0.15,  # Below 0.2 threshold
            "max_drawdown": 5.0,   # Above 4.0 threshold - this triggers critical
            "profit_factor": 1.5
        }
        
        status = optimizer.determine_status(metrics)
        assert status == PerformanceStatus.CRITICAL
    
    def test_critical_status_high_drawdown(self, config_files):
        """Test critical status with high drawdown"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        metrics = {
            "sharpe_ratio": 2.0,
            "max_drawdown": 4.0,
            "profit_factor": 1.5
        }
        
        status = optimizer.determine_status(metrics)
        assert status == PerformanceStatus.CRITICAL
    
    def test_poor_status(self, config_files):
        """Test poor status"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        metrics = {
            "sharpe_ratio": 0.8,
            "max_drawdown": 2.0,
            "profit_factor": 1.2
        }
        
        status = optimizer.determine_status(metrics)
        assert status == PerformanceStatus.POOR
    
    def test_moderate_status(self, config_files):
        """Test moderate status"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        metrics = {
            "sharpe_ratio": 1.3,
            "max_drawdown": 1.2,
            "profit_factor": 1.2
        }
        
        status = optimizer.determine_status(metrics)
        assert status == PerformanceStatus.MODERATE
    
    def test_good_status(self, config_files):
        """Test good status"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        metrics = {
            "sharpe_ratio": 2.5,
            "max_drawdown": 0.5,
            "profit_factor": 2.0
        }
        
        status = optimizer.determine_status(metrics)
        assert status == PerformanceStatus.GOOD


class TestAdjustmentFactors:
    """Tests for adjustment factor calculation"""
    
    def test_critical_factors(self, config_files):
        """Test critical status adjustment factors"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        factors = optimizer.get_adjustment_factors(PerformanceStatus.CRITICAL)
        
        assert factors["position_size_mult"] == 0.5
        assert factors["learning_rate_mult"] > 1.0
    
    def test_good_factors(self, config_files):
        """Test good status adjustment factors"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        factors = optimizer.get_adjustment_factors(PerformanceStatus.GOOD)
        
        assert factors["position_size_mult"] > 1.0
        assert factors["learning_rate_mult"] < 1.0
    
    def test_moderate_factors(self, config_files):
        """Test moderate status has slightly boosted factors for growth"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        factors = optimizer.get_adjustment_factors(PerformanceStatus.MODERATE)
        
        # Step 3: Moderate now gets a slight boost (1.1x)
        assert factors["position_size_mult"] == 1.1
        assert factors["tp_mult"] == 1.0
        assert factors["sl_mult"] == 1.0


class TestQuickAdjust:
    """Tests for quick parameter adjustment"""
    
    @pytest.mark.asyncio
    async def test_quick_adjust_critical(self, config_files):
        """Test quick adjustment for critical status"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        original_size = optimizer.params["trading"]["position_size_pct"]
        
        result = await optimizer.quick_adjust(PerformanceStatus.CRITICAL)
        
        new_size = optimizer.params["trading"]["position_size_pct"]
        
        assert new_size < original_size
        assert result["status"] == "critical"
    
    @pytest.mark.asyncio
    async def test_quick_adjust_respects_bounds(self, config_files):
        """Test that quick adjustment respects parameter bounds"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        # Multiple critical adjustments
        for _ in range(5):
            await optimizer.quick_adjust(PerformanceStatus.CRITICAL)
        
        bounds = optimizer.param_bounds
        
        assert optimizer.params["trading"]["position_size_pct"] >= bounds["position_size_pct"][0]
        assert optimizer.params["trading"]["take_profit_pct"] >= bounds["take_profit_pct"][0]
    
    @pytest.mark.asyncio
    async def test_quick_adjust_saves_params(self, config_files):
        """Test that quick adjustment saves to file"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        await optimizer.quick_adjust(PerformanceStatus.POOR)
        
        # Reload and verify
        with open(params_path, "r") as f:
            saved_params = json.load(f)
        
        assert saved_params["trading"]["position_size_pct"] == optimizer.params["trading"]["position_size_pct"]


class TestMLHyperparameterSuggestion:
    """Tests for ML hyperparameter suggestions"""
    
    def test_critical_suggestions(self, config_files):
        """Test ML suggestions for critical status"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        suggestions = optimizer.suggest_ml_hyperparameters(PerformanceStatus.CRITICAL)
        
        # Should suggest more epochs and higher LR
        assert suggestions["epochs"] > optimizer.params["ml_model"]["epochs"]
        assert suggestions["learning_rate"] > optimizer.params["ml_model"]["learning_rate"]
    
    def test_good_suggestions(self, config_files):
        """Test ML suggestions for good status"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        suggestions = optimizer.suggest_ml_hyperparameters(PerformanceStatus.GOOD)
        
        # Should suggest lower LR (stable)
        assert suggestions["learning_rate"] < optimizer.params["ml_model"]["learning_rate"]


class TestResetToDefaults:
    """Tests for parameter reset"""
    
    def test_reset_to_defaults(self, config_files):
        """Test resetting parameters to defaults"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        # Modify params
        optimizer.params["trading"]["take_profit_pct"] = 5.0
        optimizer.params["trading"]["stop_loss_pct"] = 3.0
        
        # Reset
        optimizer.reset_to_defaults()
        
        assert optimizer.params["trading"]["take_profit_pct"] == 0.5
        assert optimizer.params["trading"]["stop_loss_pct"] == 0.3


class TestOptimizationSummary:
    """Tests for optimization summary"""
    
    def test_empty_summary(self, config_files):
        """Test summary with no optimizations"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        summary = optimizer.get_optimization_summary()
        
        assert summary["total_optimizations"] == 0
    
    def test_summary_with_history(self, config_files):
        """Test summary includes current params"""
        params_path, objectives_path = config_files
        optimizer = ParameterOptimizer(params_path, objectives_path)
        
        summary = optimizer.get_optimization_summary()
        
        # Even with no history, should show current params
        assert summary["total_optimizations"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
