"""
AML-100 Source Package
Machine Learning Autonomous HFT for XYZ100-USDC on Hyperliquid
"""

from .hyperliquid_api import HyperliquidAPI, OrderBook, Position, Trade
from .data_fetcher import DataFetcher
from .risk_manager import RiskManager
from .ml_model import MLModel, HybridLSTMDQN, BacktestEnvironment
from .optimizer import ParameterOptimizer, PerformanceStatus
from .main import AMLHFTSystem

__version__ = "1.0.0"
__author__ = "AML Development Team"

__all__ = [
    "HyperliquidAPI",
    "OrderBook",
    "Position", 
    "Trade",
    "DataFetcher",
    "RiskManager",
    "MLModel",
    "HybridLSTMDQN",
    "BacktestEnvironment",
    "ParameterOptimizer",
    "PerformanceStatus",
    "AMLHFTSystem",
]
