"""
Tests for Hyperliquid API Module
"""

import asyncio
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hyperliquid_api import (
    OrderBook,
    Position,
    Trade,
    RateLimiter,
    HyperliquidAPI
)


class TestOrderBook:
    """Tests for OrderBook dataclass"""
    
    def test_empty_orderbook(self):
        """Test empty orderbook"""
        ob = OrderBook()
        
        assert ob.mid_price == 0.0
        assert ob.spread == 0.0
        assert ob.spread_bps == 0.0
    
    def test_mid_price(self):
        """Test mid price calculation"""
        ob = OrderBook(
            bids=[(100.0, 1.0), (99.0, 2.0)],
            asks=[(101.0, 1.0), (102.0, 2.0)]
        )
        
        assert ob.mid_price == 100.5
    
    def test_spread(self):
        """Test spread calculation"""
        ob = OrderBook(
            bids=[(100.0, 1.0)],
            asks=[(100.5, 1.0)]
        )
        
        assert ob.spread == 0.5
        assert ob.spread_bps == pytest.approx(49.75, rel=0.01)


class TestPosition:
    """Tests for Position dataclass"""
    
    def test_position_creation(self):
        """Test position creation"""
        pos = Position(
            symbol="BTC",
            size=0.1,
            entry_price=50000.0,
            unrealized_pnl=100.0,
            realized_pnl=50.0,
            leverage=5
        )
        
        assert pos.symbol == "BTC"
        assert pos.size == 0.1
        assert pos.leverage == 5
    
    def test_short_position(self):
        """Test short position (negative size)"""
        pos = Position(
            symbol="BTC",
            size=-0.1,
            entry_price=50000.0,
            unrealized_pnl=-50.0,
            realized_pnl=0.0,
            leverage=10
        )
        
        assert pos.size < 0


class TestTrade:
    """Tests for Trade dataclass"""
    
    def test_trade_creation(self):
        """Test trade creation"""
        trade = Trade(
            order_id="123",
            symbol="BTC",
            side="B",
            size=0.01,
            price=50000.0,
            timestamp=time.time(),
            fee=0.5
        )
        
        assert trade.symbol == "BTC"
        assert trade.side == "B"


class TestRateLimiter:
    """Tests for RateLimiter"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test rate limiter initialization"""
        limiter = RateLimiter(rate=10.0, capacity=5.0)
        
        assert limiter.rate == 10.0
        assert limiter.capacity == 5.0
        assert limiter.tokens == 5.0
    
    @pytest.mark.asyncio
    async def test_acquire_token(self):
        """Test acquiring tokens"""
        limiter = RateLimiter(rate=10.0, capacity=5.0)
        
        start = time.monotonic()
        await limiter.acquire(1.0)
        elapsed = time.monotonic() - start
        
        # Should not wait if tokens available
        assert elapsed < 0.1
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiting works"""
        limiter = RateLimiter(rate=10.0, capacity=2.0)
        
        # Exhaust tokens
        await limiter.acquire(2.0)
        
        # Next acquire should wait
        start = time.monotonic()
        await limiter.acquire(1.0)
        elapsed = time.monotonic() - start
        
        # Should have waited approximately 0.1s (1 token at 10/sec)
        assert elapsed >= 0.08


class TestHyperliquidAPI:
    """Tests for HyperliquidAPI"""
    
    @pytest.fixture
    def api_config(self, tmp_path):
        """Create test API config"""
        config = {
            "hyperliquid": {
                "api_key_env": "TEST_API_KEY",
                "api_secret_env": "TEST_API_SECRET",
                "wallet_address_env": "TEST_WALLET",
                "private_key_env": "TEST_PRIVATE_KEY",
                "mainnet_url": "https://api.hyperliquid.xyz",
                "ws_url": "wss://api.hyperliquid.xyz/ws",
                "info_url": "https://api.hyperliquid.xyz/info",
                "exchange_url": "https://api.hyperliquid.xyz/exchange",
                "rate_limits": {
                    "requests_per_second": 10,
                    "orders_per_second": 5,
                    "websocket_subscriptions_max": 100
                },
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "retry_delay_ms": 100
            },
            "symbol": "BTC",
            "asset_index": 0,
            "decimals": {
                "price": 1,
                "size": 4
            }
        }
        
        config_path = tmp_path / "api.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        return str(config_path)
    
    def test_config_loading(self, api_config):
        """Test that config loads correctly"""
        api = HyperliquidAPI(api_config)
        
        assert api.config["symbol"] == "BTC"
        assert api.config["hyperliquid"]["timeout_seconds"] == 30
    
    def test_cached_orderbook(self, api_config):
        """Test orderbook caching"""
        api = HyperliquidAPI(api_config)
        
        # Initially no cache
        assert api.get_cached_orderbook("BTC") is None
        
        # Add to cache
        api._orderbook_cache["BTC"] = OrderBook(
            bids=[(50000.0, 1.0)],
            asks=[(50001.0, 1.0)]
        )
        
        # Should return cached
        cached = api.get_cached_orderbook("BTC")
        assert cached is not None
        assert cached.mid_price == 50000.5
    
    def test_cached_position(self, api_config):
        """Test position caching"""
        api = HyperliquidAPI(api_config)
        
        # Initially no cache
        assert api.get_cached_position("BTC") is None
        
        # Add to cache
        api._position_cache["BTC"] = Position(
            symbol="BTC",
            size=0.1,
            entry_price=50000.0,
            unrealized_pnl=100.0,
            realized_pnl=0.0,
            leverage=5
        )
        
        # Should return cached
        cached = api.get_cached_position("BTC")
        assert cached is not None
        assert cached.size == 0.1
    
    def test_latency_tracking(self, api_config):
        """Test latency metric tracking"""
        api = HyperliquidAPI(api_config)
        
        # Add some samples
        for latency in [10.0, 15.0, 20.0, 25.0, 30.0]:
            api._latency_samples.append(latency)
        
        assert api.get_avg_latency() == 20.0
        assert api.get_latency_p99() == 30.0
    
    def test_empty_latency(self, api_config):
        """Test latency with no samples"""
        api = HyperliquidAPI(api_config)
        
        assert api.get_avg_latency() == 0.0
        assert api.get_latency_p99() == 0.0


class TestHyperliquidAPIAsync:
    """Async tests for HyperliquidAPI"""
    
    @pytest.fixture
    def api_config(self, tmp_path):
        """Create test API config"""
        config = {
            "hyperliquid": {
                "api_key_env": "TEST_API_KEY",
                "api_secret_env": "TEST_API_SECRET",
                "wallet_address_env": "TEST_WALLET",
                "private_key_env": "TEST_PRIVATE_KEY",
                "mainnet_url": "https://api.hyperliquid.xyz",
                "ws_url": "wss://api.hyperliquid.xyz/ws",
                "info_url": "https://api.hyperliquid.xyz/info",
                "exchange_url": "https://api.hyperliquid.xyz/exchange",
                "rate_limits": {
                    "requests_per_second": 10,
                    "orders_per_second": 5,
                    "websocket_subscriptions_max": 100
                },
                "timeout_seconds": 30,
                "retry_attempts": 3,
                "retry_delay_ms": 100
            },
            "symbol": "BTC",
            "asset_index": 0,
            "decimals": {
                "price": 1,
                "size": 4
            }
        }
        
        config_path = tmp_path / "api.json"
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        return str(config_path)
    
    @pytest.mark.asyncio
    async def test_close_without_initialize(self, api_config):
        """Test closing without initialization"""
        api = HyperliquidAPI(api_config)
        
        # Should not raise
        await api.close()
    
    @pytest.mark.asyncio
    async def test_initialization_missing_credentials(self, api_config, monkeypatch):
        """Test initialization with missing credentials"""
        # Clear environment
        monkeypatch.delenv("TEST_WALLET", raising=False)
        monkeypatch.delenv("TEST_PRIVATE_KEY", raising=False)
        
        api = HyperliquidAPI(api_config)
        
        with pytest.raises(ValueError, match="Missing Hyperliquid credentials"):
            await api.initialize()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
