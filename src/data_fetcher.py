"""
Data Fetcher Module
High-performance data fetching and storage for Hyperliquid HFT
Optimized for Apple M4 with async I/O and efficient serialization
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, asdict
import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import orjson
import joblib
import torch

from src.hyperliquid_api import HyperliquidAPI, OrderBook, Trade

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """Point-in-time market snapshot for training/backtesting"""
    timestamp: float
    symbol: str
    mid_price: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread_bps: float
    orderbook_imbalance: float
    volume_24h: float
    funding_rate: float


@dataclass 
class OHLCV:
    """OHLCV candlestick data"""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataFetcher:
    """
    High-performance data fetching and caching system
    Optimized for HFT with minimal latency data access
    
    Features:
    - In-memory caching for fast access
    - Joblib serialization for preprocessed tensor caching
    - Async I/O for non-blocking operations
    """
    
    def __init__(
        self,
        api: HyperliquidAPI,
        data_dir: str = "data",
        cache_size: int = 10000
    ):
        self.api = api
        self.data_dir = Path(data_dir)
        self.cache_size = cache_size
        
        # In-memory caches for fast access
        self._kline_cache: Dict[str, deque] = {}
        self._snapshot_cache: deque = deque(maxlen=cache_size)
        self._trade_cache: deque = deque(maxlen=cache_size)
        
        # Training data cache (preprocessed tensors)
        self._training_data_cache: Dict[str, Any] = {}
        self._tensor_cache_dir = self.data_dir / "cache"
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Background tasks
        self._fetch_tasks: List[asyncio.Task] = []
        self._running = False
        
        # XYZ100-USDC specific configuration
        self.XYZ100_MIN_HISTORY_RECORDS = 5000
        self.XYZ100_MIN_HISTORY_DAYS = 30
        self.PRIMARY_SYMBOL = "xyz:XYZ100"  # XYZ perps coin name
        self.FALLBACK_SYMBOL = "US500"  # Changed from BTC to US500 for equity correlation
        self.SECONDARY_FALLBACK = "BTC"  # BTC as last resort
        self._using_fallback = False
        self._fallback_type = None  # "US500", "synthetic_us500", or "BTC"
        self._last_xyz100_check = 0.0
        self._xyz100_check_interval = 3600  # Check hourly
        
        # SPX/S&P500 synthetic data parameters
        self.SPX_ANNUAL_VOL = 0.15  # ~15% annualized volatility for S&P500
        self.SPX_DRIFT = 0.10  # ~10% annual drift
        self.XYZ100_CORRELATION = 0.8  # Correlation with XYZ100
        
    def _ensure_directories(self) -> None:
        """Create necessary data directories"""
        subdirs = ["historical", "backtests", "trading", "backups", "cache"]
        for subdir in subdirs:
            (self.data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    async def check_xyz100_availability(self):
        """
        Check if XYZ100-USDC has sufficient historical data
        Returns (is_available, record_count)
        """
        try:
            now_ms = int(time.time() * 1000)
            thirty_days_ago = now_ms - (self.XYZ100_MIN_HISTORY_DAYS * 24 * 60 * 60 * 1000)
            
            klines = await self.api.get_historical_klines(
                symbol=self.PRIMARY_SYMBOL,
                interval="1m",
                start_time=thirty_days_ago,
                end_time=now_ms
            )
            
            record_count = len(klines) if klines else 0
            is_available = record_count >= self.XYZ100_MIN_HISTORY_RECORDS
            
            if not is_available:
                logger.warning(
                    f"XYZ100-USDC insufficient history: {record_count} records "
                    f"(need {self.XYZ100_MIN_HISTORY_RECORDS})"
                )
            else:
                logger.info(f"XYZ100-USDC available: {record_count} records")
                
            return is_available, record_count
            
        except Exception as e:
            logger.error(f"Error checking XYZ100 availability: {e}")
            return False, 0
    
    async def get_active_symbol(self):
        """
        Get the active trading symbol with fallback logic
        Returns XYZ100 if sufficient history, BTC otherwise
        """
        current_time = time.time()
        
        if current_time - self._last_xyz100_check >= self._xyz100_check_interval:
            self._last_xyz100_check = current_time
            is_available, count = await self.check_xyz100_availability()
            
            if is_available and self._using_fallback:
                logger.info("Switching from BTC fallback to XYZ100-USDC")
                self._using_fallback = False
            elif not is_available and not self._using_fallback:
                logger.warning("XYZ100-USDC insufficient history - Using BTC fallback")
                self._using_fallback = True
        
        return self.FALLBACK_SYMBOL if self._using_fallback else self.PRIMARY_SYMBOL
    
    def is_using_fallback(self):
        """Check if currently using BTC fallback"""
        return self._using_fallback
    
    def scale_btc_data_for_xyz100(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale BTC data to approximate XYZ100 equity volatility characteristics.
        XYZ100 typically has ~1.2x higher volatility than BTC due to equity nature.
        """
        if df.empty or not self._using_fallback:
            return df
            
        df = df.copy()
        XYZ100_VOL_FACTOR = 1.2  # Equity perps have higher vol
        
        # Scale price returns to match equity volatility
        close = df["close"]
        returns = close.pct_change()
        scaled_returns = returns * XYZ100_VOL_FACTOR
        
        # Reconstruct prices with scaled returns
        df["close"] = close.iloc[0] * (1 + scaled_returns).cumprod()
        
        # Scale high/low range proportionally
        price_range = df["high"] - df["low"]
        mid_price = (df["high"] + df["low"]) / 2
        scaled_range = price_range * XYZ100_VOL_FACTOR
        df["high"] = mid_price + scaled_range / 2
        df["low"] = mid_price - scaled_range / 2
        
        logger.debug(f"Scaled BTC data for XYZ100 (vol_factor={XYZ100_VOL_FACTOR})")
        return df

    async def fetch_spx_data(self, days: int = 180) -> Optional[pd.DataFrame]:
        """
        Fetch SPX (S&P 500 perpetual) data from HyperLiquid if available.
        Falls back to synthetic generation if not found.
        """
        try:
            # Try fetching SPX from HyperLiquid
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
            
            klines = await self.api.get_klines(
                symbol="SPX",
                interval="1m",
                start_time=start_ms,
                end_time=now_ms
            )
            
            if klines and len(klines) >= 1000:
                logger.info(f"Fetched {len(klines)} SPX klines from HyperLiquid")
                df = pd.DataFrame([{
                    "timestamp": k["t"] if isinstance(k, dict) else k.timestamp,
                    "open": k["o"] if isinstance(k, dict) else k.open,
                    "high": k["h"] if isinstance(k, dict) else k.high,
                    "low": k["l"] if isinstance(k, dict) else k.low,
                    "close": k["c"] if isinstance(k, dict) else k.close,
                    "volume": k["v"] if isinstance(k, dict) else k.volume
                } for k in klines])
                
                # Convert to numeric types (fix API returning strings)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna()  # Remove any rows that failed conversion
                
                self._fallback_type = "SPX"
                return df
            else:
                logger.info("SPX not available on HyperLiquid, generating synthetic data")
                return self.generate_synthetic_spx(days)
                
        except Exception as e:
            logger.warning(f"Error fetching SPX: {e}, using synthetic data")
            return self.generate_synthetic_spx(days)

    def generate_synthetic_spx(self, days: int = 90) -> pd.DataFrame:
        """
        Generate synthetic SPX-like data for extended backtesting.
        Uses S&P500 statistical properties: ~15% annual vol, ~10% drift.
        Matches XYZ100 correlation coefficient of ~0.8.
        
        M4 OPTIMIZED: Chunked generation to avoid memory spikes
        Default: 90 days (129,600 klines) to match backtest requirements
        """
        logger.info(f"Generating synthetic SPX data for {days} days")
        
        # Time parameters
        minutes_per_day = 24 * 60
        total_minutes = days * minutes_per_day  # e.g., 90 days = 129,600 klines
        
        # Convert annual params to per-minute with 1.2x vol scaling for XYZ100 correlation
        vol_scaling = 1.2  # Higher vol for equity perp
        vol_per_minute = (self.SPX_ANNUAL_VOL * vol_scaling) / np.sqrt(252 * minutes_per_day)
        drift_per_minute = self.SPX_DRIFT / (252 * minutes_per_day)
        
        # Generate timestamps
        end_time = time.time()
        start_time = end_time - (days * 24 * 60 * 60)
        
        # CHUNKED GENERATION for memory efficiency on M4
        chunk_size = 10000  # 10k klines per chunk (~7 days)
        all_chunks = []
        
        for chunk_start in range(0, total_minutes, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_minutes)
            chunk_len = chunk_end - chunk_start
            
            # Generate timestamps for chunk
            chunk_timestamps = np.linspace(
                start_time + (chunk_start * 60),
                start_time + (chunk_end * 60),
                chunk_len
            )
            
            # Geometric Brownian Motion
            np.random.seed(42 + chunk_start)  # Reproducible with different seed per chunk
            z = np.random.standard_normal(chunk_len)
            
            # Mean reversion
            phi = 0.02
            for i in range(1, len(z)):
                z[i] = z[i] * (1 - phi) + z[i-1] * phi
            
            # Generate log returns
            log_returns = drift_per_minute + vol_per_minute * z
            
            # Starting price (carry from previous chunk or default)
            if all_chunks:
                start_price = all_chunks[-1]["close"].iloc[-1]
            else:
                start_price = 4500.0  # Approx SPX level
            
            prices = start_price * np.exp(np.cumsum(log_returns))
            
            # Generate OHLC
            typical_range = vol_per_minute * 2
            chunk_df = pd.DataFrame({
                "timestamp": chunk_timestamps,
                "close": prices,
                "high": prices * (1 + np.abs(np.random.normal(0, typical_range, chunk_len))),
                "low": prices * (1 - np.abs(np.random.normal(0, typical_range, chunk_len))),
                "open": np.concatenate([[start_price], prices[:-1]]),
            })
            
            # Volume with market hours boost
            base_volume = 1000
            hour_of_day = (chunk_df["timestamp"] % 86400) / 3600
            market_hours_boost = np.where((hour_of_day >= 14.5) & (hour_of_day <= 21), 3.0, 1.0)
            chunk_df["volume"] = base_volume * market_hours_boost * np.random.lognormal(0, 0.5, chunk_len)
            
            all_chunks.append(chunk_df)
            
            # Progress logging
            progress = int((chunk_end / total_minutes) * 100)
            if progress % 20 == 0:
                logger.debug(f"Synthetic SPX generation: {progress}%")
        
        # Combine all chunks
        df = pd.concat(all_chunks, ignore_index=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        
        self._fallback_type = "synthetic_spx"
        logger.info(f"Generated {len(df)} synthetic SPX klines ({days} days)")
        return df

    async def get_fallback_data(self, days: int = 180) -> pd.DataFrame:
        """
        Get fallback data prioritizing SPX over BTC.
        Order: SPX (real) -> SPX (synthetic) -> BTC (scaled)
        """
        # Try SPX first
        df = await self.fetch_spx_data(days)
        if df is not None and len(df) > 0:
            return df
        
        # Fall back to BTC with scaling
        logger.warning("SPX unavailable, falling back to BTC with vol scaling")
        self._fallback_type = "BTC"
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
        
        klines = await self.api.get_klines(
            symbol="BTC",
            interval="1m", 
            start_time=start_ms,
            end_time=now_ms
        )
        
        if klines:
            df = pd.DataFrame([{
                "timestamp": k.timestamp,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume
            } for k in klines])
            return self.scale_btc_data_for_xyz100(df)
        
        # Last resort: pure synthetic
        logger.warning("BTC also unavailable, using pure synthetic SPX")
        return self.generate_synthetic_spx(days)
    
    async def generate_hybrid_data(self, days: int = 180, real_weight: float = 0.7) -> pd.DataFrame:
        """
        Generate hybrid real+synthetic data for robust training.
        Combines 70% real SPX data with 30% synthetic for data augmentation.
        
        Args:
            days: Total days of data to generate
            real_weight: Proportion of real data (0.7 = 70% real, 30% synthetic)
        
        Returns:
            DataFrame with hybrid data sorted by timestamp
        """
        logger.info(f"Generating hybrid data: {real_weight*100:.0f}% real + {(1-real_weight)*100:.0f}% synthetic")
        
        # Fetch real SPX data
        real_df = await self.fetch_spx_data(days)
        
        # Generate synthetic data
        synthetic_days = max(int(days * (1 - real_weight)), 30)
        synthetic_df = self.generate_synthetic_spx(synthetic_days)
        
        if real_df is None or len(real_df) < 1000:
            logger.warning("Insufficient real SPX data, using 100% synthetic")
            return self.generate_synthetic_spx(days)
        
        # Calculate actual samples
        total_samples = days * 24 * 60  # 1-minute klines
        real_samples = int(total_samples * real_weight)
        synthetic_samples = total_samples - real_samples
        
        # Sample from real data
        if len(real_df) > real_samples:
            real_df = real_df.sample(n=real_samples, random_state=42).sort_values("timestamp")
        
        # Sample from synthetic
        if len(synthetic_df) > synthetic_samples:
            synthetic_df = synthetic_df.sample(n=synthetic_samples, random_state=43).sort_values("timestamp")
        
        # Merge and sort
        hybrid_df = pd.concat([real_df, synthetic_df], ignore_index=True)
        hybrid_df = hybrid_df.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"Hybrid data: {len(real_df)} real + {len(synthetic_df)} synthetic = {len(hybrid_df)} total klines")
        return hybrid_df
    
    async def fetch_wallet_fills(
        self, 
        wallet: str = "0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584",
        days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch real wallet fills from HyperLiquid for ultra-realistic training.
        Uses HyperLiquid SDK to get actual user fills.
        """
        try:
            from hyperliquid.info import Info
            from hyperliquid.utils import constants
        except ImportError:
            logger.error("❌ hyperliquid-python-sdk not installed: pip install hyperliquid-python-sdk")
            return pd.DataFrame()
        
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        fills = []
        try:
            user_fills = info.user_fills(wallet)
            
            for fill in user_fills:
                if fill['time'] >= start_time:
                    fills.append({
                        "timestamp": fill['time'] / 1000,
                        "symbol": fill['coin'],
                        "side": fill['side'],
                        "price": float(fill['px']),
                        "size": float(fill['sz']),
                        "fee": float(fill.get('fee', 0)),
                        "closed_pnl": float(fill.get('closedPnl', 0)),
                        "is_maker": fill.get('maker', False)
                    })
            
            logger.info(f"✅ Fetched {len(fills)} wallet fills from {wallet[:10]}... (last {days} days)")
            return pd.DataFrame(fills)
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch wallet fills: {type(e).__name__}: {e}")
            return pd.DataFrame()
    
    async def generate_wallet_hybrid_data(
        self, 
        wallet: str = "0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584",
        days: int = 180
    ) -> pd.DataFrame:
        """
        Generate hybrid data with 90% wallet fills + 10% synthetic.
        Most realistic training data - learns actual HyperLiquid execution patterns.
        """
        wallet_df = await self.fetch_wallet_fills(wallet=wallet, days=days)
        
        if wallet_df.empty or len(wallet_df) < 1000:
            logger.warning(f"⚠️ Insufficient wallet data ({len(wallet_df)} fills), falling back to SPX hybrid")
            return await self.generate_hybrid_data(days, real_weight=0.7)
        
        # Supplement with synthetic (10%)
        synthetic_df = self.generate_synthetic_spx(days=int(days * 0.1))
        
        # Merge and sort by timestamp
        hybrid = pd.concat([wallet_df, synthetic_df], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"💎 Wallet hybrid: {len(wallet_df)} real fills + {len(synthetic_df)} synthetic = {len(hybrid)} total")
        return hybrid

    def _add_equity_perp_features(self, df):
        """
        Add XYZ100 equity perpetual specific features
        Optimized for equity volatility patterns
        """
        if df.empty:
            return df
            
        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df.get("volume", pd.Series(0, index=df.index))
        
        # Equity volatility (annualized, different periods)
        returns = close.pct_change()
        df["equity_vol_5"] = returns.rolling(5).std() * np.sqrt(252 * 24 * 60)
        df["equity_vol_20"] = returns.rolling(20).std() * np.sqrt(252 * 24 * 60)
        
        # Implied volatility proxy from price range
        price_range = (high - low) / close
        df["implied_vol_proxy"] = price_range.rolling(20).mean() * np.sqrt(252)
        
        # Mean reversion signal (for equity-like behavior)
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        df["mean_revert_signal"] = (close - ma_20) / std_20.clip(lower=1e-8)
        
        # Gap detection (equity market gaps)
        df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        df["gap_filled"] = (
            ((df["gap_pct"] > 0) & (df["low"] <= df["close"].shift(1))) |
            ((df["gap_pct"] < 0) & (df["high"] >= df["close"].shift(1)))
        ).astype(float)
        
        # Trend persistence (equity tends to trend more)
        df["trend_persistence"] = returns.rolling(10).apply(
            lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5,
            raw=False
        )
        
        # Volume spike detection
        vol_ma = volume.rolling(20).mean()
        df["volume_spike"] = (volume / vol_ma.clip(lower=1)).clip(upper=10)
        
        # NASDAQ/TECH CORRELATION FEATURES (for XYZ100 tech equity exposure)
        # Simulate QQQ-style behavior (tech-heavy equity index)
        # Assumes XYZ100 has 0.85 correlation with tech sector
        df["tech_momentum"] = returns.rolling(20).mean() * 1.25  # Amplified momentum for tech
        df["tech_vol_regime"] = returns.rolling(10).std() * np.sqrt(252 * 24 * 60)
        df["tech_reversal_signal"] = (close.rolling(5).mean() - close.rolling(20).mean()) / close
        
        # Intraday patterns common in tech equities
        if "timestamp" in df.columns:
            df["hour_of_day"] = pd.to_datetime(df["timestamp"], unit="s").dt.hour
            # Tech stocks often have volatility spikes at 9:30-10:30 EST and 15:30-16:00 EST
            # Convert to UTC (roughly 14:30-15:30 and 20:30-21:00)
            df["tech_open_hour"] = ((df["hour_of_day"] >= 14) & (df["hour_of_day"] <= 15)).astype(float)
            df["tech_close_hour"] = ((df["hour_of_day"] >= 20) & (df["hour_of_day"] <= 21)).astype(float)
        else:
            df["tech_open_hour"] = 0.0
            df["tech_close_hour"] = 0.0
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df

    
    async def start_background_fetching(
        self,
        symbol: str = "BTC",
        intervals: List[str] = ["1m", "5m", "15m"]
    ) -> None:
        """Start background data fetching tasks"""
        self._running = True
        
        # Start kline fetchers for each interval
        for interval in intervals:
            task = asyncio.create_task(
                self._kline_fetch_loop(symbol, interval)
            )
            self._fetch_tasks.append(task)
        
        # Start market snapshot fetcher
        task = asyncio.create_task(
            self._snapshot_fetch_loop(symbol)
        )
        self._fetch_tasks.append(task)
        
        logger.info(f"Started background fetching for {symbol}")
    
    async def stop_background_fetching(self) -> None:
        """Stop all background fetching tasks"""
        self._running = False
        for task in self._fetch_tasks:
            task.cancel()
        await asyncio.gather(*self._fetch_tasks, return_exceptions=True)
        self._fetch_tasks.clear()
        logger.info("Stopped background fetching")
    
    async def _kline_fetch_loop(
        self,
        symbol: str,
        interval: str
    ) -> None:
        """Background loop for fetching klines"""
        interval_seconds = self._parse_interval(interval)
        cache_key = f"{symbol}_{interval}"
        
        if cache_key not in self._kline_cache:
            self._kline_cache[cache_key] = deque(maxlen=1000)
        
        while self._running:
            try:
                klines = await self.api.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=100
                )
                
                for kline in klines:
                    ohlcv = OHLCV(
                        timestamp=kline.get("t", 0) / 1000,
                        open=float(kline.get("o", 0)),
                        high=float(kline.get("h", 0)),
                        low=float(kline.get("l", 0)),
                        close=float(kline.get("c", 0)),
                        volume=float(kline.get("v", 0))
                    )
                    self._kline_cache[cache_key].append(ohlcv)
                
                # Wait for next candle
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Kline fetch error ({interval}): {e}")
                await asyncio.sleep(10)  # Longer backoff on error
    
    async def _snapshot_fetch_loop(
        self,
        symbol: str
    ) -> None:
        """Background loop for market snapshots (optimized interval)"""
        consecutive_failures = 0
        max_failures = 10
        
        while self._running:
            try:
                snapshot = await self.fetch_market_snapshot(symbol)
                if snapshot is not None:
                    self._snapshot_cache.append(snapshot)
                    consecutive_failures = 0  # Reset on success
                else:
                    consecutive_failures += 1
                    logger.warning(f"Snapshot returned None (attempt {consecutive_failures}/{max_failures})")
                
                # Stop trying if API is completely down
                if consecutive_failures >= max_failures:
                    logger.error(f"Snapshot fetch failed {max_failures} times, disabling snapshots")
                    break
                
                # Update every 10 seconds to reduce API load (was 5s)
                await asyncio.sleep(10)
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Snapshot fetch error ({consecutive_failures}/{max_failures}): {e}")
                
                if consecutive_failures >= max_failures:
                    logger.error(f"Too many snapshot failures, disabling snapshots")
                    break
                
                await asyncio.sleep(30)  # Longer backoff on error (was 10s)
    
    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to seconds"""
        unit = interval[-1]
        value = int(interval[:-1])
        
        multipliers = {
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400
        }
        
        return value * multipliers.get(unit, 60)
    
    async def fetch_market_snapshot(self, symbol: str = "BTC") -> Optional[MarketSnapshot]:
        """Fetch current market snapshot"""
        try:
            orderbook = await self.api.get_orderbook(symbol)
            
            # Check for None response
            if orderbook is None or not hasattr(orderbook, 'bids'):
                logger.warning(f"Orderbook returned None or invalid data for {symbol}")
                return None
            
            # Calculate orderbook imbalance
            bid_volume = sum(size for _, size in orderbook.bids[:5]) if orderbook.bids else 0
            ask_volume = sum(size for _, size in orderbook.asks[:5]) if orderbook.asks else 0
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            return MarketSnapshot(
                timestamp=time.time(),
                symbol=symbol,
                mid_price=orderbook.mid_price if orderbook.mid_price else 0,
                bid_price=orderbook.bids[0][0] if orderbook.bids else 0,
                ask_price=orderbook.asks[0][0] if orderbook.asks else 0,
                bid_size=orderbook.bids[0][1] if orderbook.bids else 0,
                ask_size=orderbook.asks[0][1] if orderbook.asks else 0,
                spread_bps=orderbook.spread_bps,
                orderbook_imbalance=imbalance,
                volume_24h=0,  # Would need separate API call
                funding_rate=0  # Would need separate API call
            )
        except Exception as e:
            logger.warning(f"Error fetching market snapshot for {symbol}: {e}")
            return None
    
    async def fetch_wallet_fills(
        self,
        wallet_address: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch real fill history from HyperLiquid SDK for wallet address.
        Provides 90%+ real data for training with actual slippage/fills.
        
        Args:
            wallet_address: User wallet (e.g., 0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584)
            days: Lookback period
        
        Returns:
            DataFrame with timestamp, price, size, side, fee columns
        """
        try:
            logger.info(f"Fetching fill history for wallet {wallet_address[:10]}... ({days} days)")
            
            # Use HyperLiquid SDK user_fills endpoint
            fills = await self.api.get_user_fills_history(
                wallet=wallet_address,
                start_time=int(time.time() * 1000) - (days * 24 * 60 * 60 * 1000)
            )
            
            if not fills or len(fills) == 0:
                logger.warning(f"No fills found for wallet {wallet_address[:10]}...")
                return pd.DataFrame()
            
            # Convert to DataFrame
            fill_data = []
            for fill in fills:
                fill_data.append({
                    "timestamp": fill.get("time", 0) / 1000,
                    "price": float(fill.get("px", 0)),
                    "size": float(fill.get("sz", 0)),
                    "side": fill.get("side", ""),
                    "fee": float(fill.get("fee", 0)),
                    "closed_pnl": float(fill.get("closedPnl", 0))
                })
            
            df = pd.DataFrame(fill_data)
            logger.info(f"Loaded {len(df)} fills from wallet history")
            return df
            
        except Exception as e:
            logger.warning(f"Error fetching wallet fills: {e}")
            return pd.DataFrame()
    
    async def fetch_historical_klines(
        self,
        symbol: str = "BTC",
        interval: str = "1m",
        days: int = 30,
        save: bool = True
    ) -> pd.DataFrame:
        """Fetch historical klines for backtesting"""
        logger.info(f"Fetching {days} days of {interval} klines for {symbol}")
        
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 86400 * 1000)
        
        all_klines = []
        
        # For periods up to 7 days, fetch in a single request (API supports this)
        if days <= 7:
            try:
                logger.info(f"Fetching single batch: start={start_time}, end={end_time}")
                klines = await self.api.get_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time
                )
                logger.info(f"Got {len(klines) if klines else 0} klines from API")
                if klines:
                    all_klines = klines
            except Exception as e:
                logger.error(f"Error fetching klines: {e}")
        else:
            # For longer periods, fetch in weekly batches with longer delays
            current_start = start_time
            week_ms = 7 * 86400 * 1000
            batch_num = 0
            consecutive_failures = 0
            max_consecutive_failures = 1  # Fail fast for unavailable symbols
            
            while current_start < end_time:
                try:
                    batch_num += 1
                    batch_end = min(current_start + week_ms, end_time)
                    logger.info(f"Fetching batch {batch_num}: {current_start} to {batch_end}")
                    
                    klines = await self.api.get_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=current_start,
                        end_time=batch_end
                    )
                    
                    if not klines:
                        logger.warning(f"Empty response for batch {batch_num}")
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error(f"Too many consecutive failures ({consecutive_failures}), aborting")
                            break
                        # Skip to next batch to avoid infinite loop
                        current_start = batch_end + 1
                        await asyncio.sleep(2)
                        continue
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    logger.info(f"Batch {batch_num} returned {len(klines)} klines")
                    all_klines.extend(klines)
                    
                    # Move to next week
                    current_start = batch_end + 1
                    
                    # Longer delay between batches (2 seconds)
                    if current_start < end_time:
                        await asyncio.sleep(2.0)
                    
                except Exception as e:
                    logger.error(f"Error fetching klines batch {batch_num}: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Too many consecutive failures ({consecutive_failures}), aborting fetch for {symbol}")
                        break
                    # Skip to next batch to avoid infinite loop on persistent errors
                    current_start = batch_end + 1 if 'batch_end' in dir() else current_start + week_ms
                    await asyncio.sleep(5)
        
        logger.info(f"Total klines collected: {len(all_klines)}")
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "timestamp": k.get("t", 0) / 1000,
                "open": float(k.get("o", 0)),
                "high": float(k.get("h", 0)),
                "low": float(k.get("l", 0)),
                "close": float(k.get("c", 0)),
                "volume": float(k.get("v", 0))
            }
            for k in all_klines
        ])
        
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            
            if save:
                await self.save_historical_data(df, symbol, interval)
        
        logger.info(f"Fetched {len(df)} klines total")
        return df
    
    async def save_historical_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str
    ) -> None:
        """Save historical data to parquet for efficient storage/loading"""
        filename = f"{symbol}_{interval}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.parquet"
        filepath = self.data_dir / "historical" / filename
        
        # Use pyarrow for fast parquet writing
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filepath, compression="snappy")
        
        logger.info(f"Saved historical data to {filepath}")
    
    async def load_historical_data(
        self,
        symbol: str = "BTC",
        interval: str = "1m",
        days: Optional[int] = None
    ) -> pd.DataFrame:
        """Load historical data from parquet files"""
        historical_dir = self.data_dir / "historical"
        pattern = f"{symbol}_{interval}_*.parquet"
        
        files = sorted(historical_dir.glob(pattern))
        
        if not files:
            logger.warning(f"No historical data found for {symbol} {interval}")
            return pd.DataFrame()
        
        # Load and concatenate all files
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            dfs.append(df)
        
        result = pd.concat(dfs, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        
        if days:
            cutoff = time.time() - (days * 86400)
            result = result[result["timestamp"] >= cutoff]
        
        logger.info(f"Loaded {len(result)} historical records")
        return result
    
    async def save_trade(
        self,
        trade_data: Dict[str, Any]
    ) -> None:
        """Save individual trade to trading directory"""
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        filepath = self.data_dir / "trading" / f"trades_{date_str}.json"
        
        # Append to daily trades file
        trades = []
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    content = f.read()
                    if content.strip():
                        trades = json.loads(content)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Could not load existing trades file: {e}, starting fresh")
                trades = []
        
        trade_data["saved_at"] = time.time()
        trades.append(trade_data)
        
        with open(filepath, "w") as f:
            json.dump(trades, f, indent=2, default=str)
    
    async def save_backtest_results(
        self,
        results: Dict[str, Any],
        name: str = "backtest"
    ) -> None:
        """Save backtest results"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = self.data_dir / "backtests" / filename
        
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved backtest results to {filepath}")
    
    async def create_backup(self) -> None:
        """Create hourly backup of data"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = self.data_dir / "backups" / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup trading data
        trading_dir = self.data_dir / "trading"
        for f in trading_dir.glob("*.json"):
            import shutil
            shutil.copy(f, backup_dir / f.name)
        
        # Backup latest backtest
        backtest_dir = self.data_dir / "backtests"
        backtest_files = sorted(backtest_dir.glob("*.json"))
        if backtest_files:
            import shutil
            shutil.copy(backtest_files[-1], backup_dir / backtest_files[-1].name)
        
        logger.info(f"Created backup at {backup_dir}")
    
    def get_cached_klines(
        self,
        symbol: str = "BTC",
        interval: str = "1m",
        limit: int = 100
    ) -> List[OHLCV]:
        """Get cached klines for fast access"""
        cache_key = f"{symbol}_{interval}"
        if cache_key not in self._kline_cache:
            return []
        
        cache = self._kline_cache[cache_key]
        return list(cache)[-limit:]
    
    def get_cached_snapshots(self, limit: int = 100) -> List[MarketSnapshot]:
        """Get cached market snapshots"""
        return list(self._snapshot_cache)[-limit:]
    
    def get_latest_snapshot(self) -> Optional[MarketSnapshot]:
        """Get most recent market snapshot"""
        if self._snapshot_cache:
            return self._snapshot_cache[-1]
        return None
    
    async def get_training_data(
        self,
        symbol: str = "BTC",
        interval: str = "1m",
        days: int = 30,
        use_cache: bool = True,
        append_live: bool = False
    ) -> pd.DataFrame:
        """
        Get prepared training data with features.
        
        Optimizations:
        - Joblib caching for preprocessed data
        - In-memory cache for repeated access
        - Async loading from parquet files
        
        Args:
            symbol: Trading symbol
            interval: Kline interval
            days: Number of days of historical data (default 30 for better pattern coverage)
            use_cache: Use disk/memory cache
            append_live: Append recent live cached klines for freshness
        """
        cache_key = f"{symbol}_{interval}_{days}d"
        
        # Check in-memory cache first (fastest)
        if use_cache and cache_key in self._training_data_cache and not append_live:
            logger.info(f"Using in-memory cached training data: {cache_key}")
            return self._training_data_cache[cache_key].copy()
        
        # Check disk cache (second fastest)
        if use_cache and not append_live:
            cached_df = self._load_cached_training_data(symbol, interval, days)
            if cached_df is not None:
                self._training_data_cache[cache_key] = cached_df
                return cached_df.copy()
        
        # Try to load from historical parquet files
        df = await self.load_historical_data(symbol, interval, days)
        
        if df.empty:
            # Fetch fresh data from API with exception handling
            try:
                df = await self.fetch_historical_klines(symbol, interval, days)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} data: {e}")
                df = pd.DataFrame()  # Ensure fallback triggers
        
        # If primary symbol (XYZ100) data is insufficient, use fallback
        if (df.empty or len(df) < 500) and symbol == self.PRIMARY_SYMBOL:
            logger.warning(f"{symbol} data insufficient ({len(df)} rows) - trying SPX/BTC fallback")
            self._using_fallback = True
            
            # Try SPX first (better equity correlation)
            fallback_df = await self.get_fallback_data(days)
            if fallback_df is not None and len(fallback_df) >= 500:
                logger.info(f"Using {self._fallback_type} fallback data ({len(fallback_df)} rows)")
                df = fallback_df
            else:
                # Last resort: synthetic data
                logger.warning("All fallbacks failed, generating synthetic SPX data")
                df = self.generate_synthetic_spx(days)
        
        if df.empty:
            return df
        
        # STEP 3: Append recent live data if requested
        if append_live:
            live_df = self._get_recent_live_klines(symbol, interval, limit=60)
            if not live_df.empty:
                logger.info(f"Appending {len(live_df)} recent live klines to training data")
                df = pd.concat([df, live_df], ignore_index=True)
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        
        # Add technical features
        df = self._add_technical_features(df)
        
        # Add equity perpetual specific features (XYZ100 tech correlation)
        df = self._add_equity_perp_features(df)
        
        # Cache the processed data (don't cache live-appended data)
        if use_cache and not append_live:
            self._save_cached_training_data(df, symbol, interval, days)
            self._training_data_cache[cache_key] = df
        
        return df
    
    def _get_recent_live_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 60
    ) -> pd.DataFrame:
        """
        Get recent live klines from the background cache (Step 3).
        This provides fresh data for retraining without API calls.
        """
        cache_key = f"{symbol}_{interval}"
        
        if cache_key not in self._kline_cache or not self._kline_cache[cache_key]:
            return pd.DataFrame()
        
        # Get recent klines from cache
        cached_klines = list(self._kline_cache[cache_key])[-limit:]
        
        if not cached_klines:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "timestamp": k.timestamp,
            "open": k.open,
            "high": k.high,
            "low": k.low,
            "close": k.close,
            "volume": k.volume
        } for k in cached_klines])
        
        return df
    
    def _get_cache_hash(self, symbol: str, interval: str, days: int) -> str:
        """Generate cache file hash based on parameters and date"""
        # Include today's date so cache expires daily
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        cache_str = f"{symbol}_{interval}_{days}_{date_str}"
        return hashlib.md5(cache_str.encode()).hexdigest()[:12]
    
    def _load_cached_training_data(
        self,
        symbol: str,
        interval: str,
        days: int
    ) -> Optional[pd.DataFrame]:
        """Load preprocessed training data from joblib cache"""
        cache_hash = self._get_cache_hash(symbol, interval, days)
        cache_path = self._tensor_cache_dir / f"training_{cache_hash}.joblib"
        
        if cache_path.exists():
            try:
                data = joblib.load(cache_path)
                if isinstance(data, dict) and "df" in data:
                    logger.info(f"Loaded cached training data from {cache_path}")
                    return data["df"]
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
                # Remove corrupted cache
                cache_path.unlink(missing_ok=True)
        
        return None
    
    def _save_cached_training_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        days: int
    ) -> None:
        """Save preprocessed training data to joblib cache"""
        cache_hash = self._get_cache_hash(symbol, interval, days)
        cache_path = self._tensor_cache_dir / f"training_{cache_hash}.joblib"
        
        try:
            data = {
                "df": df,
                "symbol": symbol,
                "interval": interval,
                "days": days,
                "timestamp": time.time(),
                "rows": len(df)
            }
            joblib.dump(data, cache_path, compress=3)
            logger.info(f"Saved training data cache to {cache_path} ({len(df)} rows)")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def get_training_tensor(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Convert DataFrame to PyTorch tensor for training.
        Optimized for direct GPU/MPS upload.
        """
        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        features = df[available_cols].values.astype(np.float32)
        
        # Handle NaN/inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Z-score normalization
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        features = (features - mean) / std
        
        # Convert to tensor
        tensor = torch.from_numpy(features)
        
        if device is not None:
            tensor = tensor.to(device)
        
        return tensor
    
    def clear_cache(self, older_than_days: int = 7) -> int:
        """Clear old cache files to free disk space"""
        cleared = 0
        cutoff = time.time() - (older_than_days * 86400)
        
        for cache_file in self._tensor_cache_dir.glob("*.joblib"):
            try:
                if cache_file.stat().st_mtime < cutoff:
                    cache_file.unlink()
                    cleared += 1
            except Exception as e:
                logger.warning(f"Failed to clear cache file {cache_file}: {e}")
        
        # Clear in-memory cache
        self._training_data_cache.clear()
        
        logger.info(f"Cleared {cleared} old cache files")
        return cleared
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators for ML training.
        
        Step 3: Enhanced feature engineering with:
        - Standard TA indicators (RSI, MACD, BB, ATR)
        - Momentum and volatility features
        - Trend strength (ADX)
        - Price patterns (higher highs, lower lows)
        - Volume-price divergence
        """
        # Returns
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # Moving averages
        for period in [7, 14, 20, 21, 50, 100, 200]:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            df[f"ema_{period}"] = df["close"].ewm(span=period).mean()
        
        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df["close"].ewm(span=12).mean()
        exp2 = df["close"].ewm(span=26).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (std * 2)
        df["bb_lower"] = df["bb_middle"] - (std * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        
        # ATR
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"]
        
        # Volume features
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"]
        
        # Price momentum
        df["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
        
        # Volatility
        df["volatility"] = df["returns"].rolling(20).std() * np.sqrt(365 * 24 * 60)  # Annualized
        
        # Trend strength
        df["adx"] = self._calculate_adx(df)
        
        # Step 3 Enhanced: Price patterns
        df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(float)
        df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(float)
        df["higher_close"] = (df["close"] > df["close"].shift(1)).astype(float)
        
        # Step 3 Enhanced: Price-Volume divergence
        price_trend = df["close"].diff(5).apply(np.sign)
        volume_trend = df["volume"].diff(5).apply(np.sign)
        df["pv_divergence"] = (price_trend != volume_trend).astype(float)
        
        # Step 3 Enhanced: Candle patterns
        df["body_size"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8)
        df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"] + 1e-8)
        df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-8)
        
        # Step 3 Enhanced: Relative price position
        df["price_position_5"] = (df["close"] - df["low"].rolling(5).min()) / (df["high"].rolling(5).max() - df["low"].rolling(5).min() + 1e-8)
        df["price_position_20"] = (df["close"] - df["low"].rolling(20).min()) / (df["high"].rolling(20).max() - df["low"].rolling(20).min() + 1e-8)
        
        # Step 3 Enhanced: Rate of change
        df["roc_5"] = df["close"].pct_change(5)
        df["roc_10"] = df["close"].pct_change(10)
        
        # Step 3 v2: Orderbook-like features (simulated from price action)
        # These approximate orderbook dynamics when live orderbook not available
        df["bid_ask_spread_pct"] = df["atr_pct"] * 0.1  # Spread approximation
        df["bid_side_pressure"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
        df["ask_side_pressure"] = (df["high"] - df["close"]) / (df["high"] - df["low"] + 1e-8)
        df["imbalance_ratio"] = df["bid_side_pressure"] / (df["ask_side_pressure"] + 1e-8)
        df["imbalance_ratio"] = df["imbalance_ratio"].clip(0.1, 10)  # Clip extremes
        
        # Depth ratio approximation (based on volume and volatility)
        df["depth_ratio"] = df["volume_ratio"] / (df["atr_pct"] + 1e-8)
        df["depth_ratio"] = df["depth_ratio"].clip(0.1, 100)
        
        # Step 3 v2: Funding rate proxy (from momentum divergence)
        # Positive when momentum is bullish, negative when bearish
        short_mom = df["close"].pct_change(8)  # ~8 hours
        long_mom = df["close"].pct_change(24)  # ~1 day
        df["funding_proxy"] = (short_mom - long_mom) * 100  # Scaled funding proxy
        df["funding_proxy"] = df["funding_proxy"].clip(-0.1, 0.1)
        
        # Step 3 v2: Mean reversion signals
        df["zscore_20"] = (df["close"] - df["sma_20"]) / (df["close"].rolling(20).std() + 1e-8)
        df["zscore_50"] = (df["close"] - df["sma_50"]) / (df["close"].rolling(50).std() + 1e-8)
        
        # Step 3 v2: Volatility regime
        vol_short = df["returns"].rolling(5).std()
        vol_long = df["returns"].rolling(50).std()
        df["vol_regime"] = vol_short / (vol_long + 1e-8)  # >1 = high vol regime
        
        # Fill NaN values
        df = df.ffill().bfill()
        
        return df
    
    def augment_data(
        self,
        df: pd.DataFrame,
        noise_pct: float = 0.01,
        num_augments: int = 2
    ) -> pd.DataFrame:
        """
        Step 3: Data augmentation with price jitter.
        Creates synthetic variations of the data for training robustness.
        
        Args:
            df: Original DataFrame
            noise_pct: Percentage noise to add (default 1%)
            num_augments: Number of augmented copies to create
            
        Returns:
            Augmented DataFrame with original + synthetic data
        """
        augmented_dfs = [df.copy()]
        
        price_cols = ["open", "high", "low", "close"]
        
        for i in range(num_augments):
            aug_df = df.copy()
            
            # Add random noise to price columns
            for col in price_cols:
                if col in aug_df.columns:
                    noise = np.random.normal(1.0, noise_pct, len(aug_df))
                    aug_df[col] = aug_df[col] * noise
            
            # Ensure high >= low
            aug_df["high"] = aug_df[["high", "low"]].max(axis=1)
            aug_df["low"] = aug_df[["high", "low"]].min(axis=1)
            
            # Ensure close is within high-low range
            aug_df["close"] = aug_df["close"].clip(aug_df["low"], aug_df["high"])
            aug_df["open"] = aug_df["open"].clip(aug_df["low"], aug_df["high"])
            
            # Add volume noise
            if "volume" in aug_df.columns:
                vol_noise = np.random.normal(1.0, noise_pct * 2, len(aug_df))
                aug_df["volume"] = (aug_df["volume"] * vol_noise).clip(lower=0)
            
            augmented_dfs.append(aug_df)
        
        # Concatenate and re-add features
        result = pd.concat(augmented_dfs, ignore_index=True)
        result = result.sort_values("timestamp").reset_index(drop=True)
        
        logger.info(f"Augmented data: {len(df)} -> {len(result)} rows ({num_augments} augments)")
        
        return result
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (abs(minus_dm.rolling(period).mean()) / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
