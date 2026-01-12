#!/usr/bin/env python3
"""Script to update data_fetcher.py with XYZ100-USDC specific features"""

import os

# Read the current file
file_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'data_fetcher.py')

with open(file_path, 'r') as f:
    content = f.read()

# Check if already updated
if 'XYZ100_MIN_HISTORY_RECORDS' in content:
    print("File already has XYZ100 features")
    with open(file_path, 'w') as f:
        f.write(content)
    exit(0)
else:
    # Add XYZ100 config to __init__
    init_insert = '''
        
        # XYZ100-USDC specific configuration
        self.XYZ100_MIN_HISTORY_RECORDS = 5000
        self.XYZ100_MIN_HISTORY_DAYS = 30
        self.PRIMARY_SYMBOL = "XYZ100"
        self.FALLBACK_SYMBOL = "BTC"
        self._using_fallback = False
        self._last_xyz100_check = 0.0
        self._xyz100_check_interval = 3600  # Check hourly'''
    
    old_init = '        self._running = False'
    new_init = '        self._running = False' + init_insert
    content = content.replace(old_init, new_init, 1)
    
    print("Added XYZ100 config to __init__")

# Add XYZ100 methods after _ensure_directories
methods_code = '''
    
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
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
'''

if 'check_xyz100_availability' not in content:
    # Insert after _ensure_directories method
    old_pattern = '            (self.data_dir / subdir).mkdir(parents=True, exist_ok=True)'
    new_pattern = old_pattern + methods_code
    content = content.replace(old_pattern, new_pattern, 1)
    print("Added XYZ100 methods")
else:
    print("XYZ100 methods already exist")

# Write updated file
with open(file_path, 'w') as f:
    f.write(content)

print("data_fetcher.py updated successfully")
