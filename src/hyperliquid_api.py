"""
Hyperliquid API Wrapper
High-performance async interface for Hyperliquid exchange
Optimized for HFT with sub-millisecond latency targets
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque

import aiohttp
import orjson
import websockets
from eth_account import Account
from eth_account.signers.local import LocalAccount

# Logger - configuration is handled by launch.py with ColoredFormatter
logger = logging.getLogger(__name__)


@dataclass
class OrderBook:
    """Real-time orderbook representation"""
    bids: List[Tuple[float, float]] = field(default_factory=list)  # (price, size)
    asks: List[Tuple[float, float]] = field(default_factory=list)
    timestamp: float = 0.0
    
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2
        return 0.0
    
    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 0.0
    
    @property
    def spread_bps(self) -> float:
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return 0.0


@dataclass
class Position:
    """Current position data"""
    symbol: str
    size: float
    entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: int
    liquidation_price: Optional[float] = None
    margin_used: float = 0.0


@dataclass
class Trade:
    """Trade execution data"""
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    timestamp: float
    fee: float = 0.0
    is_maker: bool = False
    closed_pnl: float = 0.0  # Step 1: Realized PnL from position closure
    start_position: float = 0.0  # Position size before this trade
    dir: str = ""  # Direction indicator for position tracking


class CircuitBreaker:
    """Circuit breaker for rate limit protection (Step 5)"""
    
    def __init__(
        self,
        max_failures: int = 3,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0
    ):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half_open
        self._lock = asyncio.Lock()
    
    async def record_failure(self) -> None:
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            
            if self.failure_count >= self.max_failures:
                self.state = "open"
                logger.warning(
                    f"🔌 Circuit breaker OPEN: {self.failure_count} consecutive failures. "
                    f"Pausing requests for {self.reset_timeout}s"
                )
    
    async def record_success(self) -> None:
        async with self._lock:
            self.failure_count = 0
            self.state = "closed"
    
    async def can_proceed(self) -> bool:
        async with self._lock:
            if self.state == "closed":
                return True
            
            now = time.monotonic()
            elapsed = now - self.last_failure_time
            
            if self.state == "open":
                if elapsed >= self.reset_timeout:
                    self.state = "half_open"
                    logger.info("🔌 Circuit breaker HALF-OPEN: Allowing test request")
                    return True
                return False
            
            if self.state == "half_open":
                return True
            
            return True
    
    async def wait_if_needed(self) -> None:
        """Wait until circuit allows requests"""
        while not await self.can_proceed():
            wait_time = max(1, self.reset_timeout - (time.monotonic() - self.last_failure_time))
            logger.debug(f"Circuit breaker wait: {wait_time:.1f}s remaining")
            await asyncio.sleep(min(wait_time, 5.0))


class RateLimiter:
    """Token bucket rate limiter for API calls"""
    
    def __init__(self, rate: float, capacity: float, name: str = "default"):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
        self.name = name
        self._limit_hit_count = 0
        self._last_limit_warning = 0.0
    
    async def acquire(self, tokens: float = 1.0) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.rate
                self._limit_hit_count += 1
                
                # Log warning when rate limit is hit (max once per 10 seconds)
                if now - self._last_limit_warning > 10.0:
                    logger.warning(
                        f"⚠️ RATE LIMIT HIT [{self.name}]: Waiting {wait_time:.2f}s for tokens. "
                        f"Hit count: {self._limit_hit_count}, tokens: {self.tokens:.2f}/{self.capacity:.2f}"
                    )
                    self._last_limit_warning = now
                
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= tokens


@dataclass
class CachedResponse:
    """Cached API response with TTL"""
    data: Any
    timestamp: float
    ttl: float  # Time-to-live in seconds
    
    @property
    def is_valid(self) -> bool:
        return time.time() - self.timestamp < self.ttl


class HyperliquidAPI:
    """
    High-performance async wrapper for Hyperliquid API
    Supports REST and WebSocket connections for HFT
    """
    
    def __init__(self, config_path: str = "config/api.json"):
        self.config = self._load_config(config_path)
        self.account: Optional[LocalAccount] = None
        self.wallet_address: Optional[str] = None
        
        # Connection state
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Any = None  # WebSocket connection
        self._ws_connected = False
        self._ws_subscriptions: Dict[str, Callable] = {}
        
        # Step 1: API Response Cache - aggressive TTLs for <15 calls/min target
        self._response_cache: Dict[str, CachedResponse] = {}
        self._cache_ttl = {
            "user_state": 10.0,     # Step 1: User state cached 10s (was 5s) - primary call reduction
            "positions": 10.0,      # Step 1: Positions cached 10s (same as user_state)
            "orderbook": 3.0,       # Step 1: Orderbook cached 3s (was 2s) - for signal generation
            "open_orders": 10.0,    # Step 1: Open orders cached 10s (was 5s)
            "user_fills": 30.0,     # Step 1: Fills cached 30s (was 15s) - processed via WS callback
            "mid_price": 3.0,       # Step 1: Mid price cached 3s (was 2s)
        }
        self._cache_lock = asyncio.Lock()
        
        # Request deduplication - prevent concurrent identical requests
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        # Step 5: API call statistics for monitoring
        self._api_call_counts: Dict[str, int] = {}
        self._last_stats_log = 0.0
        self._calls_this_minute: int = 0
        self._minute_start: float = 0.0
        
        # Step 1: Rate limiters - optimized for <15 calls/min with larger burst
        self._rest_limiter = RateLimiter(
            rate=0.25,   # 0.25 req/s = 15/min target, caching keeps actual lower
            capacity=15,  # Allow burst of 15 for startup/init phases
            name="REST"
        )
        self._order_limiter = RateLimiter(
            rate=0.5,  # 0.5 orders per second - orders are infrequent
            capacity=10,  # Allow burst of 10 orders
            name="ORDER"
        )
        
        # Step 5: Circuit breaker for rate limit protection
        self._circuit_breaker = CircuitBreaker(
            max_failures=3,
            reset_timeout=15.0,  # Reduced from 60s for faster fallback
            half_open_timeout=10.0  # Reduced from 30s
        )
        
        # Track last request time for additional throttling
        self._last_request_time: float = 0.0
        self._min_request_interval: float = 0.5  # 500ms minimum between requests
        self._backoff_multiplier: float = 1.0  # Step 5: Exponential backoff multiplier
        self._max_backoff: float = 60.0  # Maximum backoff time
        
        # Step 1: Position and fill tracking for PnL realization
        self._open_positions: Dict[str, Dict[str, Any]] = {}  # Track open positions
        self._pending_orders: Dict[str, Dict[str, Any]] = {}  # Track pending orders
        self._fill_callbacks: List[Callable] = []  # Callbacks for fill events
        self._position_callbacks: List[Callable] = []  # Callbacks for position updates
        self._total_realized_pnl: float = 0.0  # Cumulative realized PnL
        self._ws_user_connected: bool = False  # Track user event subscription
        
        # Caches for low-latency access
        self._orderbook_cache: Dict[str, OrderBook] = {}
        self._position_cache: Dict[str, Position] = {}
        self._trade_history: deque = deque(maxlen=1000)
        
        # Performance metrics
        self._latency_samples: deque = deque(maxlen=100)
        
        # WebSocket message queue
        self._ws_queue: asyncio.Queue = asyncio.Queue()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load API configuration"""
        with open(config_path, "r") as f:
            return json.load(f)
    
    async def initialize(self) -> None:
        """Initialize connections and authenticate"""
        # Load credentials from environment
        private_key = os.environ.get(
            self.config["hyperliquid"]["private_key_env"]
        )
        wallet_address = os.environ.get(
            self.config["hyperliquid"]["wallet_address_env"]
        )
        
        if not private_key or not wallet_address:
            raise ValueError(
                "Missing Hyperliquid credentials. Set environment variables: "
                f"{self.config['hyperliquid']['private_key_env']}, "
                f"{self.config['hyperliquid']['wallet_address_env']}"
            )
        
        # Initialize account
        self.account = Account.from_key(private_key)
        self.wallet_address = wallet_address
        
        # Create aiohttp session with optimized settings
        timeout = aiohttp.ClientTimeout(
            total=self.config["hyperliquid"]["timeout_seconds"]
        )
        connector = aiohttp.TCPConnector(
            limit=100,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30
        )
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            json_serialize=lambda x: orjson.dumps(x).decode()
        )
        
        # Step 2: Warn if connected to testnet
        self.warn_if_testnet()
        
        logger.info(f"Initialized Hyperliquid API for wallet: {wallet_address[:10]}...")
    
    async def close(self) -> None:
        """Clean up connections gracefully"""
        try:
            if self._ws:
                try:
                    await self._ws.close()
                except Exception as e:
                    logger.debug(f"WebSocket close error (ignored): {e}")
            if self._session and not self._session.closed:
                try:
                    await self._session.close()
                except Exception as e:
                    logger.debug(f"Session close error (ignored): {e}")
        except Exception as e:
            logger.debug(f"Close cleanup error (ignored): {e}")
        finally:
            self._session = None
            self._ws = None
        logger.info("Closed all Hyperliquid connections")
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session is open, recreate if needed"""
        if self._session is None or self._session.closed:
            await self._recreate_session()
    
    async def _recreate_session(self) -> None:
        """Recreate the HTTP session after connection issues"""
        logger.info("Recreating HTTP session...")
        if self._session and not self._session.closed:
            try:
                await self._session.close()
            except Exception:
                pass
        
        timeout = aiohttp.ClientTimeout(
            total=self.config["hyperliquid"]["timeout_seconds"],
            connect=30,
            sock_read=60
        )
        connector = aiohttp.TCPConnector(
            limit=50,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            force_close=False
        )
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            json_serialize=lambda x: orjson.dumps(x).decode()
        )
        logger.info("HTTP session recreated")
    
    async def _post_request(
        self,
        url: str,
        payload: Dict[str, Any],
        signed: bool = False
    ) -> Dict[str, Any]:
        """Execute POST request with rate limiting, circuit breaker, and retries"""
        # Step 5: Check circuit breaker before proceeding
        await self._circuit_breaker.wait_if_needed()
        
        await self._rest_limiter.acquire()
        
        # Enforce minimum request interval with backoff multiplier
        now = time.time()
        effective_interval = self._min_request_interval * self._backoff_multiplier
        time_since_last = now - self._last_request_time
        if time_since_last < effective_interval:
            await asyncio.sleep(effective_interval - time_since_last)
        self._last_request_time = time.time()
        
        request_start = time.perf_counter()
        
        headers = {"Content-Type": "application/json"}
        consecutive_timeouts = 0
        consecutive_429s = 0  # Step 5: Track rate limits
        
        # Step 1: Ensure session is available before use
        await self._ensure_session()
        
        for attempt in range(self.config["hyperliquid"]["retry_attempts"]):
            try:
                # Check session before each attempt
                if self._session is None or self._session.closed:
                    await self._ensure_session()
                
                async with self._session.post(
                    url,
                    json=payload,
                    headers=headers
                ) as response:
                    latency = (time.perf_counter() - request_start) * 1000
                    self._latency_samples.append(latency)
                    consecutive_timeouts = 0  # Reset on success
                    
                    if response.status == 200:
                        # Step 5: Reset backoff on success
                        self._backoff_multiplier = 1.0
                        consecutive_429s = 0
                        await self._circuit_breaker.record_success()
                        
                        # Try to parse JSON, handle potential errors
                        try:
                            text = await response.text()
                            if not text or text.strip() == "":
                                logger.error("Empty response from API")
                                return {"status": "error", "response": "empty"}
                            result = orjson.loads(text)
                            # Check for error response from exchange
                            if isinstance(result, dict):
                                if result.get("status") == "err":
                                    logger.error(f"Exchange error: {result.get('response', 'unknown')}")
                            return result
                        except Exception as je:
                            logger.error(f"JSON parse error: {je}, response text: {text[:500]}")
                            return {"status": "error", "response": text[:500]}
                    elif response.status == 429:
                        # Step 5: Rate limited - exponential backoff with circuit breaker
                        consecutive_429s += 1
                        self._backoff_multiplier = min(
                            self._max_backoff,
                            self._backoff_multiplier * 2
                        )
                        wait_time = min(self._max_backoff, 2 ** attempt * self._backoff_multiplier)
                        logger.warning(
                            f"Rate limited (429 #{consecutive_429s}), backing off {wait_time:.1f}s "
                            f"(multiplier: {self._backoff_multiplier:.1f}x)"
                        )
                        
                        # Record failure for circuit breaker
                        await self._circuit_breaker.record_failure()
                        
                        await asyncio.sleep(wait_time)
                    elif response.status >= 500:
                        # Server error - use shorter backoff for faster fallback
                        text = await response.text()
                        server_backoff = min(5 * (2 ** attempt), 30)  # Cap at 30s for faster fallback
                        logger.error(
                            f"Server error {response.status}: {text[:200]}, "
                            f"retrying in {server_backoff}s (attempt {attempt + 1})"
                        )
                        await self._circuit_breaker.record_failure()
                        await asyncio.sleep(server_backoff)
                    else:
                        text = await response.text()
                        logger.error(f"API error {response.status}: {text[:200]}")
                        await asyncio.sleep(1)
                        
            except asyncio.TimeoutError:
                consecutive_timeouts += 1
                logger.error(f"Request timeout (attempt {attempt + 1})")
                
                # Recreate session after 3 consecutive timeouts
                if consecutive_timeouts >= 3:
                    await self._recreate_session()
                    consecutive_timeouts = 0
                
                await asyncio.sleep(
                    self.config["hyperliquid"]["retry_delay_ms"] / 1000
                )
            except aiohttp.ClientError as e:
                logger.error(f"Client error (attempt {attempt + 1}): {type(e).__name__}: {e}")
                # Recreate session on connection errors
                await self._recreate_session()
                await asyncio.sleep(
                    self.config["hyperliquid"]["retry_delay_ms"] / 1000
                )
            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}): {type(e).__name__}: {e}")
                await asyncio.sleep(
                    self.config["hyperliquid"]["retry_delay_ms"] / 1000
                )
        
        raise Exception(f"Failed to execute request after {attempt + 1} attempts")
    
    async def _cached_request(
        self,
        cache_key: str,
        url: str,
        payload: Dict[str, Any],
        ttl_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute POST request with caching and request deduplication.
        Prevents redundant API calls for identical requests within TTL.
        """
        # Check cache first
        async with self._cache_lock:
            if cache_key in self._response_cache:
                cached = self._response_cache[cache_key]
                if cached.is_valid:
                    logger.debug(f"Cache HIT [{cache_key}]: age={time.time() - cached.timestamp:.2f}s")
                    return cached.data
            
            # Check if identical request is already pending (deduplication)
            if cache_key in self._pending_requests:
                logger.debug(f"Request dedup [{cache_key}]: waiting for pending request")
                try:
                    return await self._pending_requests[cache_key]
                except Exception:
                    pass  # If pending request fails, we'll retry
        
        # Create a future for this request (for deduplication)
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_requests[cache_key] = future
        
        try:
            # Track API call counts
            self._api_call_counts[cache_key] = self._api_call_counts.get(cache_key, 0) + 1
            
            # Log stats every 60 seconds
            now = time.time()
            if now - self._last_stats_log > 60.0:
                total_calls = sum(self._api_call_counts.values())
                top_calls = sorted(self._api_call_counts.items(), key=lambda x: -x[1])[:5]
                logger.info(f"📊 API call stats (60s): total={total_calls}, top={top_calls}")
                self._api_call_counts.clear()
                self._last_stats_log = now
            
            # Make the actual request
            result = await self._post_request(url, payload)
            
            # Cache the result
            ttl = ttl_override or self._cache_ttl.get(cache_key.split(":")[0], 1.0)
            async with self._cache_lock:
                self._response_cache[cache_key] = CachedResponse(
                    data=result,
                    timestamp=time.time(),
                    ttl=ttl
                )
            
            # Complete the future for any waiting requests
            if not future.done():
                future.set_result(result)
            
            return result
            
        except Exception as e:
            if not future.done():
                future.set_exception(e)
            raise
        finally:
            # Remove from pending requests
            self._pending_requests.pop(cache_key, None)
    
    def clear_cache(self, prefix: Optional[str] = None) -> None:
        """Clear response cache, optionally by prefix"""
        if prefix:
            keys_to_remove = [k for k in self._response_cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._response_cache[k]
            logger.debug(f"Cleared {len(keys_to_remove)} cached responses with prefix '{prefix}'")
        else:
            count = len(self._response_cache)
            self._response_cache.clear()
            logger.debug(f"Cleared all {count} cached responses")
    
    def _sign_action(
        self,
        action: Dict[str, Any],
        nonce: int
    ) -> Dict[str, Any]:
        """Sign an action for authenticated requests using Hyperliquid's phantom agent pattern"""
        from eth_account.messages import encode_typed_data
        from eth_utils import keccak, to_hex
        import msgpack
        import numpy as np
        
        # Convert numpy types to native Python types recursively
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean the action of numpy types
        clean_action = convert_numpy(action)
        
        # Hash the action using msgpack (official SDK approach)
        data = msgpack.packb(clean_action)
        data += nonce.to_bytes(8, "big")
        # No vault address (add null byte)
        data += b"\x00"
        action_hash = keccak(data)
        
        # Construct phantom agent for mainnet (source = "a")
        phantom_agent = {
            "source": "a",  # "a" for mainnet, "b" for testnet
            "connectionId": action_hash
        }
        
        # L1 payload structure (official SDK format)
        l1_data = {
            "domain": {
                "chainId": 1337,
                "name": "Exchange",
                "verifyingContract": "0x0000000000000000000000000000000000000000",
                "version": "1",
            },
            "types": {
                "Agent": [
                    {"name": "source", "type": "string"},
                    {"name": "connectionId", "type": "bytes32"},
                ],
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
            },
            "primaryType": "Agent",
            "message": phantom_agent,
        }
        
        # Sign the message
        signable_message = encode_typed_data(full_message=l1_data)
        
        if self.account is None:
            raise RuntimeError("Account not initialized")
        signed = self.account.sign_message(signable_message)
        
        return {
            "action": clean_action,
            "nonce": nonce,
            "signature": {
                "r": to_hex(signed.r),
                "s": to_hex(signed.s),
                "v": signed.v
            },
            "vaultAddress": None
        }
    
    # ==================== Market Data Methods ====================
    
    async def get_meta(self) -> Dict[str, Any]:
        """Get exchange metadata including asset info"""
        payload = {"type": "meta"}
        # Add dex parameter for XYZ perps
        dex = self.config.get("hyperliquid", {}).get("dex")
        if dex:
            payload["dex"] = dex
        return await self._post_request(
            self.config["hyperliquid"]["info_url"],
            payload
        )
    
    async def get_mid_price(self, symbol: str = "BTC") -> float:
        """
        Step 1: Get mid price from orderbook for unrealized PnL calculation.
        Returns the mid price (average of best bid and best ask).
        """
        try:
            orderbook = await self.get_orderbook(symbol)
            if orderbook.bids and orderbook.asks:
                best_bid = orderbook.bids[0][0]
                best_ask = orderbook.asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                return mid_price
            else:
                # Fallback to cached orderbook if available
                if symbol in self._orderbook_cache:
                    cached = self._orderbook_cache[symbol]
                    if cached.bids and cached.asks:
                        return (cached.bids[0][0] + cached.asks[0][0]) / 2
                raise ValueError(f"No orderbook data for {symbol}")
        except Exception as e:
            logger.error(f"Failed to get mid price for {symbol}: {e}")
            raise
    
    async def get_orderbook(self, symbol: str = "BTC") -> Optional[OrderBook]:
        """Get current orderbook snapshot with caching"""
        try:
            cache_key = f"orderbook:{symbol}"
            payload = {
                "type": "l2Book",
                "coin": symbol
            }
            # Add dex parameter for XYZ perps
            dex = self.config.get("hyperliquid", {}).get("dex")
            if dex:
                payload["dex"] = dex
            
            data = await self._cached_request(
                cache_key,
                self.config["hyperliquid"]["info_url"],
                payload
            )
            
            # Check if data is valid
            if not data or "levels" not in data:
                logger.warning(f"Invalid orderbook data for {symbol}: {data}")
                return None
            
            orderbook = OrderBook(
                bids=[(float(b["px"]), float(b["sz"])) for b in data.get("levels", [[]])[0]],
                asks=[(float(a["px"]), float(a["sz"])) for a in data.get("levels", [[], []])[1]],
                timestamp=time.time()
            )
            
            self._orderbook_cache[symbol] = orderbook
            return orderbook
        except Exception as e:
            logger.debug(f"Failed to get orderbook for {symbol}: {e}")
            # Return cached orderbook if available
            if symbol in self._orderbook_cache:
                cached = self._orderbook_cache[symbol]
                age = time.time() - cached.timestamp
                if age < 60:  # Use cached data if less than 60s old
                    logger.debug(f"Using cached orderbook ({age:.0f}s old)")
                    return cached
            return None
    
    async def get_klines(
        self,
        symbol: str = "BTC",
        interval: str = "1m",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Get historical klines/candlesticks"""
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": start_time or int((time.time() - 86400) * 1000),
                "endTime": end_time or int(time.time() * 1000)
            }
        }
        # Add dex parameter for XYZ perps
        dex = self.config.get("hyperliquid", {}).get("dex")
        if dex:
            payload["dex"] = dex
        
        data = await self._post_request(
            self.config["hyperliquid"]["info_url"],
            payload
        )
        
        return data  # type: ignore[return-value]
    
    async def get_recent_trades(
        self,
        symbol: str = "BTC",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent public trades"""
        payload = {
            "type": "recentTrades",
            "coin": symbol
        }
        # Add dex parameter for XYZ perps
        dex = self.config.get("hyperliquid", {}).get("dex")
        if dex:
            payload["dex"] = dex
        
        result = await self._post_request(
            self.config["hyperliquid"]["info_url"],
            payload
        )
        return result if isinstance(result, list) else []
    
    async def get_funding_rate(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Get current funding rate"""
        payload = {
            "type": "fundingHistory",
            "coin": symbol,
            "startTime": int((time.time() - 3600) * 1000)
        }
        # Add dex parameter for XYZ perps
        dex = self.config.get("hyperliquid", {}).get("dex")
        if dex:
            payload["dex"] = dex
        
        return await self._post_request(
            self.config["hyperliquid"]["info_url"],
            payload
        )
    
    # ==================== Account Methods ====================
    
    async def get_user_state(self) -> Dict[str, Any]:
        """Get user account state including positions and balances (cached)"""
        cache_key = "user_state"
        payload = {
            "type": "clearinghouseState",
            "user": self.wallet_address
        }
        # Add dex parameter for XYZ perps clearinghouse
        dex = self.config.get("hyperliquid", {}).get("dex")
        if dex:
            payload["dex"] = dex
        
        return await self._cached_request(
            cache_key,
            self.config["hyperliquid"]["info_url"],
            payload
        )
    
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        state = await self.get_user_state()
        positions = []
        
        for pos in state.get("assetPositions", []):
            position_data = pos.get("position", {})
            if float(position_data.get("szi", 0)) != 0:
                positions.append(Position(
                    symbol=position_data.get("coin", ""),
                    size=float(position_data.get("szi", 0)),
                    entry_price=float(position_data.get("entryPx", 0)),
                    unrealized_pnl=float(position_data.get("unrealizedPnl", 0)),
                    realized_pnl=float(position_data.get("realizedPnl", 0)),
                    leverage=int(position_data.get("leverage", {}).get("value", 1)),
                    liquidation_price=float(position_data.get("liquidationPx", 0)) if position_data.get("liquidationPx") else None,
                    margin_used=float(position_data.get("marginUsed", 0))
                ))
        
        # Update cache
        for pos in positions:
            self._position_cache[pos.symbol] = pos
        
        return positions
    
    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get open orders (cached)"""
        cache_key = "open_orders"
        payload = {
            "type": "openOrders",
            "user": self.wallet_address
        }
        # Add dex parameter for XYZ perps
        dex = self.config.get("hyperliquid", {}).get("dex")
        if dex:
            payload["dex"] = dex
        
        result = await self._cached_request(
            cache_key,
            self.config["hyperliquid"]["info_url"],
            payload
        )
        return result if isinstance(result, list) else []
    
    async def get_user_fills(
        self,
        start_time: Optional[int] = None,
        limit: int = 100
    ) -> List[Trade]:
        """Get user trade fills with closed PnL tracking (cached)"""
        cache_key = f"user_fills:{limit}"
        payload = {
            "type": "userFills",
            "user": self.wallet_address
        }
        # Add dex parameter for XYZ perps
        dex = self.config.get("hyperliquid", {}).get("dex")
        if dex:
            payload["dex"] = dex
        
        data = await self._cached_request(
            cache_key,
            self.config["hyperliquid"]["info_url"],
            payload
        )
        
        trades = []
        fills_list = data if isinstance(data, list) else []
        for fill in fills_list[:limit]:  # type: ignore[index]
            # Step 1: Parse closedPnl for realized PnL tracking
            closed_pnl = float(fill.get("closedPnl", 0))
            start_position = float(fill.get("startPosition", 0))
            direction = fill.get("dir", "")
            
            trade = Trade(
                order_id=str(fill.get("oid", "")),
                symbol=fill.get("coin", ""),
                side=fill.get("side", ""),
                size=float(fill.get("sz", 0)),
                price=float(fill.get("px", 0)),
                timestamp=fill.get("time", 0) / 1000,
                fee=float(fill.get("fee", 0)),
                is_maker=fill.get("liquidation", False),
                closed_pnl=closed_pnl,
                start_position=start_position,
                dir=direction
            )
            trades.append(trade)
            
            # Step 1: Update cumulative realized PnL
            if closed_pnl != 0:
                self._total_realized_pnl += closed_pnl
        
        return trades
    
    # ==================== Order Methods ====================
    
    async def verify_order_filled(
        self,
        order_id: str,
        timeout: float = 10.0,
        poll_interval: float = 0.5
    ) -> Dict[str, Any]:
        """
        Step 2: Verify if an order was filled on mainnet.
        Polls order status until filled, canceled, or timeout.
        
        Returns:
            Dict with keys: filled (bool), status (str), fill_price (float), fill_size (float)
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                open_orders = await self.get_open_orders()
                
                # Check if order is still open
                order_found = False
                for order in open_orders:
                    if str(order.get("oid")) == order_id:
                        order_found = True
                        break
                
                if not order_found:
                    # Order not in open orders - check fills
                    fills = await self.get_user_fills(limit=10)
                    for fill in fills:
                        if fill.order_id == order_id:
                            logger.info(
                                f"✅ Order {order_id} FILLED: {fill.side} {fill.size} @ ${fill.price:.2f}, "
                                f"PnL: ${fill.closed_pnl:+.2f}"
                            )
                            return {
                                "filled": True,
                                "status": "filled",
                                "fill_price": fill.price,
                                "fill_size": fill.size,
                                "closed_pnl": fill.closed_pnl,
                                "fee": fill.fee
                            }
                    
                    # Order not found in fills or open - may be canceled
                    logger.warning(f"⚠️ Order {order_id} not found in open or fills - may be canceled")
                    return {"filled": False, "status": "unknown"}
                
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error verifying order {order_id}: {e}")
                await asyncio.sleep(poll_interval)
        
        logger.warning(f"⏰ Order {order_id} verification timeout after {timeout}s")
        return {"filled": False, "status": "timeout"}
    
    def is_mainnet(self) -> bool:
        """Step 2: Check if connected to mainnet (not testnet)"""
        exchange_url = self.config.get("hyperliquid", {}).get("exchange_url", "")
        # Mainnet uses api.hyperliquid.xyz, testnet uses api.hyperliquid-testnet.xyz
        is_main = "testnet" not in exchange_url.lower()
        return is_main
    
    def warn_if_testnet(self) -> None:
        """Step 2: Log warning if not on mainnet"""
        if not self.is_mainnet():
            logger.warning(
                "⚠️ TESTNET DETECTED: Trading on testnet, not mainnet. "
                "Trades will not affect real wallet."
            )
        else:
            logger.info("✅ Connected to Hyperliquid MAINNET")
    
    async def place_order(
        self,
        symbol: str,
        side: str,  # "B" for buy, "A" for sell
        size: float,
        price: Optional[float] = None,
        order_type: str = "limit",  # "limit" or "market"
        reduce_only: bool = False,
        time_in_force: str = "Gtc",  # "Gtc", "Ioc", "Alo"
        client_order_id: Optional[str] = None,
        verify_fill: bool = False  # Step 2: Optionally verify fill
    ) -> Dict[str, Any]:
        """
        Place an order on Hyperliquid.
        
        Step 2 Enhanced: Added verify_fill option to confirm order execution.
        """
        await self._order_limiter.acquire()
        
        # Get asset index and decimals
        asset_index = self.config.get("asset_index", 0)
        price_decimals = self.config.get("decimals", {}).get("price", 0)
        size_decimals = self.config.get("decimals", {}).get("size", 5)
        
        # Round price and size to proper decimals
        if price is not None:
            price = round(price, price_decimals)
        size = round(size, size_decimals)
        
        # Build order
        order = {
            "a": asset_index,
            "b": side == "B",
            "p": str(int(price)) if price_decimals == 0 and price else str(price) if price else "0",
            "s": str(size),
            "r": reduce_only,
            "t": {
                "limit": {"tif": time_in_force}
            } if order_type == "limit" else {"trigger": {"tpsl": "tp", "triggerPx": str(int(price) if price_decimals == 0 else price)}}
        }
        
        if client_order_id:
            order["c"] = client_order_id
        
        action = {
            "type": "order",
            "orders": [order],
            "grouping": "na"
        }
        
        # NOTE: Order actions do NOT include 'dex' field - the asset index (110000+) 
        # already identifies the builder-deployed DEX. Only info queries need 'dex'.
        
        nonce = int(time.time() * 1000)
        signed_payload = self._sign_action(action, nonce)
        
        result = await self._post_request(
            self.config["hyperliquid"]["exchange_url"],
            signed_payload,
            signed=True
        )
        
        # Step 2: Parse order result for ID
        order_id = None
        if isinstance(result, dict):
            statuses = result.get("response", {}).get("data", {}).get("statuses", [])
            if statuses and isinstance(statuses[0], dict):
                if "resting" in statuses[0]:
                    order_id = str(statuses[0]["resting"]["oid"])
                elif "filled" in statuses[0]:
                    order_id = str(statuses[0]["filled"]["oid"])
                    logger.info(f"✅ Order immediately FILLED: {side} {size} {symbol} @ {price}")
        
        # Step 2: Log order placement with mainnet confirmation
        network = "MAINNET" if self.is_mainnet() else "TESTNET"
        logger.info(
            f"📤 Order placed [{network}]: {side} {size} {symbol} @ ${price:.2f}, "
            f"order_id={order_id}, type={order_type}"
        )
        
        # Step 2: Optionally verify fill
        if verify_fill and order_id:
            fill_result = await self.verify_order_filled(order_id, timeout=5.0)
            result["fill_verification"] = fill_result
            if fill_result.get("filled"):
                logger.info(f"✅ Fill confirmed: order {order_id}")
            else:
                logger.warning(f"⚠️ Fill not confirmed for order {order_id}: {fill_result.get('status')}")
        
        result["order_id"] = order_id
        return result
    
    async def place_market_order(
        self,
        symbol: str,
        side: str,
        size: float,
        slippage_pct: float = 0.1
    ) -> Dict[str, Any]:
        """Place a market order with slippage protection"""
        # Get current orderbook for price reference
        orderbook = await self.get_orderbook(symbol)
        
        # Get price decimals for rounding
        price_decimals = self.config.get("decimals", {}).get("price", 0)
        
        if side == "B":
            # Buy - use ask price with slippage
            price = orderbook.asks[0][0] * (1 + slippage_pct / 100)
        else:
            # Sell - use bid price with slippage
            price = orderbook.bids[0][0] * (1 - slippage_pct / 100)
        
        # Round to proper decimals (BTC uses whole dollars)
        price = round(price, price_decimals)
        if price_decimals == 0:
            price = int(price)
        
        return await self.place_order(
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            order_type="limit",
            time_in_force="Ioc"  # Immediate or cancel
        )
    
    async def place_trigger_order(
        self,
        symbol: str,
        side: str,  # "B" for buy, "A" for sell
        size: float,
        trigger_price: float,
        limit_price: Optional[float] = None,
        tpsl: str = "sl",  # "tp" for take profit, "sl" for stop loss
        is_market: bool = True,
        reduce_only: bool = True
    ) -> Dict[str, Any]:
        """
        Place a trigger order (stop loss or take profit) on Hyperliquid.
        
        These orders are placed on the exchange and will execute automatically
        when the trigger price is reached, even if the bot is offline.
        
        Args:
            symbol: Trading symbol (e.g., "BTC")
            side: "B" for buy, "A" for sell (opposite of position for closing)
            size: Order size
            trigger_price: Price at which the order triggers
            limit_price: Limit price for limit trigger orders (None for market)
            tpsl: "tp" for take profit, "sl" for stop loss
            is_market: If True, execute as market order when triggered
            reduce_only: If True, only reduce position (default True for TP/SL)
        
        Returns:
            Order result dict with order_id
        """
        await self._order_limiter.acquire()
        
        asset_index = self.config.get("asset_index", 0)
        price_decimals = self.config.get("decimals", {}).get("price", 0)
        size_decimals = self.config.get("decimals", {}).get("size", 5)
        
        # Round trigger price and size
        trigger_price = round(trigger_price, price_decimals)
        if price_decimals == 0:
            trigger_price = int(trigger_price)
        size = round(size, size_decimals)
        
        # For market trigger orders, limit price equals trigger price
        if limit_price is None:
            limit_price = trigger_price
        else:
            limit_price = round(limit_price, price_decimals)
            if price_decimals == 0:
                limit_price = int(limit_price)
        
        # Build trigger order
        order = {
            "a": asset_index,
            "b": side == "B",
            "p": str(limit_price),
            "s": str(size),
            "r": reduce_only,
            "t": {
                "trigger": {
                    "isMarket": is_market,
                    "triggerPx": str(trigger_price),
                    "tpsl": tpsl
                }
            }
        }
        
        action = {
            "type": "order",
            "orders": [order],
            "grouping": "na"
        }
        
        # NOTE: Order actions do NOT include 'dex' field - asset index identifies DEX
        
        nonce = int(time.time() * 1000)
        signed_payload = self._sign_action(action, nonce)
        
        result = await self._post_request(
            self.config["hyperliquid"]["exchange_url"],
            signed_payload,
            signed=True
        )
        
        # Parse order ID
        order_id = None
        if isinstance(result, dict):
            statuses = result.get("response", {}).get("data", {}).get("statuses", [])
            if statuses and isinstance(statuses[0], dict):
                if "resting" in statuses[0]:
                    order_id = str(statuses[0]["resting"]["oid"])
                elif "error" in statuses[0]:
                    logger.error(f"Trigger order error: {statuses[0]['error']}")
        
        network = "MAINNET" if self.is_mainnet() else "TESTNET"
        tpsl_label = "TAKE PROFIT" if tpsl == "tp" else "STOP LOSS"
        logger.info(
            f"📤 {tpsl_label} order placed [{network}]: {side} {size} {symbol} @ trigger ${trigger_price}, "
            f"order_id={order_id}"
        )
        
        result["order_id"] = order_id
        return result
    
    async def place_stop_loss(
        self,
        symbol: str,
        position_side: str,  # "B" for long position, "A" for short position
        size: float,
        stop_price: float
    ) -> Dict[str, Any]:
        """
        Place a stop loss order for an open position.
        
        Args:
            symbol: Trading symbol
            position_side: Current position side ("B" for long, "A" for short)
            size: Position size to protect
            stop_price: Stop loss trigger price
        
        Returns:
            Order result
        """
        # SL order side is opposite of position
        order_side = "A" if position_side == "B" else "B"
        return await self.place_trigger_order(
            symbol=symbol,
            side=order_side,
            size=size,
            trigger_price=stop_price,
            tpsl="sl",
            is_market=True,
            reduce_only=True
        )
    
    async def place_take_profit(
        self,
        symbol: str,
        position_side: str,  # "B" for long position, "A" for short position
        size: float,
        take_profit_price: float
    ) -> Dict[str, Any]:
        """
        Place a take profit order for an open position.
        
        Args:
            symbol: Trading symbol
            position_side: Current position side ("B" for long, "A" for short)
            size: Position size to take profit on
            take_profit_price: Take profit trigger price
        
        Returns:
            Order result
        """
        # TP order side is opposite of position
        order_side = "A" if position_side == "B" else "B"
        return await self.place_trigger_order(
            symbol=symbol,
            side=order_side,
            size=size,
            trigger_price=take_profit_price,
            tpsl="tp",
            is_market=True,
            reduce_only=True
        )
    
    async def cancel_order(
        self,
        symbol: str,
        order_id: int
    ) -> Dict[str, Any]:
        """Cancel an order"""
        asset_index = self.config.get("asset_index", 0)
        
        action = {
            "type": "cancel",
            "cancels": [{"a": asset_index, "o": order_id}]
        }
        
        # NOTE: Cancel actions do NOT include 'dex' field - asset index identifies DEX
        
        nonce = int(time.time() * 1000)
        signed_payload = self._sign_action(action, nonce)
        
        return await self._post_request(
            self.config["hyperliquid"]["exchange_url"],
            signed_payload,
            signed=True
        )
    
    async def cancel_all_orders(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Cancel all open orders for a symbol"""
        open_orders = await self.get_open_orders()
        
        cancels = []
        for order in open_orders:
            if order.get("coin") == symbol:
                cancels.append({
                    "a": self.config.get("asset_index", 0),
                    "o": order.get("oid")
                })
        
        if not cancels:
            return {"status": "no_orders"}
        
        action = {
            "type": "cancel",
            "cancels": cancels
        }
        
        # NOTE: Cancel actions do NOT include 'dex' field - asset index identifies DEX
        
        nonce = int(time.time() * 1000)
        signed_payload = self._sign_action(action, nonce)
        
        return await self._post_request(
            self.config["hyperliquid"]["exchange_url"],
            signed_payload,
            signed=True
        )
    
    async def close_position(self, symbol: str = "BTC") -> Optional[Dict[str, Any]]:
        """Close current position for a symbol"""
        positions = await self.get_positions()
        
        for pos in positions:
            if pos.symbol == symbol and pos.size != 0:
                side = "A" if pos.size > 0 else "B"
                return await self.place_market_order(
                    symbol=symbol,
                    side=side,
                    size=abs(pos.size)
                )
        
        return None
    
    # ==================== WebSocket Methods ====================
    
    async def connect_websocket(self) -> None:
        """Establish WebSocket connection"""
        ws_url = self.config["hyperliquid"]["ws_url"]
        
        self._ws = await websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024  # 10MB
        )
        
        self._ws_connected = True
        logger.info("WebSocket connected")
        
        # Start message handler
        asyncio.create_task(self._ws_message_handler())
    
    async def _ws_message_handler(self) -> None:
        """Handle incoming WebSocket messages"""
        while self._ws_connected and self._ws is not None:
            try:
                message = await self._ws.recv()
                data = orjson.loads(message)
                
                channel = data.get("channel")
                if channel and channel in self._ws_subscriptions:
                    await self._ws_subscriptions[channel](data)
                
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                await self._reconnect_websocket()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
    
    async def _reconnect_websocket(self) -> None:
        """Reconnect WebSocket with exponential backoff"""
        retry_delay = 1
        max_delay = 60
        
        while True:
            try:
                await self.connect_websocket()
                
                # Resubscribe to all channels - subscriptions dict has channel:callback
                # Recreate subscription from channel name
                for channel in self._ws_subscriptions:
                    parts = channel.split(":")
                    if len(parts) == 2:
                        sub_type, symbol = parts
                        await self._subscribe({"type": sub_type, "coin": symbol})
                
                break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)
    
    async def _subscribe(self, subscription: Dict[str, Any]) -> None:
        """Send subscription message"""
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(orjson.dumps({
            "method": "subscribe",
            "subscription": subscription
        }))
    
    async def subscribe_orderbook(
        self,
        symbol: str,
        callback: Callable
    ) -> None:
        """Subscribe to orderbook updates"""
        subscription = {
            "type": "l2Book",
            "coin": symbol
        }
        
        channel = f"l2Book:{symbol}"
        self._ws_subscriptions[channel] = callback
        
        await self._subscribe(subscription)
        logger.info(f"Subscribed to {symbol} orderbook")
    
    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable
    ) -> None:
        """Subscribe to trade updates"""
        subscription = {
            "type": "trades",
            "coin": symbol
        }
        
        channel = f"trades:{symbol}"
        self._ws_subscriptions[channel] = callback
        
        await self._subscribe(subscription)
        logger.info(f"Subscribed to {symbol} trades")
    
    async def subscribe_user_events(
        self,
        callback: Callable
    ) -> None:
        """Subscribe to user account events (fills, orders)"""
        subscription = {
            "type": "userEvents",
            "user": self.wallet_address
        }
        
        channel = f"userEvents:{self.wallet_address}"
        self._ws_subscriptions[channel] = callback
        self._ws_user_connected = True
        
        await self._subscribe(subscription)
        logger.info("Subscribed to user events")
    
    def register_fill_callback(self, callback: Callable) -> None:
        """Step 1: Register a callback for fill events"""
        self._fill_callbacks.append(callback)
        logger.debug(f"Registered fill callback: {callback.__name__}")
    
    def register_position_callback(self, callback: Callable) -> None:
        """Step 1: Register a callback for position updates"""
        self._position_callbacks.append(callback)
        logger.debug(f"Registered position callback: {callback.__name__}")
    
    async def _process_user_event(self, event_data: Dict[str, Any]) -> None:
        """Step 1: Process incoming user events from WebSocket"""
        try:
            event_type = event_data.get("type", "")
            data = event_data.get("data", {})
            
            if event_type == "fill" or "fills" in data:
                # Process fill events
                fills = data.get("fills", [event_data]) if "fills" in data else [data]
                for fill in fills:
                    await self._handle_fill_event(fill)
            
            elif event_type == "orderUpdate" or "order" in data:
                # Process order updates
                await self._handle_order_update(data.get("order", data))
            
            elif event_type == "positionUpdate" or "positions" in data:
                # Process position updates
                positions = data.get("positions", [data])
                for pos in positions:
                    await self._handle_position_update(pos)
                    
        except Exception as e:
            logger.error(f"Error processing user event: {e}")
    
    async def _handle_fill_event(self, fill: Dict[str, Any]) -> None:
        """Step 1: Handle a fill event - confirm execution and update PnL"""
        try:
            closed_pnl = float(fill.get("closedPnl", 0))
            size = float(fill.get("sz", 0))
            price = float(fill.get("px", 0))
            side = fill.get("side", "")
            symbol = fill.get("coin", "BTC")
            fee = float(fill.get("fee", 0))
            
            # Log fill confirmation
            side_str = "BUY" if side == "B" else "SELL"
            logger.info(f"✅ Fill confirmed: {side_str} {size} {symbol} @ ${price:,.2f} (fee: ${fee:.4f})")
            
            if closed_pnl != 0:
                self._total_realized_pnl += closed_pnl
                logger.info(f"💰 Realized PnL: ${closed_pnl:+.2f} (cumulative: ${self._total_realized_pnl:+.2f})")
            
            # Trigger fill callbacks
            for callback in self._fill_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(fill)
                    else:
                        callback(fill)
                except Exception as cb_err:
                    logger.error(f"Fill callback error: {cb_err}")
                    
        except Exception as e:
            logger.error(f"Error handling fill event: {e}")
    
    async def _handle_order_update(self, order: Dict[str, Any]) -> None:
        """Step 1: Handle order status updates"""
        try:
            oid = order.get("oid", "")
            status = order.get("status", "")
            
            if status in ("filled", "partiallyFilled"):
                logger.debug(f"Order {oid} status: {status}")
            elif status == "canceled":
                logger.debug(f"Order {oid} canceled")
            elif status == "rejected":
                logger.warning(f"Order {oid} rejected: {order.get('msg', 'unknown')}")
                
        except Exception as e:
            logger.error(f"Error handling order update: {e}")
    
    async def _handle_position_update(self, pos_data: Dict[str, Any]) -> None:
        """Step 1: Handle position updates"""
        try:
            symbol = pos_data.get("coin", "BTC")
            size = float(pos_data.get("szi", 0))
            entry_px = float(pos_data.get("entryPx", 0))
            unrealized = float(pos_data.get("unrealizedPnl", 0))
            
            # Update position cache
            self._open_positions[symbol] = {
                "size": size,
                "entry_price": entry_px,
                "unrealized_pnl": unrealized,
                "timestamp": time.time()
            }
            
            # Trigger position callbacks
            for callback in self._position_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(pos_data)
                    else:
                        callback(pos_data)
                except Exception as cb_err:
                    logger.error(f"Position callback error: {cb_err}")
            
            if size != 0:
                logger.debug(f"Position update: {symbol} {size} @ {entry_px}, uPnL: ${unrealized:.2f}")
                    
        except Exception as e:
            logger.error(f"Error handling position update: {e}")
    
    def get_total_realized_pnl(self) -> float:
        """Step 1: Get cumulative realized PnL"""
        return self._total_realized_pnl
    
    async def confirm_order_filled(self, order_id: str, timeout: float = 10.0) -> Optional[Trade]:
        """Step 1: Wait for order fill confirmation"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            fills = await self.get_user_fills(limit=20)
            for fill in fills:
                if str(fill.order_id) == str(order_id):
                    return fill
            await asyncio.sleep(0.5)
        
        logger.warning(f"Order {order_id} fill confirmation timeout after {timeout}s")
        return None
    
    # ==================== Utility Methods ====================
    
    def get_cached_orderbook(self, symbol: str = "BTC") -> Optional[OrderBook]:
        """Get cached orderbook for low-latency access"""
        return self._orderbook_cache.get(symbol)
    
    def get_cached_position(self, symbol: str = "BTC") -> Optional[Position]:
        """Get cached position for low-latency access"""
        return self._position_cache.get(symbol)
    
    def get_avg_latency(self) -> float:
        """Get average API latency in milliseconds"""
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)
    
    def get_latency_p99(self) -> float:
        """Get 99th percentile latency"""
        if not self._latency_samples:
            return 0.0
        sorted_samples = sorted(self._latency_samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[idx]


# Async context manager for clean resource management
class HyperliquidClient:
    """Context manager for Hyperliquid API"""
    
    def __init__(self, config_path: str = "config/api.json"):
        self.api = HyperliquidAPI(config_path)
    
    async def __aenter__(self) -> HyperliquidAPI:
        await self.api.initialize()
        return self.api
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.api.close()
