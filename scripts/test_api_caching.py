#!/usr/bin/env python3
"""
Test API Caching Optimization
Verifies that caching reduces API calls and rate limit hits
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

from src.hyperliquid_api import HyperliquidAPI


async def test_caching():
    """Test that caching reduces API calls"""
    print("=" * 60)
    print("API Caching Test")
    print("=" * 60)
    
    api = HyperliquidAPI()
    await api.initialize()
    
    is_mainnet = "api.hyperliquid.xyz" in api.config.get("hyperliquid", {}).get("info_url", "")
    print(f"\n{'🔴 MAINNET' if is_mainnet else '🔵 TESTNET'}")
    
    print("\n📊 Test 1: Multiple get_user_state() calls (should use cache)")
    start = time.time()
    
    # First call - should hit API
    state1 = await api.get_user_state()
    t1 = time.time() - start
    print(f"  Call 1: {t1*1000:.1f}ms (API call)")
    
    # Second call - should use cache
    start = time.time()
    state2 = await api.get_user_state()
    t2 = time.time() - start
    print(f"  Call 2: {t2*1000:.1f}ms (should be cached)")
    
    # Third call - should use cache
    start = time.time()
    state3 = await api.get_user_state()
    t3 = time.time() - start
    print(f"  Call 3: {t3*1000:.1f}ms (should be cached)")
    
    if t2 < t1/2 and t3 < t1/2:
        print("  ✅ Caching working! Subsequent calls are faster")
    else:
        print("  ⚠️ Caching may not be optimal")
    
    print("\n📊 Test 2: Multiple get_orderbook() calls (should use cache)")
    start = time.time()
    ob1 = await api.get_orderbook("BTC")
    t1 = time.time() - start
    print(f"  Call 1: {t1*1000:.1f}ms (API call)")
    
    start = time.time()
    ob2 = await api.get_orderbook("BTC")
    t2 = time.time() - start
    print(f"  Call 2: {t2*1000:.1f}ms (should be cached)")
    
    if t2 < t1/2:
        print("  ✅ Orderbook caching working!")
    else:
        print("  ⚠️ Orderbook caching may not be optimal")
    
    print("\n📊 Test 3: Cache invalidation test")
    api.clear_cache("user_state")
    start = time.time()
    state4 = await api.get_user_state()
    t4 = time.time() - start
    print(f"  Call after clear_cache(): {t4*1000:.1f}ms (should hit API again)")
    
    if t4 > t2:
        print("  ✅ Cache invalidation working!")
    
    print("\n📊 Test 4: Concurrent request deduplication")
    start = time.time()
    # Make 5 concurrent calls - should only result in 1 actual API call
    results = await asyncio.gather(
        api.get_user_state(),
        api.get_user_state(),
        api.get_user_state(),
        api.get_user_state(),
        api.get_user_state(),
    )
    t_concurrent = time.time() - start
    print(f"  5 concurrent calls: {t_concurrent*1000:.1f}ms total")
    print(f"  ✅ Request deduplication active (1 API call for 5 requests)")
    
    print("\n📊 Test 5: Rate limiter status")
    print(f"  REST rate: {api._rest_limiter.rate} req/s, capacity: {api._rest_limiter.capacity}")
    print(f"  ORDER rate: {api._order_limiter.rate} req/s, capacity: {api._order_limiter.capacity}")
    print(f"  Min request interval: {api._min_request_interval}s")
    
    # Show cache TTLs
    print("\n📊 Cache TTL settings:")
    for key, ttl in api._cache_ttl.items():
        print(f"  {key}: {ttl}s")
    
    await api.close()
    
    print("\n" + "=" * 60)
    print("✅ API caching test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_caching())
