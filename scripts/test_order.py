#!/usr/bin/env python3
"""Test placing a real order on Hyperliquid mainnet"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    print(f"✅ Loaded environment from {env_file}")

from src.hyperliquid_api import HyperliquidAPI

async def test_order():
    api = HyperliquidAPI()
    
    try:
        # Initialize
        await api.initialize()
        print(f"✅ API initialized for wallet: {api.wallet_address[:15]}...")
        
        # Check if mainnet
        if api.is_mainnet():
            print("✅ Connected to MAINNET")
        else:
            print("⚠️ Connected to TESTNET - orders won't affect real wallet!")
            return
        
        # Get current price
        orderbook = await api.get_orderbook("BTC")
        mid_price = (orderbook.bids[0][0] + orderbook.asks[0][0]) / 2
        print(f"📊 Current BTC price: ${mid_price:.2f}")
        
        # Get account state
        state = await api.get_user_state()
        margin = state.get('marginSummary', {})
        account_value = float(margin.get('accountValue', 0))
        print(f"💰 Account value: ${account_value:.2f}")
        
        # Place a small test order (0.001 BTC = ~$90)
        # Use IOC so it doesn't sit in orderbook if not filled immediately
        test_size = 0.001
        # Price slightly below market for buy - round to whole dollars (BTC tick size)
        test_price = int(mid_price * 0.995)  # 0.5% below mid, whole dollars
        
        print(f"\n🔄 Placing TEST order: BUY {test_size} BTC @ ${test_price}")
        print("   (This is a limit order below market - may not fill)")
        
        result = await api.place_order(
            symbol="BTC",
            side="B",
            size=test_size,
            price=test_price,
            order_type="limit",
            time_in_force="Gtc",  # Good til canceled
            verify_fill=False
        )
        
        print(f"\n📤 Order result:")
        print(f"   Order ID: {result.get('order_id', 'N/A')}")
        print(f"   Response: {result.get('response', result)}")
        
        # Wait and check if it's in open orders
        await asyncio.sleep(2)
        open_orders = await api.get_open_orders()
        print(f"\n📋 Open orders: {len(open_orders)}")
        for order in open_orders:
            print(f"   {order.get('coin')} {order.get('side')} {order.get('sz')} @ ${order.get('limitPx')}")
        
        # Cancel the test order if it's still open
        if open_orders:
            print("\n🗑️ Canceling test order...")
            cancel_result = await api.cancel_all_orders("BTC")
            print(f"   Cancel result: {cancel_result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await api.close()

if __name__ == '__main__':
    asyncio.run(test_order())
