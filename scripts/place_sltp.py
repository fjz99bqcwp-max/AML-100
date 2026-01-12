#!/usr/bin/env python3
"""
Place SL/TP orders for existing open positions that don't have them.
This script checks for open positions and places protective orders on Hyperliquid.
"""

import asyncio
import json
import sys
sys.path.insert(0, '.')

from src.hyperliquid_api import HyperliquidAPI


async def place_sltp_for_positions():
    """Place SL/TP orders for all open positions without protection"""
    
    # Load params for SL/TP percentages
    with open('config/params.json') as f:
        params = json.load(f)
    
    sl_pct = params['trading']['stop_loss_pct']
    tp_pct = params['trading']['take_profit_pct']
    
    print(f"SL%: {sl_pct}%, TP%: {tp_pct}%")
    
    api = HyperliquidAPI()
    await api.initialize()
    
    try:
        # Get current positions
        positions = await api.get_positions()
        
        # Get current open orders
        open_orders = await api.get_open_orders()
        print(f"\nCurrent open orders: {len(open_orders)}")
        for order in open_orders:
            print(f"  {order}")
        
        for pos in positions:
            if pos.size == 0:
                continue
            
            print(f"\n=== Position: {pos.symbol} ===")
            print(f"  Size: {pos.size}")
            print(f"  Entry: ${pos.entry_price}")
            print(f"  Unrealized PnL: ${pos.unrealized_pnl:.2f}")
            
            # Determine position side and SL/TP prices
            if pos.size > 0:  # Long position
                position_side = "B"
                sl_price = pos.entry_price * (1 - sl_pct / 100)
                tp_price = pos.entry_price * (1 + tp_pct / 100)
            else:  # Short position
                position_side = "A"
                sl_price = pos.entry_price * (1 + sl_pct / 100)
                tp_price = pos.entry_price * (1 - tp_pct / 100)
            
            # Round prices to whole dollars for BTC
            sl_price = int(sl_price)
            tp_price = int(tp_price)
            
            print(f"  Position Side: {'LONG' if position_side == 'B' else 'SHORT'}")
            print(f"  Stop Loss: ${sl_price} ({'-' if position_side == 'B' else '+'}{sl_pct}%)")
            print(f"  Take Profit: ${tp_price} ({'+' if position_side == 'B' else '-'}{tp_pct}%)")
            
            # Cancel existing orders first
            print("\n  Cancelling existing orders...")
            cancel_result = await api.cancel_all_orders(pos.symbol)
            print(f"  Cancel result: {cancel_result}")
            
            # Place SL order
            print(f"\n  Placing Stop Loss at ${sl_price}...")
            sl_result = await api.place_stop_loss(
                symbol=pos.symbol,
                position_side=position_side,
                size=abs(pos.size),
                stop_price=sl_price
            )
            print(f"  SL Result: order_id={sl_result.get('order_id')}")
            if sl_result.get("status") == "err":
                print(f"  SL Error: {sl_result.get('response')}")
            
            # Place TP order
            print(f"\n  Placing Take Profit at ${tp_price}...")
            tp_result = await api.place_take_profit(
                symbol=pos.symbol,
                position_side=position_side,
                size=abs(pos.size),
                take_profit_price=tp_price
            )
            print(f"  TP Result: order_id={tp_result.get('order_id')}")
            if tp_result.get("status") == "err":
                print(f"  TP Error: {tp_result.get('response')}")
        
        # Verify orders were placed
        print("\n=== Verifying Orders ===")
        await asyncio.sleep(1)  # Wait for orders to propagate
        open_orders = await api.get_open_orders()
        print(f"Open orders after placement: {len(open_orders)}")
        for order in open_orders:
            print(f"  {order}")
        
    finally:
        await api.close()


if __name__ == '__main__':
    asyncio.run(place_sltp_for_positions())
