#!/usr/bin/env python3
"""Check current position and orders on Hyperliquid"""

import asyncio
import json
import sys
sys.path.insert(0, '.')

async def check_current_state():
    from src.hyperliquid_api import HyperliquidAPI
    api = HyperliquidAPI()
    await api.initialize()
    
    # Get positions
    positions = await api.get_positions()
    print('=== Current Positions ===')
    for p in positions:
        print(f'  {p.symbol}: size={p.size}, entry={p.entry_price}, unrealized={p.unrealized_pnl}')
    
    if not positions or all(p.size == 0 for p in positions):
        print('  No open positions')
    
    # Get open orders
    open_orders = await api.get_open_orders()
    print(f'\n=== Open Orders ({len(open_orders)}) ===')
    if not open_orders:
        print('  ⚠️ No open orders (no SL/TP orders on exchange!)')
    else:
        sl_orders = []
        tp_orders = []
        for order in open_orders:
            if order.get('reduceOnly'):
                limit_px = float(order.get('limitPx', 0))
                # For a short position, SL is higher than entry, TP is lower
                # For a long position, SL is lower than entry, TP is higher
                for p in positions:
                    if p.symbol == order.get('coin') and p.size != 0:
                        if p.size < 0:  # Short
                            if limit_px > p.entry_price:
                                sl_orders.append(order)
                            else:
                                tp_orders.append(order)
                        else:  # Long
                            if limit_px < p.entry_price:
                                sl_orders.append(order)
                            else:
                                tp_orders.append(order)
        
        print(f'  Stop Loss orders: {len(sl_orders)}')
        for o in sl_orders:
            print(f'    {o.get("side")} {o.get("sz")} @ ${o.get("limitPx")} (oid: {o.get("oid")})')
        print(f'  Take Profit orders: {len(tp_orders)}')
        for o in tp_orders:
            print(f'    {o.get("side")} {o.get("sz")} @ ${o.get("limitPx")} (oid: {o.get("oid")})')
    
    # Get user state for more details
    state = await api.get_user_state()
    
    print('\n=== Account State ===')
    margin_summary = state.get('marginSummary', {})
    print(f'  Account Value: ${float(margin_summary.get("accountValue", 0)):.2f}')
    print(f'  Total Margin Used: ${float(margin_summary.get("totalMarginUsed", 0)):.2f}')
    print(f'  Total Notional Position: ${float(margin_summary.get("totalNtlPos", 0)):.2f}')
    
    if 'assetPositions' in state:
        for ap in state['assetPositions']:
            pos = ap.get('position', {})
            if float(pos.get('szi', 0)) != 0:
                print(f'\n=== Position Details ===')
                print(f'  Size: {pos.get("szi")}')
                print(f'  Entry: {pos.get("entryPx")}')
                print(f'  Leverage: {pos.get("leverage")}')
                print(f'  Liquidation: {pos.get("liquidationPx")}')
                print(f'  Unrealized PnL: ${float(pos.get("unrealizedPnl", 0)):.2f}')
    
    # Recent fills
    print('\n=== Recent Fills (last 5) ===')
    fills = await api.get_user_fills(limit=5)
    total_fees = 0
    for f in fills:
        total_fees += f.fee
        print(f'  {f.side} {f.size} @ ${f.price:.0f}, PnL: ${f.closed_pnl:.2f}, Fee: ${f.fee:.4f}')
    
    await api.close()

if __name__ == '__main__':
    asyncio.run(check_current_state())
