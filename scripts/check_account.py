#!/usr/bin/env python3
"""Quick account check script"""
import asyncio
from src.hyperliquid_api import HyperliquidAPI

async def main():
    api = HyperliquidAPI('config/api.json')
    await api.initialize()
    
    state = await api.get_user_state()
    print('=== ACCOUNT STATE ===')
    margin = state.get('marginSummary', {})
    print(f"Account Value: ${float(margin.get('accountValue', 0)):,.2f}")
    print(f"Total PnL: ${float(margin.get('totalRawUsd', 0)):,.2f}")
    
    positions = state.get('assetPositions', [])
    print('\n=== POSITIONS ===')
    has_pos = False
    for pos in positions:
        p = pos.get('position', {})
        size = float(p.get('szi', 0))
        if size != 0:
            has_pos = True
            print(f"{p.get('coin')}: {size} @ {p.get('entryPx')} (uPnL: {p.get('unrealizedPnl')})")
    if not has_pos:
        print('No open positions')
    
    fills = await api.get_user_fills(limit=50)
    print(f'\n=== RECENT FILLS (Last 50) ===')
    if fills:
        buys = sum(1 for f in fills if f.side == 'B')
        sells = len(fills) - buys
        print(f'Buys: {buys}, Sells: {sells}')
        total_fees = sum(f.fee for f in fills)
        print(f'Total fees: ${total_fees:.4f}')
        for f in fills[:15]:
            side = 'BUY' if f.side == 'B' else 'SELL'
            print(f'  {f.symbol}: {side} {f.size} @ ${f.price:,.2f} (fee: ${f.fee:.4f})')
        if len(fills) > 15:
            print(f'  ... and {len(fills)-15} more')
    else:
        print('No recent fills')
    
    orders = await api.get_open_orders()
    print(f'\n=== OPEN ORDERS: {len(orders)} ===')
    
    await api.close()

if __name__ == "__main__":
    asyncio.run(main())
