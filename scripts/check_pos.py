#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from hyperliquid.info import Info
from dotenv import load_dotenv
import os

load_dotenv()
address = '0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584'
info = Info(skip_ws=True)
user_state = info.user_state(address)

print('=== POSITIONS ===')
positions = user_state.get('assetPositions', [])
has_pos = False
for asset_pos in positions:
    pos = asset_pos.get('position', {})
    size = float(pos.get('szi', 0))
    if size != 0:
        has_pos = True
        coin = pos.get('coin', 'Unknown')
        entry = pos.get('entryPx', 0)
        upnl = pos.get('unrealizedPnl', 0)
        margin = pos.get('marginUsed', 0)
        liq_px = pos.get('liquidationPx', 0)
        print(f'{coin}: {size} @ ${entry} | uPnL: ${upnl} | Margin: ${margin} | Liq: ${liq_px}')

if not has_pos:
    print('No open positions')

margin = user_state.get('marginSummary', {})
print(f"\nAccount Value: ${float(margin.get('accountValue', 0)):,.2f}")
print(f"Total Margin Used: ${float(margin.get('totalMarginUsed', 0)):,.2f}")
print(f"Withdrawable: ${float(user_state.get('withdrawable', 0)):,.2f}")
