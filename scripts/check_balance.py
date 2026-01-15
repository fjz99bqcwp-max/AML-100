#!/usr/bin/env python3
"""Check wallet balances for main perps and XYZ perps."""

from hyperliquid.info import Info
from hyperliquid.utils import constants

wallet = '0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584'
info = Info(constants.MAINNET_API_URL, skip_ws=True)

print('=== Main Perps ===')
main_state = info.user_state(wallet)
margin = main_state.get('marginSummary', {}) if main_state else {}
print(f'Account Value: ${float(margin.get("accountValue", 0)):,.2f}')

print()
print('=== Spot/XYZ Perps ===')
spot = info.spot_user_state(wallet)
if spot:
    for b in spot.get('balances', []):
        print(f'  {b["coin"]}: {b["total"]}')
