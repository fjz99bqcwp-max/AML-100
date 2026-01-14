#!/usr/bin/env python3
"""Check both Perp and Spot balances"""
import sys
sys.path.insert(0, '.')
from hyperliquid.info import Info
from dotenv import load_dotenv
import os

load_dotenv()

address = os.getenv('HYPERLIQUID_WALLET_ADDRESS', '0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584')
info = Info(skip_ws=True)

# Check Perp account
perp_state = info.user_state(address)
print(f'=== PERP ACCOUNT ({address}) ===')
if perp_state and 'marginSummary' in perp_state:
    margin = perp_state['marginSummary']
    print(f'Account Value: ${float(margin.get("accountValue", 0)):,.2f}')
    print(f'Withdrawable: ${float(perp_state.get("withdrawable", 0)):,.2f}')
else:
    print('No Perp balance')

# Check Spot account
try:
    spot_state = info.spot_user_state(address)
    print(f'\n=== SPOT ACCOUNT ===')
    if spot_state and 'balances' in spot_state:
        total_value = 0
        for bal in spot_state['balances']:
            token = bal.get('coin', 'Unknown')
            amount = float(bal.get('total', 0))
            hold = float(bal.get('hold', 0))
            if amount > 0:
                print(f'{token}: {amount:,.4f} (hold: {hold:,.4f})')
                if token == 'USDC':
                    total_value = amount
        print(f'\nTotal USDC: ${total_value:,.2f}')
    else:
        print('No Spot balances')
except Exception as e:
    print(f'Spot check error: {e}')
