#!/usr/bin/env python3
"""Check XYZ transfers and perp ledger"""

import asyncio
import aiohttp
import json

WALLET = '0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584'

async def check():
    async with aiohttp.ClientSession() as session:
        # Check ledger updates
        print("=== userNonFundingLedgerUpdates ===")
        async with session.post(
            'https://api.hyperliquid.xyz/info', 
            json={'type': 'userNonFundingLedgerUpdates', 'user': WALLET, 'startTime': 0}
        ) as resp:
            data = await resp.json()
            # Filter for xyz/XYZ entries and show recent ones
            xyz_data = [d for d in data if 'xyz' in str(d).lower()]
            print(f"Found {len(xyz_data)} XYZ ledger entries")
            for d in xyz_data[:10]:
                print(f"  {d}")
        print()
        
        # Check deposits
        print("=== Deposits (internalTransfer) ===")
        transfers = [d for d in data if d.get('delta', {}).get('type') in ['internalTransfer', 'deposit', 'spotTransfer']]
        recent_transfers = transfers[-20:]
        for t in recent_transfers:
            print(f"  {t}")
        print()

asyncio.run(check())
