#!/usr/bin/env python3
"""Check Hyperliquid XYZ Perps wallet state - Find correct API"""

import asyncio
import aiohttp
import json

# Your wallet address
WALLET = '0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584'

# XYZ Perp deployer address (trade.xyz uses this for their clearinghouse)
XYZ_PERP_DEPLOYER = '0x010461c14e146AC35fE42271bdc1134EE31C703a'

async def check_xyz_wallet():
    async with aiohttp.ClientSession() as session:
        print("=" * 60)
        print("CHECKING HYPERLIQUID ACCOUNTS")
        print("=" * 60)
        print(f"Wallet: {WALLET}")
        print()
        
        # 1. Standard clearinghouseState (main HL perps)
        print("=== Main Hyperliquid Perps ===")
        async with session.post(
            'https://api.hyperliquid.xyz/info',
            json={'type': 'clearinghouseState', 'user': WALLET}
        ) as resp:
            data = await resp.json()
            margin = data.get('marginSummary', {})
            print(f"Account Value: ${float(margin.get('accountValue', 0)):.2f}")
        print()
        
        # 2. Try perp meta for XYZ deployer
        print("=== Perp Meta (XYZ Deployer) ===")
        async with session.post(
            'https://api.hyperliquid.xyz/info',
            json={'type': 'perpMeta', 'perpDeployer': XYZ_PERP_DEPLOYER}
        ) as resp:
            text = await resp.text()
            try:
                data = json.loads(text)
                universe = data.get('universe', [])
                print(f"Found {len(universe)} perps on XYZ")
                for asset in universe[:5]:
                    print(f"  - {asset.get('name')}")
            except:
                print(f"Response: {text[:500]}")
        print()
        
        # 3. Try clearinghouseState with perpDeployer
        print("=== XYZ Clearinghouse State (with perpDeployer) ===")
        async with session.post(
            'https://api.hyperliquid.xyz/info',
            json={
                'type': 'clearinghouseState', 
                'user': WALLET,
                'perpDeployer': XYZ_PERP_DEPLOYER
            }
        ) as resp:
            text = await resp.text()
            try:
                data = json.loads(text)
                margin = data.get('marginSummary', {})
                print(f"Account Value: ${float(margin.get('accountValue', 0)):.2f}")
                print(f"Total Margin: ${float(margin.get('totalMarginUsed', 0)):.2f}")
                print(f"Withdrawable: ${float(margin.get('withdrawable', 0)):.2f}")
            except:
                print(f"Response: {text[:500]}")
        print()
        
        # 4. User fills to see xyz:XYZ100 trades
        print("=== Recent XYZ100 Fills ===")
        async with session.post(
            'https://api.hyperliquid.xyz/info',
            json={'type': 'userFills', 'user': WALLET}
        ) as resp:
            fills = await resp.json()
            xyz_fills = [f for f in fills if 'xyz' in f.get('coin', '').lower()]
            print(f"Total XYZ fills: {len(xyz_fills)}")
            for fill in xyz_fills[:5]:
                from datetime import datetime
                ts = fill.get('time', 0)
                dt = datetime.fromtimestamp(ts / 1000) if ts else 'unknown'
                print(f"  {dt} - {fill.get('coin')} {fill.get('side')} {fill.get('sz')} @ ${fill.get('px')}")
        print()
        
        # 5. Try to get user funds
        print("=== User Funding / Transfers ===")
        async with session.post(
            'https://api.hyperliquid.xyz/info',
            json={'type': 'userFunding', 'user': WALLET, 'startTime': 0}
        ) as resp:
            text = await resp.text()
            try:
                data = json.loads(text)
                xyz_funding = [f for f in data if 'xyz' in str(f).lower()][:5]
                print(f"XYZ funding entries: {len(xyz_funding)}")
                for f in xyz_funding:
                    print(f"  {f}")
            except:
                print(f"Response: {text[:300]}")
        print()
        
        print("=" * 60)

if __name__ == '__main__':
    asyncio.run(check_xyz_wallet())
