#!/usr/bin/env python3
"""Find the correct API for XYZ perps clearinghouse"""

import asyncio
import aiohttp
import json

WALLET = '0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584'

async def check():
    async with aiohttp.ClientSession() as session:
        # The transfer shows destinationDex='xyz', so we need to query the xyz clearinghouse
        # Try different API patterns
        
        print("=== Method 1: clearinghouseState with dex parameter ===")
        async with session.post(
            'https://api.hyperliquid.xyz/info',
            json={'type': 'clearinghouseState', 'user': WALLET, 'dex': 'xyz'}
        ) as resp:
            text = await resp.text()
            try:
                data = json.loads(text)
                margin = data.get('marginSummary', {})
                print(f"Account Value: ${float(margin.get('accountValue', 0)):.2f}")
            except:
                print(f"Error: {text[:200]}")
        
        print("\n=== Method 2: perpsPerUser ===")
        async with session.post(
            'https://api.hyperliquid.xyz/info',
            json={'type': 'perpsPerUser', 'user': WALLET}
        ) as resp:
            text = await resp.text()
            try:
                data = json.loads(text)
                print(f"Response: {json.dumps(data, indent=2)[:500]}")
            except:
                print(f"Error: {text[:200]}")
        
        print("\n=== Method 3: userPerps ===")
        async with session.post(
            'https://api.hyperliquid.xyz/info',
            json={'type': 'userPerps', 'user': WALLET}
        ) as resp:
            text = await resp.text()
            try:
                data = json.loads(text)
                print(f"Response: {json.dumps(data, indent=2)[:500]}")
            except:
                print(f"Error: {text[:200]}")
        
        print("\n=== Method 4: multiClearinghouseState ===")
        async with session.post(
            'https://api.hyperliquid.xyz/info',
            json={'type': 'multiClearinghouseState', 'users': [WALLET]}
        ) as resp:
            text = await resp.text()
            try:
                data = json.loads(text)
                print(f"Response: {json.dumps(data, indent=2)[:1000]}")
            except:
                print(f"Error: {text[:200]}")
        
        print("\n=== Method 5: XYZ API endpoint ===")
        # Try trade.xyz specific API
        async with session.post(
            'https://api.trade.xyz/info',
            json={'type': 'clearinghouseState', 'user': WALLET}
        ) as resp:
            text = await resp.text()
            try:
                data = json.loads(text)
                margin = data.get('marginSummary', {})
                print(f"Account Value: ${float(margin.get('accountValue', 0)):.2f}")
            except:
                print(f"Response: {text[:300]}")

asyncio.run(check())
