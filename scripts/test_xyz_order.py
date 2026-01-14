#!/usr/bin/env python3
"""Test XYZ perps order structure"""

import asyncio
import aiohttp
import json
import os
import time
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak, to_hex
import msgpack

WALLET = '0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584'
PRIVATE_KEY = os.environ.get('HYPERLIQUID_API_SECRET')

async def test_order():
    if not PRIVATE_KEY:
        print("Missing HYPERLIQUID_API_SECRET env var")
        return
    
    account = Account.from_key(PRIVATE_KEY)
    
    async with aiohttp.ClientSession() as session:
        # Get orderbook first
        async with session.post(
            'https://api.hyperliquid.xyz/info',
            json={'type': 'l2Book', 'coin': 'xyz:XYZ100', 'dex': 'xyz'}
        ) as resp:
            ob = await resp.json()
            best_bid = float(ob['levels'][0][0]['px'])
            best_ask = float(ob['levels'][1][0]['px'])
            print(f"Orderbook: Bid=${best_bid:.0f}, Ask=${best_ask:.0f}")
        
        # Prepare order - XYZ100 has 0 price decimals, 4 size decimals
        # Asset index 0 for XYZ100
        price = int(best_bid)  # Limit at bid for safety
        size = "0.0001"  # Minimum size
        
        order = {
            "a": 0,  # Asset index (XYZ100 is 0 on xyz dex)
            "b": True,  # Buy
            "p": str(price),  # Price as string
            "s": size,  # Size
            "r": False,  # Not reduce only
            "t": {"limit": {"tif": "Gtc"}}  # Good till cancel
        }
        
        action = {
            "type": "order",
            "orders": [order],
            "grouping": "na",
            "dex": "xyz"  # CRITICAL: Must specify xyz dex
        }
        
        print(f"\nAction to sign: {json.dumps(action, indent=2)}")
        
        # Sign the action
        nonce = int(time.time() * 1000)
        
        data = msgpack.packb(action)
        data += nonce.to_bytes(8, "big")
        data += b"\x00"  # No vault
        action_hash = keccak(data)
        
        phantom_agent = {
            "source": "a",
            "connectionId": action_hash
        }
        
        l1_data = {
            "domain": {
                "chainId": 1337,
                "name": "Exchange",
                "verifyingContract": "0x0000000000000000000000000000000000000000",
                "version": "1",
            },
            "types": {
                "Agent": [
                    {"name": "source", "type": "string"},
                    {"name": "connectionId", "type": "bytes32"},
                ],
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
            },
            "primaryType": "Agent",
            "message": phantom_agent,
        }
        
        signable_message = encode_typed_data(full_message=l1_data)
        signed = account.sign_message(signable_message)
        
        payload = {
            "action": action,
            "nonce": nonce,
            "signature": {
                "r": to_hex(signed.r),
                "s": to_hex(signed.s),
                "v": signed.v
            },
            "vaultAddress": None
        }
        
        print(f"\nSubmitting order...")
        
        async with session.post(
            'https://api.hyperliquid.xyz/exchange',
            json=payload,
            headers={'Content-Type': 'application/json'}
        ) as resp:
            result = await resp.text()
            print(f"Response: {result}")

if __name__ == '__main__':
    asyncio.run(test_order())
