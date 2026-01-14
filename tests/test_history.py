#!/usr/bin/env python3
"""Test how much historical data Hyperliquid provides"""
import asyncio
import aiohttp
import time
import pytest

@pytest.mark.slow
async def test_history():
    url = 'https://api.hyperliquid.xyz/info'
    
    # Try different time ranges
    for days in [1, 7, 14, 30]:
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 86400 * 1000)
        
        payload = {
            'type': 'candleSnapshot',
            'req': {
                'coin': 'BTC',
                'interval': '1m',
                'startTime': start_time,
                'endTime': end_time
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                if isinstance(data, list):
                    print(f'{days} days: {len(data)} klines')
                else:
                    print(f'{days} days: Error - {str(data)[:200]}')
        
        await asyncio.sleep(2)

if __name__ == '__main__':
    asyncio.run(test_history())
