#!/usr/bin/env python3
"""Test fetching specific time ranges to understand API behavior"""
import asyncio
import aiohttp
import time

async def test_ranges():
    url = 'https://api.hyperliquid.xyz/info'
    
    end_time = int(time.time() * 1000)
    
    # Test different starting points
    tests = [
        ("Last 1 day", end_time - (1 * 86400 * 1000), end_time),
        ("Last 7 days", end_time - (7 * 86400 * 1000), end_time),
        ("7-14 days ago", end_time - (14 * 86400 * 1000), end_time - (7 * 86400 * 1000)),
        ("14-21 days ago", end_time - (21 * 86400 * 1000), end_time - (14 * 86400 * 1000)),
        ("21-30 days ago", end_time - (30 * 86400 * 1000), end_time - (21 * 86400 * 1000)),
    ]
    
    async with aiohttp.ClientSession() as session:
        for name, start, end in tests:
            payload = {
                'type': 'candleSnapshot',
                'req': {
                    'coin': 'BTC',
                    'interval': '1m',
                    'startTime': start,
                    'endTime': end
                }
            }
            
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                if isinstance(data, list):
                    print(f'{name}: {len(data)} klines')
                else:
                    print(f'{name}: Error - {str(data)[:100]}')
            
            await asyncio.sleep(1.5)

if __name__ == '__main__':
    asyncio.run(test_ranges())
