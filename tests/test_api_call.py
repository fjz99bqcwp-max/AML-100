#!/usr/bin/env python3
"""Test Hyperliquid API to check kline format"""
import asyncio
import aiohttp
import json
import time
import os
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)

async def test_raw_api():
    """Test direct API call"""
    url = 'https://api.hyperliquid.xyz/info'
    
    end_time = int(time.time() * 1000)
    start_time = end_time - (24 * 60 * 60 * 1000)
    
    print(f'Raw API Test - Start: {start_time}, End: {end_time}')
    
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
            print('Response type:', type(data))
            if isinstance(data, list) and len(data) > 0:
                print('First element:', json.dumps(data[0], indent=2))
                print('Total elements:', len(data))
            else:
                print('Full response:', json.dumps(data, indent=2)[:1000])

async def test_our_api():
    """Test our API wrapper"""
    from src.hyperliquid_api import HyperliquidAPI
    from src.data_fetcher import DataFetcher
    
    print('\n--- Testing our API wrapper ---')
    api = HyperliquidAPI()
    await api.initialize()
    
    try:
        klines = await api.get_klines(symbol='BTC', interval='1m', limit=100)
        print(f'Our API returned {len(klines) if isinstance(klines, list) else "non-list"} klines')
        if isinstance(klines, list) and len(klines) > 0:
            print('First kline:', json.dumps(klines[0], indent=2))
        else:
            print('Result:', klines)
        
        # Wait a bit before next test
        print('\nWaiting 2 seconds before data fetcher test...')
        await asyncio.sleep(2)
        
        print('\n--- Testing Data Fetcher ---')
        fetcher = DataFetcher(api)
        df = await fetcher.fetch_historical_klines(symbol='BTC', interval='1m', days=1, save=False)
        print(f'Data fetcher returned {len(df)} rows')
        if len(df) > 0:
            print(df.head())
    finally:
        await api.close()

async def main():
    """Run all tests in single event loop"""
    await test_raw_api()
    
    # Wait before next test to let rate limits reset
    print('\nWaiting 3 seconds before API wrapper test...')
    await asyncio.sleep(3)
    
    await test_our_api()

if __name__ == '__main__':
    asyncio.run(main())
