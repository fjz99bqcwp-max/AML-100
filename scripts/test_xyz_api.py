#!/usr/bin/env python3
"""Test script to check correct API format for XYZ perps."""

import requests
import json

BASE_URL = "https://api.hyperliquid.xyz/info"
HEADERS = {"Content-Type": "application/json"}

def test_request(name: str, payload: dict) -> dict:
    """Test an API request and return the response."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 60)
    
    try:
        response = requests.post(BASE_URL, json=payload, headers=HEADERS, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            # Truncate output for readability
            result_str = json.dumps(data, indent=2)
            if len(result_str) > 500:
                print(f"Response (truncated): {result_str[:500]}...")
            else:
                print(f"Response: {result_str}")
            return {"success": True, "data": data, "status": response.status_code}
        else:
            print(f"Error: {response.text[:200]}")
            return {"success": False, "error": response.text, "status": response.status_code}
    except Exception as e:
        print(f"Exception: {e}")
        return {"success": False, "error": str(e), "status": 0}

def main():
    print("=" * 60)
    print("Testing Hyperliquid XYZ Perps API Formats")
    print("=" * 60)
    
    # Test 1: l2Book WITHOUT dex parameter (expected to fail for XYZ perps)
    test_request(
        "l2Book for XYZ100 WITHOUT dex parameter",
        {"type": "l2Book", "coin": "XYZ100"}
    )
    
    # Test 2: l2Book WITH dex parameter (correct format for XYZ perps)
    test_request(
        "l2Book for XYZ100 WITH dex='xyz' parameter",
        {"type": "l2Book", "coin": "XYZ100", "dex": "xyz"}
    )
    
    # Test 3: For comparison, test BTC (main perps, should work without dex)
    test_request(
        "l2Book for BTC (main perps, no dex needed)",
        {"type": "l2Book", "coin": "BTC"}
    )
    
    # Test 4: Test meta endpoint with dex parameter
    test_request(
        "meta for XYZ perps WITH dex='xyz' parameter",
        {"type": "meta", "dex": "xyz"}
    )
    
    # Test 5: Test metaAndAssetCtxs with dex parameter
    test_request(
        "metaAndAssetCtxs for XYZ perps WITH dex='xyz' parameter",
        {"type": "metaAndAssetCtxs", "dex": "xyz"}
    )
    
    # Test 6: candleSnapshot for XYZ perps
    import time
    test_request(
        "candleSnapshot for XYZ100 WITH dex='xyz' parameter",
        {
            "type": "candleSnapshot",
            "req": {
                "coin": "XYZ100",
                "interval": "1m",
                "startTime": int((time.time() - 3600) * 1000),
                "endTime": int(time.time() * 1000)
            },
            "dex": "xyz"
        }
    )
    
    print("\n" + "=" * 60)
    print("Summary of correct API format for XYZ perps:")
    print("=" * 60)
    print("""
For HIP-3 / XYZ perps, you MUST include the 'dex' parameter in the payload.

Correct format for l2Book:
{
    "type": "l2Book",
    "coin": "XYZ100",
    "dex": "xyz"
}

Correct format for candleSnapshot:
{
    "type": "candleSnapshot",
    "req": {
        "coin": "XYZ100",
        "interval": "1m",
        "startTime": <timestamp_ms>,
        "endTime": <timestamp_ms>
    },
    "dex": "xyz"
}

The 'dex' parameter should be at the TOP LEVEL of the payload, not nested.
""")

if __name__ == "__main__":
    main()
