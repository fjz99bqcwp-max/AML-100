#!/bin/bash
# Test Hyperliquid XYZ Perps API

echo "=============================================="
echo "Testing Hyperliquid XYZ Perps API Formats"
echo "=============================================="

echo ""
echo "Test 1: l2Book for XYZ100 WITHOUT dex parameter (expected to fail)"
echo "----------------------------------------------"
curl -s -X POST https://api.hyperliquid.xyz/info \
  -H "Content-Type: application/json" \
  -d '{"type": "l2Book", "coin": "XYZ100"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2)[:300])" 2>&1 || echo "Failed or empty response"

echo ""
echo "Test 2: l2Book for XYZ100 WITH dex='xyz' parameter (correct format)"
echo "----------------------------------------------"
curl -s -X POST https://api.hyperliquid.xyz/info \
  -H "Content-Type: application/json" \
  -d '{"type": "l2Book", "coin": "XYZ100", "dex": "xyz"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2)[:300])" 2>&1 || echo "Failed or empty response"

echo ""
echo "Test 3: l2Book for BTC (main perps, no dex needed)"
echo "----------------------------------------------"
curl -s -X POST https://api.hyperliquid.xyz/info \
  -H "Content-Type: application/json" \
  -d '{"type": "l2Book", "coin": "BTC"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2)[:300])" 2>&1 || echo "Failed or empty response"

echo ""
echo "Test 4: Get all perp dexes to verify xyz dex exists"
echo "----------------------------------------------"
curl -s -X POST https://api.hyperliquid.xyz/info \
  -H "Content-Type: application/json" \
  -d '{"type": "perpDexs"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2)[:500])" 2>&1 || echo "Failed or empty response"

echo ""
echo "Test 5: meta for xyz dex to check available assets"
echo "----------------------------------------------"
curl -s -X POST https://api.hyperliquid.xyz/info \
  -H "Content-Type: application/json" \
  -d '{"type": "meta", "dex": "xyz"}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2)[:500])" 2>&1 || echo "Failed or empty response"

echo ""
echo "=============================================="
echo "Conclusion: For XYZ perps (HIP-3), always include 'dex': 'xyz' at the top level"
echo "=============================================="
