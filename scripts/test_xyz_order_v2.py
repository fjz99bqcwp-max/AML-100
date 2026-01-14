#!/usr/bin/env python3
"""Test XYZ100 order with correct asset index (110000)"""

import os
import time
import requests
import json
import msgpack
from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import keccak

# Load config
with open('config/api.json') as f:
    config = json.load(f)

print(f'Asset index from config: {config["asset_index"]}')  # Should be 110000

# Get private key
pk = os.getenv('HYPERLIQUID_API_SECRET')
wallet = Account.from_key(pk)
address = wallet.address
print(f'Wallet: {address}')

# Get orderbook for current price
r = requests.post('https://api.hyperliquid.xyz/info', json={
    'type': 'l2Book',
    'coin': 'xyz:XYZ100',
    'dex': 'xyz'
})
book = r.json()
best_bid = float(book['levels'][0][0]['px'])
best_ask = float(book['levels'][1][0]['px'])
print(f'Orderbook: Bid=${best_bid:.0f}, Ask=${best_ask:.0f}')

# Build order with correct asset index 110000 for XYZ perps
order = {
    'a': 110000,  # XYZ perps offset (110000) + position 0 = 110000
    'b': True,    # Buy
    'p': str(int(best_bid - 100)),  # Well below market (won't fill)
    's': '0.0004',  # Min $10 value: $25723 * 0.0004 = $10.29
    'r': False,
    't': {'limit': {'tif': 'Gtc'}}
}

action = {
    'type': 'order',
    'orders': [order],
    'grouping': 'na'
    # NO dex field - asset index already identifies the DEX
}

print(f'Order action: {json.dumps(action, indent=2)}')

# Sign the action
def action_hash(action, vault_address, nonce):
    data = msgpack.packb(action)
    hash_input = data + nonce.to_bytes(8, 'big') + (b'\x01' if vault_address else b'\x00')
    return keccak(hash_input)

def construct_phantom_agent(hash_val, is_mainnet):
    return {
        'source': 'a' if is_mainnet else 'b',
        'connectionId': hash_val
    }

nonce = int(time.time() * 1000)
hash_val = action_hash(action, None, nonce)
phantom = construct_phantom_agent(hash_val, True)  # True = mainnet

typed_data = {
    'types': {
        'EIP712Domain': [
            {'name': 'name', 'type': 'string'},
            {'name': 'version', 'type': 'string'},
            {'name': 'chainId', 'type': 'uint256'},
            {'name': 'verifyingContract', 'type': 'address'}
        ],
        'Agent': [
            {'name': 'source', 'type': 'string'},
            {'name': 'connectionId', 'type': 'bytes32'}
        ]
    },
    'primaryType': 'Agent',
    'domain': {
        'name': 'Exchange',
        'version': '1',
        'chainId': 1337,
        'verifyingContract': '0x0000000000000000000000000000000000000000'
    },
    'message': phantom
}

signable = encode_typed_data(full_message=typed_data)
signed = wallet.sign_message(signable)

signature = {
    'r': hex(signed.r),
    's': hex(signed.s),
    'v': signed.v
}

payload = {
    'action': action,
    'nonce': nonce,
    'signature': signature
}

print(f'\nSending order to https://api.hyperliquid.xyz/exchange...')
r = requests.post('https://api.hyperliquid.xyz/exchange', json=payload)
print(f'Status: {r.status_code}')
result = r.json()
print(f'Response: {json.dumps(result, indent=2)}')

# Check if order was successful
if result.get('status') == 'ok':
    statuses = result.get('response', {}).get('data', {}).get('statuses', [])
    if statuses:
        status = statuses[0]
        if 'resting' in status:
            print(f'\n✅ ORDER PLACED SUCCESSFULLY!')
            print(f'   Order ID: {status["resting"]["oid"]}')
            # Cancel the test order
            print(f'\n   Canceling test order...')
            cancel_action = {
                'type': 'cancel',
                'cancels': [{'a': 110000, 'o': status['resting']['oid']}]
            }
            cancel_nonce = int(time.time() * 1000)
            cancel_hash = action_hash(cancel_action, None, cancel_nonce)
            cancel_phantom = construct_phantom_agent(cancel_hash, True)
            cancel_typed_data = typed_data.copy()
            cancel_typed_data['message'] = cancel_phantom
            cancel_signable = encode_typed_data(full_message=cancel_typed_data)
            cancel_signed = wallet.sign_message(cancel_signable)
            cancel_sig = {'r': hex(cancel_signed.r), 's': hex(cancel_signed.s), 'v': cancel_signed.v}
            cancel_payload = {'action': cancel_action, 'nonce': cancel_nonce, 'signature': cancel_sig}
            cancel_r = requests.post('https://api.hyperliquid.xyz/exchange', json=cancel_payload)
            print(f'   Cancel status: {cancel_r.status_code}')
            print(f'   Cancel response: {cancel_r.text}')
        elif 'filled' in status:
            print(f'\n✅ ORDER FILLED IMMEDIATELY!')
            print(f'   Total Size: {status["filled"]["totalSz"]}')
elif result.get('status') == 'err':
    print(f'\n❌ ORDER FAILED: {result.get("response")}')
else:
    print(f'\n⚠️ Unexpected response format')
