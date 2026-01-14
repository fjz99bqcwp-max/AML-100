"""
Check XYZ clearinghouse balance
"""
import os
from hyperliquid.info import Info
from hyperliquid.utils import constants

# Load wallet address
wallet = os.getenv("HYPERLIQUID_WALLET_ADDRESS")
if not wallet:
    print("❌ HYPERLIQUID_WALLET_ADDRESS not set")
    exit(1)

print(f"Checking XYZ clearinghouse for wallet: {wallet}\n")

# Get XYZ clearinghouse state
info = Info(constants.MAINNET_API_URL, skip_ws=True)
xyz_state = info.user_state(wallet, dex="xyz")

margin = xyz_state["marginSummary"]
account_value = float(margin["accountValue"])
margin_used = float(margin["totalMarginUsed"])
withdrawable = float(xyz_state["withdrawable"])

print("=== XYZ CLEARINGHOUSE (xyz:XYZ100-USDC) ===")
print(f"Account Value: ${account_value:,.2f}")
print(f"Margin Used: ${margin_used:,.2f}")
print(f"Withdrawable: ${withdrawable:,.2f}")
print()

# Check positions
positions = xyz_state.get("assetPositions", [])
has_position = False
for pos in positions:
    p = pos.get("position", {})
    size = float(p.get("szi", 0))
    if size != 0:
        has_position = True
        print(f"Position: {p.get('coin')} - Size: {size}, Entry: ${float(p.get('entryPx', 0)):,.2f}")

if not has_position:
    print("No open positions")
