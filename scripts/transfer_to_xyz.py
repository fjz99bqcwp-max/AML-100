"""
Transfer USDC from Hyperliquid L1 to XYZ clearinghouse
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account


async def transfer_to_xyz():
    """Transfer USDC from L1 to XYZ clearinghouse"""
    
    # Load credentials
    private_key = os.getenv("HYPERLIQUID_API_SECRET")
    if not private_key:
        print("❌ HYPERLIQUID_API_SECRET not set")
        return
    
    account = Account.from_key(private_key)
    address = account.address
    
    print(f"Wallet: {address}")
    print()
    
    # Get amount to transfer
    amount = input("Enter amount of USDC to transfer to XYZ clearinghouse (or 'max'): ").strip()
    
    if amount.lower() == "max":
        # Get available balance from L1
        from hyperliquid.info import Info
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        state = info.user_state(address)
        max_amount = float(state["withdrawable"])
        amount = max_amount
        print(f"Transferring maximum: ${amount:,.2f}")
    else:
        try:
            amount = float(amount)
        except ValueError:
            print("❌ Invalid amount")
            return
    
    if amount <= 0:
        print("❌ Amount must be positive")
        return
    
    # Confirm
    print()
    print(f"⚠️  About to transfer ${amount:,.2f} USDC from Hyperliquid L1 to XYZ clearinghouse")
    confirm = input("Type 'yes' to confirm: ").strip().lower()
    
    if confirm != "yes":
        print("❌ Cancelled")
        return
    
    # Execute transfer
    exchange = Exchange(account, constants.MAINNET_API_URL)
    
    # USD transfer to XYZ dex
    result = exchange.usd_transfer(amount, "xyz")
    
    print()
    print(f"✅ Transfer submitted: ${amount:,.2f} USDC → XYZ clearinghouse")
    print(f"Result: {result}")
    print()
    print("Verifying balance...")
    
    # Wait a moment for settlement
    await asyncio.sleep(2)
    
    # Check new balance
    from hyperliquid.info import Info
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    xyz_state = info.user_state(address, dex="xyz")
    xyz_balance = float(xyz_state["marginSummary"]["accountValue"])
    
    print(f"✅ XYZ clearinghouse balance: ${xyz_balance:,.2f}")
    

if __name__ == "__main__":
    asyncio.run(transfer_to_xyz())
