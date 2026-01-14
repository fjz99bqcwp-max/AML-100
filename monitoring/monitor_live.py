"""
Live Trading Monitor
Terminal-based monitoring of live trading data from Hyperliquid
Refreshes every 5 minutes aligned to UTC+1 clock (00:00, 00:05, etc.)
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# UTC+1 timezone
UTC_PLUS_1 = timezone(timedelta(hours=1))

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_env():
    """Load environment variables from .env file"""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


load_env()

from src.hyperliquid_api import HyperliquidAPI


class Colors:
    """ANSI color codes for terminal output"""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"


def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


def get_utc_plus_1_now() -> datetime:
    """Get current time in UTC+1"""
    return datetime.now(UTC_PLUS_1)


def get_seconds_until_next_5min() -> float:
    """Calculate seconds until next 5-minute mark in UTC+1"""
    now = get_utc_plus_1_now()
    minutes_to_next = 5 - (now.minute % 5)
    if minutes_to_next == 5:
        minutes_to_next = 0
    next_time = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_next)
    if minutes_to_next == 0:
        next_time += timedelta(minutes=5)
    return (next_time - now).total_seconds()


def format_pnl(value: float) -> str:
    """Format PnL with color"""
    if value > 0:
        return f"{Colors.GREEN}+${value:,.2f}{Colors.RESET}"
    elif value < 0:
        return f"{Colors.RED}-${abs(value):,.2f}{Colors.RESET}"
    return f"${value:,.2f}"


def format_pct(value: float) -> str:
    """Format percentage with color"""
    if value > 0:
        return f"{Colors.GREEN}+{value:.2f}%{Colors.RESET}"
    elif value < 0:
        return f"{Colors.RED}{value:.2f}%{Colors.RESET}"
    return f"{value:.2f}%"


async def fetch_all_data(api: HyperliquidAPI) -> dict:
    """Fetch all data at once for synchronized refresh"""
    data = {
        "state": None,
        "orders": None,
        "fills": None,
        "orderbook": None,
        "error": None
    }
    
    try:
        # Fetch all data concurrently
        results = await asyncio.gather(
            api.get_user_state(),
            api.get_open_orders(),
            api.get_user_fills(limit=10),
            api.get_orderbook("BTC"),
            return_exceptions=True
        )
        
        data["state"] = results[0] if not isinstance(results[0], Exception) else None
        data["orders"] = results[1] if not isinstance(results[1], Exception) else None
        data["fills"] = results[2] if not isinstance(results[2], Exception) else None
        data["orderbook"] = results[3] if not isinstance(results[3], Exception) else None
        
        for result in results:
            if isinstance(result, Exception):
                data["error"] = str(result)
                break
                
    except Exception as e:
        data["error"] = str(e)
    
    return data


def display_dashboard(data: dict, api: HyperliquidAPI, next_refresh: float):
    """Display the dashboard with fetched data"""
    clear_screen()
    
    now = get_utc_plus_1_now()
    
    # Wallet address for reference
    WALLET = os.environ.get("HYPERLIQUID_WALLET_ADDRESS", "0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584")
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}")
    print("  AML-100 HFT - Live Trading Monitor (XYZ100-USDC)")
    print("=" * 60 + f"{Colors.RESET}")
    print(f"  Time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC+1")
    print(f"  Wallet: {WALLET[:10]}...{WALLET[-6:]}")
    print(f"  Next refresh in: {int(next_refresh)}s")
    print()
    
    if data["error"]:
        print(f"{Colors.RED}Error fetching data: {data['error']}{Colors.RESET}")
        return
    
    # TRADE DENSITY ALERT SECTION (NEW)
    fills = data.get("fills", [])
    if fills and len(fills) > 0:
        # Calculate trades per hour from recent fills
        # Note: Trade objects have timestamp attribute (in seconds), not "time" key
        recent_fills = [f for f in fills if time.time() - getattr(f, 'timestamp', 0) < 3600]
        trades_per_hour = len(recent_fills)
        
        print(f"{Colors.BOLD}🚨 Trade Density Monitor{Colors.RESET}")
        print("-" * 40)
        
        # Alert thresholds
        if trades_per_hour < 3:
            alert_color = Colors.RED
            alert_msg = "CRITICAL: Very low activity"
        elif trades_per_hour < 5:
            alert_color = Colors.YELLOW
            alert_msg = "WARNING: Below target"
        else:
            alert_color = Colors.GREEN
            alert_msg = "GOOD: Active trading"
        
        print(f"  Trades/Hour:      {alert_color}{trades_per_hour} {alert_msg}{Colors.RESET}")
        print(f"  Target:           5+ trades/hour")
        print(f"  Last 10 fills:    {min(len(fills), 10)} trades")
        print()
    
    state = data["state"]
    if state:
        # Account Summary
        print(f"{Colors.BOLD}📊 Account Summary{Colors.RESET}")
        print("-" * 40)
        
        margin_summary = state.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0))
        total_margin_used = float(margin_summary.get("totalMarginUsed", 0))
        
        print(f"  Account Value:    ${account_value:,.2f}")
        print(f"  Margin Used:      ${total_margin_used:,.2f}")
        print(f"  Available:        ${account_value - total_margin_used:,.2f}")
        print()
        
        # Positions
        print(f"{Colors.BOLD}📈 Open Positions{Colors.RESET}")
        print("-" * 40)
        
        positions = state.get("assetPositions", [])
        has_positions = False
        
        for pos in positions:
            position = pos.get("position", {})
            size = float(position.get("szi", 0))
            
            if size != 0:
                has_positions = True
                coin = position.get("coin", "???")
                entry_px = float(position.get("entryPx", 0))
                unrealized_pnl = float(position.get("unrealizedPnl", 0))
                leverage = position.get("leverage", {}).get("value", 1)
                liq_px = position.get("liquidationPx")
                
                side = "LONG" if size > 0 else "SHORT"
                side_color = Colors.GREEN if size > 0 else Colors.RED
                
                print(f"  {Colors.BOLD}{coin}{Colors.RESET}")
                print(f"    Side:         {side_color}{side}{Colors.RESET}")
                print(f"    Size:         {abs(size):.4f}")
                print(f"    Entry Price:  ${entry_px:,.2f}")
                print(f"    Leverage:     {leverage}x")
                print(f"    Unrealized:   {format_pnl(unrealized_pnl)}")
                if liq_px:
                    print(f"    Liquidation:  ${float(liq_px):,.2f}")
                print()
        
        if not has_positions:
            print(f"  {Colors.YELLOW}No open positions{Colors.RESET}")
            print()
    
    # Open Orders
    print(f"{Colors.BOLD}📋 Open Orders{Colors.RESET}")
    print("-" * 40)
    
    orders = data["orders"]
    if orders:
        for order in orders[:5]:
            coin = order.get("coin", "???")
            side = order.get("side", "???")
            sz = float(order.get("sz", 0))
            px = float(order.get("limitPx", 0))
            
            side_color = Colors.GREEN if side == "B" else Colors.RED
            side_name = "BUY" if side == "B" else "SELL"
            
            print(f"  {coin}: {side_color}{side_name}{Colors.RESET} {sz:.4f} @ ${px:,.2f}")
    else:
        print(f"  {Colors.YELLOW}No open orders{Colors.RESET}")
    print()
    
    # Recent Trades (Real Trading Only - last 24h)
    print(f"{Colors.BOLD}💹 Recent Trades (Live){Colors.RESET}")
    print("-" * 40)
    
    fills = data["fills"]
    if fills:
        recent_fills = []
        cutoff = time.time() - 86400  # 24 hours
        
        for trade in fills:
            if hasattr(trade, 'timestamp') and trade.timestamp > cutoff:
                recent_fills.append(trade)
        
        if recent_fills:
            for trade in recent_fills[:5]:
                ts = trade.timestamp
                time_str = datetime.fromtimestamp(ts, tz=UTC_PLUS_1).strftime("%H:%M:%S")
                side = "BUY" if trade.side == "B" else "SELL"
                side_color = Colors.GREEN if trade.side == "B" else Colors.RED
                
                print(f"  {time_str} | {trade.symbol} | {side_color}{side}{Colors.RESET} | "
                      f"{trade.size:.4f} @ ${trade.price:,.2f}")
        else:
            print(f"  {Colors.YELLOW}No trades in last 24h{Colors.RESET}")
    else:
        print(f"  {Colors.YELLOW}No recent trades{Colors.RESET}")
    print()
    
    # HFT Mode Status - load from system state if available
    system_state_file = PROJECT_ROOT / "logs" / "system_state.json"
    if system_state_file.exists():
        try:
            with open(system_state_file) as f:
                system_state = json.load(f)
            
            risk_metrics = system_state.get("risk_metrics", {})
            hold_stats = system_state.get("hold_time_stats", {})
            
            print(f"{Colors.BOLD}⏱️  HFT Hold Time Stats{Colors.RESET}")
            print("-" * 40)
            
            if hold_stats:
                avg_hold = hold_stats.get("avg", 0)
                min_hold = hold_stats.get("min", 0)
                max_hold = hold_stats.get("max", 0)
                target = hold_stats.get("target", 30)
                within_target = hold_stats.get("within_target_pct", 0)
                
                # Color code based on target
                avg_color = Colors.GREEN if avg_hold <= target else Colors.YELLOW
                
                print(f"  Avg Hold Time:    {avg_color}{avg_hold:.1f}s{Colors.RESET} (target: {target}s)")
                print(f"  Min/Max:          {min_hold:.1f}s / {max_hold:.1f}s")
                print(f"  Within Target:    {within_target:.1f}%")
            else:
                print(f"  {Colors.YELLOW}No hold time data yet{Colors.RESET}")
            print()
        except Exception:
            pass  # Skip HFT stats if file read fails
    
    # Market Data
    print(f"{Colors.BOLD}📊 Market Data (BTC){Colors.RESET}")
    print("-" * 40)
    
    orderbook = data["orderbook"]
    if orderbook:
        print(f"  Mid Price:  ${orderbook.mid_price:,.2f}")
        print(f"  Spread:     {orderbook.spread_bps:.2f} bps")
        if orderbook.bids and orderbook.asks:
            print(f"  Best Bid:   ${orderbook.bids[0][0]:,.2f} ({orderbook.bids[0][1]:.4f})")
            print(f"  Best Ask:   ${orderbook.asks[0][0]:,.2f} ({orderbook.asks[0][1]:.4f})")
    else:
        print(f"  {Colors.YELLOW}Orderbook unavailable{Colors.RESET}")
    print()
    
    # API Latency
    print(f"{Colors.BOLD}⚡ Performance{Colors.RESET}")
    print("-" * 40)
    print(f"  Avg Latency:  {api.get_avg_latency():.2f} ms")
    print(f"  P99 Latency:  {api.get_latency_p99():.2f} ms")
    print()
    
    print(f"{Colors.YELLOW}Press Ctrl+C to exit | Refreshes every 5min (aligned to UTC+1){Colors.RESET}")


async def monitor_live():
    """Main monitoring loop - refreshes every 5 minutes aligned to UTC+1"""
    api = HyperliquidAPI("config/api.json")
    
    try:
        await api.initialize()
        print(f"{Colors.GREEN}Connected to Hyperliquid{Colors.RESET}")
        
        while True:
            # Fetch all data at once
            data = await fetch_all_data(api)
            
            # Calculate time until next 5-minute mark
            next_refresh = get_seconds_until_next_5min()
            
            # Display dashboard
            display_dashboard(data, api, next_refresh)
            
            # Wait until next 5-minute mark
            await asyncio.sleep(next_refresh)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Monitoring stopped{Colors.RESET}")
    finally:
        await api.close()


if __name__ == "__main__":
    asyncio.run(monitor_live())
