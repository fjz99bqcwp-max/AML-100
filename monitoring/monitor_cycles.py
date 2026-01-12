"""
Cycle Monitor
Monitors current settings, performance status, and cycle information
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
                    os.environ.setdefault(key.strip(), value.strip())


load_env()


class Colors:
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


def get_status_color(status: str) -> str:
    colors = {
        "critical": Colors.RED,
        "poor": Colors.YELLOW,
        "moderate": Colors.BLUE,
        "good": Colors.GREEN
    }
    return colors.get(status.lower(), Colors.WHITE)


def format_time_ago(timestamp: float) -> str:
    if not timestamp:
        return "Never"
    delta = time.time() - timestamp
    if delta < 60:
        return f"{int(delta)}s ago"
    elif delta < 3600:
        return f"{int(delta/60)}m ago"
    else:
        return f"{int(delta/3600)}h {int((delta%3600)/60)}m ago"


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


def load_all_data() -> dict:
    """Load all data at once for synchronized refresh"""
    config_dir = Path("config")
    monitoring_dir = Path("monitoring")
    data_dir = Path("data")
    
    data = {
        "params": None,
        "objectives": None,
        "dashboard": None,
        "backtest": None,
        "trades": None,
        "errors": []
    }
    
    # Load parameters
    try:
        with open(config_dir / "params.json", "r") as f:
            data["params"] = json.load(f)
    except Exception as e:
        data["errors"].append(f"params: {e}")
    
    # Load objectives
    try:
        with open(config_dir / "objectives.json", "r") as f:
            data["objectives"] = json.load(f)
    except Exception as e:
        data["errors"].append(f"objectives: {e}")
    
    # Load dashboard data
    dashboard_file = monitoring_dir / "dashboard_data.json"
    if dashboard_file.exists():
        try:
            with open(dashboard_file, "r") as f:
                data["dashboard"] = json.load(f)
        except Exception as e:
            data["errors"].append(f"dashboard: {e}")
    
    # Load latest backtest (deduplicated - only most recent)
    backtest_dir = data_dir / "backtests"
    if backtest_dir.exists():
        backtest_files = sorted(backtest_dir.glob("backtest_*.json"))
        if backtest_files:
            try:
                with open(backtest_files[-1], "r") as f:
                    data["backtest"] = json.load(f)
                    data["backtest"]["_filename"] = backtest_files[-1].name
            except Exception as e:
                data["errors"].append(f"backtest: {e}")
    
    # Load trades (deduplicated - only from latest file)
    trades_dir = data_dir / "trading"
    if trades_dir.exists():
        trade_files = sorted(trades_dir.glob("trades_*.json"))
        if trade_files:
            try:
                with open(trade_files[-1], "r") as f:
                    trades = json.load(f)
                    # Deduplicate trades by timestamp+side+size
                    seen = set()
                    unique_trades = []
                    for t in trades:
                        key = (t.get("timestamp", 0), t.get("side", ""), t.get("size", 0))
                        if key not in seen:
                            seen.add(key)
                            unique_trades.append(t)
                    data["trades"] = unique_trades
                    data["trades_filename"] = trade_files[-1].name
            except Exception:
                pass
    
    return data


def display_cycles_dashboard(data: dict, next_refresh: float):
    """Display the cycles dashboard"""
    clear_screen()
    
    now = get_utc_plus_1_now()
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}")
    print("  AML HFT - Cycle & Settings Monitor")
    print("=" * 60 + f"{Colors.RESET}")
    print(f"  Time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC+1")
    print(f"  Next refresh in: {int(next_refresh)}s")
    print()
    
    # Display errors if any
    for error in data["errors"]:
        print(f"{Colors.RED}Error loading {error}{Colors.RESET}")
    
    # Trading Parameters
    params = data["params"]
    if params:
        print(f"{Colors.BOLD}⚙️  Current Trading Parameters{Colors.RESET}")
        print("-" * 40)
        
        trading = params.get("trading", {})
        print(f"  Take Profit:      {Colors.GREEN}{trading.get('take_profit_pct', 0):.2f}%{Colors.RESET}")
        print(f"  Stop Loss:        {Colors.RED}{trading.get('stop_loss_pct', 0):.2f}%{Colors.RESET}")
        print(f"  Position Size:    {trading.get('position_size_pct', 0):.2f}%")
        print(f"  Max Positions:    {trading.get('max_positions', 1)}")
        print(f"  Leverage:         {trading.get('leverage', 1)}x")
        print()
        
        ml_config = params.get("ml_model", {})
        print(f"{Colors.BOLD}🤖 ML Model Settings{Colors.RESET}")
        print("-" * 40)
        print(f"  Model Type:       {ml_config.get('type', 'N/A')}")
        print(f"  Learning Rate:    {ml_config.get('learning_rate', 0):.6f}")
        print(f"  Batch Size:       {ml_config.get('batch_size', 0)}")
        print(f"  Sequence Length:  {ml_config.get('sequence_length', 0)}")
        print(f"  LSTM Hidden:      {ml_config.get('lstm_hidden_size', 0)}")
        print()
    
    # Objectives
    objectives = data["objectives"]
    if objectives:
        print(f"{Colors.BOLD}🎯 Performance Objectives{Colors.RESET}")
        print("-" * 40)
        print(f"  Monthly Min:      {objectives.get('monthly_performance_min', 0)}%")
        print(f"  Monthly Max:      {objectives.get('monthly_performance_max', 0)}%")
        print(f"  Sharpe Min:       {objectives.get('sharpe_ratio_min', 0)}")
        print(f"  Profit Factor:    {objectives.get('profit_factor_min', 0)}")
        print(f"  Max Drawdown:     {objectives.get('drawdown_max', 0)}%")
        print(f"  Auto-Stop DD:     {objectives.get('auto_stop_drawdown', 0)}%")
        print()
    
    # Dashboard/Cycle Status
    dashboard = data["dashboard"]
    if dashboard:
        print(f"{Colors.BOLD}📊 Current Cycle Status{Colors.RESET}")
        print("-" * 40)
        
        status = dashboard.get("status", "unknown")
        status_color = get_status_color(status)
        
        print(f"  Current Cycle:    #{dashboard.get('current_cycle', 0)}")
        print(f"  Status:           {status_color}{Colors.BOLD}{status.upper()}{Colors.RESET}")
        print(f"  Last Update:      {format_time_ago(dashboard.get('last_update', 0))}")
        print(f"  Cycle PnL:        ${dashboard.get('cycle_pnl', 0):.2f}")
        print()
        
        # Risk status
        risk = dashboard.get("risk_status", {})
        if risk:
            print(f"{Colors.BOLD}⚠️  Risk Status{Colors.RESET}")
            print("-" * 40)
            
            halted = risk.get("is_halted", False)
            if halted:
                print(f"  {Colors.RED}⛔ TRADING HALTED{Colors.RESET}")
                print(f"  Reason: {risk.get('halt_reason', 'Unknown')}")
            else:
                print(f"  {Colors.GREEN}✓ Trading Active{Colors.RESET}")
            
            print(f"  Risk Level:       {risk.get('risk_level', 'N/A')}")
            print(f"  Current Capital:  ${risk.get('current_capital', 0):,.2f}")
            print(f"  Total PnL:        ${risk.get('total_pnl', 0):,.2f} ({risk.get('total_pnl_pct', 0):.2f}%)")
            print(f"  Current DD:       {risk.get('current_drawdown', 0):.2f}%")
            print(f"  Max DD:           {risk.get('max_drawdown', 0):.2f}%")
            print(f"  Win Rate:         {risk.get('win_rate', 0):.1f}%")
            print(f"  Profit Factor:    {risk.get('profit_factor', 0):.2f}")
            print(f"  Sharpe Ratio:     {risk.get('sharpe_ratio', 0):.2f}")
            print(f"  Trade Count:      {risk.get('trade_count', 0)}")
            print()
    else:
        print(f"{Colors.YELLOW}No cycle data available yet{Colors.RESET}")
        print()
    
    # Latest Backtest (deduplicated)
    backtest = data["backtest"]
    if backtest:
        print(f"{Colors.BOLD}📈 Latest Backtest{Colors.RESET}")
        print("-" * 40)
        print(f"  File:             {backtest.get('_filename', 'N/A')}")
        print(f"  Return:           {backtest.get('total_return_pct', 0):.2f}%")
        print(f"  Sharpe:           {backtest.get('sharpe_ratio', 0):.2f}")
        print(f"  Max DD:           {backtest.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Win Rate:         {backtest.get('win_rate', 0):.1f}%")
        print(f"  Profit Factor:    {backtest.get('profit_factor', 0):.2f}")
        print(f"  Total Trades:     {backtest.get('total_trades', 0)}")
        
        objectives_met = backtest.get("objectives_met", False)
        if objectives_met:
            print(f"  Objectives:       {Colors.GREEN}✓ MET{Colors.RESET}")
        else:
            print(f"  Objectives:       {Colors.RED}✗ NOT MET{Colors.RESET}")
        print()
    
    # Recent Trades (deduplicated)
    trades = data.get("trades")
    if trades:
        print(f"{Colors.BOLD}📊 Recent Trades ({data.get('trades_filename', 'N/A')}){Colors.RESET}")
        print("-" * 40)
        
        # Summary stats
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losses = len(trades) - wins
        
        print(f"  Total Trades:     {len(trades)}")
        print(f"  Wins/Losses:      {wins}/{losses}")
        print(f"  Total PnL:        ${total_pnl:.2f}")
        
        # Show last 3 trades
        if len(trades) > 0:
            print(f"\n  {Colors.CYAN}Last 3 trades:{Colors.RESET}")
            for t in trades[-3:]:
                side = "BUY" if t.get("side") == "B" else "SELL"
                pnl = t.get("pnl", 0)
                pnl_color = Colors.GREEN if pnl > 0 else Colors.RED
                print(f"    {side}: {pnl_color}${pnl:.2f}{Colors.RESET}")
        print()
    
    print(f"{Colors.YELLOW}Press Ctrl+C to exit | Refreshes every 5min (aligned to UTC+1){Colors.RESET}")


async def monitor_cycles():
    """Monitor trading cycles and settings - refreshes every 5 minutes aligned to UTC+1"""
    while True:
        # Load all data at once
        data = load_all_data()
        
        # Calculate time until next 5-minute mark
        next_refresh = get_seconds_until_next_5min()
        
        # Display dashboard
        display_cycles_dashboard(data, next_refresh)
        
        # Wait until next 5-minute mark
        await asyncio.sleep(next_refresh)


if __name__ == "__main__":
    try:
        asyncio.run(monitor_cycles())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Monitoring stopped{Colors.RESET}")
