"""
Dashboard Health
Monitors and displays system health metrics in terminal
Refreshes every 5 minutes aligned to UTC+1 clock (00:00, 00:05, etc.)
"""

import asyncio
import json
import os
import psutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import sys

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


def format_bytes(bytes_val: float) -> str:
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def format_uptime(seconds: float) -> str:
    """Format uptime to human readable"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds/60)}m {int(seconds%60)}s"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"
    else:
        days = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        return f"{days}d {hours}h"


class SystemHealth:
    """Monitors system health and connection status"""
    
    def __init__(self, output_path: str = "monitoring/health.json"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.health = {
            "last_update": 0,
            "system_status": "unknown",
            "api_connected": False,
            "websocket_connected": False,
            "last_api_latency_ms": 0,
            "avg_api_latency_ms": 0,
            "p99_api_latency_ms": 0,
            "api_errors_1h": 0,
            "websocket_errors_1h": 0,
            "last_trade_time": None,
            "last_heartbeat": 0,
            "uptime_seconds": 0,
            "cpu_percent": 0,
            "memory_percent": 0,
            "memory_mb": 0,
            "disk_percent": 0,
            "network_sent_mb": 0,
            "network_recv_mb": 0,
            "process_threads": 0,
            "warnings": [],
            "errors": []
        }
        
        self.start_time = time.time()
        self._error_log = []
        self._warning_log = []
    
    def update_connection_status(
        self,
        api_connected: bool,
        websocket_connected: bool
    ) -> None:
        """Update connection status"""
        self.health["api_connected"] = api_connected
        self.health["websocket_connected"] = websocket_connected
        
        if api_connected and websocket_connected:
            self.health["system_status"] = "healthy"
        elif api_connected or websocket_connected:
            self.health["system_status"] = "degraded"
        else:
            self.health["system_status"] = "disconnected"
        
        self._save()
    
    def update_latency(
        self,
        last_latency: float,
        avg_latency: float,
        p99_latency: float
    ) -> None:
        """Update API latency metrics"""
        self.health["last_api_latency_ms"] = last_latency
        self.health["avg_api_latency_ms"] = avg_latency
        self.health["p99_api_latency_ms"] = p99_latency
        
        # Warn on high latency
        if last_latency > 1000:
            self.add_warning(f"High API latency: {last_latency:.0f}ms")
        
        self._save()
    
    def record_error(self, error_type: str, message: str) -> None:
        """Record an error"""
        error = {
            "timestamp": time.time(),
            "type": error_type,
            "message": message
        }
        self._error_log.append(error)
        
        # Keep only last hour
        cutoff = time.time() - 3600
        self._error_log = [e for e in self._error_log if e["timestamp"] > cutoff]
        
        if error_type == "api":
            self.health["api_errors_1h"] = len([e for e in self._error_log if e["type"] == "api"])
        elif error_type == "websocket":
            self.health["websocket_errors_1h"] = len([e for e in self._error_log if e["type"] == "websocket"])
        
        self.health["errors"] = self._error_log[-10:]  # Last 10 errors
        self._save()
    
    def add_warning(self, message: str) -> None:
        """Add a warning"""
        warning = {
            "timestamp": time.time(),
            "message": message
        }
        self._warning_log.append(warning)
        
        # Keep only last hour
        cutoff = time.time() - 3600
        self._warning_log = [w for w in self._warning_log if w["timestamp"] > cutoff]
        
        self.health["warnings"] = self._warning_log[-10:]
        self._save()
    
    def record_trade(self) -> None:
        """Record that a trade occurred"""
        self.health["last_trade_time"] = time.time()
        self._save()
    
    def heartbeat(self) -> None:
        """Record heartbeat"""
        self.health["last_heartbeat"] = time.time()
        self._save()
    
    def update_system_metrics(self) -> None:
        """Update system resource metrics"""
        try:
            # CPU usage
            self.health["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            
            # Memory
            memory = psutil.virtual_memory()
            self.health["memory_percent"] = memory.percent
            self.health["memory_mb"] = memory.used / (1024 * 1024)
            
            # Disk
            disk = psutil.disk_usage("/")
            self.health["disk_percent"] = disk.percent
            
            # Network
            net = psutil.net_io_counters()
            self.health["network_sent_mb"] = net.bytes_sent / (1024 * 1024)
            self.health["network_recv_mb"] = net.bytes_recv / (1024 * 1024)
            
            # Process info
            process = psutil.Process()
            self.health["process_threads"] = process.num_threads()
            
            # Uptime
            self.health["uptime_seconds"] = time.time() - self.start_time
            
            # Check for issues
            if self.health["cpu_percent"] > 90:
                self.add_warning(f"High CPU usage: {self.health['cpu_percent']}%")
            if self.health["memory_percent"] > 90:
                self.add_warning(f"High memory usage: {self.health['memory_percent']}%")
                
        except Exception as e:
            self.record_error("system", str(e))
        
        self._save()
    
    def _save(self) -> None:
        """Save health data to JSON"""
        self.health["last_update"] = time.time()
        self.health["last_update_str"] = datetime.now(UTC_PLUS_1).isoformat()
        
        with open(self.output_path, "w") as f:
            json.dump(self.health, f, indent=2, default=str)
    
    def get_health(self) -> Dict[str, Any]:
        """Get current health data"""
        return self.health.copy()
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return self.health["system_status"] == "healthy"


def load_all_health_data() -> dict:
    """Load all health-related data at once"""
    data = {
        "system": {},
        "api": {},
        "process": {},
        "health_file": None,
        "logs": [],
        "errors": []
    }
    
    try:
        # System metrics
        data["system"]["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        
        memory = psutil.virtual_memory()
        data["system"]["memory_percent"] = memory.percent
        data["system"]["memory_used"] = memory.used
        data["system"]["memory_total"] = memory.total
        
        disk = psutil.disk_usage("/")
        data["system"]["disk_percent"] = disk.percent
        data["system"]["disk_used"] = disk.used
        data["system"]["disk_total"] = disk.total
        
        net = psutil.net_io_counters()
        data["system"]["network_sent"] = net.bytes_sent
        data["system"]["network_recv"] = net.bytes_recv
        
    except Exception as e:
        data["errors"].append(f"System metrics: {e}")
    
    try:
        # Current process info
        process = psutil.Process()
        data["process"]["threads"] = process.num_threads()
        data["process"]["memory"] = process.memory_info().rss
        data["process"]["cpu"] = process.cpu_percent()
        data["process"]["create_time"] = process.create_time()
    except Exception as e:
        data["errors"].append(f"Process info: {e}")
    
    # Load health.json if exists
    health_file = Path("monitoring/health.json")
    if health_file.exists():
        try:
            with open(health_file, "r") as f:
                data["health_file"] = json.load(f)
        except Exception as e:
            data["errors"].append(f"health.json: {e}")
    
    # Check recent log files
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = sorted(logs_dir.glob("*.log"))
        for log_file in log_files[-3:]:  # Last 3 log files
            try:
                stat = log_file.stat()
                data["logs"].append({
                    "name": log_file.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
            except Exception:
                pass
    
    return data


def display_health_dashboard(data: dict, next_refresh: float):
    """Display the health dashboard"""
    clear_screen()
    
    now = get_utc_plus_1_now()
    
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}")
    print("  AML HFT - System Health Monitor")
    print("=" * 60 + f"{Colors.RESET}")
    print(f"  Time: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC+1")
    print(f"  Next refresh in: {int(next_refresh)}s")
    print()
    
    # Display errors if any
    for error in data["errors"]:
        print(f"{Colors.RED}Error: {error}{Colors.RESET}")
    
    # System Resources
    system = data["system"]
    print(f"{Colors.BOLD}💻 System Resources{Colors.RESET}")
    print("-" * 40)
    
    # CPU
    cpu = system.get("cpu_percent", 0)
    cpu_color = Colors.GREEN if cpu < 70 else (Colors.YELLOW if cpu < 90 else Colors.RED)
    print(f"  CPU Usage:        {cpu_color}{cpu:.1f}%{Colors.RESET}")
    
    # Memory
    mem_pct = system.get("memory_percent", 0)
    mem_color = Colors.GREEN if mem_pct < 70 else (Colors.YELLOW if mem_pct < 90 else Colors.RED)
    mem_used = format_bytes(system.get("memory_used", 0))
    mem_total = format_bytes(system.get("memory_total", 0))
    print(f"  Memory:           {mem_color}{mem_pct:.1f}%{Colors.RESET} ({mem_used} / {mem_total})")
    
    # Disk
    disk_pct = system.get("disk_percent", 0)
    disk_color = Colors.GREEN if disk_pct < 80 else (Colors.YELLOW if disk_pct < 95 else Colors.RED)
    disk_used = format_bytes(system.get("disk_used", 0))
    disk_total = format_bytes(system.get("disk_total", 0))
    print(f"  Disk:             {disk_color}{disk_pct:.1f}%{Colors.RESET} ({disk_used} / {disk_total})")
    
    # Network
    net_sent = format_bytes(system.get("network_sent", 0))
    net_recv = format_bytes(system.get("network_recv", 0))
    print(f"  Network Sent:     {net_sent}")
    print(f"  Network Recv:     {net_recv}")
    print()
    
    # Process Info
    process = data["process"]
    print(f"{Colors.BOLD}🔧 AML Process{Colors.RESET}")
    print("-" * 40)
    
    proc_mem = format_bytes(process.get("memory", 0))
    proc_threads = process.get("threads", 0)
    proc_cpu = process.get("cpu", 0)
    
    create_time = process.get("create_time", 0)
    if create_time:
        uptime = time.time() - create_time
        uptime_str = format_uptime(uptime)
    else:
        uptime_str = "N/A"
    
    print(f"  Process Memory:   {proc_mem}")
    print(f"  Process CPU:      {proc_cpu:.1f}%")
    print(f"  Threads:          {proc_threads}")
    print(f"  Uptime:           {uptime_str}")
    print()
    
    # Health File Status
    health = data["health_file"]
    if health:
        print(f"{Colors.BOLD}🩺 Health Status{Colors.RESET}")
        print("-" * 40)
        
        status = health.get("system_status", "unknown")
        if status == "healthy":
            status_color = Colors.GREEN
        elif status == "degraded":
            status_color = Colors.YELLOW
        else:
            status_color = Colors.RED
        
        print(f"  Status:           {status_color}{Colors.BOLD}{status.upper()}{Colors.RESET}")
        
        api_conn = health.get("api_connected", False)
        ws_conn = health.get("websocket_connected", False)
        api_color = Colors.GREEN if api_conn else Colors.RED
        ws_color = Colors.GREEN if ws_conn else Colors.RED
        
        print(f"  API Connected:    {api_color}{'✓ Yes' if api_conn else '✗ No'}{Colors.RESET}")
        print(f"  WebSocket:        {ws_color}{'✓ Yes' if ws_conn else '✗ No'}{Colors.RESET}")
        
        # Latency
        avg_latency = health.get("avg_api_latency_ms", 0)
        p99_latency = health.get("p99_api_latency_ms", 0)
        latency_color = Colors.GREEN if avg_latency < 100 else (Colors.YELLOW if avg_latency < 500 else Colors.RED)
        print(f"  Avg Latency:      {latency_color}{avg_latency:.1f}ms{Colors.RESET}")
        print(f"  P99 Latency:      {p99_latency:.1f}ms")
        
        # Errors
        api_errors = health.get("api_errors_1h", 0)
        ws_errors = health.get("websocket_errors_1h", 0)
        error_color = Colors.GREEN if (api_errors + ws_errors) == 0 else Colors.YELLOW
        print(f"  Errors (1h):      {error_color}API: {api_errors}, WS: {ws_errors}{Colors.RESET}")
        
        # Last trade
        last_trade = health.get("last_trade_time")
        if last_trade:
            trade_ago = time.time() - last_trade
            trade_str = format_uptime(trade_ago) + " ago"
        else:
            trade_str = "None"
        print(f"  Last Trade:       {trade_str}")
        
        # Warnings
        warnings = health.get("warnings", [])
        if warnings:
            print(f"\n  {Colors.YELLOW}Warnings:{Colors.RESET}")
            for w in warnings[-3:]:
                msg = w.get("message", "Unknown") if isinstance(w, dict) else str(w)
                print(f"    ⚠️  {msg}")
        print()
    else:
        print(f"{Colors.YELLOW}No health.json file found{Colors.RESET}")
        print()
    
    # Log Files
    logs = data["logs"]
    if logs:
        print(f"{Colors.BOLD}📋 Recent Log Files{Colors.RESET}")
        print("-" * 40)
        
        for log in logs:
            name = log["name"]
            size = format_bytes(log["size"])
            modified = datetime.fromtimestamp(log["modified"], tz=UTC_PLUS_1).strftime("%H:%M:%S")
            print(f"  {name}: {size} (modified {modified})")
        print()
    
    print(f"{Colors.YELLOW}Press Ctrl+C to exit | Refreshes every 5min (aligned to UTC+1){Colors.RESET}")


async def health_monitor_loop():
    """Health monitoring loop - refreshes every 5 minutes aligned to UTC+1"""
    while True:
        # Load all data at once
        data = load_all_health_data()
        
        # Calculate time until next 5-minute mark
        next_refresh = get_seconds_until_next_5min()
        
        # Display dashboard
        display_health_dashboard(data, next_refresh)
        
        # Wait until next 5-minute mark
        await asyncio.sleep(next_refresh)


if __name__ == "__main__":
    try:
        asyncio.run(health_monitor_loop())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Health monitor stopped{Colors.RESET}")
