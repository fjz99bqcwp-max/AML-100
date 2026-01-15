#!/usr/bin/env python3
"""
AML-100 Launch Script - Autonomous HFT System
Zurich CET timezone | Apple M4 MPS Optimized
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Zurich CET timezone (UTC+1, UTC+2 in summer)
CET = timezone(timedelta(hours=1))

# ANSI Colors - Optimized for dark terminal themes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Dark-theme optimized - brighter, more visible colors
    RED = "\033[38;5;203m"        # Soft red (not too harsh)
    GREEN = "\033[38;5;114m"      # Soft green
    YELLOW = "\033[38;5;221m"     # Warm yellow
    BLUE = "\033[38;5;111m"       # Bright blue
    MAGENTA = "\033[38;5;176m"    # Soft magenta
    CYAN = "\033[38;5;80m"        # Teal cyan
    ORANGE = "\033[38;5;215m"     # Orange for force trades
    GRAY = "\033[38;5;245m"       # Dim gray for timestamps
    WHITE = "\033[38;5;255m"      # Bright white for important


def get_cet_timestamp() -> str:
    """Get current timestamp in Zurich CET (time-only for cleaner logs)"""
    return datetime.now(CET).strftime("%H:%M:%S")


def log_info(msg: str) -> None:
    """Log info message with dark-theme optimized format"""
    print(f"{Colors.GRAY}[{get_cet_timestamp()}]{Colors.RESET} {Colors.CYAN}[INFO]{Colors.RESET} {Colors.DIM}{'launch':10}{Colors.RESET} {Colors.CYAN}{msg}{Colors.RESET}")


def log_success(msg: str) -> None:
    """Log success message with dark-theme optimized format"""
    print(f"{Colors.GRAY}[{get_cet_timestamp()}]{Colors.RESET} {Colors.GREEN}[INFO]{Colors.RESET} {Colors.DIM}{'launch':10}{Colors.RESET} {Colors.GREEN}{msg}{Colors.RESET}")


def log_warning(msg: str) -> None:
    """Log warning message with dark-theme optimized format"""
    print(f"{Colors.GRAY}[{get_cet_timestamp()}]{Colors.RESET} {Colors.YELLOW}[WARN]{Colors.RESET} {Colors.DIM}{'launch':10}{Colors.RESET} {Colors.YELLOW}{msg}{Colors.RESET}")


def log_error(msg: str) -> None:
    """Log error message with dark-theme optimized format"""
    print(f"{Colors.GRAY}[{get_cet_timestamp()}]{Colors.RESET} {Colors.RED}[ERR!]{Colors.RESET} {Colors.DIM}{'launch':10}{Colors.RESET} {Colors.RED}{msg}{Colors.RESET}")


def log_phase(num: int, name: str) -> None:
    """Log phase transition with dark-theme optimized format"""
    print(f"\n{Colors.GRAY}[{get_cet_timestamp()}]{Colors.RESET} {Colors.BOLD}{Colors.MAGENTA}--- Phase {num}: {name} ---{Colors.RESET}")


class CETFormatter(logging.Formatter):
    """Custom formatter with CET timezone and dark-theme optimized colors"""
    
    LEVEL_COLORS = {
        'DEBUG': Colors.GRAY,
        'INFO': Colors.CYAN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.RED + Colors.BOLD,
    }
    
    LEVEL_TAGS = {
        'DEBUG': 'DBUG',
        'INFO': 'INFO',
        'WARNING': 'WARN',
        'ERROR': 'ERR!',
        'CRITICAL': 'CRIT',
    }
    
    def format(self, record):
        # Get CET timestamp
        ct = datetime.fromtimestamp(record.created, tz=CET)
        timestamp = ct.strftime("%H:%M:%S")  # Shorter time-only format
        
        # Get color and tag for level
        level_color = self.LEVEL_COLORS.get(record.levelname, Colors.RESET)
        level_tag = self.LEVEL_TAGS.get(record.levelname, 'INFO')
        
        # Shorten module name
        module = record.name
        if module.startswith('src.'):
            module = module[4:]  # Remove 'src.' prefix
        module = module[:15].ljust(15)  # Fixed width for hyperliquid_api (15 chars)
        
        # Format message - strip icons for cleaner output
        msg = record.getMessage()
        
        # Context-aware message coloring for dark theme
        msg_color = Colors.RESET
        if 'executed' in msg.lower() or 'success' in msg.lower():
            msg_color = Colors.GREEN
        elif 'FORCE' in msg:
            msg_color = Colors.ORANGE
        elif 'cycle' in msg.lower():
            msg_color = Colors.CYAN
        elif 'BUY' in msg or 'SELL' in msg:
            msg_color = Colors.BLUE
        elif 'skip' in msg.lower():
            msg_color = Colors.YELLOW
        elif 'HOLD' in msg:
            msg_color = Colors.GRAY
        elif 'halted' in msg.lower() or 'stop' in msg.lower() or 'error' in msg.lower():
            msg_color = Colors.RED
        elif 'profit' in msg.lower() or 'PnL' in msg:
            msg_color = Colors.GREEN if '+' in msg else Colors.RED if '-' in msg else msg_color
        
        # Console output: [time] [LEVEL] module  message
        formatted = f"{Colors.GRAY}[{timestamp}]{Colors.RESET} {level_color}[{level_tag}]{Colors.RESET} {Colors.DIM}{module}{Colors.RESET} {msg_color}{msg}{Colors.RESET}"
        
        return formatted


class FileFormatter(logging.Formatter):
    """Plain formatter for file output with CET timezone - no icons, clean text"""
    
    # Unicode icons to strip from log messages
    ICONS_TO_STRIP = [
        '✅', '❌', '⚠️', '🔄', '📈', '📉', '🚦', '🟡', '🟢', '🔴',
        '💀', '🔌', '⏩', '🎯', '💰', '📊', 'ℹ️', '✓', '✗', '⚠',
        '│', '·', '━', '▶', '◀', '●', '○', '◉', '◎', '★', '☆',
        '🔒', '⏱️', '🔋', '💡', '🚀', '⚡', '🎲', '🔧', '📝', '🔍'
    ]
    
    LEVEL_TAGS = {
        'DEBUG': 'DBUG',
        'INFO': 'INFO',
        'WARNING': 'WARN',
        'ERROR': 'ERR!',
        'CRITICAL': 'CRIT',
    }
    
    def format(self, record):
        ct = datetime.fromtimestamp(record.created, tz=CET)
        timestamp = ct.strftime("%Y-%m-%d %H:%M:%S")
        
        module = record.name
        if module.startswith('src.'):
            module = module[4:]
        module = module[:12].ljust(12)
        
        level = self.LEVEL_TAGS.get(record.levelname, 'INFO')
        msg = record.getMessage()
        
        # Strip all icons from file log output
        for icon in self.ICONS_TO_STRIP:
            msg = msg.replace(icon, '')
        
        # Clean up extra whitespace from icon removal
        msg = ' '.join(msg.split())
        
        return f"[{timestamp}] [{level}] {module} {msg}"


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with CET timezone and colored output"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now(CET).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"aml100_{timestamp}.log"
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CETFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler without colors
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(FileFormatter())
    root_logger.addHandler(file_handler)
    
    # Reduce noise from libraries
    for lib in ["websockets", "asyncio", "urllib3", "optuna", "httpx", "httpcore"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def load_env() -> None:
    """Load environment variables from .env"""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


def reset_to_defaults() -> None:
    """Reset params.json and objectives.json to SPEC defaults"""
    log_info("Resetting to SPEC defaults...")
    
    # SPEC defaults for params.json
    params = {
        "trading": {
            "take_profit_pct": 0.5,
            "stop_loss_pct": 0.25,
            "position_size_pct": 40.0,
            "leverage": 1,
            "max_leverage": 5,
            "dynamic_tp_sl": False,
            "vol_scaling": False,
            "max_hold_seconds": 3600,
            "hft_mode": False
        },
        "ml_model": {
            "type": "hybrid_lstm_dqn",
            "lstm_hidden_size": 128,
            "lstm_num_layers": 2,
            "dqn_hidden_size": 256,
            "enable_torch_compile": True,
            "learning_rate": 0.0001,
            "batch_size": 32,
            "epochs": 180,
            "sequence_length": 60,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.997,
            "memory_size": 10000,
            "target_update_freq": 10,
            "dropout": 0.2,
            "early_stop_patience": 30,
            "num_workers": 6,
            "use_supervised_guidance": False,
            "use_fgsm": False,
            "rl_algo": "DQN"
        },
        "backtest": {
            "commission_pct": 0.0005,
            "slippage_pct": 0.05,
            "latency_ms_min": 5,
            "latency_ms_max": 15,
            "enable_multiprocessing": True,
            "max_workers": 6
        },
        "risk": {
            "max_daily_loss_pct": 2.0,
            "max_position_pct": 40.0
        }
    }
    
    # SPEC defaults for objectives.json
    objectives = {
        "monthly_return_pct_min": 15.0,
        "sharpe_ratio_min": 1.5,
        "drawdown_max": 5.0,
        "auto_stop_drawdown": 5.0,
        "starting_capital_backtest": 1000,
        "starting_capital_live": 1469,
        "backtest_interval_hours": 1,
        "cycle_duration_seconds": 300,
        "wallet_address": "0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584"
    }
    
    with open(PROJECT_ROOT / "config" / "params.json", "w") as f:
        json.dump(params, f, indent=2)
    
    with open(PROJECT_ROOT / "config" / "objectives.json", "w") as f:
        json.dump(objectives, f, indent=2)
    
    log_success("Reset params.json and objectives.json to SPEC defaults")


def check_environment() -> bool:
    """Verify environment setup"""
    required = ["HYPERLIQUID_WALLET_ADDRESS", "HYPERLIQUID_API_SECRET"]
    missing = [v for v in required if not os.environ.get(v)]
    
    if missing:
        log_error(f"Missing env vars: {', '.join(missing)}")
        return False
    return True


async def run_backtest(days: int = 180, data_source: str = "hybrid") -> dict:
    """Run Phase 1: Backtest"""
    from src.main import AMLHFTSystem
    
    system = AMLHFTSystem()
    await system.setup()
    
    try:
        log_phase(1, f"Backtest ({data_source.upper()}, {days} days)")
        
        # Get hybrid data (90% real SPX + 10% synthetic)
        if data_source == "hybrid":
            historical_data = await system._data_fetcher.generate_hybrid_data(days, real_weight=0.9)
        else:
            historical_data = None
        
        results = await system.run_backtest(days=days, historical_data=historical_data)
        
        # Display results
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}  BACKTEST RESULTS ({data_source.upper()}){Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 50}{Colors.RESET}")
        
        ret_color = Colors.GREEN if results.get('total_return_pct', 0) > 0 else Colors.RED
        print(f"  Total Return:    {ret_color}{results.get('total_return_pct', 0):>8.2f}%{Colors.RESET}")
        print(f"  Sharpe Ratio:    {results.get('sharpe_ratio', 0):>8.2f}")
        print(f"  Max Drawdown:    {results.get('max_drawdown_pct', 0):>8.2f}%")
        print(f"  Win Rate:        {results.get('win_rate', 0):>8.1f}%")
        print(f"  Profit Factor:   {results.get('profit_factor', 0):>8.2f}")
        print(f"  Total Trades:    {results.get('total_trades', 0):>8}")
        
        obj_met = results.get('objectives_met', False)
        obj_color = Colors.GREEN if obj_met else Colors.RED
        print(f"  Objectives Met:  {obj_color}{'YES' if obj_met else 'NO':>8}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 50}{Colors.RESET}\n")
        
        return results
    finally:
        await system.shutdown()


async def run_optimization(n_trials: int = 50) -> dict:
    """Run Phase 2: Bayesian Parameter Optimization"""
    from src.main import AMLHFTSystem
    
    system = AMLHFTSystem()
    await system.setup()
    
    try:
        log_phase(2, "Bayesian Optimization")
        results = await system.run_optimization(n_trials=n_trials)
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}  OPTIMIZATION RESULTS{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 50}{Colors.RESET}")
        print(f"  Improvement:     {results.get('improvement', 0):>8.2f}%")
        print(f"  Best Score:      {results.get('best_score', 0):>8.4f}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 50}{Colors.RESET}\n")
        
        return results
    finally:
        await system.shutdown()


async def run_training(days: int = 180, epochs: int = 180) -> dict:
    """Run Phase 3: Model Training"""
    from src.main import AMLHFTSystem
    
    system = AMLHFTSystem()
    await system.setup()
    
    try:
        log_phase(3, f"ML Training ({epochs} epochs max)")
        results = await system.train_model(days=days, epochs=epochs)
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}  TRAINING RESULTS{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 50}{Colors.RESET}")
        print(f"  Epochs:          {results.get('epochs', 0):>8}")
        print(f"  Best Reward:     {results.get('best_reward', 0):>8.4f}")
        print(f"  Training Time:   {results.get('training_time', 0):>8.1f}s")
        print(f"  Early Stopped:   {'Yes' if results.get('early_stopped') else 'No':>8}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 50}{Colors.RESET}\n")
        
        return results
    finally:
        await system.shutdown()


async def run_validation(days: int = 30) -> dict:
    """Run Phase 4: Validation Backtest"""
    log_phase(4, "Validation")
    return await run_backtest(days=days, data_source="hybrid")


async def run_live(wallet: str) -> None:
    """Run Phase 5: Live Trading"""
    from src.main import AMLHFTSystem
    
    system = AMLHFTSystem()
    await system.setup()
    
    try:
        log_phase(5, f"Live Trading (wallet: {wallet[:10]}...)")
        await system.run_live_trading()
    finally:
        await system.shutdown()


async def run_autonomous(asset: str, wallet: str, cycle_hours: int = 1, skip_backtest: bool = False) -> None:
    """
    Full autonomous mode: backtest → optimize → train → validate → live
    Loops hourly with automatic parameter adjustment
    
    Enhanced with:
    - Model bias detection (>90% HOLD triggers retrain even with skip_backtest)
    - Synthetic fallback awareness
    """
    from src.main import AMLHFTSystem
    
    log_info(f"Starting AUTONOMOUS MODE for {asset}")
    log_info(f"   Wallet: {wallet}")
    log_info(f"   Cycle: Every {cycle_hours} hour(s)")
    log_info(f"   Timezone: Zurich CET")
    if skip_backtest:
        log_info(f"   Skip Backtest: Enabled (bias check still enforced)")
    
    cycle = 0
    running = True
    
    def handle_shutdown(sig, frame):
        nonlocal running
        log_warning("Shutdown signal received...")
        running = False
    
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    while running:
        cycle += 1
        cycle_start = datetime.now(CET)
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}  AUTONOMOUS CYCLE {cycle} - {cycle_start.strftime('%Y-%m-%d %H:%M:%S CET')}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'═' * 60}{Colors.RESET}\n")
        
        try:
            # Initialize system for bias check
            system = AMLHFTSystem()
            await system.setup()
            
            # BIAS CHECK: Even with skip_backtest, check model bias before live trading
            needs_retrain = False
            if hasattr(system, '_ml_model') and system._ml_model:
                bias_status = system._ml_model.check_bias_status()
                if bias_status.get('needs_retrain'):
                    log_warning(f"MODEL BIAS DETECTED: {bias_status.get('reason')}")
                    log_warning(f"   Hold%: {bias_status.get('hold_pct', 0):.1%}, Severity: {bias_status.get('bias_severity')}")
                    needs_retrain = True
                elif bias_status.get('bias_severity') in ['high', 'moderate']:
                    log_info(f"Model bias check: {bias_status.get('bias_severity')} ({bias_status.get('hold_pct', 0):.1%} HOLD)")
            
            # Check if skipping backtest
            if skip_backtest and not needs_retrain:
                log_info("Skipping backtest - proceeding to live trading")
                objectives_met = True
            elif skip_backtest and needs_retrain:
                log_warning("Skip-backtest overridden due to model bias - forcing retrain")
                objectives_met = False
                
                # Force retrain
                log_phase(3, "Forced Retrain (bias detected)")
                training_results = await run_training(days=180, epochs=200)
                
                # Reset bias flags
                if hasattr(system, '_ml_model') and system._ml_model:
                    system._ml_model.reset_bias_flags()
                
                # Validate after retrain
                validation = await run_validation(days=30)
                objectives_met = validation.get('objectives_met', False)
            else:
                # Normal flow: Phase 1: Backtest with hybrid data
                await system.shutdown()  # Close for backtest to reopen
                backtest_results = await run_backtest(days=180, data_source="hybrid")
                
                objectives_met = backtest_results.get('objectives_met', False)
                sharpe = backtest_results.get('sharpe_ratio', 0)
                drawdown = backtest_results.get('max_drawdown_pct', 100)
                
                if not objectives_met:
                    log_warning(f"Objectives not met (Sharpe={sharpe:.2f}, DD={drawdown:.2f}%)")
                    
                    # Phase 2: Optimize parameters
                    await run_optimization(n_trials=30)
                    
                    # Phase 3: Retrain model
                    await run_training(days=180, epochs=180)
                    
                    # Phase 4: Validate
                    validation = await run_validation(days=30)
                    objectives_met = validation.get('objectives_met', False)
                    
                # Reinitialize for live trading
                system = AMLHFTSystem()
                await system.setup()
            
            if objectives_met:
                log_success("Objectives met! Starting live trading...")
                
                # Phase 5: Live trading (runs for cycle_hours)
                try:
                    await asyncio.wait_for(
                        system.run_live_trading(),
                        timeout=cycle_hours * 3600
                    )
                except asyncio.TimeoutError:
                    log_info(f"Cycle {cycle} complete, starting next cycle...")
                finally:
                    await system.shutdown()
            else:
                log_warning("Objectives still not met after optimization. Retrying next cycle...")
                await asyncio.sleep(60)  # Wait 1 minute before retry
                
        except Exception as e:
            log_error(f"Cycle {cycle} error: {e}")
            await asyncio.sleep(60)
    
    log_info("Autonomous mode stopped")


def main():
    parser = argparse.ArgumentParser(
        description="AML-100 - Autonomous ML Trading System for HyperLiquid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/launch.py --mode backtest --days 180
  python scripts/launch.py --mode train --epochs 180
  python scripts/launch.py --mode autonomous --asset XYZ100 --wallet 0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584
  caffeinate python scripts/launch.py --mode autonomous --asset XYZ100
        """
    )
    
    parser.add_argument("--mode", type=str, required=True,
                        choices=["backtest", "optimize", "train", "validate", "live", "autonomous"],
                        help="Execution mode")
    parser.add_argument("--asset", type=str, default="XYZ100",
                        help="Trading asset (default: XYZ100)")
    parser.add_argument("--wallet", type=str, 
                        default="0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584",
                        help="Wallet address for live trading")
    parser.add_argument("--days", type=int, default=180,
                        help="Days of historical data (default: 180)")
    parser.add_argument("--epochs", type=int, default=180,
                        help="Training epochs (default: 180)")
    parser.add_argument("--trials", type=int, default=50,
                        help="Optimization trials (default: 50)")
    parser.add_argument("--cycle-hours", type=int, default=1,
                        help="Hours per autonomous cycle (default: 1)")
    parser.add_argument("--reset-defaults", action="store_true",
                        help="Reset to SPEC defaults before running")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip backtest and go directly to live trading")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.log_level)
    load_env()
    
    if args.reset_defaults:
        reset_to_defaults()
    
    # Print banner
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  AML-100 - Autonomous ML Trading for HyperLiquid{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  Asset: {args.asset} | Mode: {args.mode.upper()}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"  {Colors.DIM}Started:{Colors.RESET} {get_cet_timestamp()}")
    print(f"  {Colors.DIM}Wallet:{Colors.RESET}  {args.wallet[:20]}...")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}\n")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run selected mode
    try:
        if args.mode == "backtest":
            asyncio.run(run_backtest(days=args.days))
        elif args.mode == "optimize":
            asyncio.run(run_optimization(n_trials=args.trials))
        elif args.mode == "train":
            asyncio.run(run_training(days=args.days, epochs=args.epochs))
        elif args.mode == "validate":
            asyncio.run(run_validation(days=args.days))
        elif args.mode == "live":
            asyncio.run(run_live(wallet=args.wallet))
        elif args.mode == "autonomous":
            asyncio.run(run_autonomous(
                asset=args.asset, 
                wallet=args.wallet, 
                cycle_hours=args.cycle_hours,
                skip_backtest=args.skip_backtest
            ))
            
    except KeyboardInterrupt:
        log_warning("Interrupted by user")
    except Exception as e:
        log_error(f"Fatal error: {e}")
        sys.exit(1)
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"  {Colors.DIM}Stopped:{Colors.RESET} {get_cet_timestamp()}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}\n")


if __name__ == "__main__":
    main()
