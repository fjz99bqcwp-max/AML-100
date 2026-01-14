#!/usr/bin/env python3
"""
MLA HFT System - Launch Script
Entry point for autonomous trading system
Handles backtests → optimization → tests → live trading
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path (AML-100.py is in the project root, so use .parent only)
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    print(f"✅ Loaded environment from {ENV_FILE}")
else:
    print(f"⚠️ No .env file found at {ENV_FILE}")

# Note: src.main is imported after setup_logging() to ensure colored formatter is applied


# ============================================================
# ANSI Color Codes
# ============================================================
class Colors:
    """ANSI color codes for terminal output"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


def get_timestamp() -> str:
    """Get formatted timestamp for logs"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def log_info(message: str, icon: str = "ℹ️") -> None:
    """Print info message with timestamp and color"""
    ts = get_timestamp()
    print(f"{Colors.DIM}[{ts}]{Colors.RESET} {icon}  {Colors.CYAN}{message}{Colors.RESET}")


def log_success(message: str, icon: str = "✓") -> None:
    """Print success message with timestamp and color"""
    ts = get_timestamp()
    print(f"{Colors.DIM}[{ts}]{Colors.RESET} {Colors.GREEN}{icon}  {message}{Colors.RESET}")


def log_warning(message: str, icon: str = "⚠") -> None:
    """Print warning message with timestamp and color"""
    ts = get_timestamp()
    print(f"{Colors.DIM}[{ts}]{Colors.RESET} {Colors.YELLOW}{icon}  {message}{Colors.RESET}")


def log_error(message: str, icon: str = "✗") -> None:
    """Print error message with timestamp and color"""
    ts = get_timestamp()
    print(f"{Colors.DIM}[{ts}]{Colors.RESET} {Colors.RED}{icon}  {message}{Colors.RESET}")


def log_phase(phase_num: int, phase_name: str) -> None:
    """Print phase header with styling"""
    ts = get_timestamp()
    print()
    print(f"{Colors.DIM}[{ts}]{Colors.RESET} {Colors.BOLD}{Colors.MAGENTA}━━━ Phase {phase_num}: {phase_name} ━━━{Colors.RESET}")


def log_section(title: str) -> None:
    """Print section header"""
    ts = get_timestamp()
    print(f"{Colors.DIM}[{ts}]{Colors.RESET} {Colors.BOLD}{Colors.BLUE}▸ {title}{Colors.RESET}")


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging"""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"mla_hft_{timestamp}.log"
    
    # Custom formatter with colors for console
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with colors"""
        
        FORMATS = {
            logging.DEBUG: f"{Colors.DIM}%(asctime)s{Colors.RESET} {Colors.BLUE}DEBUG{Colors.RESET}    %(name)-20s %(message)s",
            logging.INFO: f"{Colors.DIM}%(asctime)s{Colors.RESET} {Colors.GREEN}INFO{Colors.RESET}     %(name)-20s %(message)s",
            logging.WARNING: f"{Colors.DIM}%(asctime)s{Colors.RESET} {Colors.YELLOW}WARNING{Colors.RESET}  %(name)-20s %(message)s",
            logging.ERROR: f"{Colors.DIM}%(asctime)s{Colors.RESET} {Colors.RED}ERROR{Colors.RESET}    %(name)-20s %(message)s",
            logging.CRITICAL: f"{Colors.DIM}%(asctime)s{Colors.RESET} {Colors.BG_RED}{Colors.WHITE}CRITICAL{Colors.RESET} %(name)-20s %(message)s",
        }
        
        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
            formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
            return formatter.format(record)
    
    # Plain formatter for file
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    
    # File handler without colors
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[console_handler, file_handler]
    )
    
    # Reduce noise from libraries
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)  # Suppress Optuna trial logs


def check_environment() -> bool:
    """Check that environment is properly configured"""
    required_vars = [
        "HYPERLIQUID_WALLET_ADDRESS",
        "HYPERLIQUID_API_SECRET"
    ]
    
    missing = []
    for var in required_vars:
        if not os.environ.get(var):
            missing.append(var)
    
    if missing:
        print("\n❌ Missing required environment variables:")
        for var in missing:
            print(f"   - {var}")
        print("\nPlease set these in your .env file or environment.")
        return False
    
    return True


def check_dependencies() -> bool:
    """Verify all dependencies are installed"""
    required = [
        "torch",
        "pandas",
        "numpy",
        "websockets",
        "aiohttp"
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("\n❌ Missing Python packages:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nRun: pip install -r requirements.txt")
        return False
    
    return True


def load_env_file() -> None:
    """Load environment variables from .env file"""
    env_file = PROJECT_ROOT / ".env"
    
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


async def run_backtest_only(days: int = 30, data_source: str = "xyz100") -> None:
    """Run backtest only without live trading
    
    Args:
        days: Number of days of historical data
        data_source: Data source - xyz100, spx, btc, synthetic, or hybrid
    """
    from src.main import AMLHFTSystem
    system = AMLHFTSystem()
    
    try:
        await system.setup()
        log_phase(1, f"Backtest ({data_source.upper()} data)")
        
        # Get appropriate data based on source
        historical_data = None
        if data_source == "hybrid":
            log_info(f"Generating hybrid data (70% real SPX + 30% synthetic, {days} days)")
            historical_data = await system._data_fetcher.generate_hybrid_data(days, real_weight=0.7)
        elif data_source == "spx":
            log_info("Using SPX fallback data for backtest")
            historical_data = await system._data_fetcher.get_fallback_data(days)
        elif data_source == "synthetic":
            log_info(f"Generating synthetic SPX data ({days} days)")
            historical_data = system._data_fetcher.generate_synthetic_spx(days)
        elif data_source == "btc":
            log_info("Using BTC data for backtest")
            historical_data = await system._data_fetcher.fetch_historical_klines("BTC", "1m", days)
        # else: xyz100 - let run_backtest fetch normally
        
        # Run backtest with provided data (if any)
        results = await system.run_backtest(days=days, historical_data=historical_data)
        
        # Display results
        print()
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}  BACKTEST RESULTS ({data_source.upper()}){Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        
        ret_color = Colors.GREEN if results.get('total_return_pct', 0) > 0 else Colors.RED
        print(f"  {'Total Return:':<18} {ret_color}{results.get('total_return_pct', 0):>10.2f}%{Colors.RESET}")
        print(f"  {'Sharpe Ratio:':<18} {results.get('sharpe_ratio', 0):>10.2f}")
        print(f"  {'Max Drawdown:':<18} {results.get('max_drawdown_pct', 0):>10.2f}%")
        print(f"  {'Win Rate:':<18} {results.get('win_rate', 0):>10.1f}%")
        print(f"  {'Profit Factor:':<18} {results.get('profit_factor', 0):>10.2f}")
        print(f"  {'Total Trades:':<18} {results.get('total_trades', 0):>10}")
        
        obj_met = results.get('objectives_met')
        obj_color = Colors.GREEN if obj_met else Colors.RED
        obj_icon = "✓ YES" if obj_met else "✗ NO"
        print(f"  {'Objectives Met:':<18} {obj_color}{obj_icon:>10}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        
    finally:
        await system.shutdown()


async def run_optimization_only(n_trials: int = 50) -> None:
    """Run optimization only"""
    from src.main import AMLHFTSystem
    system = AMLHFTSystem()
    
    try:
        await system.setup()
        log_phase(2, "Parameter Optimization")
        results = await system.run_optimization(n_trials=n_trials)
        
        # Display results
        print()
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}  OPTIMIZATION RESULTS{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        
        imp_color = Colors.GREEN if results.get('improvement', 0) > 0 else Colors.YELLOW
        print(f"  {'Improvement:':<18} {imp_color}{results.get('improvement', 0):>10.2f}%{Colors.RESET}")
        print(f"  {'Best Score:':<18} {results.get('best_score', 0):>10.4f}")
        print(f"  {'Iterations:':<18} {results.get('iterations', 0):>10}")
        print()
        print(f"  {Colors.BOLD}Optimized Parameters:{Colors.RESET}")
        for k, v in results.get('optimized_params', {}).items():
            if isinstance(v, float):
                print(f"    {k:<20} {Colors.GREEN}{v:>10.4f}{Colors.RESET}")
            else:
                print(f"    {k:<20} {Colors.GREEN}{v:>10}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        
    finally:
        await system.shutdown()


async def run_training_only(days: int = 30, epochs: int = 100, quick: bool = False) -> None:
    """Run model training only"""
    from src.main import AMLHFTSystem
    system = AMLHFTSystem()
    
    try:
        await system.setup()
        assert system.data_fetcher is not None
        assert system.ml_model is not None
        
        log_phase(3, "ML Model Training")
        
        if quick:
            # Use quick training for cycle retrains
            df = await system.data_fetcher.get_training_data(
                symbol=system.api_config["symbol"],
                interval="1m",
                days=days
            )
            results = await system.ml_model.quick_train(df, epochs=epochs)
        else:
            results = await system.train_model(days=days, epochs=epochs)
        
        # Display results
        print()
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}  TRAINING RESULTS{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        
        mode_str = "Quick Train" if quick else "Full Train"
        print(f"  {'Mode:':<18} {Colors.MAGENTA}{mode_str:>10}{Colors.RESET}")
        print(f"  {'Epochs:':<18} {results.get('epochs', 0):>10}/{results.get('epochs_requested', epochs)}")
        
        reward = results.get('best_reward', 0)
        reward_color = Colors.GREEN if reward > 0 else Colors.YELLOW
        print(f"  {'Best Reward:':<18} {reward_color}{reward:>10.4f}{Colors.RESET}")
        print(f"  {'Training Time:':<18} {results.get('training_time', 0):>10.2f}s")
        print(f"  {'Avg Epoch Time:':<18} {results.get('avg_epoch_time', 0):>10.2f}s")
        
        early = results.get('early_stopped')
        early_color = Colors.GREEN if early else Colors.DIM
        print(f"  {'Early Stopped:':<18} {early_color}{'Yes' if early else 'No':>10}{Colors.RESET}")
        print(f"  {'Final LR:':<18} {results.get('final_lr', 0):>10.6f}")
        print(f"  {'Features:':<18} {results.get('feature_count', 0):>10}")
        print(f"  {'Samples:':<18} {results.get('sample_count', 0):>10}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        
    finally:
        await system.shutdown()


async def run_profiling(days: int = 7, num_batches: int = 50) -> None:
    """Run training profiler to identify bottlenecks"""
    from src.main import AMLHFTSystem
    system = AMLHFTSystem()
    
    try:
        await system.setup()
        assert system.data_fetcher is not None
        assert system.ml_model is not None
        
        log_section("Running Training Profiler")
        
        # Fetch training data
        df = await system.data_fetcher.get_training_data(
            symbol=system.api_config["symbol"],
            interval="1m",
            days=days
        )
        
        results = system.ml_model.profile_training(
            df,
            num_batches=num_batches,
            output_path="profiling/trace.json"
        )
        
        # Display results
        print()
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}  PROFILING RESULTS{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        print(f"  {'Batches:':<18} {results.get('num_batches', 0):>10}")
        print(f"  {'Device:':<18} {Colors.MAGENTA}{results.get('device', 'unknown'):>10}{Colors.RESET}")
        print(f"  {'Avg Forward:':<18} {results.get('avg_forward_ms', 0):>10.2f}ms")
        print(f"  {'Avg Backward:':<18} {results.get('avg_backward_ms', 0):>10.2f}ms")
        print(f"  {'Avg Total:':<18} {results.get('avg_total_ms', 0):>10.2f}ms")
        print(f"  {'Trace File:':<18} {results.get('trace_path', 'N/A')}")
        print()
        print(f"  {Colors.DIM}View trace with: chrome://tracing{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
        
    finally:
        await system.shutdown()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MLA HFT - Autonomous Machine Learning Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py                       # Run full autonomous system
  python launch.py --backtest            # Run backtest only
  python launch.py --backtest --data spx  # Backtest with SPX fallback data
  python launch.py --optimize            # Run optimization only
  python launch.py --train               # Run training only
  python launch.py --train --epochs 10   # Quick train with 10 epochs
  python launch.py --phase 3 --epochs 50 # Run Phase 3 training only
  python launch.py --quick-train         # Cycle retrain (<2 min)
  python launch.py --profile             # Profile training bottlenecks
  python launch.py --days 60             # Use 60 days of data
  python launch.py --hft                 # Enable HFT mode (tighter TP/SL, shorter holds)
  python launch.py --log-level DEBUG     # Enable debug logging
        """
    )
    
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest only"
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["xyz100", "spx", "btc", "synthetic", "hybrid"],
        default="xyz100",
        help="Data source for backtest: xyz100 (default), spx, btc, synthetic, or hybrid (70%% real+30%% synthetic)"
    )
    parser.add_argument(
        "--hft",
        action="store_true",
        help="Enable HFT mode with tighter TP/SL and shorter hold times"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run optimization only"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training only"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific phase only (1=backtest, 2=optimize, 3=train, 4=validate, 5=live)"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run training profiler to identify bottlenecks"
    )
    parser.add_argument(
        "--quick-train",
        action="store_true",
        help="Run quick training (reduced epochs for cycle retrains)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of historical data (default: 30)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Optimization trials (default: 50)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip environment and dependency checks"
    )
    
    args = parser.parse_args()
    
    # Setup logging first (before importing main to apply ColoredFormatter)
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Import after logging is configured
    from src.main import AMLHFTSystem, main as run_main
    
    # HFT mode: override params if --hft flag is set
    if args.hft:
        import json
        params_path = PROJECT_ROOT / "config" / "params.json"
        with open(params_path) as f:
            params = json.load(f)
        
        # Enable HFT mode with RSI mean reversion settings
        # Use time-based exit (60 bars) instead of TP/SL
        params["trading"]["hft_mode"] = True
        params["trading"]["take_profit_pct"] = 99.0  # Disabled - use time exit
        params["trading"]["stop_loss_pct"] = 99.0   # Disabled - use time exit
        params["trading"]["max_hold_seconds"] = 3600  # 60 minutes (60 bars)
        params["trading"]["target_hold_seconds"] = 3600
        
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
        log_info("HFT mode enabled: RSI mean reversion with 60-bar time exit")
    
    # Print banner
    start_time = datetime.now(timezone.utc)
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  AML-100 HFT - Machine Learning Autonomous Trading System{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  Hyperliquid XYZ100-USDC{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"  {Colors.DIM}Started:{Colors.RESET}  {Colors.WHITE}{start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC{Colors.RESET}")
    print(f"  {Colors.DIM}Mode:{Colors.RESET}     {Colors.WHITE}{_get_mode_name(args)}{Colors.RESET}")
    print(f"  {Colors.DIM}Data:{Colors.RESET}     {Colors.WHITE}{args.data.upper()}{Colors.RESET}")
    print(f"  {Colors.DIM}HFT:{Colors.RESET}      {Colors.WHITE}{'ENABLED' if args.hft else 'DISABLED'}{Colors.RESET}")
    print(f"  {Colors.DIM}Log:{Colors.RESET}      {Colors.WHITE}{args.log_level}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print()
    
    # Load environment
    load_env_file()
    
    # Pre-flight checks
    if not args.skip_checks:
        log_section("Pre-flight Checks")
        
        if not check_dependencies():
            log_error("Dependency check failed")
            sys.exit(1)
        log_success("Dependencies verified")
        
        if not check_environment():
            log_error("Environment check failed")
            sys.exit(1)
        log_success("Environment configured")
    
    log_success("All checks passed")
    print()
    
    # Run appropriate mode
    try:
        if args.profile:
            log_info("Running training profiler")
            asyncio.run(run_profiling(args.days))
        
        elif args.quick_train:
            quick_epochs = 10  # Default for quick train
            log_info(f"Running quick training mode ({quick_epochs} epochs)")
            asyncio.run(run_training_only(args.days, quick_epochs, quick=True))
        
        elif args.phase:
            log_info(f"Running Phase {args.phase} only")
            if args.phase == 1:
                asyncio.run(run_backtest_only(args.days, args.data))
            elif args.phase == 2:
                asyncio.run(run_optimization_only(args.trials))
            elif args.phase == 3:
                asyncio.run(run_training_only(args.days, args.epochs))
            elif args.phase == 4:
                # Validation = backtest after training
                asyncio.run(run_backtest_only(args.days, args.data))
            elif args.phase == 5:
                log_info("Phase 5 (live) requires full autonomous mode")
                asyncio.run(run_main())
        
        elif args.backtest:
            log_info(f"Running backtest mode ({args.days} days, {args.data} data)")
            asyncio.run(run_backtest_only(args.days, args.data))
            
        elif args.optimize:
            log_info(f"Running optimization mode ({args.trials} trials)")
            asyncio.run(run_optimization_only(args.trials))
            
        elif args.train:
            log_info(f"Running training mode ({args.epochs} epochs)")
            asyncio.run(run_training_only(args.days, args.epochs))
            
        else:
            log_info("Running full autonomous mode")
            asyncio.run(run_main())
            
    except KeyboardInterrupt:
        print()
        log_warning("Shutdown requested by user")
        logger.info("Shutdown requested by user")
        
    except Exception as e:
        log_error(f"Fatal error: {e}")
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
    
    # Print completion
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"  {Colors.DIM}Stopped:{Colors.RESET}  {Colors.WHITE}{end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC{Colors.RESET}")
    print(f"  {Colors.DIM}Duration:{Colors.RESET} {Colors.WHITE}{duration:.1f}s{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print()


def _get_mode_name(args) -> str:
    """Get human-readable mode name from args"""
    if args.profile:
        return "Profiling"
    elif args.quick_train:
        return "Quick Train"
    elif args.phase:
        phases = {1: "Backtest", 2: "Optimize", 3: "Train", 4: "Validate", 5: "Live"}
        return f"Phase {args.phase} ({phases.get(args.phase, '')})"
    elif args.backtest:
        return "Backtest Only"
    elif args.optimize:
        return "Optimization Only"
    elif args.train:
        return "Training Only"
    else:
        return "Full Autonomous"


if __name__ == "__main__":
    main()
