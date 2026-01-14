#!/bin/bash
# AML-100 HFT Launch Script
# Autonomous trading system for HyperLiquid XYZ100-USDC
# Optimized for Apple M4 Mac (24GB RAM)

set -e

# Project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️ No .venv found. Creating..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

# Load environment
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✅ Environment loaded"
fi

# Default mode
MODE="${1:-autonomous}"
ASSET="${2:-XYZ100}"
WALLET="${3:-0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584}"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  AML-100 - Autonomous ML Trading for HyperLiquid"
echo "  Mode: $MODE | Asset: $ASSET"
echo "  Wallet: ${WALLET:0:20}..."
echo "═══════════════════════════════════════════════════════"
echo ""

# Run with caffeinate to prevent sleep
if [ "$MODE" == "autonomous" ]; then
    echo "🚀 Starting AUTONOMOUS MODE (hourly cycles)..."
    echo "   Press Ctrl+C to stop"
    echo ""
    caffeinate -i python scripts/launch.py --mode autonomous --asset "$ASSET" --wallet "$WALLET" --reset-defaults
elif [ "$MODE" == "backtest" ]; then
    echo "📊 Running backtest..."
    python scripts/launch.py --mode backtest --days 180 --reset-defaults
elif [ "$MODE" == "train" ]; then
    echo "🧠 Training model..."
    python scripts/launch.py --mode train --epochs 180 --reset-defaults
else
    echo "Running mode: $MODE"
    python scripts/launch.py --mode "$MODE" --asset "$ASSET" --wallet "$WALLET"
fi
