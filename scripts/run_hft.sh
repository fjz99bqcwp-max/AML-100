#!/bin/bash
# Run AML-100 HFT mode in background

cd /Users/nheosdisplay/VSC/AML/AML-100

# Kill any existing AML-100 processes
pkill -f "AML-100.py" 2>/dev/null
sleep 1

# Start AML-100 in HFT mode with proper signal handling
nohup .venv/bin/python AML-100.py --hft > logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "Started AML-100 HFT with PID $PID"
echo "Log: logs/run_$(date +%Y%m%d_%H%M%S).log"
echo "To stop: kill $PID"

# Disown from shell to prevent signal propagation
disown $PID
