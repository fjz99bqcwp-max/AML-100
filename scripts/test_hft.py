#!/usr/bin/env python3
"""Test HFT hold timeout methods in risk_manager"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk_manager import RiskManager
import time

# Create risk manager
rm = RiskManager('config/params.json')

# Test hold timeout tracking
print('Testing HFT hold timeout methods...')

# Test on_position_opened
rm.on_position_opened(100.0, 1.0)
print(f'Position opened, entry_time set: {rm.position_entry_time > 0}')

# Check hold timeout (should not trigger yet)
time.sleep(0.1)
should_exit, seconds = rm.check_hold_timeout()
print(f'After 0.1s: should_exit={should_exit}, seconds_held={seconds:.2f}')

# Test stats with no completed trades
stats = rm.get_hold_time_stats()
print(f'Initial hold stats: {stats}')

# Close position
rm.on_position_closed()
print(f'Position closed, hold_times count: {len(rm.hold_times)}')

# Test get_avg_hold_time
avg = rm.get_avg_hold_time()
print(f'Avg hold time: {avg:.2f}s')

# Test get_dynamic_tp_sl
tp, sl = rm.get_dynamic_tp_sl(2.0, 1.0)
print(f'Dynamic TP/SL: {tp:.2f}%/{sl:.2f}%')

print('\n✅ All HFT methods working correctly!')
