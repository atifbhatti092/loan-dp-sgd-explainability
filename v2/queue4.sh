#!/bin/bash
cd "$(dirname "$0")"
sleep 3
while pgrep -f "queue2.sh" > /dev/null || pgrep -f "queue3.sh" > /dev/null; do sleep 5; done
echo "=== $(date +%T) run_lc.py" >> results/queue4.log; python3 run_lc.py 100000 >> results/queue4.log 2>&1
echo "=== ALL DONE4 $(date +%T)" >> results/queue4.log
