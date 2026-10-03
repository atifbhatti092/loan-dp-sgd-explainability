#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "queue.sh" > /dev/null || pgrep -f "queue2.sh" > /dev/null; do sleep 5; done
run() { echo "=== $(date +%T) $*" >> results/queue3.log; python3 "$@" >> results/queue3.log 2>&1; }
run run_cv.py taiwan 100000
run run_expl.py taiwan 5 100000
run run_mia.py taiwan 100000
run run_fair.py taiwan 5
echo "=== ALL DONE3 $(date +%T)" >> results/queue3.log
