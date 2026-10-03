#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "queue.sh" > /dev/null; do sleep 5; done
run() { echo "=== $(date +%T) $*" >> results/queue2.log; python3 "$@" >> results/queue2.log 2>&1; }
run run_mlp_tune.py loan 10
run run_mlp_tune.py german 10
echo "=== ALL DONE2 $(date +%T)" >> results/queue2.log
