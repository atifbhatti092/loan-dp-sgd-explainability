#!/bin/bash
cd "$(dirname "$0")"
run() { echo "=== $(date +%T) $*" >> results/queue5.log; python3 "$@" >> results/queue5.log 2>&1; }
run run_expl.py taiwan 5 100000
run run_mia.py taiwan 100000
run run_fair.py taiwan 5
run run_lc.py 100000
echo "=== ALL DONE5 $(date +%T)" >> results/queue5.log
