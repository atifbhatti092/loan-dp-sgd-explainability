#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "run_cv.py german" > /dev/null; do sleep 3; done
run() { echo "=== $(date +%T) $*" >> results/queue.log; python3 "$@" >> results/queue.log 2>&1; }
run run_misc.py
run run_cv.py loan 100000
run run_expl.py german 10 100000
run run_mia.py german 100000
run run_fair.py german 10
run run_expl.py loan 10 100000
run run_mia.py loan 100000
run run_fair.py loan 10
run run_cv.py taiwan 100000
run run_expl.py taiwan 5 100000
run run_mia.py taiwan 100000
run run_fair.py taiwan 5
run run_lc.py 100000
echo "=== ALL DONE $(date +%T)" >> results/queue.log
