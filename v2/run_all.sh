#!/bin/bash
# Re-creates every result of the corrected study from raw data. Roughly 2-3 hours on one CPU core.
set -e
cd "$(dirname "$0")"
python3 tests.py
python3 run_misc.py
for d in german loan taiwan; do python3 run_cv.py $d 1000000; done
for d in german loan; do python3 run_mlp_tune.py $d 10; done
python3 run_expl.py german 10 1000000; python3 run_expl.py loan 10 1000000; python3 run_expl.py taiwan 5 1000000
for d in german loan taiwan; do python3 run_mia.py $d 1000000; done
python3 run_fair.py german 10; python3 run_fair.py loan 10; python3 run_fair.py taiwan 5
python3 run_lc.py 1000000
python3 export_paper_data.py
python3 make_manifest.py
