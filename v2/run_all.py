"""Cross-platform re-run of the whole study (Windows / Linux / macOS).  Usage, from inside the v2 folder:
    python run_all.py            # backs up the shipped results, clears them, re-runs everything, compares with the backup
    python run_all.py --dry      # only prints what it would do
    python run_all.py --resume   # continue an interrupted run WITHOUT clearing anything
    python run_all.py --compare  # only compares results/ with results_assistant_backup/
The scripts resume from existing files in results/, so results/ must be empty to really re-run (this script does that, after making the backup)."""
import os, sys, json, glob, shutil, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); PY = sys.executable
RES, BAK = "results", "results_assistant_backup"
STEPS = [["tests.py"], ["run_misc.py"], ["run_cv.py", "german", "1000000"], ["run_cv.py", "loan", "1000000"], ["run_cv.py", "taiwan", "1000000"],
         ["run_mlp_tune.py", "loan", "10"], ["run_mlp_tune.py", "german", "10"],
         ["run_expl.py", "german", "10", "1000000"], ["run_expl.py", "loan", "10", "1000000"], ["run_expl.py", "taiwan", "5", "1000000"],
         ["run_mia.py", "german", "1000000"], ["run_mia.py", "loan", "1000000"], ["run_mia.py", "taiwan", "1000000"],
         ["run_fair.py", "german", "10"], ["run_fair.py", "loan", "10"], ["run_fair.py", "taiwan", "5"], ["run_lc.py", "1000000"],
         ["export_paper_data.py"]]
def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "timing": continue
            yield from leaves(v, p + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from leaves(v, p + f"[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool): yield p, float(o)
def compare():
    print("\nComparison of your results with the assistant's results (timing fields ignored):")
    worst_all = 0.0
    for f in sorted(glob.glob(os.path.join(BAK, "*.json"))):
        name = os.path.basename(f)
        if name in ("sigma_cache.json",) or name.startswith(("exp1", "exp2")): continue
        g = os.path.join(RES, name)
        if not os.path.exists(g): print(f"  {name:22s} MISSING in your results"); continue
        a, b = dict(leaves(json.load(open(f)))), dict(leaves(json.load(open(g))))
        keys = set(a) & set(b); worst = max((abs(a[k] - b[k]) for k in keys), default=0.0); n_bad = sum(abs(a[k] - b[k]) > 1e-6 for k in keys)
        print(f"  {name:22s} {len(keys):7d} numbers | largest difference {worst:.2e} | {n_bad} differ by more than 1e-6" + ("" if len(a) == len(b) else f" | structure differs ({len(a)} vs {len(b)})"))
        worst_all = max(worst_all, worst)
    print("\nVERDICT:", "identical within 1e-6" if worst_all < 1e-6 else "small numerical differences (normal across CPUs/library versions) - check the largest value above; below ~1e-3 is fine")
if __name__ == "__main__":
    if "--compare" in sys.argv: compare(); sys.exit()
    dry = "--dry" in sys.argv; resume = "--resume" in sys.argv
    if not dry and not resume:
        if not os.path.exists(BAK): shutil.copytree(RES, BAK); print("backup of the shipped results ->", BAK)
        for f in glob.glob(os.path.join(RES, "*")):
            os.remove(f)
        os.makedirs(RES, exist_ok=True); print("results/ cleared")
    t0 = time.time()
    for i, s in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] python {' '.join(s)}   ({(time.time()-t0)/60:.0f} min elapsed)", flush=True)
        if not dry:
            r = subprocess.run([PY] + s)
            if r.returncode != 0: print("STEP FAILED:", s); sys.exit(1)
    if not dry: compare(); print("\nFinished in %.0f minutes." % ((time.time() - t0)/60))
