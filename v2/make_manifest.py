"""Writes MANIFEST.json: SHA-256 of every data, code and result file, library versions, git commit and a UTC timestamp."""
import os, sys, json, hashlib, platform, subprocess, datetime
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def collect():
    files = {}
    pats = [("Train.csv",), ("data_extra",), ("v2",), ("notebooks",)]
    for top in ("Train.csv", "data_extra", "v2", "notebooks", "prep.py", "dpmlp.py", "run_experiments.py", "results.json"):
        p = os.path.join(ROOT, top)
        if os.path.isfile(p): files[top] = sha(p)
        elif os.path.isdir(p):
            for d, _, fs in os.walk(p):
                if "__pycache__" in d or "figures" in d and False: continue
                for f in sorted(fs):
                    if f.endswith((".pyc", ".log")) or f in ("MANIFEST.json", "paper_data.json"): continue
                    if f.endswith(".ipynb"): continue          # notebooks change when executed; their inputs are hashed instead
                    fp = os.path.join(d, f); files[os.path.relpath(fp, ROOT)] = sha(fp)
    return files
if __name__ == "__main__":
    import numpy, scipy, sklearn, pandas, catboost, xgboost, interpret, dp_accounting, importlib.metadata as im
    m = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "python": platform.python_version(), "platform": platform.platform(),
         "versions": {"numpy": numpy.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "pandas": pandas.__version__, "catboost": catboost.__version__, "xgboost": xgboost.__version__,
                      "interpret": im.version("interpret"), "dp-accounting": im.version("dp-accounting")},
         "git_commit": subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(), "sha256": collect()}
    json.dump(m, open(os.path.join(ROOT, "v2", "MANIFEST.json"), "w"), indent=1); print("manifest:", len(m["sha256"]), "files")
