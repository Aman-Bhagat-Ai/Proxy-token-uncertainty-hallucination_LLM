"""SelfCheckGPT-NLI contradiction matrix (sentence x 20 samples) + equivalence check vs official package."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, yaml
from tuq.data import load_prepared
from tuq.selfcheck import nli_matrix, official_check

cfg = yaml.safe_load(open("configs/experiment.yaml"))
passages = load_prepared("data/wikibio_prepared.json")
if not os.path.exists("results/nli.json"):
    meta = nli_matrix(passages, "results/nli.json", cfg["selfcheck_nli"]["model"], cfg["selfcheck_nli"]["batch_size"])
    print(json.dumps(meta)); json.dump(meta, open("results/logs/04_nli.json", "w"), indent=1)
check = {}
try:
    off = official_check(passages, 3)
    mine = json.load(open("results/nli.json"))["p_contradict"]
    diffs = [abs(a - float(np.mean(b))) for pi in range(3) for a, b in zip(off[pi], mine[pi])]
    check = {"n_sentences": len(diffs), "max_abs_diff": max(diffs), "mean_abs_diff": float(np.mean(diffs)),
             "official": off, "batched": [[float(np.mean(b)) for b in mine[pi]] for pi in range(3)]}
except Exception as e:  # package missing etc.
    check = {"error": repr(e)}
print(json.dumps({k: v for k, v in check.items() if k not in ("official", "batched")}))
json.dump(check, open("results/logs/04_nli_official_check.json", "w"), indent=1)
