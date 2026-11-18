"""Per-token log-prob and entropy of the GPT-3 passages under each proxy LM."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import yaml, torch
from tuq.data import load_prepared
from tuq.proxy import score_model

cfg = yaml.safe_load(open("configs/experiment.yaml"))
ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="*", default=cfg["proxy_lms"])
args = ap.parse_args()
torch.manual_seed(0)
passages = load_prepared("data/wikibio_prepared.json")
os.makedirs("results/proxy", exist_ok=True)
for m in args.models:
    out = f"results/proxy/{m.replace('/', '__')}.json.gz"
    if os.path.exists(out):
        print("skip", m); continue
    meta = score_model(m, passages, cfg["proxy_prefix"], out, cfg["proxy_dtype"], cfg["max_length"])
    print(json.dumps(meta))
    json.dump(meta, open(f"results/logs/02_proxy_{m.replace('/', '__')}.json", "w"), indent=1)
