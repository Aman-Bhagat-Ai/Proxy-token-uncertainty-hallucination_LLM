"""SAR word relevance with a cross-encoder STS model."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import yaml
from tuq.data import load_prepared
from tuq.relevance import compute_relevance

cfg = yaml.safe_load(open("configs/experiment.yaml"))
passages = load_prepared("data/wikibio_prepared.json")
meta = compute_relevance(passages, "results/relevance.json", cfg["relevance"]["cross_encoder"])
print(json.dumps(meta)); json.dump(meta, open("results/logs/03_relevance.json", "w"), indent=1)
