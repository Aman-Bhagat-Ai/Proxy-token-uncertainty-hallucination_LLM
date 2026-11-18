"""Download SelfCheckGPT WikiBio, locate sentences, compute word spans and semantic anchors."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import yaml
from tuq.data import prepare

cfg = yaml.safe_load(open("configs/experiment.yaml"))
os.makedirs("data", exist_ok=True); os.makedirs("results/logs", exist_ok=True)
info = prepare("data/wikibio_prepared.json", cfg["anchors"]["spacy_model"],
               cfg["anchors"]["pos"], cfg["anchors"]["entity_labels"])
print(json.dumps(info, indent=1))
json.dump(info, open("results/logs/01_prepare.json", "w"), indent=1)
