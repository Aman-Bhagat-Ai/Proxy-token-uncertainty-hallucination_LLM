"""Equivalence check of the batched fp16 SelfCheckGPT-NLI against an unbatched fp32 reference that follows the
official selfcheckgpt 0.1.7 SelfCheckNLI.predict code path (one (sentence, sample) pair per forward pass,
softmax over the 2 logits, P(contradiction) = probs[:, 1], mean over samples).
The official package itself could not be imported under transformers 5.x (it calls the removed
tokenizer.batch_encode_plus), so its logic is reproduced here line by line."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tuq.data import load_prepared

N_PASSAGES = 5
P = load_prepared("data/wikibio_prepared.json")[:N_PASSAGES]
tok = AutoTokenizer.from_pretrained("potsawee/deberta-v3-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("potsawee/deberta-v3-large-mnli").to("cuda").eval()
mine = json.load(open("results/nli.json"))["p_contradict"]
ref, ours = [], []
with torch.no_grad():
    for pi, p in enumerate(P):
        for si, s in enumerate(p["sentences"]):
            vals = []
            for sample in p["samples"]:
                enc = tok(s["text"], sample, padding="longest", truncation=True, return_tensors="pt").to("cuda")
                vals.append(torch.softmax(model(**enc).logits, dim=-1)[0, 1].item())
            ref.append(float(np.mean(vals)))
            ours.append(float(np.mean(mine[pi][si])))
d = np.abs(np.array(ref) - np.array(ours))
out = {"n_passages": N_PASSAGES, "n_sentences": len(ref), "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
       "pearson": float(np.corrcoef(ref, ours)[0, 1])}
print(json.dumps(out))
json.dump(out, open("results/logs/04b_nli_reference_check.json", "w"), indent=1)
