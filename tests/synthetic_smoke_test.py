"""Smoke test of the CPU evaluation pipeline on synthetic data (no GPU, no download).
Run from repo root in a temp copy: python tests/synthetic_smoke_test.py"""
import gzip, json, os, random, sys
sys.path.insert(0, "src")
import numpy as np
from tuq.data import locate_sentences, words_of
rng = random.Random(0)
labs = ["accurate", "minor_inaccurate", "major_inaccurate"]
P, prox, rel, nli = [], [], [], []
for pid in range(40):
    sents = [" ".join(f"w{rng.randint(0,99)}" for _ in range(rng.randint(4, 12))) + "." for _ in range(rng.randint(3, 9))]
    text = " ".join(sents)
    spans = locate_sentences(text, sents)
    assert all(s is not None for s in spans)
    ss, ps, rs, ns = [], [], [], []
    allmajor = pid % 10 == 0
    for s, sp in zip(sents, spans):
        lab = "major_inaccurate" if allmajor else rng.choice(labs)
        w = words_of(text, *sp)
        anc = [rng.randint(0, 1) for _ in w]
        ss.append({"text": s, "start": sp[0], "end": sp[1], "label": lab,
                   "score": {"accurate": 0, "minor_inaccurate": .5, "major_inaccurate": 1}[lab], "words": w, "anchor": anc})
        bump = 0 if lab == "accurate" else 1
        nt = len(w) + 2
        ps.append({"logp": [-abs(rng.gauss(1 + bump, 1)) for _ in range(nt)], "ent": [abs(rng.gauss(2 + bump, 1)) for _ in range(nt)],
                   "word": [min(i // 1, len(w) - 1) for i in range(nt)]})
        rs.append([rng.random() for _ in w]); ns.append([min(1, max(0, rng.gauss(0.2 + 0.5 * bump, .2))) for _ in range(20)])
    P.append({"pid": pid, "text": text, "samples": ["x"] * 20, "sentences": ss}); prox.append(ps); rel.append(rs); nli.append(ns)
os.makedirs("data", exist_ok=True); os.makedirs("results/proxy", exist_ok=True)
json.dump(P, open("data/wikibio_prepared.json", "w"))
for m in ["Qwen/Qwen2.5-3B", "facebook/opt-1.3b"]:
    with gzip.open(f"results/proxy/{m.replace('/', '__')}.json.gz", "wt") as f:
        json.dump({"meta": {}, "passages": prox}, f)
json.dump({"meta": {}, "relevance": rel}, open("results/relevance.json", "w"))
json.dump({"meta": {}, "p_contradict": nli}, open("results/nli.json", "w"))
print("synthetic data written")
