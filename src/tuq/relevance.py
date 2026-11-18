"""SAR-style relevance of each word in a sentence (Duan et al., ACL 2024), computed at word level.

relevance(w_i) = 1 - sim(sentence, sentence without w_i),  sim from a cross-encoder STS model in [0, 1].
"""
import json
import time


def compute_relevance(passages, out_path, model_name="cross-encoder/stsb-roberta-large", batch_size=128):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_name, device="cuda")
    pairs, index = [], []
    for pi, p in enumerate(passages):
        text = p["text"]
        for si, s in enumerate(p["sentences"]):
            words = [text[a:b] for a, b in s["words"]]
            full = " ".join(words)
            for wi in range(len(words)):
                reduced = " ".join(words[:wi] + words[wi + 1:])
                pairs.append((full, reduced))
                index.append((pi, si, wi))
    t0 = time.time()
    sims = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    rel = [[[0.0] * len(s["words"]) for s in p["sentences"]] for p in passages]
    for (pi, si, wi), sim in zip(index, sims):
        rel[pi][si][wi] = round(float(1.0 - sim), 5)
    meta = {"model": model_name, "pairs": len(pairs), "seconds": round(time.time() - t0, 1)}
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "relevance": rel}, f)
    return meta
