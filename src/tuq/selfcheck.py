"""Batched SelfCheckGPT-NLI (Manakul et al., EMNLP 2023).

P(contradict | r_i, S_n) = softmax([z_e, z_c])[c] from potsawee/deberta-v3-large-mnli (neutral removed),
S_NLI(i) = mean_n P(contradict | r_i, S_n). We keep the full [sentence x sample] matrix so that the
score for N < 20 samples can be read off without re-running the model (first N samples).
Pair order follows the official selfcheckgpt package: (sentence, sample).
"""
import json
import time


def nli_matrix(passages, out_path, model_name="potsawee/deberta-v3-large-mnli", batch_size=32, device="cuda"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval().half()
    pairs, index = [], []
    for pi, p in enumerate(passages):
        for si, s in enumerate(p["sentences"]):
            for ni, sample in enumerate(p["samples"]):
                pairs.append((s["text"], sample))
                index.append((pi, si, ni))
    # sort by length to reduce padding; restore order afterwards
    order = sorted(range(len(pairs)), key=lambda k: len(pairs[k][0]) + len(pairs[k][1]))
    probs = [0.0] * len(pairs)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([pairs[k][0] for k in idx], [pairs[k][1] for k in idx], padding="longest",
                      truncation=True, max_length=512, return_tensors="pt").to(device)
            logits = model(**enc).logits.float()
            pc = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            for k, v in zip(idx, pc):
                probs[k] = v
    mat = [[[0.0] * len(p["samples"]) for _ in p["sentences"]] for p in passages]
    for (pi, si, ni), v in zip(index, probs):
        mat[pi][si][ni] = round(v, 5)
    meta = {"model": model_name, "pairs": len(pairs), "seconds": round(time.time() - t0, 1),
            "num_labels": int(model.config.num_labels), "id2label": model.config.id2label}
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "p_contradict": mat}, f)
    return meta


def official_check(passages, n_passages=3):
    """Run the official selfcheckgpt package on a few passages for an equivalence check."""
    import torch
    from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
    m = SelfCheckNLI(device=torch.device("cuda"))
    out = []
    for p in passages[:n_passages]:
        sc = m.predict(sentences=[s["text"] for s in p["sentences"]], sampled_passages=p["samples"])
        out.append([float(x) for x in sc])
    return out
