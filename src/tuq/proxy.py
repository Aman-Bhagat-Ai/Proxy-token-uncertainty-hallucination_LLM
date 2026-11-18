"""Re-score the GPT-3 passages with an open proxy LM and keep per-token log-prob and entropy.

For token y_t of the GPT-3 text (conditioned on a fixed prefix and the earlier GPT-3 tokens):
  logp_t = log p(y_t | prefix, y_<t)
  ent_t  = -sum_v p(v | prefix, y_<t) log p(v | prefix, y_<t)    (full vocabulary)
Each token is assigned to the sentence and whitespace word containing its first non-space character.
"""
import gzip
import json
import time


def _first_nonspace(text, s, e):
    for i in range(s, e):
        if not text[i].isspace():
            return i
    return None


def _locate(pos, spans):
    for k, (a, b) in enumerate(spans):
        if a <= pos < b:
            return k
    return None


def score_model(model_name, passages, prefix, out_path, dtype="float16", max_length=1024, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype).to(device).eval()
    t0 = time.time()
    results, n_tok_total, n_unassigned, n_trunc = [], 0, 0, 0
    for p in passages:
        text = p["text"]
        full = prefix + text
        enc = tok(full, return_offsets_mapping=True, add_special_tokens=True,
                  truncation=True, max_length=max_length, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"].to(device)
        if offsets[-1][1] < len(full) - 1:
            n_trunc += 1
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=enc["attention_mask"].to(device)).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)                      # [T, V]
        ent = -(logprobs.exp() * logprobs).sum(-1)                         # [T]
        tgt = ids[0, 1:]
        tok_logp = logprobs[:-1].gather(1, tgt[:, None])[:, 0]             # logp of token j given <j
        tok_ent = ent[:-1]
        tok_logp, tok_ent = tok_logp.cpu().tolist(), tok_ent.cpu().tolist()

        sent_spans = [(s["start"], s["end"]) for s in p["sentences"]]
        per_sent = [{"logp": [], "ent": [], "word": []} for _ in p["sentences"]]
        L = len(prefix)
        for j in range(1, len(offsets)):
            s, e = offsets[j]
            if e <= L or s == e:
                continue
            s_txt, e_txt = max(s, L) - L, e - L
            pos = _first_nonspace(text, s_txt, min(e_txt, len(text)))
            if pos is None:
                continue
            k = _locate(pos, sent_spans)
            if k is None:
                n_unassigned += 1
                continue
            w = _locate(pos, [tuple(x) for x in p["sentences"][k]["words"]])
            per_sent[k]["logp"].append(round(tok_logp[j - 1], 5))
            per_sent[k]["ent"].append(round(tok_ent[j - 1], 5))
            per_sent[k]["word"].append(-1 if w is None else w)
            n_tok_total += 1
        results.append(per_sent)
    elapsed = time.time() - t0
    meta = {"model": model_name, "prefix": prefix, "dtype": dtype, "tokens_scored": n_tok_total,
            "tokens_unassigned": n_unassigned, "truncated_passages": n_trunc,
            "seconds": round(elapsed, 1), "vocab_size": int(logits.shape[-1]),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    with gzip.open(out_path, "wt") as f:
        json.dump({"meta": meta, "passages": results}, f)
    del model
    torch.cuda.empty_cache()
    return meta


def load_scores(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)
