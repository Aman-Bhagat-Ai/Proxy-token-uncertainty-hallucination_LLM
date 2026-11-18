"""Sentence-level aggregation of token-level uncertainty (pure numpy; no GPU needed).

Every score is oriented so that HIGHER = more likely hallucinated.
"""
import numpy as np


def _wmean(x, w):
    w = np.asarray(w, dtype=float)
    if w.sum() <= 0:
        return float(np.mean(x))
    return float(np.sum(w * x) / w.sum())


def sentence_scores(tok, words_anchor, words_rel, alphas=(0.0, 0.1, 0.25, 0.5, 1.0)):
    """tok: {'logp': [...], 'ent': [...], 'word': [...]} for one sentence.
    words_anchor: 0/1 per word; words_rel: SAR relevance per word (or None)."""
    nlp = -np.asarray(tok["logp"], dtype=float)
    ent = np.asarray(tok["ent"], dtype=float)
    wid = np.asarray(tok["word"], dtype=int)
    out = {}
    if len(nlp) == 0:
        return None
    out["avg_nlp"] = float(nlp.mean())
    out["max_nlp"] = float(nlp.max())
    out["avg_ent"] = float(ent.mean())
    out["max_ent"] = float(ent.max())
    # ---- semantic-anchor weighting (this study's targeted modification) ----
    anc = np.array([words_anchor[w] if 0 <= w < len(words_anchor) else 0 for w in wid], dtype=float)
    out["n_anchor_tok"] = int(anc.sum())
    for a in alphas:
        wts = anc + a * (1.0 - anc)
        out[f"anchor_nlp_a{a}"] = _wmean(nlp, wts)
        out[f"anchor_ent_a{a}"] = _wmean(ent, wts)
    out["anchor_nlp"] = out["anchor_nlp_a0.0"]
    out["anchor_ent"] = out["anchor_ent_a0.0"]
    out["anchor_max_nlp"] = float(nlp[anc > 0].max()) if anc.sum() > 0 else out["max_nlp"]
    # ---- SAR / TokenSAR-style relevance weighting (published comparator) ----
    if words_rel is not None:
        rel = np.array([max(words_rel[w], 0.0) if 0 <= w < len(words_rel) else 0.0 for w in wid])
        out["sar_nlp"] = _wmean(nlp, rel)
        out["sar_ent"] = _wmean(ent, rel)
    return out


def build_table(passages, proxy_scores, relevance=None, nli=None, alphas=(0.0, 0.1, 0.25, 0.5, 1.0),
                n_list=(1, 2, 3, 5, 10, 20)):
    """Returns list of row dicts, one per sentence, with label info and every score column."""
    rows = []
    for pi, p in enumerate(passages):
        for si, s in enumerate(p["sentences"]):
            row = {"pid": pi, "sid": si, "label": s["label"], "score": s["score"],
                   "n_words": len(s["words"]), "n_anchor_words": int(sum(s["anchor"]))}
            for mname, ps in proxy_scores.items():
                tok = ps["passages"][pi][si]
                rel = relevance["relevance"][pi][si] if relevance is not None else None
                sc = sentence_scores(tok, s["anchor"], rel, alphas)
                if sc is None:
                    continue
                for k, v in sc.items():
                    row[f"{mname}::{k}"] = v
            if nli is not None:
                m = np.asarray(nli["p_contradict"][pi][si], dtype=float)
                for n in n_list:
                    row[f"nli_n{n}"] = float(m[:n].mean())
                row["nli"] = float(m.mean())
            rows.append(row)
    return rows
