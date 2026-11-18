"""EXPLORATORY diagnostics, added after the frozen evaluation showed that every proxy score was at or below
the random baseline (logged in research_decisions.md, D9). Nothing here is used to select a method.

Writes results/diagnostics_direction.csv     sign-flipped and within-passage-centred scores
       results/diagnostics_specificity.csv   is the proxy signal explained by lexical specificity?
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tuq.metrics import aucpr

df = pd.read_csv("results/sentence_scores.csv")
y = (df.label != "accurate").astype(int).to_numpy()
models = sorted({c.split("::")[0] for c in df.columns if "::" in c})
rows = []
for m in models:
    for s in ["avg_nlp", "max_nlp", "avg_ent", "anchor_nlp", "anchor_ent", "sar_ent"]:
        c = f"{m}::{s}"
        v = df[c].to_numpy()
        centred = v - df.groupby("pid")[c].transform("mean").to_numpy()
        rows.append({"model": m, "score": s,
                     "AUCPR_as_specified": aucpr(y, v),
                     "AUCPR_sign_flipped": aucpr(y, -v),
                     "AUROC_as_specified": 100 * roc_auc_score(y, v),
                     "AUCPR_within_passage_centred": aucpr(y, centred),
                     "AUROC_within_passage_centred": 100 * roc_auc_score(y, centred),
                     "passage_mean_vs_halluc_rate_pearson": 100 * np.corrcoef(
                         df.groupby("pid")[c].mean(), df.groupby("pid")["score"].mean())[0, 1]})
pd.DataFrame(rows).to_csv("results/diagnostics_direction.csv", index=False, float_format="%.2f")

# Specificity: anchor-word fraction alone as a "detector", and per-label means
df["anchor_frac"] = df.n_anchor_words / df.n_words.clip(lower=1)
spec = [{"feature": "anchor_word_fraction", "AUCPR": aucpr(y, df.anchor_frac), "AUCPR_flipped": aucpr(y, -df.anchor_frac),
         "AUROC": 100 * roc_auc_score(y, df.anchor_frac)}]
for lab, sub in df.groupby("label"):
    spec.append({"feature": f"mean_anchor_frac[{lab}]", "AUCPR": sub.anchor_frac.mean()})
    spec.append({"feature": f"mean_n_words[{lab}]", "AUCPR": sub.n_words.mean()})
for m in models:
    spec.append({"feature": f"corr(anchor_frac, {m}::avg_nlp)", "AUCPR": 100 * np.corrcoef(df.anchor_frac, df[f"{m}::avg_nlp"])[0, 1]})
pd.DataFrame(spec).to_csv("results/diagnostics_specificity.csv", index=False, float_format="%.3f")
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print(pd.DataFrame(spec).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
