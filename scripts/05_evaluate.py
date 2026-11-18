"""Build the per-sentence score table and every result file used in the paper (CPU only).

Writes to results/:
  sentence_scores.csv      one row per sentence, every score column (traceable raw evidence)
  validation.json          baseline-validation checks (label counts, random baseline, NLI vs paper)
  main_table.csv           all methods x all metrics
  bootstrap_ci.csv         95% passage-bootstrap CIs (NonFact AUC-PR) + paired differences
  alpha_sensitivity.csv    anchor weight alpha sweep
  nli_n_sensitivity.csv    SelfCheckGPT-NLI with N samples
  cascade.csv              two-stage screening curve
  error_analysis.csv       AUC-PR on subsets (position, anchors, length)
  threshold_transfer.csv   cross-model threshold transfer (F1)
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from tuq.aggregate import build_table
from tuq.data import load_prepared
from tuq.metrics import aucpr, bootstrap_aucpr, cascade, evaluate, random_baseline, task_views
from tuq.proxy import load_scores

cfg = yaml.safe_load(open("configs/experiment.yaml"))
SHORT = {"Qwen/Qwen2.5-0.5B": "qwen0.5b", "Qwen/Qwen2.5-1.5B": "qwen1.5b",
         "Qwen/Qwen2.5-3B": "qwen3b", "facebook/opt-1.3b": "opt1.3b",
         "EleutherAI/pythia-1.4b": "pythia1.4b", "Qwen/Qwen2.5-1.5B-Instruct": "qwen1.5b-inst"}
alphas = tuple(cfg["anchors"]["alpha_sensitivity"])
passages = load_prepared("data/wikibio_prepared.json")
proxy = {}
for m in cfg["proxy_lms"]:
    path = f"results/proxy/{m.replace('/', '__')}.json.gz"
    if os.path.exists(path):
        proxy[SHORT.get(m, m)] = load_scores(path)
rel = json.load(open("results/relevance.json")) if os.path.exists("results/relevance.json") else None
nli = json.load(open("results/nli.json")) if os.path.exists("results/nli.json") else None
df = pd.DataFrame(build_table(passages, proxy, rel, nli, alphas, tuple(cfg["selfcheck_nli"]["n_sensitivity"])))
df.to_csv("results/sentence_scores.csv", index=False, float_format="%.5f")

# ---------------- validation ----------------
val = {"n_passages": int(df.pid.nunique()), "n_sentences": len(df),
       "label_counts": df.label.value_counts().to_dict(),
       "random_baseline": random_baseline(df),
       "random_baseline_paper": {"NonFact": 72.96, "NonFact*": 29.72, "Factual": 27.04},
       "nonfact_star_kept_sentences": int(task_views(df)["NonFact*"][0].sum())}
rng = np.random.default_rng(cfg["evaluation"]["seed"])
y = (df.label != "accurate").astype(int).to_numpy()
val["random_scores_aucpr_mean_of_10"] = float(np.mean([aucpr(y, rng.random(len(df))) for _ in range(10)]))
if nli is not None:
    val["selfcheck_nli_ours"] = evaluate(df, "nli")
    val["selfcheck_nli_paper"] = {"NonFact": 92.50, "NonFact*": 45.17, "Factual": 66.08, "Pearson": 74.14, "Spearman": 73.78}
json.dump(val, open("results/validation.json", "w"), indent=1)
print(json.dumps(val, indent=1))

# ---------------- main table ----------------
base_scores = ["avg_nlp", "max_nlp", "avg_ent", "max_ent", "sar_nlp", "sar_ent",
               "anchor_nlp", "anchor_ent", "anchor_max_nlp"]
rows = []
for m in proxy:
    for s in base_scores:
        col = f"{m}::{s}"
        if col in df:
            rows.append({"model": m, "score": s, **evaluate(df, col)})
if nli is not None:
    rows.append({"model": "selfcheck", "score": "nli_n20", **evaluate(df, "nli")})
main = pd.DataFrame(rows)
main.to_csv("results/main_table.csv", index=False, float_format="%.2f")
print(main.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# ---------------- bootstrap CIs and paired differences ----------------
cols = [f"{m}::{s}" for m in proxy for s in ["avg_nlp", "avg_ent", "sar_nlp", "sar_ent", "anchor_nlp", "anchor_ent", "max_nlp"]]
cols = [c for c in cols if c in df] + (["nli"] if nli is not None else [])
B = bootstrap_aucpr(df, cols, cfg["evaluation"]["bootstrap_resamples"], cfg["evaluation"]["seed"])
brow = []
for c in cols:
    brow.append({"comparison": c, "point": aucpr(y, df[c]), "lo": np.percentile(B[c], 2.5),
                 "hi": np.percentile(B[c], 97.5)})
for m in proxy:
    for a, b in [("anchor_nlp", "avg_nlp"), ("anchor_ent", "avg_ent"), ("sar_nlp", "avg_nlp"),
                 ("sar_ent", "avg_ent"), ("anchor_ent", "sar_ent"), ("anchor_nlp", "sar_nlp")]:
        ca, cb = f"{m}::{a}", f"{m}::{b}"
        if ca in B and cb in B:
            d = B[ca] - B[cb]
            brow.append({"comparison": f"{ca} - {cb}", "point": aucpr(y, df[ca]) - aucpr(y, df[cb]),
                         "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5),
                         "p_le_0": float((d <= 0).mean())})
pd.DataFrame(brow).to_csv("results/bootstrap_ci.csv", index=False, float_format="%.3f")

# ---------------- anchor alpha sensitivity ----------------
arow = []
for m in proxy:
    for kind in ["nlp", "ent"]:
        for a in alphas:
            col = f"{m}::anchor_{kind}_a{a}"
            arow.append({"model": m, "kind": kind, "alpha": a, **evaluate(df, col)})
pd.DataFrame(arow).to_csv("results/alpha_sensitivity.csv", index=False, float_format="%.2f")

# ---------------- NLI sample-count sensitivity ----------------
if nli is not None:
    nrow = [{"N": n, **evaluate(df, f"nli_n{n}")} for n in cfg["selfcheck_nli"]["n_sensitivity"]]
    pd.DataFrame(nrow).to_csv("results/nli_n_sensitivity.csv", index=False, float_format="%.2f")

# ---------------- two-stage cascade ----------------
if nli is not None:
    mname, sname = cfg["cascade"]["stage1_score"].split("::")
    s1col = f"{SHORT.get(mname, mname)}::{sname}"
    if s1col in df:
        crow = cascade(df, s1col, "nli", cfg["cascade"]["flag_fractions"], cfg["cascade"]["random_control_seeds"])
        pd.DataFrame(crow).to_csv("results/cascade.csv", index=False, float_format="%.4f")

# ---------------- error analysis on subsets ----------------
df["position"] = np.where(df.sid == 0, "first", "later")
df["has_anchor"] = np.where(df.n_anchor_words > 0, "anchor", "no_anchor")
df["length_bin"] = pd.qcut(df.n_words, 3, labels=["short", "medium", "long"])
total_h = df.groupby("pid")["label"].transform(lambda x: (x == "major_inaccurate").all())
df["passage_type"] = np.where(total_h, "total_halluc", "mixed")
erow = []
ea_cols = [c for c in df.columns if "::" in c and c.split("::")[1] in ("avg_nlp", "avg_ent", "max_nlp", "sar_ent", "anchor_ent", "anchor_nlp")]
ea_cols += (["nli"] if nli is not None else [])
for factor in ["position", "has_anchor", "length_bin", "passage_type"]:
    for level, sub in df.groupby(factor, observed=True):
        ys = (sub.label != "accurate").astype(int).to_numpy()
        if ys.min() == ys.max():
            continue
        for c in ea_cols:
            erow.append({"factor": factor, "level": str(level), "n": len(sub), "pos_rate": 100 * ys.mean(),
                         "method": c, "NonFact": aucpr(ys, sub[c])})
pd.DataFrame(erow).to_csv("results/error_analysis.csv", index=False, float_format="%.2f")
# score vs length correlation (a confound check)
lc = {c: float(np.corrcoef(df.n_words, df[c])[0, 1]) for c in ea_cols}
json.dump(lc, open("results/length_correlation.json", "w"), indent=1)

# ---------------- cross-model threshold transfer ----------------
trow = []
for kind in ["avg_ent", "avg_nlp"]:
    ms = [m for m in proxy if f"{m}::{kind}" in df]
    for src in ms:
        s_src = df[f"{src}::{kind}"].to_numpy()
        cands = np.quantile(s_src, np.linspace(0.01, 0.99, 99))
        f1s = [f1_score(y, s_src >= t) for t in cands]
        thr = cands[int(np.argmax(f1s))]
        for tgt in ms:
            s_tgt = df[f"{tgt}::{kind}"].to_numpy()
            own = max(f1_score(y, s_tgt >= t) for t in np.quantile(s_tgt, np.linspace(0.01, 0.99, 99)))
            trow.append({"score": kind, "threshold_from": src, "applied_to": tgt, "threshold": thr,
                         "F1_transferred": f1_score(y, s_tgt >= thr), "F1_own_best": own,
                         "flag_rate": float((s_tgt >= thr).mean())})
pd.DataFrame(trow).to_csv("results/threshold_transfer.csv", index=False, float_format="%.4f")

# per-model score distribution summary
drow = []
for m in proxy:
    for kind in ["avg_ent", "avg_nlp"]:
        for lab, sub in df.groupby("label"):
            drow.append({"model": m, "score": kind, "label": lab, "mean": sub[f"{m}::{kind}"].mean(),
                         "std": sub[f"{m}::{kind}"].std(), "n": len(sub)})
pd.DataFrame(drow).to_csv("results/score_distributions.csv", index=False, float_format="%.4f")
print("done")
