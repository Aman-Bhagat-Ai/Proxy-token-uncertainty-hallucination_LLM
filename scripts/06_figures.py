"""Publication figures from the result CSVs (no GPU). Output: paper/figures/*.pdf and *.png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("paper/figures", exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
ORDER = ["qwen0.5b", "qwen1.5b", "qwen3b", "qwen1.5b-inst", "opt1.3b", "pythia1.4b"]
LABEL = {"qwen0.5b": "Qwen2.5-0.5B", "qwen1.5b": "Qwen2.5-1.5B", "qwen3b": "Qwen2.5-3B", "opt1.3b": "OPT-1.3B", "qwen1.5b-inst": "Qwen2.5-1.5B-Inst", "pythia1.4b": "Pythia-1.4B"}
COL = {"qwen0.5b": "#9ecae1", "qwen1.5b": "#4292c6", "qwen3b": "#08519c", "opt1.3b": "#e6550d", "qwen1.5b-inst": "#6baed6", "pythia1.4b": "#31a354"}


def save(fig, name):
    fig.savefig(f"paper/figures/{name}.pdf")
    fig.savefig(f"paper/figures/{name}.png", dpi=200)
    plt.close(fig)


main = pd.read_csv("results/main_table.csv")
val = pd.read_json("results/validation.json", typ="series")
rand = val["random_baseline"]["NonFact"]

# Fig 1: NonFact AUC-PR per scoring rule and proxy LM, with SelfCheckGPT-NLI and random reference lines
scores = ["avg_nlp", "max_nlp", "avg_ent", "max_ent", "sar_nlp", "sar_ent", "anchor_nlp", "anchor_ent"]
names = ["Avg(-logp)", "Max(-logp)", "Avg(H)", "Max(H)", "SAR(-logp)", "SAR(H)", "Anchor(-logp)", "Anchor(H)"]
models = [m for m in ORDER if m in set(main.model)]
fig, ax = plt.subplots(figsize=(6.6, 2.6))
w = 0.8 / len(models)
x = np.arange(len(scores))
for i, m in enumerate(models):
    vals = [main[(main.model == m) & (main.score == s)]["NonFact"].values[0] for s in scores]
    ax.bar(x + (i - (len(models) - 1) / 2) * w, vals, w, label=LABEL[m], color=COL[m])
ax.axhline(rand, color="grey", ls=":", lw=1)
ax.text(len(scores) - 0.5, rand + 0.4, f"random {rand:.2f}", ha="right", va="bottom", fontsize=7, color="grey")
nli = main[main.model == "selfcheck"]["NonFact"]
if len(nli):
    ax.axhline(nli.values[0], color="black", ls="--", lw=1)
    ax.text(len(scores) - 0.5, nli.values[0] - 0.4, f"SelfCheckGPT-NLI {nli.values[0]:.2f}", ha="right", va="top", fontsize=7)
ax.set_xticks(x, names, rotation=20)
ax.set_ylabel("AUC-PR, NonFact (%)")
lo = min(main[main.model.isin(models)]["NonFact"].min(), rand) - 3
ax.set_ylim(lo, 100)
ax.legend(ncol=3, frameon=False, loc="upper left", fontsize=7)
save(fig, "fig_main_aucpr")

# Fig 2: anchor alpha sensitivity
al = pd.read_csv("results/alpha_sensitivity.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2), sharey=False)
for ax, kind, ttl in zip(axes, ["nlp", "ent"], ["-log p", "entropy H"]):
    for m in models:
        d = al[(al.model == m) & (al.kind == kind)].sort_values("alpha")
        ax.plot(d.alpha, d.NonFact, marker="o", ms=3, label=LABEL[m], color=COL[m])
    ax.set_xlabel(r"non-anchor weight $\alpha$ ($\alpha$=1: plain average)")
    ax.set_title(f"Anchor-weighted mean of {ttl}", fontsize=8)
    ax.axhline(rand, color="grey", ls=":", lw=1)
axes[0].set_ylabel("AUC-PR, NonFact (%)")
axes[1].legend(frameon=False, fontsize=7)
save(fig, "fig_alpha_sensitivity")

# Fig 3: cascade cost-quality curve + NLI sample count
cas = pd.read_csv("results/cascade.csv")
nn = pd.read_csv("results/nli_n_sensitivity.csv")
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3))
ax = axes[0]
s1 = cas[cas.selector == "stage1"].sort_values("q")
r = cas[cas.selector == "random"].groupby("q").NonFact.agg(["mean", "std"]).reset_index()
ax.plot(100 * s1.q, s1.NonFact, marker="o", ms=3, color="#08519c", label="flag by token score (Qwen2.5-3B Avg(H))")
ax.errorbar(100 * r.q, r["mean"], yerr=r["std"], marker="s", ms=3, color="#e6550d", capsize=2, label="flag at random (10 seeds)")
ax.set_xlabel("sentences sent to SelfCheckGPT-NLI (%)")
ax.set_ylabel("AUC-PR, NonFact (%)")
ax.legend(frameon=False, fontsize=6.5)
ax.set_title("Two-stage screening", fontsize=8)
ax = axes[1]
ax.plot(nn.N, nn.NonFact, marker="o", ms=3, color="black", label="NonFact")
ax.set_xscale("log")
ax.set_xticks(nn.N, [str(v) for v in nn.N])
ax.set_xlabel("number of stochastic samples N")
ax.set_ylabel("AUC-PR, NonFact (%)")
ax.set_title("SelfCheckGPT-NLI vs. sample budget", fontsize=8)
save(fig, "fig_cascade_nli")

# Fig 4: score distributions by label (Avg(H)) per proxy -> calibration differs across models
ss = pd.read_csv("results/sentence_scores.csv") if os.path.exists("results/sentence_scores.csv") else None
if ss is not None:
    fig, axes = plt.subplots(1, len(models), figsize=(6.6, 1.9), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):
        data = [ss[ss.label == lab][f"{m}::avg_ent"].values for lab in ["accurate", "minor_inaccurate", "major_inaccurate"]]
        ax.boxplot(data, showfliers=False, widths=0.6)
        ax.set_xticks([1, 2, 3], ["acc.", "minor", "major"])
        ax.set_title(LABEL[m], fontsize=7.5)
    np.atleast_1d(axes)[0].set_ylabel("Avg(H) per sentence (nats)")
    save(fig, "fig_score_distributions")
print("figures written")

# Fig 5 (exploratory diagnostic): passage-level confound + within-passage centring
if ss is not None and os.path.exists("results/diagnostics_direction.csv"):
    dd = pd.read_csv("results/diagnostics_direction.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    ax = axes[0]
    g = ss.groupby("pid").agg(h=("score", "mean"), s=("qwen3b::avg_nlp", "mean"))
    ax.scatter(g.h, g.s, s=6, alpha=0.6, color="#08519c", edgecolor="none")
    r = np.corrcoef(g.h, g.s)[0, 1]
    ax.set_xlabel("passage hallucination score (human, 0-1)")
    ax.set_ylabel("passage mean Avg(-log p), Qwen2.5-3B")
    ax.set_title(f"Between passages: r = {r:.2f}", fontsize=8)
    ax = axes[1]
    sub = dd[dd.score.isin(["avg_nlp", "avg_ent", "anchor_ent"])]
    labels = {"avg_nlp": "Avg(-logp)", "avg_ent": "Avg(H)", "anchor_ent": "Anchor(H)"}
    x = np.arange(len(models))
    for k, (s, mk) in enumerate(zip(["avg_nlp", "avg_ent", "anchor_ent"], ["o", "s", "^"])):
        a = [sub[(sub.model == m) & (sub.score == s)]["AUROC_as_specified"].values[0] for m in models]
        b = [sub[(sub.model == m) & (sub.score == s)]["AUROC_within_passage_centred"].values[0] for m in models]
        ax.scatter(x + (k - 1) * 0.2, a, marker=mk, color="#e6550d", s=14, label=f"{labels[s]} raw" if True else None)
        ax.scatter(x + (k - 1) * 0.2, b, marker=mk, color="#31a354", s=14, label=f"{labels[s]} within-passage")
    ax.axhline(50, color="grey", ls=":", lw=1)
    ax.set_xticks(x, [LABEL[m].replace("Qwen2.5-", "Q-") for m in models], rotation=25, fontsize=6.5)
    ax.set_ylabel("AUROC, NonFact (%)")
    ax.set_title("Raw vs. passage-centred scores", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.38))
    save(fig, "fig_passage_confound")
    print("diagnostic figure written")
