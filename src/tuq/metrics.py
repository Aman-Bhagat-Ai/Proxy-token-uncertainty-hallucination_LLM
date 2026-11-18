"""Evaluation exactly as in SelfCheckGPT: AUC-PR via precision_recall_curve + trapezoidal auc."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def aucpr(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    p, r, _ = precision_recall_curve(y, s)
    return 100.0 * auc(r, p)


def task_views(df):
    """Return (mask, y, sign) for the three SelfCheckGPT sentence-level tasks."""
    major = (df["label"] == "major_inaccurate").to_numpy()
    nonfact = (df["label"] != "accurate").to_numpy()
    # NonFact*: drop passages that are total hallucinations (every sentence major-inaccurate)
    total_h = df.groupby("pid")["label"].transform(lambda x: (x == "major_inaccurate").all()).to_numpy()
    keep = ~total_h
    return {
        "NonFact": (np.ones(len(df), bool), nonfact.astype(int), 1.0),
        "NonFact*": (keep, major.astype(int), 1.0),
        "Factual": (np.ones(len(df), bool), (~nonfact).astype(int), -1.0),
    }


def random_baseline(df):
    return {t: 100.0 * y[m].mean() for t, (m, y, _) in task_views(df).items()}


def passage_corr(df, col):
    g = df.groupby("pid").agg(h=("score", "mean"), s=(col, "mean"))
    return 100.0 * pearsonr(g["h"], g["s"])[0], 100.0 * spearmanr(g["h"], g["s"])[0]


def evaluate(df, col):
    out = {}
    for t, (m, y, sign) in task_views(df).items():
        out[t] = aucpr(y[m], sign * df[col].to_numpy()[m])
    y = (df["label"] != "accurate").astype(int).to_numpy()
    out["AUROC"] = 100.0 * roc_auc_score(y, df[col].to_numpy())
    out["Pearson"], out["Spearman"] = passage_corr(df, col)
    return out


def bootstrap_aucpr(df, cols, n_boot=1000, seed=0):
    """Passage-level bootstrap of NonFact AUC-PR for each column. Returns {col: array[n_boot]}."""
    rng = np.random.default_rng(seed)
    pids = df["pid"].unique()
    groups = {pid: idx for pid, idx in df.groupby("pid").indices.items()}
    y = (df["label"] != "accurate").astype(int).to_numpy()
    vals = {c: df[c].to_numpy() for c in cols}
    res = {c: np.empty(n_boot) for c in cols}
    for b in range(n_boot):
        pick = rng.choice(pids, size=len(pids), replace=True)
        idx = np.concatenate([groups[p] for p in pick])
        for c in cols:
            res[c][b] = aucpr(y[idx], vals[c][idx])
    return res


def cascade(df, stage1_col, stage2_col, fractions, seeds=(0,)):
    """Two-stage screening: the top-q fraction of sentences by stage-1 score are re-scored by stage 2.
    Flagged sentences are ranked above unflagged ones (score = 1 + stage-2 score); unflagged keep the
    stage-1 percentile in [0, 1). A random-selection control flags a random q fraction instead."""
    y = (df["label"] != "accurate").astype(int).to_numpy()
    s1 = df[stage1_col].to_numpy()
    s2 = df[stage2_col].to_numpy()
    n = len(df)
    pct = np.argsort(np.argsort(s1)) / n               # percentile rank in [0, 1)
    rows = []
    for q in fractions:
        k = int(round(q * n))
        flag = np.zeros(n, bool)
        flag[np.argsort(-s1)[:k]] = True
        score = np.where(flag, 1.0 + s2, pct)
        rows.append({"q": q, "selector": "stage1", "seed": -1, "NonFact": aucpr(y, score),
                     "stage2_calls_frac": flag.mean(), "positives_flagged": y[flag].sum() / y.sum()})
        for sd in seeds:
            rng = np.random.default_rng(sd)
            flag = np.zeros(n, bool)
            flag[rng.permutation(n)[:k]] = True
            score = np.where(flag, 1.0 + s2, pct)
            rows.append({"q": q, "selector": "random", "seed": sd, "NonFact": aucpr(y, score),
                         "stage2_calls_frac": flag.mean(), "positives_flagged": y[flag].sum() / y.sum()})
    return rows
