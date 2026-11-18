# Baseline validation report (Prompt 10)

Reference: Manakul, Liusie & Gales, *SelfCheckGPT*, EMNLP 2023 — Table 2 values re-read from arXiv 2303.08896.

## VERIFIED

| Check | Expected (paper) | Measured (this repo) | Source file |
|---|---|---|---|
| Passages / sentences | 238 / 1,908 | 238 / 1,908, 0 unlocated | `results/logs/01_prepare.json` |
| Label counts | positive rate 72.96 % | 761 major, 631 minor, 516 accurate → 72.956 % | `results/validation.json` |
| Random AUC-PR NonFact / NonFact* / Factual | 72.96 / 29.72 / 27.04 | 72.96 / 29.72 / 27.04 | `results/validation.json` |
| NonFact* definition (drop passages where every sentence is major-inaccurate) | 29.72 % | 29.72 % on the 1,632 kept sentences | `results/validation.json` |
| SelfCheckGPT-NLI NonFact / NonFact* / Factual | 92.50 / 45.17 / 66.08 | 92.50 / 45.16 / 66.08 | `results/main_table.csv` |
| SelfCheckGPT-NLI passage Pearson / Spearman | 74.14 / 73.78 | 74.13 / 73.77 | `results/main_table.csv` |
| Batched fp16 NLI vs. unbatched fp32 reference (official code path) | identical up to numerics | max \|Δ\| = 4.4e-4, mean 8.6e-5, r = 0.9999998 (45 sentences) | `results/logs/04b_nli_reference_check.json` |
| Metric code | `precision_recall_curve` + `auc` (SelfCheckGPT demo) | same; smoke-tested on synthetic data | `src/tuq/metrics.py`, `tests/` |
| Proxy pipeline sanity | no truncation | 0 truncated passages; 15–19 unassignable tokens per model (whitespace between sentences) | `results/logs/02_proxy_*.json` |

## UNCERTAIN
* **Proxy-LM numbers cannot be compared exactly with the paper.** The paper used LLaMA/OPT/GPT-J/NeoX proxies (7B–30B, plus OPT down to 125M) and a prompt with the subject's name. This study uses 0.5B–3B proxies and a generic prefix. The *direction* matches (smaller proxies near or below random; negative passage correlation for −log p), but exact replication is neither expected nor claimed.
* Padded vocabularies (OPT 50,272, Pythia 50,304) put a negligible amount of probability mass into the entropy.

## NEEDS CORRECTION
* None open. Two issues were found and fixed during the run (see `ai_assistance_log.md`): the transformers 5 `dtype` rename, and one corrupted line in a hand-relayed config file (caught by the YAML parser and closed by MD5 comparison of all files).

## Conclusion
The evaluation pipeline (labels, task definitions, metric) and the strong comparator (SelfCheckGPT-NLI) reproduce the published values to within 0.01 AUC-PR. Extensions and comparisons were run only after this check.
