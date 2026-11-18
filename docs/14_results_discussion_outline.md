# Prompt 14: evidence-based Results and Discussion outline

All numbers are from `results/` (main_table.csv, bootstrap_ci.csv, cascade.csv, nli_n_sensitivity.csv,
diagnostics_*.csv).

## RQ1: Do proxy token scores (0.5B–3B) detect non-factual sentences?
* **Direct observation.** 4 of 48 proxy × score combinations exceed random (72.96), by at most 1.23 points (best:
  Qwen2.5-3B Max(H) 74.19). Avg(−log p) is 66.3–68.4 with AUROC 40–43, below chance for all six proxies. The passage
  correlation is negative (−39 to −49).
* **Possible mechanism (INFERRED).** Fully fabricated biographies are generic and easy to predict, while true
  biographies contain specific, rarer details.
* **Alternative explanations.** The fixed prefix; tokenizer effects; the passage base rate (73 % positive).
* **Unsupported claims to avoid.** "Token uncertainty does not work". That is only shown for ≤ 3B proxies on WikiBio,
  not for white-box probabilities.
* **Hypothesis status.** "Proxy scores detect hallucination": **contradicted** in this setting.

## RQ2: Does anchor weighting or SAR beat the plain average?
* **Direct observation.** Anchor − Avg (−log p) is +2.3 to +3.3 in all six proxies, with every CI above 0. Anchor beats
  SAR for −log p in all six and for H in four. The α ablation is monotone (α = 0 best).
* **Mechanism.** AUROC moves from below 50 towards 50, and passage correlations move towards 0. Anchor weighting
  removes an **inverted** signal; it does not add detection power.
* **Alternative explanation tested.** Anchor density alone: it differs little by label and is uncorrelated with
  Avg(−log p).
* **Unsupported claims to avoid.** "Anchor weighting improves hallucination detection" without qualification.
* **Hypothesis status.** Review §8.2: **unresolved** for white-box probabilities; **not supported** as a detector with
  small proxies.

## RQ3: Does a token-score cascade beat random routing?
* **Direct observation.** The pre-specified Qwen2.5-3B Avg(H) selector is 0.25–1.24 points **below** random routing at
  every budget (q = 0.5: 81.65 vs. 82.72 ± 0.51).
* **Sensitivity (exploratory, added in the peer-review round).** With the best observed proxy score (Qwen2.5-3B Max(H))
  as the selector, the difference to random routing is between −0.54 and +0.27 points (within about 1 s.d. of the
  random runs). The conclusion does not change (`results/cascade_sensitivity_maxH.csv`).
* **Cost.** NLI with N = 5 keeps 91.70 of 92.50; N = 3 keeps 90.98. A proxy pass is 5–14 ms per sentence vs. 0.36 s for
  NLI with N = 20.
* **Hypothesis status.** Review §8.3 (cheap screen saves cost at little quality loss): **contradicted** with small
  proxies; reducing N is the better cost lever.

## Error / failure analysis (post hoc, D9)
* The inversion is between passages. After centring on the passage mean, AUROC is 48–53 for every proxy.
* Sentences without anchors: proxy scores fall to 49–60 AUC-PR, while NLI keeps 83.3.
* First sentence of each passage: Qwen2.5-3B Max(−log p) reaches 76.95 (n = 238). This is only a hypothesis.

## Limitations
One generator, one domain (English biographies) and one prefix; proxies ≤ 3B; word-level SAR; 238 passages (CIs 8–9
points wide); the AUC-PR base rate is 73 %.

## Follow-up experiments (ranked by information per cost)
1. White-box: generate with an open model, score with the same model, and repeat the anchor and cascade tests.
2. A 7–8B proxy in 4-bit to test whether the inversion disappears with scale.
3. A passage-centred calibrator trained on another dataset and tested on WikiBio.
