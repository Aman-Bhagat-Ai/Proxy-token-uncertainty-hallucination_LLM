# Prompt 09: prompt-to-code log (one module at a time)

The code was generated with an AI coding assistant (Claude; see [`../ai_assistance_log.md`](../ai_assistance_log.md)).
Each module was specified first, generated, checked with a sanity test and accepted only after real output. The
evidence column points to files produced by actual runs on the Colab T4.

| # | Module | Specification and assumptions | Code | Sanity test | Real output (evidence) |
|---|---|---|---|---|---|
| 1 | Data loading and validation | Load the HF split; find every sentence's character span in the passage; whitespace words; spaCy anchor mask (PROPN/NUM + 18 entity types). Assumes sentences occur verbatim in the passage. | `src/tuq/data.py`, `scripts/01_prepare_data.py` | Count unlocated sentences (must be 0); label counts must reproduce 72.96 %. | `results/logs/01_prepare.json`: 238 / 1,908, 0 unlocated; `results/validation.json`: 761/631/516 |
| 2 | Proxy LM scoring | One fp16 forward pass per passage with a fixed prefix; per-token −log p and full-vocabulary entropy; tokens mapped to sentence and word through fast-tokenizer offsets. | `src/tuq/proxy.py`, `scripts/02_proxy_scores.py` | No truncated passages; only inter-sentence whitespace tokens unassigned. | `results/logs/02_proxy_*.json`: 0 truncated, 15–19 unassigned, 9–27 s per model |
| 3 | SAR relevance | R = 1 − sim(s, s without the word) with `cross-encoder/stsb-roberta-large`, at word level (deviation D8). | `src/tuq/relevance.py`, `scripts/03_relevance.py` | Pair count = total words. | `results/logs/03_relevance.json`: 33,372 pairs, 439 s |
| 4 | SelfCheckGPT-NLI | Batched fp16; P(contradiction) = softmax index 1, averaged over samples. | `src/tuq/selfcheck.py`, `scripts/04_selfcheck_nli.py` | Must reproduce 92.50; equivalence with the unbatched fp32 path of the official logic. | `results/logs/04_nli.json`; `04b_nli_reference_check.json`: max \|Δ\| 4.4e-4, r = 0.9999998 |
| 5 | Aggregation | Avg/Max, SAR-weighted mean and anchor-weighted mean with α; plain mean when a sentence has no anchors. | `src/tuq/aggregate.py` | α = 1 must equal the plain average. | `results/sentence_scores.csv` (anchor_*_a1.0 columns equal avg_* columns) |
| 6 | Metrics and evaluation | AUC-PR as in SelfCheckGPT; NonFact* view; passage correlations; passage bootstrap; cascade. | `src/tuq/metrics.py`, `scripts/05_evaluate.py` | Synthetic data with a known signal before real data. | `tests/synthetic_smoke_test.py` passed; `results/logs/05_evaluate.txt` |
| 7 | Figures | Read CSVs only; never hard-code numbers. | `scripts/06_figures.py` | Visual check against the CSV values. | `paper/figures/*.pdf/png` |
| 8 | Diagnostics (post hoc, D9) | Sign flip, within-passage centring, anchor density, subsets. | `scripts/07_diagnostics.py` | — | `results/diagnostics_*.csv`, `results/error_analysis.csv` |

**Deliberate deviations from the reference methods:**
* word-level SAR (D8);
* batched fp16 NLI instead of the package's unbatched path (checked in `04b`);
* a generic prefix instead of a prompt with the subject's name.
