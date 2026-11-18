# Prompt 03: literature → executable experiment

Gaps: G1 anchor weighting, G2 two-stage screening and G3 proxy setting, from [`02_gap_matrix.md`](02_gap_matrix.md).
Hardware: Google Colab, 1× Tesla T4 (16 GB). Time: one lab cycle (about a day).

## 1. Three published methods that can be implemented

| | SelfCheckGPT (Manakul et al., EMNLP 2023) | SAR (Duan et al., ACL 2024) | Semantic entropy (Kuhn et al., ICLR 2023; Farquhar et al., Nature 2024) |
|---|---|---|---|
| Algorithm | Sample N extra answers; score each sentence by inconsistency with the samples (NLI variant: mean P(contradiction)). The paper also reports grey-box and proxy token scores. | Weight token uncertainty by relevance = 1 − sim(sentence, sentence without the token). | Sample answers, cluster them by bidirectional entailment, entropy over the clusters. |
| Dataset | WikiBio GPT-3 hallucination set, with 20 samples included. | TriviaQA, CoQA, SciQ and others (short QA). | Short QA (TriviaQA, CoQA, …). |
| Code / weights | `selfcheckgpt` 0.1.7 on PyPI; `potsawee/deberta-v3-large-mnli` on HF. **VERIFIED:** downloaded and used. The package does not import under Transformers 5 (D10). | Paper description. **UNKNOWN:** the official repository was not run. `cross-encoder/stsb-roberta-large`: **VERIFIED** downloaded. | Public code exists according to the papers. **UNKNOWN:** not checked. |
| Metric | AUC-PR (NonFact, NonFact*, Factual), passage Pearson/Spearman. | AUROC. | AUROC. |
| Known limits | N samples cost N generations; NLI is the cost bottleneck. | Needs a second neural model; SAR is defined on short answers. | Built for short answers; needs sampling; clustering long biographies is ill-defined. |
| Compute on T4 | 38,160 NLI pairs ≈ 11 min (measured). | 33,372 cross-encoder pairs ≈ 7.5 min (measured). | Not run. |

## 2. Three candidate studies

| Study | Smallest working baseline | Testable question | Burden | Evidence required |
|---|---|---|---|---|
| **A. Reproduction + extension** | SelfCheckGPT-NLI on WikiBio (must give 92.50). | Does the result hold under a new condition (e.g. fewer samples)? | Low | Matching published numbers; N-sensitivity curve. |
| **B. Controlled comparison** | Proxy Avg/Max(−log p), Avg/Max(H) on WikiBio. | How do 6 proxies × 4 scores compare on one frozen protocol? | Low | AUC-PR table with bootstrap CIs. |
| **C. Targeted modification** | Plain token average (B) + SAR comparator. | Does anchor-only weighting beat the plain average and SAR? | Low | Paired bootstrap differences; α ablation. |

**Chosen:** a **B + C hybrid**, with A as the validation step. A cascade analysis (G2) reuses the stored NLI scores.
Semantic entropy was not chosen, because its clustering step does not transfer cleanly to long-form biographies.

## 3. Bridge: original papers → selected method → executable baseline → planned test

| Original paper | Selected method | Executable baseline | Planned test |
|---|---|---|---|
| Manakul et al. 2023 | SelfCheckGPT-NLI, N = 20 | `scripts/04_selfcheck_nli.py` and the reference check `04b` | Reproduce 92.50; N-sensitivity; stage 2 of the cascade |
| Manakul et al. 2023 (Table 8); Malinin & Gales 2021 | Proxy token scores Avg/Max of −log p and H | `scripts/02_proxy_scores.py` | RQ1: AUC-PR vs. random 72.96 |
| Duan et al. 2024 | SAR relevance weighting (word level) | `scripts/03_relevance.py` | RQ2 comparator |
| Review §8.2 | Semantic-anchor weighting (spaCy PROPN/NUM + 18 entity types) | `src/tuq/aggregate.py` | RQ2: paired bootstrap, α ablation |
| Review §8.3 | Two-stage cascade | `src/tuq/metrics.py::cascade` | RQ3: vs. random routing, 10 seeds |

## 4. What I had to verify myself before coding (all done)
- [x] Dataset fields and labels (761/631/516) → `results/validation.json`
- [x] No GPT-3 log-probabilities in the release (decides the black-box setting, D3)
- [x] The NLI model loads and the softmax index for "contradiction" is 1 → `04b` check
- [x] A 7B model does not fit on a T4 in fp16 (D6)
- [x] SelfCheckGPT Table 2/8 numbers re-read from the original paper
