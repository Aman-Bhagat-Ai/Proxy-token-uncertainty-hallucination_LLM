# Prompt 04: feasibility, novelty and resource access

**Candidate study.** Token-level uncertainty from small open proxy LMs on SelfCheckGPT WikiBio, with semantic-anchor
weighting and a two-stage cascade to SelfCheckGPT-NLI.

## 1. Novelty-verification plan (search strings; no titles invented)
Suggested searches on Google Scholar, Semantic Scholar, the ACL Anthology and arXiv:
* `"proxy model" OR "surrogate model" token probability hallucination detection black-box`
* `named entity OR "content words" weighted entropy hallucination detection LLM`
* `token-level uncertainty "part-of-speech" OR "entity" weighting uncertainty quantification`
* `cascade OR "two-stage" OR "selective" hallucination detection self-consistency cost`
* `SelfCheckGPT WikiBio proxy LLaMA OPT log-probability` (papers citing SelfCheckGPT)

Neighbouring work to inspect: relevance-weighted UQ (SAR and follow-ups), claim-level / entity-level hallucination
detection, black-box UQ for API models, and cost-aware or adaptive self-consistency.
**Status: UNKNOWN.** A full search was not completed, so the paper claims no novelty beyond being the first test of
*the review's own* proposals.

## 2. Matrix

| Published method | VERIFIED contribution (source) | Limitation | Proposed difference | Evidence still missing |
|---|---|---|---|---|
| SelfCheckGPT proxy scores | Proxy LMs near/below random; LLaMA-30B best (Manakul et al. 2023, Table 8) | 7B–30B proxies; prompt includes the subject's name | Modern 0.5B–3B proxies (Qwen2.5, OPT, Pythia), generic prefix, AUROC and correlation reported | Whether ≥7B modern proxies behave differently |
| SelfCheckGPT-NLI | 92.50 NonFact AUC-PR with N = 20 (Table 2) | Cost of N samples and N NLI calls | N-sensitivity curve and use as stage 2 of a cascade | — |
| SAR | Relevance-weighted token uncertainty improves AUROC on short QA (Duan et al. 2024) | Needs a similarity model; short answers | Anchor weighting via a POS/NER tagger; SAR as the comparator at word level | Token-level SAR on long-form text |
| Semantic entropy | Strong detection on short QA (Kuhn 2023; Farquhar 2024) | Sampling + clustering; short answers | Not used (does not transfer to long-form passages) | — |

## 3. Access check

| Resource | Access | License / terms | Label |
|---|---|---|---|
| `potsawee/wiki_bio_gpt3_hallucination` | HF Hub, downloaded in the run | CC-BY-SA-3.0 (dataset card) | VERIFIED |
| `potsawee/deberta-v3-large-mnli` | HF Hub, downloaded | as on the model card | access VERIFIED; license not re-checked |
| `selfcheckgpt` 0.1.7 | PyPI, installed | as on PyPI | installed VERIFIED; **does not import under Transformers 5** (D10) |
| Qwen2.5-0.5B/1.5B/1.5B-Instruct/3B, OPT-1.3B, Pythia-1.4B | HF Hub, downloaded | per model card (the licenses differ between models; weights are not redistributed here) | access VERIFIED; license terms should be checked per card before any reuse |
| `cross-encoder/stsb-roberta-large`, spaCy `en_core_web_sm` | downloaded | per card | access VERIFIED |
| Benchmark protocol | SelfCheckGPT evaluation code path (`precision_recall_curve` + `auc`) | — | VERIFIED |

## 4. Compute and time (T4, measured afterwards)
Proxy scoring 9–27 s per model, SAR relevance 439 s, NLI 680 s and evaluation ~3 min. The full pipeline takes about
30 min, which is well inside a one-day budget.

## 5. Decision
**GO**, narrowed to proxies ≤ 3B (a 7B model does not fit on a T4 in fp16).

Fallbacks:
* If the NLI package is unusable, re-implement the scoring logic and check it against the official logic (this was needed: D10).
* If proxy scores are at chance, report the negative result and diagnose it (this happened: D9).
* If the NLI model is unavailable, run a proxy-only study.
