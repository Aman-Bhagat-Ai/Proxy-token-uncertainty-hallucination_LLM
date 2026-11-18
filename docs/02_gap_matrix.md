# Prompt 02: evidence-based gap matrix

> **History.** The first gap matrix was made on 22 Oct 2026 ([`prompts_01_02_original_2026-10-22.pdf`](prompts_01_02_original_2026-10-22.pdf)). It ranked relevance/anchor weighting 1st, early prediction 2nd and decoding robustness 3rd, and it set the two-stage hybrid and cross-model calibration aside. The ranking below is the revised one used for the experiment; the reasons for the change are logged as D12 in [`../research_decisions.md`](../research_decisions.md).

Each candidate gap gives the passage in the review and separates what the original papers report (**reported**) from
what is my own reading (**inference**). No novelty is claimed: a systematic search beyond the review was not done (see
[`04_feasibility_novelty_matrix.md`](04_feasibility_novelty_matrix.md)).

| # | Gap | Review evidence | Reported vs. inference | Experimental question | Evidence that would test it |
|---|---|---|---|---|---|
| G1 | **Semantic-anchor weighting is untested** | §8.2 says factual tokens (entities, numbers, dates) should count more than function words; POS/entity tags could make a cheaper, single-pass version of SAR. §6.2: averages are distorted by function words and tokenization. | **Reported:** SAR re-weights by relevance using a similarity model and gains AUROC (Duan et al. 2024). **Inference:** a POS/NER tagger can stand in for the similarity model. | Does averaging uncertainty over anchor tokens only beat the plain average and SAR at sentence level? | Paired passage-bootstrap AUC-PR difference on labelled sentences; ablation over the non-anchor weight α. |
| G2 | **Hybrid / two-stage detectors are untested** | §7 (last paragraph): hybrid detectors running a cheap token check and an expensive semantic check only when needed "haven't been found systematically". §8.3 describes the design. | **Reported:** SelfCheckGPT needs 5–20 samples (Manakul et al. 2023). **Inference:** a token score can route sentences to it. | At a fixed budget of semantic checks, does token-score routing keep more quality than random routing? | Cascade AUC-PR curve vs. a random-routing control (several seeds). |
| G3 | **Model dependence; the generator's probabilities are often unavailable** | §6.4: thresholds fitted to one model may fail on another. §7: no accepted cross-model calibration method. §8.4: per-model calibration. | **Reported:** SelfCheckGPT Table 8: open proxy LMs (LLaMA, OPT, GPT-J, NeoX) are near or below random, and the best LLaMA-30B proxy reaches 75.43–80.92. **Inference:** this is the realistic setting for closed APIs and has not been tested with modern small proxies. | How well do token scores from small open proxy LMs (0.5B–3B) detect GPT-3 hallucinations, and how does this vary with size and family? | AUC-PR / AUROC / passage correlation per proxy and score on one frozen protocol. |
| G4 | Decoding dependence | §6.4: temperature, top-p and decoding reshape distributions; §7: effect "hasn't been pored over". | **Reported:** semantic entropy AUROC 0.758 (sampling) vs. 0.735 (beam) (Kuhn et al. 2023). **Inference:** the size of this effect for token scores is unknown. | How does decoding temperature change the token-uncertainty/hallucination link? | Needs **new generations with new human labels**, so it is not testable with released data. |
| G5 | Early prediction and claim-level localisation | §7 (first paragraph): checks score finished answers; token scores are rarely mapped onto atomic claims. | **Inference** only. | Do the first high-risk tokens predict a false claim later in the answer? | Needs token- or claim-level labels; WikiBio has sentence labels only. |

## Ranking (significance × resources × feasibility on one Colab T4 in about a day)
1. **G3**: it decides whether G1 and G2 matter at all for API models. It uses released data and cheap forward passes.
2. **G1**: one isolated change with a direct published comparator (SAR); single-pass.
3. **G2**: needs only the stored NLI scores, so the extra compute is almost zero.

G4 and G5 were rejected because they need new human annotation, which is not available in the time frame.

## Claims checked against the original papers before coding
* 72.96 / 87.51 / 93.42 AUC-PR (SelfCheckGPT Table 2): **VERIFIED**.
* Proxy-LM rows (SelfCheckGPT Table 8): **VERIFIED**.
* SAR uses `cross-encoder/stsb-roberta-large` for similarity: **INFERRED** from the SAR paper description; the SAR repository was not run.
