# Prompt 01: review scope

**Source.** Aman Bhagat, *Token-Level Uncertainty as a Predictor of Hallucination in Large Language Models: A Thematic
Literature Review and Research Agenda* (course manuscript, IIIT Allahabad, 2026). The full PDF was read; nothing below
relies on unseen content.

| Item | Content of the review | Label |
|---|---|---|
| Question | Whether next-token log-probability and entropy can predict hallucinated claims, and where they fail. | VERIFIED (review §1) |
| Structure | §2 background; §3 six themes (token probabilities, entropy, relevance weighting / SAR, semantic uncertainty, output signatures, learning from distribution sequences); §4 comparison table; §5 trends; §6 challenges; §7 gaps; §8 future directions. | VERIFIED |
| Core sources | Malinin & Gales 2021; SelfCheckGPT (Manakul et al. 2023); SAR (Duan et al. 2024); semantic entropy (Kuhn et al. 2023; Farquhar et al. 2024); LM-Polygraph (Vashurin et al. 2025); the uncertainty-quantification survey by Shorinwa et al. 2025; seven fingerprinting / distribution records from the supplied spreadsheet (Appendix A). | VERIFIED (review reference list) |
| Key numbers it relies on | WikiBio NonFact AUC-PR: random 72.96, GPT-3 Max(−log p) 87.51, SelfCheckGPT-Prompt 93.42. | VERIFIED: re-read from SelfCheckGPT Table 2 (arXiv 2303.08896) before coding |
| Conclusion | Token uncertainty is a real but **partial** predictor; it cannot separate a familiar falsehood from a familiar truth (§6.1, §9). | VERIFIED |
| Agenda | §8.1 calibrated risk predictor; **§8.2 semantic-anchor weighting**; **§8.3 two-stage early-warning architecture**; §8.4 model-conditioned calibration; §8.5 evaluation protocol; §8.6 combining evidence. | VERIFIED |
| Out of scope for the review | No experiments of its own; only English QA and biography benchmarks were discussed. | VERIFIED (§7) |

Decisions that follow from this scope are logged in [`../research_decisions.md`](../research_decisions.md) (D1–D11).
