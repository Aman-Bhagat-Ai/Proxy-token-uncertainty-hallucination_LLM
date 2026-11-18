# Prompt 16: technical peer review and revision log

**Manuscript reviewed:** `paper/main.tex` / `paper/main.pdf` (10 pages including references and appendix; body ≈ 8.5
pages).
**Reviewer:** an AI assistant (Claude, Anthropic), acting at the author's request as a critical reviewer. This is
disclosed in `ai_assistance_log.md`. The reviewer re-checked numbers against `results/` and section references against
the compiled review PDF. No literature was added.

## Summary
The paper tests two proposals from the author's review (anchor weighting, two-stage screening) in a black-box proxy
setting on SelfCheckGPT WikiBio. Its strengths are a validated baseline (random exact; NLI within 0.01), a frozen
protocol, paired bootstrap CIs and an honest negative result with a diagnosis. The main problems are traceability of
the references to the review, paraphrasing artefacts that change or obscure claims, and the scope of the conclusions.

## MAJOR concerns

| ID | Location | Concern | Why it matters | Remedy |
|---|---|---|---|---|
| M1 | §1 Introduction, p. 1 | The review is cited as "Sec. 7.2", "Sec. 7.3" and "Secs. 5.4, 6". In the compiled review, anchor weighting is **§8.2**, the two-stage architecture **§8.3**, model dependence **§6.4** and the gaps section **§7**. | Citation traceability: an examiner following the pointers lands on the wrong sections. | Replace with `Sec.~8.2`, `Sec.~8.3` and `Secs.~6.4, 7`. |
| M2 | §3.1, p. 2 | "Factual detects precise sentences, employing the undid score". This should be *accurate* sentences, using the *negated* score. | It states the metric definition incorrectly. | Replace with "\emph{Factual} detects accurate sentences, using the negated score." |
| M3 | §3.2, p. 3 | "plain mean if a sentence anchors" (missing "has no") and "Moving on to that to SAR this needs no second neural model". | The method definition is unreadable or wrong. | "plain mean if a sentence has no anchors"; "Unlike SAR, this needs no second neural model beyond a small tagger." |
| M4 | §5.1, p. 4 | "All four are Max(H) or Anchor(H) and exceed it…" comes **before** the sentence that defines "four" and "it". | The key RQ1 result is hard to read. | Put "Of the 48 combinations … only four exceed the random baseline." first. |
| M5 | §5.2, p. 6 | "The AUROC and correlation columns don't." comes after the numbers it introduces. | This is the central interpretive step of RQ2 (gain = cancelled inversion). | "Read naively, this would validate the review's hypothesis. The AUROC and correlation columns don't: Anchor(H) moves AUROC from …". |
| M6 | §5.3, p. 6 | The cost paragraph is scrambled ("3 samples are keeping 90.98) …", "(; so, a full proxy pass took …"). | The efficiency analysis (a required deeper evaluation) cannot be read. | "A cheaper way to reduce cost is shown by the right-hand panel: SelfCheckGPT-NLI with N=5 keeps 91.70 of its 92.50 AUC-PR (and 3 samples keep 90.98) at a quarter of the cost. On the T4, NLI took 680 s for 38,160 pairs (0.36 s per sentence at N=20), while a full proxy pass took 9–27 s for 1,908 sentences (5–14 ms per sentence). The token score is about 25–70 times cheaper, but cost only matters once a score carries signal." |
| M7 | §5.3, title and text | "a two-stage screen cannot be built on an inverted score" is shown for **one** pre-specified selector (Qwen2.5-3B Avg(H)). | The general claim is stronger than the evidence. | Done as an exploratory check: with the best observed score (Qwen2.5-3B Max(H)) as the selector, the cascade is −0.54 to +0.27 points vs. random routing (`results/cascade_sensitivity_maxH.csv`). Add one sentence citing this as exploratory. |
| M8 | Hedging words throughout | The paraphrase added qualifiers that change claims: "is **often** 0.25–1.24 points below" (the data show *every* budget), "is **normally** inverted", "the less weight … **perhaps**, the higher", "consistently (specifically in this setting)". | Claims must match the evidence exactly (handout §10). | Remove "often", "normally" and "perhaps" in these sentences; keep "in this setting" once. |

## MINOR concerns

| ID | Location | Concern | Remedy |
|---|---|---|---|
| m1 | §3.5 p. 3; Ack. p. 9; App. A p. 10 | "coming with store" / "coming with repo" / "Raw indication" | "accompanying repository" (twice); "Raw evidence" |
| m2 | §6 Conclusion, p. 9 | The last sentence ends "…where they were put forward:." | "The review's ideas remain to be tested where they were put forward: on the generator's own distribution." |
| m3 | App. A, p. 10 | "Commands - every step (01--07, is run by … including the reference check 04b); in turn, in real-world scenarios" | "Commands: every step (01–07, including the reference check 04b) is run by `bash scripts/run_all.sh`; each step can also be run on its own." |
| m4 | Throughout | Filler phrases ("in actual application", "to a significant degree", "broadly speaking", "on the whole") add length and no information. | Delete where possible. |
| m5 | §1 p. 1 | "fitting on a free GPU, a graphics processor" | "fitting on a free GPU" |
| m6 | §2 p. 2 | "SelfCheckGPT.." (double period from `\paragraph{SelfCheckGPT.}`) | `\paragraph{SelfCheckGPT}` |
| m7 | Contributions (iii) | "The first test of the review's anchor-weighting proposal". This is true of the review's own proposal, but no external novelty search was done (see `docs/04`). | Keep the wording scoped to "the review's proposal"; do not generalise. |
| m8 | §3.1 | AUC-PR is computed with trapezoidal `auc`, which is slightly optimistic vs. average precision. | This is kept deliberately for comparability with SelfCheckGPT; add a short note. |
| m9 | `references.bib` | The key `qwen2025qwen25` has year 2024. | Cosmetic; it can stay. |

## Checks with no concern found
* Table 1 and Table 2 values match `results/validation.json` and `results/main_table.csv`.
* Paired differences (Table 4) match `results/bootstrap_ci.csv`.
* No test-set tuning: the only data-dependent choices are labelled post hoc (D9).
* No leakage: there is no training.
* All 15 cited entries are in the .bib.
* The quotation from the review in §6 is verbatim.

## Prioritised revision checklist and author-response table

| Priority | ID | Response | Status |
|---|---|---|---|
| 1 | M1 | Accepted. The same wrong section numbers were also in `docs/research_brief.md`, `research_decisions.md` (D1) and `ai_assistance_log.md` (row 1). | **Fixed** in the docs and in `paper/main.tex` |
| 2 | M2–M6, m1–m3 | Accepted (paraphrasing artefacts). | **Fixed** in `paper/main.tex` |
| 3 | M7 | Accepted. The exploratory sensitivity was run on the stored scores (`results/cascade_sensitivity_maxH.csv`); the conclusion is unchanged. | **Fixed**: one sentence added in §5.3 |
| 4 | M8, m5 | Accepted ("often", "normally" and "perhaps" removed; one "in this setting" kept). | **Fixed** in `paper/main.tex` |
| 5 | m4 | Partly accepted: the remaining filler phrases do not change any claim. | Left as is |
| 6 | m6 | Accepted. | **Fixed** |
| 7 | m7, m8, m9 | Noted; no change needed. | — |

The pre-revision text is kept as `paper/old/main_paraphrased_original.tex`. **`paper/main.pdf` must be recompiled from
the revised `paper/main.tex` (Overleaf) so the PDF matches the source.**
