# AI-assistance and research-integrity log (Prompt 18)

**Disclosure.** As the course allows, I used an AI tool, Claude (Anthropic), as an AI coding and research assistant
for this study. It was used to turn my literature review into a study design, generate and debug the code, run it in
Google Colab (T4 GPU) and draft the paper and documentation. Every number in the paper and in `results/` comes from
logged executions; nothing below is invented.

| # | Task | Tool | AI proposal | Verification step | Outcome |
|---|---|---|---|---|---|
| 1 | Gap selection | Claude | Use the review's Sec. 6–8 gaps and agenda: anchor weighting (8.2), two-stage screening (8.3), model dependence (6.4). | Gap list checked against the review text (`main.tex`, Sec. 6–8). | Three RQs in `docs/research_brief.md`. |
| 2 | Verify baseline numbers | Claude + web fetch of arXiv 2303.08896 | Take SelfCheckGPT Table 2 / Table 8 values as validation targets. | Numbers re-read from the original paper (not from the review) before coding. | Targets: random 72.96/29.72/27.04; NLI 92.50; proxy LLaMA rows. |
| 3 | Code (all modules) | Claude | `src/tuq/*.py`, `scripts/0*.py`. | CPU parts unit-tested on synthetic data (`tests/synthetic_smoke_test.py`) before any real run. | Smoke test passed; pipeline ran end-to-end. |
| 4 | Environment | Colab T4 | Versions recorded from `pip list` in the first cell. | Real console output captured (`requirements.txt`). | torch 2.14.1, transformers 5.22.1, etc. |
| 5 | **Debug: API change** | Claude | transformers 5.x renamed `torch_dtype` → `dtype` in `from_pretrained`. | Patched before first model load; checksums compared. | No effect on results. |
| 6 | **Debug: corrupted file transfer** | Claude | Code was copied into Colab as a base64 archive. The second transfer arrived with one corrupted line in `configs/experiment.yaml` (`WORK{rF_ART` instead of `WORK_OF_ART`), which made YAML parsing fail. | Detected from the YAML parser error; fixed with `sed`; then the **MD5 of every file in Colab was compared with the local copy**, and all 17 files matched. The data-preparation step (01) had run with the first, uncorrupted config, so the anchor labels were unaffected. | Lesson: copied code needs checksum verification. |
| 7 | Protocol change (logged) | Claude | Added Pythia-1.4B and Qwen2.5-1.5B-Instruct as proxies after the first four models were *scored* but **before any metric was computed**. | Recorded in `research_decisions.md` (D7). | Six proxies in the final protocol. |
| 8 | **Baseline equivalence (AI limitation found)** | Claude | Check the batched NLI against the official `selfcheckgpt` 0.1.7 package. | The package failed twice: first a missing dependency (`bert_score`), then `batch_encode_plus`, which transformers 5 removed. The official scoring logic was re-implemented line by line (unbatched, fp32) and compared on 5 passages / 45 sentences. | Max. absolute difference 4.4e-4; end-to-end NLI AUC-PR 92.50 vs. 92.50 published. |
| 9 | **Interpretation (first reading corrected)** | Claude | Anchor weighting raised AUC-PR significantly in all 6 proxies, which first looked like support for the review's hypothesis. | Checking AUROC and passage correlations showed the gain moves an *inverted* signal towards chance rather than detecting hallucinations. The paper reports it that way, and the exploratory diagnostics (D9) are labelled post hoc. | Paper claim: "less wrong, not a detector". |
| 10 | Writing | Claude | Draft of the paper, README and decision log. | Every number in the paper cross-checked against `results/*.csv` (see `baseline_validation.md`). | — |
| 11 | Process records and peer review (Prompts 01–04, 07, 09, 14, 16) | Claude | Write the missing per-prompt records in `docs/` and act as a critical reviewer of the manuscript. | Review section numbers re-checked against the compiled review PDF (this found wrong cross-references: Sec. 7.2/7.3/5.4 → 8.2/8.3/6.4); the reviewer's request for a cascade sensitivity was run on the stored scores (the frozen cascade was reproduced first, max diff 0.004 AUC-PR from CSV rounding). | `docs/README.md` index; `results/cascade_sensitivity_maxH.csv`; paper fixes listed as OPEN in `docs/16_peer_review_and_revision_log.md`. |

## Known limitations of the AI assistance
* The 7B+ proxies used in the original paper could not be run (T4 memory), so conclusions are limited to ≤3B proxies.
* AI drafting can produce fluent text that overstates results; the Results section was written only from the CSV files.
