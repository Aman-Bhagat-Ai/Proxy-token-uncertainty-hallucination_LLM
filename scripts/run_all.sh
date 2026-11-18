#!/usr/bin/env bash
# Full pipeline. Run from the repository root on a CUDA GPU (tested: Colab, Tesla T4 16 GB).
set -euo pipefail
python scripts/01_prepare_data.py          # download + sentence offsets + spaCy anchors      (~1 min)
python scripts/02_proxy_scores.py          # 6 proxy LMs, per-token logp & entropy             (~5 min incl. downloads)
python scripts/03_relevance.py             # SAR word relevance, 33,372 cross-encoder pairs   (~7.5 min)
python scripts/04_selfcheck_nli.py         # SelfCheckGPT-NLI, 38,160 pairs                   (~11.5 min)
python scripts/04b_nli_reference_check.py  # unbatched fp32 reference check on 5 passages     (~1.5 min)
python scripts/05_evaluate.py              # all tables / CSVs (CPU)                          (~3 min)
python scripts/07_diagnostics.py           # exploratory diagnostics (CPU)
python scripts/06_figures.py               # paper figures (CPU)
