"""Print every small result file (for copying results out of a notebook session)."""
import glob, json, os
for p in sorted(glob.glob("results/logs/*.json")) + ["results/validation.json", "results/length_correlation.json"]:
    if os.path.exists(p):
        print(f"### {p}\n{open(p).read()}")
for p in sorted(glob.glob("results/*.csv")):
    if "sentence_scores" not in p:
        print(f"### {p}\n{open(p).read()}")
