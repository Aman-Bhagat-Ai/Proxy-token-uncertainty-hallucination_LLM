# Data

**Dataset:** SelfCheckGPT WikiBio hallucination benchmark, Hugging Face `potsawee/wiki_bio_gpt3_hallucination`
(Manakul et al., EMNLP 2023). 238 GPT-3 (text-davinci-003) Wikipedia-style biographies, 1,908 sentences, each
annotated `accurate` / `minor_inaccurate` / `major_inaccurate`, plus 20 stochastic GPT-3 samples per passage.
Licence on the dataset card: CC-BY-SA-3.0. The only split (`evaluation`) is used purely as a test set — nothing is
trained or tuned on it.

`scripts/01_prepare_data.py` downloads it automatically and writes `data/wikibio_prepared.json`
(sentence character offsets, whitespace words, spaCy semantic-anchor mask). Raw data is not committed.

Observed counts in our run: 761 major-inaccurate, 631 minor-inaccurate, 516 accurate (1,392 / 1,908 = 72.96 %
non-factual), all 1,908 sentences located in the passage text.
