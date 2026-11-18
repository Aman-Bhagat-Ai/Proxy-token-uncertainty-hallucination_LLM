"""Load the SelfCheckGPT WikiBio data and attach character offsets, words and anchors.

Output (data/wikibio_prepared.json): list of passages
  {pid, text, samples[20], sentences:[{text, start, end, label, score, words:[[s,e],...], anchor:[0/1,...]}]}
Offsets are character offsets into `text` (= gpt3_text).
"""
import json
import re

LABEL_SCORE = {"accurate": 0.0, "minor_inaccurate": 0.5, "major_inaccurate": 1.0}
WORD_RE = re.compile(r"\S+")


def load_raw():
    from datasets import load_dataset
    ds = load_dataset("potsawee/wiki_bio_gpt3_hallucination")
    split = list(ds.keys())[0]
    return ds[split], split


def locate_sentences(text, sentences):
    """Find each sentence in text, scanning left to right. Returns list of (start, end) or None."""
    spans, cursor = [], 0
    for s in sentences:
        s_strip = s.strip()
        i = text.find(s_strip, cursor)
        if i < 0:
            # whitespace-normalised fallback: match on the first 25 chars
            i = text.find(s_strip[:25], cursor)
            if i < 0:
                spans.append(None)
                continue
            spans.append((i, min(len(text), i + len(s_strip))))
        else:
            spans.append((i, i + len(s_strip)))
        cursor = spans[-1][1]
    return spans


def words_of(text, start, end):
    return [[start + m.start(), start + m.end()] for m in WORD_RE.finditer(text[start:end])]


def anchor_mask(nlp, text, words, pos_set, ent_set):
    """1 if a whitespace word overlaps a spaCy token whose POS is in pos_set or which lies in an entity of ent_set."""
    doc = nlp(text)
    marked = []
    for tok in doc:
        if tok.pos_ in pos_set or (tok.ent_type_ and tok.ent_type_ in ent_set):
            marked.append((tok.idx, tok.idx + len(tok.text)))
    mask = []
    for ws, we in words:
        mask.append(int(any(ms < we and me > ws for ms, me in marked)))
    return mask


def prepare(out_path, spacy_model="en_core_web_sm", pos=("PROPN", "NUM"), ents=()):
    import spacy
    nlp = spacy.load(spacy_model)
    raw, split = load_raw()
    passages, n_missing = [], 0
    for pid, row in enumerate(raw):
        text = row["gpt3_text"]
        sents = row["gpt3_sentences"]
        labels = row["annotation"]
        assert len(sents) == len(labels), pid
        spans = locate_sentences(text, sents)
        out = []
        for s, lab, sp in zip(sents, labels, spans):
            if sp is None:
                n_missing += 1
                sp = (-1, -1)
                words = []
                anc = []
            else:
                words = words_of(text, *sp)
                # anchors computed on the sentence string itself, offsets shifted back
                local = [[a - sp[0], b - sp[0]] for a, b in words]
                anc = anchor_mask(nlp, text[sp[0]:sp[1]], local, set(pos), set(ents))
            out.append({"text": s, "start": sp[0], "end": sp[1], "label": lab,
                        "score": LABEL_SCORE[lab], "words": words, "anchor": anc})
        passages.append({"pid": pid, "text": text, "samples": list(row["gpt3_text_samples"]),
                         "wiki_bio_test_idx": row.get("wiki_bio_test_idx"), "sentences": out})
    with open(out_path, "w") as f:
        json.dump(passages, f)
    n_sent = sum(len(p["sentences"]) for p in passages)
    return {"split": split, "passages": len(passages), "sentences": n_sent, "unlocated_sentences": n_missing}


def load_prepared(path):
    with open(path) as f:
        return json.load(f)
