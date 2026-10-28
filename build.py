#!/usr/bin/env python3
"""Export the trained chain to model.json so the static site can run it without the corpus text.

Usage:
    python3 markov/web/build.py      # rerun whenever the corpus or FOLDER_WEIGHTS change
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from generate import CORPUS_DIR, QUOTE_STREAK, build_model, build_quote_index, load_corpus  # noqa: E402

OUT = Path(__file__).parent / "model.json"
MAX_ORDER = 2  # highest lookback the page offers


def fnv1a(text):
    """32-bit FNV-1a hash of the UTF-8 bytes; markov.js computes the same thing."""
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return h


def main():
    documents = load_corpus(CORPUS_DIR)
    model = build_model(documents, MAX_ORDER)
    # Followers are stored as indexes into one shared word list to keep the file small.
    words = sorted({word for followers in model.values() for word in followers})
    index = {word: i for i, word in enumerate(words)}
    data = {
        "maxOrder": MAX_ORDER,
        "quoteStreak": QUOTE_STREAK,
        "words": words,
        # Context (lowercased words joined by spaces) -> flat [word index, weight, word index, weight, ...].
        "contexts": {" ".join(context): [x for word, weight in followers.items() for x in (index[word], weight)]
                     for context, followers in model.items()},
        # Hashes rather than the word runs themselves, so the quote check doesn't ship the text.
        "quoteHashes": sorted(fnv1a(" ".join(run)) for run in build_quote_index(documents, QUOTE_STREAK)),
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT} ({len(model):,} contexts, {len(words):,} words, {OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
