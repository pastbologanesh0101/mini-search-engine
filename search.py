#!/usr/bin/env python3
"""Command-line interface for the mini search engine.

Usage:
    python search.py "some query"

Prints the top 5 ranked documents for the query, with their TF-IDF /
cosine-similarity score and a short snippet of matching text.
"""

import os
import sys

from engine.index import InvertedIndex
from engine.ranker import Ranker
from engine.tokenizer import tokenize

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "docs")

SNIPPET_WIDTH = 160


def make_snippet(text, query_tokens, width=SNIPPET_WIDTH):
    """Return a short excerpt of ``text`` centered on the first query match."""
    flat = " ".join(text.split())
    lowered_flat = flat.lower()

    match_pos = -1
    for tok in query_tokens:
        idx = lowered_flat.find(tok)
        if idx != -1:
            match_pos = idx
            break

    if match_pos == -1:
        snippet = flat[:width]
        return snippet + ("..." if len(flat) > width else "")

    start = max(0, match_pos - width // 2)
    end = min(len(flat), start + width)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(flat) else ""
    return f"{prefix}{flat[start:end]}{suffix}"


def build_default_index(data_dir=DATA_DIR):
    index = InvertedIndex()
    index.build_from_directory(data_dir)
    return index


def run_query(query, index=None, top_k=5):
    if index is None:
        index = build_default_index()
    ranker = Ranker(index)
    return index, ranker.score_query(query, top_k=top_k)


def main(argv):
    if len(argv) < 2 or not " ".join(argv[1:]).strip():
        print('Usage: python search.py "your query here"')
        return 1

    query = " ".join(argv[1:])
    index, results = run_query(query)

    if not results:
        print(f"No results found for: {query!r}")
        return 0

    query_tokens = tokenize(query)
    print(f"Top {len(results)} result(s) for: {query!r}\n")
    for rank, (doc_id, score) in enumerate(results, start=1):
        doc = index.documents[doc_id]
        snippet = make_snippet(doc["text"], query_tokens)
        print(f"{rank}. {doc['title']}  [doc_id={doc_id}, score={score:.4f}]")
        print(f"   {snippet}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
