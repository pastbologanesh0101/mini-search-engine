"""Inverted index with incremental document insertion and TF-IDF helpers."""

import math
import os
from collections import defaultdict

from .tokenizer import tokenize


class InvertedIndex:
    """Maps terms to the documents (and term frequencies) they appear in.

    ``postings`` is a dict of the form::

        {term: {doc_id: term_frequency, ...}, ...}

    which is the classic inverted-index shape: term -> list of
    (doc_id, term_frequency) pairs, represented here as a dict for O(1)
    lookups when scoring a specific document.
    """

    def __init__(self):
        self.postings = defaultdict(dict)
        self.doc_lengths = {}
        self.documents = {}
        self.doc_ids = []
        self.doc_norms = {}
        self._norms_dirty = True

    def add_document(self, doc_id, text, title=None, path=None):
        """Add (or replace) a single document in the index.

        This is the incremental-indexing entry point: it only touches the
        postings for terms that appear in ``text`` and does not require
        rebuilding the index from scratch.
        """
        tokens = tokenize(text)
        counts = defaultdict(int)
        for tok in tokens:
            counts[tok] += 1

        for term, count in counts.items():
            self.postings[term][doc_id] = count

        self.doc_lengths[doc_id] = len(tokens)
        self.documents[doc_id] = {
            "text": text,
            "title": title or doc_id,
            "path": path,
        }
        if doc_id not in self.doc_ids:
            self.doc_ids.append(doc_id)
        self._norms_dirty = True

    def build_from_directory(self, dir_path):
        """Index every ``.txt`` file in ``dir_path``.

        The file's stem (without extension) becomes the doc_id, and the
        first line of the file is used as a human-readable title.
        """
        for fname in sorted(os.listdir(dir_path)):
            if not fname.endswith(".txt"):
                continue
            full_path = os.path.join(dir_path, fname)
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
            doc_id = os.path.splitext(fname)[0]
            stripped = text.strip()
            title = stripped.splitlines()[0] if stripped else doc_id
            self.add_document(doc_id, text, title=title, path=full_path)

    def num_documents(self):
        return len(self.doc_ids)

    def document_frequency(self, term):
        """Number of documents containing ``term``."""
        return len(self.postings.get(term, {}))

    def term_frequency(self, term, doc_id):
        """Raw count of ``term`` inside ``doc_id`` (0 if absent)."""
        return self.postings.get(term, {}).get(doc_id, 0)

    def idf(self, term):
        """Smoothed inverse document frequency for ``term``.

        Returns 0.0 for terms that never appear in the corpus, which lets
        callers treat unknown terms as contributing no weight at all.
        """
        df = self.document_frequency(term)
        if df == 0:
            return 0.0
        n = self.num_documents()
        return math.log((n + 1) / (df + 1)) + 1.0

    def tfidf(self, term, doc_id):
        """TF-IDF weight of ``term`` in ``doc_id`` using log-scaled tf."""
        tf = self.term_frequency(term, doc_id)
        if tf == 0:
            return 0.0
        return (1.0 + math.log(tf)) * self.idf(term)

    def build_doc_norms(self):
        """(Re)compute the TF-IDF vector norm for every document.

        Cached and only recomputed when documents have changed since the
        last call, so repeated searches against a static index stay cheap.
        """
        if not self._norms_dirty:
            return
        sums_of_squares = defaultdict(float)
        for term, docs in self.postings.items():
            idf = self.idf(term)
            if idf == 0.0:
                continue
            for doc_id, tf in docs.items():
                weight = (1.0 + math.log(tf)) * idf
                sums_of_squares[doc_id] += weight * weight
        self.doc_norms = {
            doc_id: math.sqrt(value) for doc_id, value in sums_of_squares.items()
        }
        self._norms_dirty = False
