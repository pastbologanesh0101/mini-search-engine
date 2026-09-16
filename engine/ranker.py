"""TF-IDF query scoring via cosine similarity between query and document vectors."""

import math
from collections import defaultdict

from .tokenizer import tokenize


class Ranker:
    """Ranks documents in an :class:`InvertedIndex` against a query string."""

    def __init__(self, index):
        self.index = index

    def score_query(self, query, top_k=5):
        """Return up to ``top_k`` ``(doc_id, score)`` pairs, best first.

        Scoring uses cosine similarity between the TF-IDF vector of the
        query and the TF-IDF vector of each candidate document. Queries
        that are empty, whitespace-only, or contain only terms unknown to
        the index simply return an empty list rather than raising.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_counts = defaultdict(int)
        for tok in query_tokens:
            query_counts[tok] += 1

        query_vector = {}
        for term, count in query_counts.items():
            idf = self.index.idf(term)
            if idf == 0.0:
                continue  # term never seen in the corpus - contributes nothing
            query_vector[term] = (1.0 + math.log(count)) * idf

        if not query_vector:
            return []

        query_norm = math.sqrt(sum(w * w for w in query_vector.values()))
        if query_norm == 0.0:
            return []

        candidate_doc_ids = set()
        for term in query_vector:
            candidate_doc_ids.update(self.index.postings.get(term, {}).keys())

        if not candidate_doc_ids:
            return []

        self.index.build_doc_norms()

        scored = []
        for doc_id in candidate_doc_ids:
            doc_norm = self.index.doc_norms.get(doc_id, 0.0)
            if doc_norm == 0.0:
                continue
            dot_product = 0.0
            for term, query_weight in query_vector.items():
                tf = self.index.term_frequency(term, doc_id)
                if tf == 0:
                    continue
                doc_weight = (1.0 + math.log(tf)) * self.index.idf(term)
                dot_product += query_weight * doc_weight
            score = dot_product / (query_norm * doc_norm)
            if score > 0.0:
                scored.append((doc_id, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
