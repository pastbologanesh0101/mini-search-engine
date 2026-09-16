"""Unit tests for the mini search engine (tokenizer, index, and ranker)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.index import InvertedIndex
from engine.ranker import Ranker
from engine.tokenizer import tokenize


class TokenizerTests(unittest.TestCase):
    def test_lowercase_and_strip_punctuation(self):
        tokens = tokenize("Hello, World! This is a TEST.")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertIn("test", tokens)
        for tok in tokens:
            self.assertTrue(tok.isalnum())

    def test_removes_stopwords(self):
        tokens = tokenize("the quick brown fox is in the garden")
        self.assertNotIn("the", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("in", tokens)
        self.assertIn("quick", tokens)
        self.assertIn("brown", tokens)
        self.assertIn("fox", tokens)
        self.assertIn("garden", tokens)


class InvertedIndexTests(unittest.TestCase):
    def setUp(self):
        self.index = InvertedIndex()
        self.index.add_document("doc1", "cats and dogs are common household pets")
        self.index.add_document("doc2", "dogs love to play fetch in the park")

    def test_postings_have_correct_term_frequencies(self):
        self.assertIn("dogs", self.index.postings)
        self.assertEqual(self.index.postings["dogs"].get("doc1"), 1)
        self.assertEqual(self.index.postings["dogs"].get("doc2"), 1)
        # "fetch" only appears in doc2, never doc1.
        self.assertNotIn("doc1", self.index.postings.get("fetch", {}))
        self.assertEqual(self.index.term_frequency("fetch", "doc2"), 1)

    def test_document_frequency_counts_distinct_documents(self):
        self.assertEqual(self.index.document_frequency("dogs"), 2)
        self.assertEqual(self.index.document_frequency("fetch"), 1)
        self.assertEqual(self.index.document_frequency("nonexistentterm"), 0)

    def test_incremental_add_document_without_full_rebuild(self):
        # Snapshot postings for an untouched term to prove we didn't rebuild.
        dogs_before = dict(self.index.postings["dogs"])
        self.index.add_document("doc3", "birds fly south for the winter")
        self.assertIn("doc3", self.index.doc_ids)
        self.assertEqual(self.index.term_frequency("birds", "doc3"), 1)
        self.assertEqual(self.index.num_documents(), 3)
        self.assertEqual(dict(self.index.postings["dogs"]), dogs_before)


class RankerTests(unittest.TestCase):
    def setUp(self):
        self.index = InvertedIndex()
        self.index.add_document(
            "relevant",
            "Python is a popular programming language used for data "
            "analysis, data visualization, and data driven decision making.",
        )
        self.index.add_document(
            "irrelevant",
            "The chef prepared a delicious meal using fresh data from the "
            "local market this morning.",
        )
        self.index.add_document(
            "neutral",
            "Mountains rise sharply along the coastline near the old "
            "fishing village.",
        )
        self.ranker = Ranker(self.index)

    def test_relevant_doc_ranks_above_irrelevant_doc(self):
        results = self.ranker.score_query("python data analysis", top_k=5)
        ranked_ids = [doc_id for doc_id, _ in results]
        self.assertIn("relevant", ranked_ids)
        self.assertEqual(ranked_ids[0], "relevant")

        scores = dict(results)
        self.assertIn("irrelevant", scores)
        self.assertGreater(scores["relevant"], scores["irrelevant"])

    def test_empty_query_returns_no_results(self):
        self.assertEqual(self.ranker.score_query(""), [])
        self.assertEqual(self.ranker.score_query("    "), [])

    def test_unknown_terms_return_no_results(self):
        results = self.ranker.score_query("xylophonic quibberflaxx")
        self.assertEqual(results, [])

    def test_top_k_limits_number_of_results(self):
        results = self.ranker.score_query("data", top_k=1)
        self.assertLessEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
