# Mini Search Engine

A small, fully working text search engine built from scratch in pure Python
(standard library only — no Whoosh, no Elasticsearch, no external search
packages). It indexes a bundled corpus of short documents, builds an
inverted index, scores queries with TF-IDF, and ranks results by cosine
similarity.

## How it works

**1. Tokenization** (`engine/tokenizer.py`)
Text is lowercased, punctuation is stripped (only alphanumeric runs are kept
as tokens), and common English stopwords ("the", "and", "is", ...) are
removed.

**2. Inverted index** (`engine/index.py`)
`InvertedIndex` stores `postings[term][doc_id] = term_frequency`, the
classic "term -> list of (doc_id, term_frequency)" structure, represented
as a dict-of-dicts for fast lookups. Documents can be added one at a time
with `add_document()` — this is **incremental indexing**: adding a new
document only touches the postings for terms found in that document, it
never rebuilds the whole index.

**3. TF-IDF scoring** (`engine/index.py`, `engine/ranker.py`)
- Term frequency uses log-dampened counts: `1 + log(tf)`.
- IDF is smoothed: `log((N + 1) / (df + 1)) + 1`, where `N` is the number
  of documents and `df` is how many documents contain the term. This keeps
  IDF positive and well-defined even for terms that appear in every (or
  almost every) document.
- A query is turned into a TF-IDF vector the same way as a document.
- Documents are ranked by **cosine similarity** between the query vector
  and each candidate document's TF-IDF vector, i.e. `dot(query, doc) /
  (||query|| * ||doc||)`. Only documents sharing at least one query term
  are considered candidates, so unknown query terms simply contribute no
  matches.

**4. CLI** (`search.py`)
Builds the index from `data/docs/*.txt`, scores the query, and prints the
top 5 ranked documents with their score and a short snippet centered on
the first matching term.

## Project layout

```
mini-search-engine/
├── search.py              # CLI entry point
├── engine/
│   ├── tokenizer.py        # lowercase, strip punctuation, stopword removal
│   ├── index.py             # InvertedIndex: postings, tf, idf, doc norms
│   └── ranker.py            # Ranker: TF-IDF cosine similarity scoring
├── data/docs/*.txt          # ~45 short original documents (the corpus)
├── tests/test_engine.py     # unit tests (unittest)
└── .github/workflows/tests.yml
```

## Running it

Requires Python 3.11+ (only the standard library is used).

```bash
python search.py "stars and black holes in space"
```

Example output:

```
Top 4 result(s) for: 'stars and black holes in space'

1. Black Holes and Gravity  [doc_id=astronomy_black_holes, score=0.3868]
   ...vent horizon. Astronomers detect black holes indirectly by observing how nearby stars and gas orbit an invisible companion.

2. Milestones in Space Exploration  [doc_id=space_exploration, score=0.1517]
   Milestones in Space Exploration Space exploration progressed from early satellites to crewed missions that landed astronauts on the moon. Robotic probes later e...

3. How Stars Are Born  [doc_id=astronomy_stars, score=0.1206]
   How Stars Are Born Stars form inside vast clouds of gas and dust called nebulae. Gravity slowly pulls the material together until the core becomes hot and dense...

4. A Short Guide to Tea  [doc_id=cooking_tea, score=0.0751]
   ...single plant, Camellia sinensis, but different processing methods create green, black, oolong, and white varieties. Oxidation is the key step that darkens the l...
```

A query with only unknown terms (e.g. `python search.py "zzzxnotaword"`)
prints `No results found for: ...` instead of raising an error, and running
with no query prints a usage message.

## Running the tests

```bash
python -m unittest discover -s tests -v
```

All 9 tests should pass. They cover:
- Tokenization producing lowercase, punctuation-free tokens.
- Stopword removal.
- Inverted index postings having correct per-document term frequencies.
- Document frequency counts across documents.
- Incremental `add_document()` not disturbing existing postings.
- A relevant document ranking above an irrelevant one for a known query.
- Empty/whitespace-only queries returning no results gracefully.
- Unknown query terms returning no results.
- `top_k` correctly limiting the number of results.

Continuous integration runs the same suite on every push and pull request
via `.github/workflows/tests.yml`, against Python 3.11 and 3.12.

## Using the library directly

```python
from engine.index import InvertedIndex
from engine.ranker import Ranker

index = InvertedIndex()
index.add_document("doc1", "Cats and dogs are common household pets.")
index.add_document("doc2", "The stock market fell sharply this morning.")

ranker = Ranker(index)
for doc_id, score in ranker.score_query("pets and dogs"):
    print(doc_id, score)
```

## About the corpus

`data/docs/` contains about 45 short (2-4 sentence) original paragraphs
written for this project, covering varied topics — astronomy, cooking,
history, sports, technology, nature, music, art, biology, economics, and
more — purely to give the index a reasonably diverse vocabulary to search
over.

## License

MIT — see [LICENSE](LICENSE).
