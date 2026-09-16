"""Simple tokenizer: lowercasing, punctuation stripping, and stopword removal."""

import re

# A compact list of very common English words that carry little search value.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "this", "these", "those", "but", "or", "not",
    "have", "had", "do", "does", "did", "can", "could", "would", "should",
    "i", "you", "your", "we", "they", "them", "his", "her", "their", "our",
    "if", "than", "then", "so", "such", "there", "which", "who", "whom",
    "what", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "no", "nor", "only", "own",
    "same", "too", "very", "just", "also", "into", "about", "over",
    "under", "again", "further", "once", "up", "down", "out", "off",
    "above", "below", "between", "am", "being", "been", "being",
}

# Match runs of letters/digits; this both lowercases handling and strips
# punctuation because anything that isn't alphanumeric is simply not matched.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase, strip punctuation, and remove stopwords from ``text``.

    Returns a list of tokens (with repeats preserved, so callers can compute
    term frequencies directly from the output).
    """
    if not text:
        return []
    lowered = text.lower()
    raw_tokens = _TOKEN_RE.findall(lowered)
    return [tok for tok in raw_tokens if tok not in STOPWORDS]
