"""Fast text normalisation shared by indexing and querying.

A deliberately small pipeline: lowercase, word tokenisation, stopword removal
and a conservative suffix stemmer. It keeps the vocabulary compact, which keeps
both the sparse and the dense indexes small and fast, and it is identical at
index time and query time so there is no train and serve skew.
"""

import re
from functools import lru_cache

_WORD = re.compile(r"[^\W_]+")

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she should
so some such than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves also get got keep keeps kept us really much many lot lots way still
""".split())

_SUFFIXES = ("ational", "ization", "isation", "ations", "ation", "ments", "ment", "ness", "ities",
             "ity", "ating", "ated", "ates", "ings", "ing", "ied", "ies", "ed", "es", "ly", "s")


@lru_cache(maxsize=200_000)
def stem(word):
    if len(word) <= 4 or word.isdigit():
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix):
            root = word.removesuffix(suffix)
            if len(root) >= 3:
                if suffix in ("ied", "ies"):
                    return root + "y"
                return root
    return word


def tokenize(text):
    """Lowercase, split into words, drop stopwords and stem."""
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


def with_bigrams(tokens):
    """Unigrams plus adjacent bigrams, used by the dense semantic model."""
    grams = list(tokens)
    grams.extend(a + "_" + b for a, b in zip(tokens, tokens[1:]))
    return grams
