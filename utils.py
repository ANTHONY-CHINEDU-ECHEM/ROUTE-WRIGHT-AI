"""Shared utilities.

Gantry AI follows a strict house style: no hyphen or dash characters anywhere in
the project, including its source code. Arithmetic therefore uses the operator
and numpy function forms (sub, np.subtract, np.negative) and every piece of text
that leaves the system passes through clean_dashes.
"""

import math
import operator
import re
import time
from contextlib import contextmanager

import numpy as np

sub = operator.sub
EPS = 1.0 / 1e9

HYPHEN = chr(45)
DASH_CHARS = (chr(8208), chr(8209), chr(8210), chr(8211), chr(8212), chr(8213), chr(8722))
FORBIDDEN_CHARS = (HYPHEN,) + DASH_CHARS

_BULLET = re.compile(r"(?m)^(\s*)" + re.escape(HYPHEN) + r"\s+")
_NEG_NUMBER = re.compile(r"(?<![\w.])" + re.escape(HYPHEN) + r"(\d)")
_SPACED = re.compile(r"\s+" + "[" + re.escape("".join(FORBIDDEN_CHARS)) + r"]+\s+")
_JOINED = re.compile(r"(?<=\w)" + "[" + re.escape("".join(FORBIDDEN_CHARS)) + r"]+(?=\w)")


def clean_dashes(text):
    """Remove every hyphen and dash from text while keeping it readable.

    Bullets become asterisks, spaced dashes become commas, joined words are split
    with a space and negative numbers are spelled out.
    """
    if not text:
        return text
    text = _BULLET.sub(lambda m: m.group(1) + "* ", text)
    text = _NEG_NUMBER.sub(lambda m: "minus " + m.group(1), text)
    text = _SPACED.sub(", ", text)
    text = _JOINED.sub(" ", text)
    for ch in FORBIDDEN_CHARS:
        text = text.replace(ch, " ")
    return re.sub(r"[ \t]{2,}", " ", text)


def contains_dash(text):
    return any(ch in text for ch in FORBIDDEN_CHARS)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.negative(x)))


def softmax_rows(logits):
    shifted = np.subtract(logits, logits.max(axis=1, keepdims=True))
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def wilson_lower(successes, n, z=1.96):
    """Lower bound of the Wilson score interval, vectorised and safe for n equal to zero."""
    successes = np.asarray(successes, dtype=float)
    n = np.asarray(n, dtype=float)
    safe_n = np.maximum(n, 1.0)
    p = successes / safe_n
    z2 = z * z
    centre = p + z2 / (2.0 * safe_n)
    margin = z * np.sqrt(p * np.subtract(1.0, p) / safe_n + z2 / (4.0 * safe_n * safe_n))
    lower = np.subtract(centre, margin) / (1.0 + z2 / safe_n)
    return np.where(n > 0, np.clip(lower, 0.0, 1.0), 0.0)


def top_k_indices(scores, k):
    """Indices of the k largest scores in descending order, using argpartition."""
    k = int(min(k, scores.shape[0]))
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    part = np.argpartition(np.negative(scores), sub(k, 1))[:k]
    return part[np.argsort(np.negative(scores[part]), kind="stable")]


def last(seq):
    return seq[sub(len(seq), 1)]


def parse_kv(args):
    """Parse command line tokens of the form key=value into a dict plus positionals."""
    options, positionals = {}, []
    for token in args:
        if "=" in token and not token.startswith("="):
            key, value = token.split("=", 1)
            options[key.strip()] = value.strip()
        else:
            positionals.append(token)
    return options, positionals


def fmt_money(value):
    value = float(value)
    if value >= 1e9:
        return "GBP {:.2f}bn".format(value / 1e9)
    if value >= 1e6:
        return "GBP {:.1f}m".format(value / 1e6)
    if value >= 1e3:
        return "GBP {:.0f}k".format(value / 1e3)
    return "GBP {:.0f}".format(value)


def pct(value, digits=0):
    return ("{:." + str(digits) + "f}%").format(100.0 * float(value))


def safe_float(value, digits=4):
    """Round for JSON output so tiny values never render in scientific notation."""
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(("{:." + str(digits) + "f}").format(value))


class Stopwatch:
    """Collects named timings in milliseconds."""

    def __init__(self):
        self.timings = {}

    @contextmanager
    def lap(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = sub(time.perf_counter(), start) * 1000.0
            self.timings[name] = round(elapsed, 3)

    def total(self):
        return round(sum(self.timings.values()), 3)
