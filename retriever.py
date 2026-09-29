"""Query understanding and hybrid retrieval.

Pipeline for one question:

1. Analyse the query: detect likely incident types from the domain thesaurus
   and pick up service type, area type and fleet context.
2. Expand the query with the canonical names of the detected incidents.
3. Score every incident with BM25 and with dense semantic similarity.
4. Fuse the two rankings with Reciprocal Rank Fusion.
5. Apply soft metadata boosts from the analysis and hard filters from the caller.
6. Diagnose with a similarity weighted vote over the top incidents.
7. Select a diverse context set with Maximal Marginal Relevance.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .text import tokenize
from .utils import sub, top_k_indices
from .vocab import FIXES, INCIDENT_KEYS, INCIDENTS

SERVICE_SYNONYMS = {
    "Parcel Courier": ["parcel", "parcels", "courier", "couriers", "multi drop"],
    "Ecommerce Same Day": ["same day", "ecommerce", "online orders", "quick commerce"],
    "Grocery Home Delivery": ["grocery", "groceries", "supermarket", "food shopping", "chilled"],
    "Two Person Furniture Delivery": ["furniture", "sofa", "sofas", "two person", "white goods", "bulky"],
    "Pharmacy and Healthcare": ["pharmacy", "prescriptions", "medicines", "healthcare", "nhs"],
    "Food Service Wholesale": ["food service", "wholesale", "restaurants", "catering", "pubs"],
    "Builders Merchant": ["builders merchant", "building materials", "construction sites", "timber", "plasterboard"],
    "Retail Store Replenishment": ["store replenishment", "store deliveries", "shops", "retail stores"],
    "Pallet Network Final Mile": ["pallet", "pallets", "pallet network", "freight"],
    "Meal Kit Subscription": ["meal kit", "meal kits", "recipe boxes", "subscription boxes"],
}
AREA_SYNONYMS = {
    "Dense Urban": ["city centre", "central london", "dense urban", "inner city", "downtown"],
    "Urban": ["urban", "city", "town centre", "metropolitan"],
    "Suburban": ["suburban", "suburbs", "commuter towns", "housing estates"],
    "Rural": ["rural", "countryside", "villages", "farms", "remote", "highlands"],
    "Mixed Region": ["mixed region", "whole region", "regional network"],
}
FLEET_SYNONYMS = {
    "Electric Vans": ["electric vans", "ev fleet", "electric fleet", "evs"],
    "Diesel Vans": ["diesel vans", "diesel"],
    "Mixed Van Fleet": ["mixed fleet"],
    "Rigid Trucks": ["rigid", "rigids", "lorries", "lorry", "trucks", "hgv", "hgvs"],
    "Cargo Bikes and Vans": ["cargo bike", "cargo bikes", "e cargo"],
}


@lru_cache(maxsize=4096)
def _pattern(phrase):
    return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")


def _phrase_hits(text, phrases):
    hits = 0
    for phrase in phrases:
        p = phrase.strip()
        if p and p in text and _pattern(p).search(text):
            hits += 1 + p.count(" ")
    return hits


def _detect(text, table):
    scores = {k: _phrase_hits(text, v) for k, v in table.items()}
    best = max(scores, key=scores.get) if scores else None
    return best if best and scores[best] > 0 else None


@lru_cache(maxsize=1)
def _stemmed_keywords():
    """Thesaurus normalised with the query tokenizer, so "quitting" matches "quit"."""
    table = {}
    for key in INCIDENT_KEYS:
        phrases = INCIDENTS[key]["keywords"] + [INCIDENTS[key]["name"].lower()]
        table[key] = sorted({" ".join(tokenize(p)) for p in phrases if tokenize(p)})
    return table


@dataclass
class QueryAnalysis:
    text: str
    tokens: list
    prior: np.ndarray
    detected: list
    service_type: str = None
    area_type: str = None
    fleet_type: str = None
    mentioned_fixes: list = field(default_factory=list)

    def as_dict(self):
        return {
            "detected_incidents": [INCIDENTS[k]["name"] for k in self.detected],
            "service_type": self.service_type, "area_type": self.area_type, "fleet_type": self.fleet_type,
            "mentioned_fixes": [FIXES[k]["name"] for k in self.mentioned_fixes],
        }


_NEGATED = re.compile(
    r"\b(\w+)\s+(?:is|are|looks|look|seems|seem|stayed|stays|remains|remain|is still|are still)\s+"
    r"(?:normal|fine|stable|steady|unchanged|ok|okay|healthy|flat|as usual)\b"
    r"|\b(\w+)\s+(?:has|have)\s+not\s+changed\b")


def strip_negated(question):
    """Remove symptoms the user explicitly rules out, such as "traffic is normal".

    Without this, a mention of a healthy metric pulls retrieval towards incidents
    about that metric, the classic negation failure of keyword and embedding search.
    """
    return _NEGATED.sub(" ", question.lower())


def analyse(question):
    cleaned = strip_negated(question)
    text = " " + cleaned + " "
    stemmed = " " + " ".join(tokenize(cleaned)) + " "
    table = _stemmed_keywords()
    hits = np.array([_phrase_hits(stemmed, table[k]) for k in INCIDENT_KEYS], dtype=float)
    prior = hits / hits.sum() if hits.sum() > 0 else np.zeros(len(INCIDENT_KEYS))
    order = np.argsort(np.negative(hits), kind="stable")
    detected = [INCIDENT_KEYS[i] for i in order[:3] if hits[i] > 0]
    mentioned = [k for k, v in FIXES.items() if _phrase_hits(text, v["keywords"]) > 0]
    return QueryAnalysis(
        text=question, tokens=tokenize(cleaned), prior=prior, detected=detected,
        service_type=_detect(text, SERVICE_SYNONYMS), area_type=_detect(text, AREA_SYNONYMS),
        fleet_type=_detect(text, FLEET_SYNONYMS), mentioned_fixes=mentioned,
    )


@dataclass
class RetrievalResult:
    analysis: QueryAnalysis
    candidates: np.ndarray
    fused: np.ndarray
    bm25_top: np.ndarray
    dense_top: np.ndarray
    agreement: float
    diagnosis: list
    context: np.ndarray


class HybridRetriever:
    def __init__(self, index, settings):
        self.index = index
        self.settings = settings
        labels = index.labels["incident_type"]
        by_name = {INCIDENTS[k]["name"]: i for i, k in enumerate(INCIDENT_KEYS)}
        self._code_to_key = np.array([by_name[lab] for lab in labels])

    def _mask(self, filters):
        mask = np.ones(self.index.size, dtype=bool)
        for column, wanted in (filters or {}).items():
            if column not in self.index.codes or wanted in (None, "", []):
                continue
            values = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
            codes = [c for c in (self.index.code_of(column, v) for v in values) if c is not None]
            mask &= np.isin(self.index.codes[column], codes)
        return mask

    def keys_of(self, rows):
        return self._code_to_key[self.index.codes["incident_type"][rows]]

    def retrieve(self, question, filters=None, k=None, pool=None, analysis=None, mode="hybrid"):
        s = self.settings
        k = k or s.context_cases
        pool = pool or s.candidate_pool
        qa = analysis or analyse(question)
        expansion = []
        if mode == "hybrid":
            for key in qa.detected[:2]:
                expansion.extend(tokenize(INCIDENTS[key]["name"]))
        tokens = qa.tokens + expansion

        mask = self._mask(filters)
        floor = np.float32(np.negative(1e9))
        rrf_k = float(s.rrf_k)
        ranks = np.arange(1, pool + 1, dtype=np.float32)
        fused = np.zeros(self.index.size, dtype=np.float32)
        bm_top = de_top = np.zeros(0, dtype=np.int64)

        if mode in ("hybrid", "fusion", "bm25"):
            bm = np.where(mask, self.index.bm25_scores(tokens), floor)
            bm_top = top_k_indices(bm, pool)
            bm_top = bm_top[bm[bm_top] > 0]
            fused[bm_top] += 1.0 / (rrf_k + ranks[:len(bm_top)])
        if mode in ("hybrid", "fusion", "dense"):
            de = np.where(mask, self.index.dense_scores(tokens), floor)
            de_top = top_k_indices(de, pool)
            de_top = de_top[de[de_top] > 0]
            fused[de_top] += 1.0 / (rrf_k + ranks[:len(de_top)])

        if mode == "hybrid":
            code_prior = qa.prior[self._code_to_key]
            boost = 1.0 + 0.6 * code_prior[self.index.codes["incident_type"]]
            for column, value, weight in (("service_type", qa.service_type, 0.15), ("area_type", qa.area_type, 0.10),
                                          ("fleet_type", qa.fleet_type, 0.06)):
                if value:
                    code = self.index.code_of(column, value)
                    boost = boost + weight * (self.index.codes[column] == code)
            fused = fused * boost.astype(np.float32)
        fused = np.where(mask, fused, 0.0).astype(np.float32)

        candidates = top_k_indices(fused, pool)
        candidates = candidates[fused[candidates] > 0]
        top_n = 50
        agreement = 0.0
        if len(bm_top) and len(de_top):
            agreement = len(np.intersect1d(bm_top[:top_n], de_top[:top_n])) / float(top_n)
        diagnosis = self._diagnose(candidates, fused, qa)
        context = self._mmr(candidates[:60], fused, k)
        return RetrievalResult(qa, candidates, fused, bm_top, de_top, agreement, diagnosis, context)

    def _diagnose(self, candidates, fused, qa, top=25):
        head = candidates[:top]
        if len(head) == 0:
            order = np.argsort(np.negative(qa.prior))
            return [(INCIDENT_KEYS[i], float(qa.prior[i])) for i in order[:2] if qa.prior[i] > 0]
        weights = fused[head].astype(float)
        vote = np.bincount(self.keys_of(head), weights=weights, minlength=len(INCIDENT_KEYS))
        vote = vote / max(vote.sum(), 1.0 / 1e9)
        share = 0.7 * vote + 0.3 * qa.prior if qa.prior.sum() > 0 else vote
        order = np.argsort(np.negative(share), kind="stable")
        result = [(INCIDENT_KEYS[order[0]], float(share[order[0]]))]
        for nxt in order[1:3]:
            if share[nxt] >= 0.18:
                result.append((INCIDENT_KEYS[nxt], float(share[nxt])))
        return result

    def _mmr(self, candidates, fused, k):
        if len(candidates) <= k:
            return candidates
        lam = self.settings.mmr_lambda
        emb = np.asarray(self.index.embeddings[candidates])
        rel = fused[candidates].astype(float)
        rel = rel / max(float(rel.max()), 1.0 / 1e9)
        sims = emb @ emb.T
        chosen = [0]
        max_sim = sims[0].copy()
        for _ in range(sub(k, 1)):
            score = np.subtract(lam * rel, sub(1.0, lam) * max_sim)
            score[chosen] = np.negative(np.inf)
            nxt = int(np.argmax(score))
            chosen.append(nxt)
            max_sim = np.maximum(max_sim, sims[nxt])
        return candidates[chosen]
