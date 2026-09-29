import time
import unittest

import numpy as np

from routewright_ai.retriever import analyse, strip_negated
from routewright_ai.text import stem, tokenize, with_bigrams
from routewright_ai.utils import sub, top_k_indices, wilson_lower
from tests.helpers import engine


class TextTests(unittest.TestCase):
    def test_tokenize_removes_stopwords_and_stems(self):
        self.assertEqual(tokenize("The parcels are being delivered"), ["parcel", "deliver"])

    def test_stem_is_symmetric_for_common_forms(self):
        self.assertEqual(stem("quitting"), stem("quitted"))
        self.assertEqual(stem("loaded"), stem("loading"))

    def test_bigrams(self):
        self.assertEqual(with_bigrams(["a", "b", "c"]), ["a", "b", "c", "a_b", "b_c"])


class QueryUnderstandingTests(unittest.TestCase):
    def test_negated_symptoms_are_removed(self):
        self.assertNotIn("traffic", strip_negated("drops are late but traffic is normal"))
        self.assertNotIn("volume", strip_negated("routes overrun though volume has not changed"))

    def test_context_detection(self):
        a = analyse("our grocery vans in the city centre keep missing booked slots, and our lorries too")
        self.assertEqual(a.service_type, "Grocery Home Delivery")
        self.assertEqual(a.area_type, "Dense Urban")
        self.assertEqual(a.fleet_type, "Rigid Trucks")
        self.assertEqual(a.detected[0], "window_misses")


class UtilityTests(unittest.TestCase):
    def test_top_k_matches_full_sort(self):
        scores = np.random.default_rng(0).random(5000)
        self.assertEqual(list(top_k_indices(scores, 20)), list(np.flip(np.argsort(scores))[:20]))

    def test_wilson_prefers_more_evidence(self):
        small, large = wilson_lower([9, 160], [10, 200])
        self.assertGreater(large, small)
        self.assertEqual(float(wilson_lower(0, 0)), 0.0)


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = engine()

    def test_bm25_and_dense_find_battery_incidents(self):
        idx = self.eng.index
        toks = tokenize("electric vans running out of battery charge")
        for scores in (idx.bm25_scores(toks), idx.dense_scores(toks)):
            labels = [idx.label_of("incident_type", r) for r in top_k_indices(scores, 10)]
            self.assertGreaterEqual(labels.count("Electric van range shortfall"), 6)

    def test_unknown_words_score_zero(self):
        self.assertEqual(float(self.eng.index.bm25_scores(["zzzqqq"]).max()), 0.0)

    def test_retrieval_is_fast(self):
        start = time.perf_counter()
        for _ in range(50):
            self.eng.retriever.retrieve("vans leave the depot late every morning")
        self.assertLess(sub(time.perf_counter(), start) / 50 * 1000, 50.0)

    def test_filters_are_respected(self):
        res = self.eng.retriever.retrieve("missed time windows", filters={"service_type": "Builders Merchant"})
        labels = {self.eng.index.label_of("service_type", r) for r in res.candidates[:50]}
        self.assertEqual(labels, {"Builders Merchant"})


if __name__ == "__main__":
    unittest.main()
