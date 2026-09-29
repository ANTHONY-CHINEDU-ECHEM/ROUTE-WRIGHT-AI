import unittest

from routewright_ai.vocab import EFFICACY_MAP, FIXES, INCIDENTS
from tests.helpers import engine


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = engine()

    def test_top_fix_is_strong_for_every_incident(self):
        for key, inc in INCIDENTS.items():
            top = self.eng.recommend(inc["name"])["recommended"][0]["fix"]
            strong = [FIXES[k]["name"] for k in EFFICACY_MAP[key]["strong"]]
            self.assertIn(top, strong, "{} got {}".format(inc["name"], top))

    def test_harmful_fixes_never_recommended(self):
        for key, inc in INCIDENTS.items():
            recommended = {s["fix"] for s in self.eng.recommend(inc["name"])["recommended"]}
            harmful = {FIXES[k]["name"] for k in EFFICACY_MAP[key]["harmful"]}
            self.assertFalse(recommended & harmful)

    def test_reference_class_narrows_only_when_large_enough(self):
        block = self.eng.recommend("Route overruns", service_type="Parcel Courier")
        count = int(block["reference_class"].split()[0])
        self.assertGreaterEqual(count, 200 if block["narrowed_by"] else 1)


class AskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = engine()
        cls.question = "Nobody is home when our parcel drivers turn up so we keep going back twice"
        cls.result = cls.eng.ask(cls.question)

    def test_diagnosis(self):
        self.assertEqual(self.result["diagnosis"][0]["incident"], "Failed first attempt deliveries")

    def test_answer_is_grounded(self):
        v = self.result["verification"]
        self.assertTrue(v["passed"])
        self.assertEqual(v["invalid_ids"], [])
        self.assertEqual(v["citation_precision"], 1.0)

    def test_answer_structure(self):
        for heading in ("### Diagnosis", "### Recommended fixes", "### What to expect", "### Confidence"):
            self.assertIn(heading, self.result["answer"])

    def test_cache(self):
        self.assertTrue(self.eng.ask(self.question)["cached"])

    def test_negation_changes_the_diagnosis(self):
        r = self.eng.ask("Vans are leaving the depot late because loading is slow, though traffic is normal")
        self.assertEqual(r["diagnosis"][0]["incident"], "Late depot departures")

    def test_routing_problems_point_to_the_optimiser(self):
        r = self.eng.ask("Our drivers keep arriving outside the time slot customers booked")
        self.assertIn("routewright.py optimise", r["answer"])
        self.assertTrue(r["verification"]["passed"])

    def test_multi_incident_question(self):
        r = self.eng.ask("Our electric vans run out of battery and parcels are being stolen from doorsteps")
        names = {d["incident"] for d in r["diagnosis"]}
        self.assertTrue({"Electric van range shortfall", "Proof of delivery disputes"} & names)

    def test_off_topic_question_is_handled(self):
        r = self.eng.ask("What is the capital of France?")
        self.assertTrue(not r["diagnosis"] or r["confidence"]["label"] == "Low")

    def test_assess(self):
        out = self.eng.assess({"planning_tool": "Manual Planning", "visibility_level": "Basic Reporting",
                               "agency_driver_share_pct": 40})
        for risk in out["risks"].values():
            self.assertTrue(0.0 < risk["probability"] < 1.0)
        self.assertEqual(len(out["playbook"]), 3)


if __name__ == "__main__":
    unittest.main()
