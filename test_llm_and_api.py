import os
import unittest

import numpy as np
from fastapi.testclient import TestClient

from routewright_ai.api import app
from routewright_ai.config import settings
from routewright_ai.llm import generate_answer, verify
from routewright_ai.risk_model import auc
from routewright_ai.utils import sub
from tests.helpers import engine


class FabricatingProvider:
    name = "fake"

    def generate(self, question, result, pack):
        return "Extend every shift. It worked in 97% of cases [INC999999]."


class BrokenProvider:
    name = "broken"

    def generate(self, question, result, pack):
        raise RuntimeError("network down")


class VerifierTests(unittest.TestCase):
    def test_detects_invented_ids_and_numbers(self):
        check = verify("See [INC000001] and [INC123456]; success 88%.", "evidence INC000001 71%", ["INC000001"])
        self.assertEqual(check["invalid_ids"], ["INC123456"])
        self.assertEqual(check["unsupported_percentages"], [88.0])
        self.assertFalse(check["passed"])

    def test_strict_mode_replaces_fabricated_answer(self):
        result = engine().ask("Routes keep running over the end of shift", provider="extractive")
        out = generate_answer("q", result, settings, provider=FabricatingProvider(), strict=True)
        self.assertEqual(out["provider"], "extractive")
        self.assertTrue(out["verification"]["passed"])
        self.assertTrue(out["notes"])

    def test_provider_failure_falls_back(self):
        result = engine().ask("Routes keep running over the end of shift", provider="extractive")
        self.assertEqual(generate_answer("q", result, settings, provider=BrokenProvider())["provider"], "extractive")

    def test_auc(self):
        self.assertAlmostEqual(auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])), 1.0)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop("ROUTEWRIGHT_API_KEY", None)
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertGreaterEqual(body["incidents"], 16000)

    def test_ask(self):
        r = self.client.post("/ask", json={"question": "Customers say parcels never arrived but show as delivered", "k": 5})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["diagnosis"][0]["incident"], "Proof of delivery disputes")

    def test_validation(self):
        self.assertEqual(self.client.post("/ask", json={"question": ""}).status_code, 422)
        self.assertEqual(self.client.post("/search", json={"query": "x y", "mode": "magic"}).status_code, 422)

    def test_search_recommend_assess_incident(self):
        self.assertEqual(len(self.client.post("/search", json={"query": "depot loading delays", "k": 4}).json()["results"]), 4)
        self.assertEqual(self.client.post("/recommend", json={"incident": "Route overruns"}).status_code, 200)
        self.assertEqual(self.client.post("/recommend", json={"incident": "Unknown"}).status_code, 404)
        self.assertEqual(self.client.post("/assess", json={"agency_driver_share_pct": 30}).status_code, 200)
        self.assertEqual(self.client.get("/incidents/INC000010").json()["incident_id"], "INC000010")
        self.assertEqual(self.client.get("/incidents/INC999999").status_code, 404)

    def test_optimise_demo_and_custom_stops(self):
        r = self.client.post("/optimise", json={"demo_stops": 40, "area_type": "Suburban", "seconds": 0.2,
                                                "include_routes": False})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["plan"]["on_time_pct"], 100.0)
        stops = [{"stop_id": "A{}".format(i), "x_km": 1 + i % 5, "y_km": 1 + i // 5, "demand": 2} for i in range(15)]
        r = self.client.post("/optimise", json={"stops": stops, "vehicles": 2, "capacity": 20, "seconds": 0.2})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["plan"]["stops_served"], 15)
        bad = [{"stop_id": "B", "x_km": sub(0.0, 1.0), "y_km": 1.0}]
        self.assertEqual(self.client.post("/optimise", json={"stops": bad}).status_code, 422)
        self.assertEqual(self.client.post("/optimise", json={"area_type": "Moon"}).status_code, 422)

    def test_bearer_auth(self):
        os.environ["ROUTEWRIGHT_API_KEY"] = "secret"
        try:
            self.assertEqual(self.client.get("/stats").status_code, 401)
            self.assertEqual(self.client.get("/stats", headers={"Authorization": "Bearer secret"}).status_code, 200)
            self.assertEqual(self.client.get("/health").status_code, 200)
        finally:
            os.environ.pop("ROUTEWRIGHT_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
