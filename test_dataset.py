import re
import unittest

import pandas as pd

from routewright_ai.config import settings
from routewright_ai.generator import COLUMNS, generate
from routewright_ai.utils import FORBIDDEN_CHARS
from routewright_ai.vocab import FIX_NAMES, INCIDENT_NAMES
from tests.helpers import engine


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        engine()
        cls.frame = pd.read_csv(settings.dataset_path)

    def test_minimum_shape(self):
        rows, cols = self.frame.shape
        self.assertGreaterEqual(rows, 16000)
        self.assertGreaterEqual(cols, 45)
        self.assertEqual(list(self.frame.columns), COLUMNS)

    def test_ids_unique_and_complete(self):
        self.assertTrue(self.frame["incident_id"].is_unique)
        self.assertFalse(self.frame.isnull().any().any())

    def test_no_dash_characters_in_any_cell(self):
        blob = self.frame.to_csv(index=False)
        for ch in FORBIDDEN_CHARS:
            self.assertNotIn(ch, blob)

    def test_all_numbers_non_negative(self):
        self.assertTrue((self.frame.select_dtypes("number") >= 0).all().all())

    def test_acronyms_keep_their_case(self):
        text = " ".join(self.frame["incident_summary"].head(4000)) + " " + " ".join(self.frame["lessons_learned"].head(4000))
        broken = [w for w in text.split() if len(w) > 2 and w[0].islower() and w[1:3].isupper()]
        self.assertEqual(broken, [])

    def test_internal_consistency(self):
        f = self.frame
        ratio = f["incident_on_time_rate_pct"] / f["baseline_on_time_rate_pct"]
        self.assertLess((ratio.sub(f["on_time_retention_ratio"])).abs().max(), 0.01)
        self.assertTrue((f["on_time_retention_ratio"] < 1).all())
        self.assertTrue(f.loc[f["recovery_status"] == "Not Recovered", "days_to_recover"].eq(90).all())
        self.assertTrue((f["vehicles_in_fleet"] * f["avg_stops_per_route"] >= f["daily_drops"]).all())

    def test_dates_are_ordered(self):
        pattern = re.compile(r"^\d{4}/\d{2}/\d{2}$")
        for col in ("detection_date", "resolution_date"):
            self.assertTrue(self.frame[col].str.match(pattern).all())
        self.assertTrue((self.frame["detection_date"] <= self.frame["resolution_date"]).all())

    def test_vocabularies_and_balance(self):
        self.assertTrue(set(self.frame["incident_type"]) <= set(INCIDENT_NAMES))
        self.assertTrue(set(self.frame["fix_strategy"]) <= set(FIX_NAMES))
        counts = self.frame["incident_type"].value_counts()
        self.assertEqual(len(counts), 15)
        self.assertGreater(counts.min(), 600)

    def test_narratives_quote_their_own_numbers(self):
        for _, row in self.frame.head(300).iterrows():
            self.assertIn("{:.1f}".format(row["post_fix_on_time_rate_pct"]), row["resolution_narrative"])

    def test_generator_is_deterministic(self):
        pd.testing.assert_frame_equal(generate(300, seed=5), generate(300, seed=5))

    def test_efficacy_signal_present(self):
        f = self.frame
        full = f["recovery_status"].eq("Full Recovery")
        instinctive = f["fix_strategy"].isin(["Extended driver shifts", "Loading more stops onto each route"])
        self.assertGreater(full[~instinctive].mean(), 2 * full[instinctive].mean())

    def test_visibility_drives_detection(self):
        lag = self.frame.groupby("visibility_level")["detection_lag_hours"].median()
        self.assertGreater(lag["Basic Reporting"], lag["Daily KPI Dashboards"])
        self.assertGreater(lag["Daily KPI Dashboards"], lag["Real Time Control Tower"])


if __name__ == "__main__":
    unittest.main()
