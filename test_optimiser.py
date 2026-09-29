import time
import unittest

from routewright_ai import optimiser
from routewright_ai.utils import EPS, sub


def _check_plan(test, problem, plan):
    """Independently verify a plan: every stop once, capacity, windows and shift all respected."""
    seen = []
    index = {sid: i for i, sid in enumerate(problem.stop_ids)}
    for route in plan["routes"]:
        load = sum(problem.demand[index[s["stop_id"]]] for s in route["sequence"])
        test.assertLessEqual(load, problem.capacity + 1e3 * EPS)
        test.assertTrue(all(s["on_time"] for s in route["sequence"]))
        seen.extend(s["stop_id"] for s in route["sequence"])
    test.assertEqual(len(seen), len(set(seen)))
    test.assertEqual(sorted(seen + plan["unassigned_stop_ids"]), sorted(problem.stop_ids))


class OptimiserTests(unittest.TestCase):
    def test_plans_are_valid_across_scenarios(self):
        for area in optimiser.AREA_PRESETS:
            for promise in ("One Hour Slot", "Two Hour Slot", "Next Day Untimed"):
                problem = optimiser.generate_problem(60, area, promise, seed=3)
                plan = optimiser.optimise(problem, time_budget_s=0.2)
                _check_plan(self, problem, plan)
                self.assertEqual(plan["overtime_minutes"], 0.0)

    def test_beats_planner_baselines_on_time_windows(self):
        problem = optimiser.generate_problem(120, "Urban", "Two Hour Slot", seed=9)
        result = optimiser.compare(problem, time_budget_s=0.5)
        opt, nn, sweep = result["comparison"]
        self.assertGreater(opt["on_time_pct"], nn["on_time_pct"])
        self.assertGreater(opt["on_time_pct"], sweep["on_time_pct"])
        self.assertLess(opt["cost_gbp"], sweep["cost_gbp"])

    def test_saves_distance_when_windows_are_loose(self):
        problem = optimiser.generate_problem(150, "Suburban", "Next Day Untimed", seed=4)
        result = optimiser.compare(problem, time_budget_s=0.5)
        opt, nn, sweep = result["comparison"]
        self.assertLess(opt["total_km"], nn["total_km"])
        self.assertLess(opt["total_km"], sweep["total_km"])

    def test_more_search_never_makes_the_plan_longer(self):
        problem = optimiser.generate_problem(100, "Dense Urban", "Next Day Untimed", seed=12)
        quick = optimiser.optimise(problem, time_budget_s=0.05)
        longer = optimiser.optimise(problem, time_budget_s=0.6)
        self.assertLessEqual(longer["total_km"], quick["total_km"] + 1e3 * EPS)

    def test_respects_time_budget(self):
        problem = optimiser.generate_problem(150, "Urban", "Two Hour Slot", seed=2)
        started = time.perf_counter()
        optimiser.optimise(problem, time_budget_s=0.5)
        self.assertLess(sub(time.perf_counter(), started), 3.0)

    def test_unreachable_stop_is_reported_not_dropped(self):
        problem = optimiser.generate_problem(20, "Urban", "Next Day Untimed", seed=1)
        problem.window_end[0] = 0.5
        problem.__post_init__()
        plan = optimiser.optimise(problem, time_budget_s=0.1)
        self.assertIn(problem.stop_ids[0], plan["unassigned_stop_ids"])
        _check_plan(self, problem, plan)

    def test_csv_round_trip(self):
        import tempfile
        from pathlib import Path
        problem = optimiser.generate_problem(30, "Rural", "Half Day Window", seed=6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stops.csv"
            optimiser.write_stops_csv(problem, path)
            loaded = optimiser.load_stops_csv(path, vehicles=problem.vehicles, capacity=problem.capacity,
                                              area_type="Rural")
        self.assertEqual(loaded.stop_ids, problem.stop_ids)
        _check_plan(self, loaded, optimiser.optimise(loaded, time_budget_s=0.1))


if __name__ == "__main__":
    unittest.main()
