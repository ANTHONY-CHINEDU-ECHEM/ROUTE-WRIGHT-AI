"""Command line interface.

Usage: python routewright.py <command> [key=value ...]

Options are written as key=value pairs so the whole toolchain works without
flag syntax. Run python routewright.py help for the full list of commands.
"""

import json
import sys
import textwrap
import time

from .config import settings
from .utils import parse_kv, sub

HELP = """
Routewright AI command line

  python routewright.py all                generate, build, evaluate, check style and test in one go
  python routewright.py generate rows=16000 seed=42
                                           create the synthetic delivery incident dataset
  python routewright.py build              build the hybrid index and train the risk models
  python routewright.py ask "question" provider=extractive k=8 json=0
                                           diagnose a delivery problem and get evidence ranked fixes
  python routewright.py search "query" k=10 mode=hybrid
                                           find the most similar historical incidents
  python routewright.py recommend incident="Route overruns" service_type="Parcel Courier"
                                           evidence table for every fix to a known incident
  python routewright.py assess planning_tool="Manual Planning" visibility_level="Basic Reporting"
                                           predict incident exposure for an operator profile
  python routewright.py optimise stops=120 area="Urban" promise="Two Hour Slot" seconds=1.5
  python routewright.py optimise file=data/sample_stops.csv vehicles=9 capacity=74 area="Urban"
                                           plan routes with time windows and compare with planner baselines
  python routewright.py evaluate queries=400
                                           run the benchmark and write docs/EVALUATION.md
  python routewright.py export_sft grounded=1000
                                           write fine tuning data to data/sft/routewright_sft.jsonl
  python routewright.py serve host=0.0.0.0 port=8000
                                           start the REST API (docs at /docs)
  python routewright.py ui                 start the web interface
  python routewright.py test               run the test suite
  python routewright.py check_style        confirm no hyphen or dash characters exist in the project
"""


def _engine():
    from .engine import RoutewrightEngine
    return RoutewrightEngine.shared()


def cmd_generate(opts, args):
    from .engine import build_all
    build_all(settings, rows=int(opts.get("rows", settings.rows)), seed=int(opts.get("seed", settings.seed)),
              regenerate=True)


def cmd_build(opts, args):
    from .engine import build_all
    build_all(settings, regenerate=opts.get("regenerate", "0") == "1")


def cmd_ask(opts, args):
    question = " ".join(args) or opts.get("q", "")
    if not question:
        print("Ask a question, for example: python routewright.py ask \"our vans keep missing the booked delivery slots\"")
        return
    filters = {k: v for k, v in opts.items() if k in ("service_type", "area_type", "fleet_type", "region",
                                                       "operator_size_band", "planning_tool", "visibility_level")}
    result = _engine().ask(question, filters=filters or None, k=int(opts.get("k", settings.context_cases)),
                           provider=opts.get("provider"))
    if opts.get("json", "0") == "1":
        print(json.dumps({k: v for k, v in result.items()}, indent=2, default=str))
        return
    print()
    print(result["answer"])
    print()
    v = result["verification"]
    print("Provider {} | grounding {} | citations verified {} | {} ms".format(
        result["provider"], v["grounding_score"], "yes" if not v["invalid_ids"] else "no",
        result["timings_ms"]["total"]))
    for note in result["notes"]:
        print("Note: " + note)


def cmd_search(opts, args):
    query = " ".join(args) or opts.get("q", "")
    out = _engine().search(query, k=int(opts.get("k", 10)), mode=opts.get("mode", "hybrid"))
    for hit in out["results"]:
        print("{}  {:.3f}  {} | {} | {} | {} | {}".format(
            hit["incident_id"], hit["score"], hit["service_type"], hit["area_type"], hit["incident_type"],
            hit["fix_strategy"], hit["recovery_status"]))
        print(textwrap.indent(textwrap.fill(hit["incident_summary"], 100), "    "))
    print("{} ms".format(out["timings_ms"]["retrieve"]))


def cmd_recommend(opts, args):
    incident = opts.get("incident") or " ".join(args)
    block = _engine().recommend(incident, service_type=opts.get("service_type"), area_type=opts.get("area_type"),
                                fleet_type=opts.get("fleet_type"))
    print("Reference class: " + block["reference_class"])
    print("Baseline full recovery: {:.0%}".format(block["baseline_full_recovery"]))
    for s in block["all_fixes"]:
        print("  {:<55} n={:<5} full {:>4.0%}  lower bound {:>4.0%}  median avoided GBP {:,.0f}".format(
            s["fix"], s["cases"], s["full_recovery_rate"], s["confidence_lower_bound"], s["median_cost_avoided_gbp"]))


def cmd_assess(opts, args):
    numeric = {"daily_drops", "avg_stops_per_route", "avg_route_km", "avg_consignment_weight_kg",
               "time_window_minutes", "agency_driver_share_pct", "depot_count", "electric_share_pct"}
    profile = {k: (float(v) if k in numeric else v) for k, v in opts.items()}
    print(json.dumps(_engine().assess(profile), indent=2))


def cmd_optimise(opts, args):
    from . import optimiser
    seconds = float(opts.get("seconds", 1.5))
    area = opts.get("area", "Urban")
    if "file" in opts:
        problem = optimiser.load_stops_csv(opts["file"], vehicles=int(opts.get("vehicles", 10)),
                                           capacity=float(opts.get("capacity", 80)),
                                           shift_minutes=float(opts.get("shift", 600)), area_type=area)
    else:
        problem = optimiser.generate_problem(int(opts.get("stops", 120)), area, opts.get("promise", "Two Hour Slot"),
                                             seed=int(opts.get("seed", 7)))
    result = optimiser.compare(problem, time_budget_s=seconds)
    print("Problem: {stops} stops, {vehicles} vehicles of capacity {capacity}, shift {shift_minutes:.0f} minutes".format(
        **result["problem"]))
    print("{:<38} {:>8} {:>9} {:>9} {:>9} {:>10} {:>9}".format("Method", "On time", "Km", "Vehicles", "Overtime",
                                                                 "Cost GBP", "Runtime"))
    for row in result["comparison"]:
        print("{:<38} {:>7.1f}% {:>9.1f} {:>9} {:>9.0f} {:>10.2f} {:>7.0f}ms".format(
            row["method"], row["on_time_pct"], row["total_km"], row["vehicles_used"], row["overtime_minutes"],
            row["cost_gbp"], row["runtime_ms"]))
    print("On time gain over the best baseline: {} points. Distance saving against sweep: {}%.".format(
        result["on_time_gain_points"], result["km_saving_pct_vs_sweep"]))
    for route in result["plan"]["routes"][:int(opts.get("show", 3))]:
        seq = ", ".join("{} {}".format(s["stop_id"], s["arrival"]) for s in route["sequence"][:8])
        print("  Vehicle {}: {} stops, {} km, back at {}. {}{}".format(
            route["vehicle"], route["stops"], route["km"], route["finish"], seq,
            " ..." if route["stops"] > 8 else ""))
    if "out" in opts:
        with open(opts["out"], "w", encoding="utf8") as fh:
            json.dump(result, fh, indent=2)
        print("Wrote " + opts["out"])


def cmd_evaluate(opts, args):
    from . import evaluate
    report = evaluate.run(_engine(), int(opts.get("queries", 400)))
    evaluate.write(report, settings.report_dir, settings.docs_dir)
    print("Wrote reports/evaluation.json and docs/EVALUATION.md")


def cmd_export_sft(opts, args):
    from . import sft
    path = settings.data_dir / "sft" / "routewright_sft.jsonl"
    cases = opts.get("cases")
    sft.export(_engine(), path, grounded=int(opts.get("grounded", 1000)),
               cases=int(cases) if cases else None)
    print("Wrote " + str(path))


def cmd_serve(opts, args):
    import uvicorn
    from .api import app
    _engine()
    uvicorn.run(app, host=opts.get("host", "127.0.0.1"), port=int(opts.get("port", 8000)),
                workers=1, log_level=opts.get("log", "info"))


def cmd_ui(opts, args):
    from pathlib import Path
    from streamlit.web import cli as stcli
    app_path = str(Path(__file__).resolve().parent / "ui_app.py")
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())


def cmd_test(opts, args):
    import unittest
    from .config import ROOT
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


def cmd_check_style(opts, args):
    from .style import scan
    problems = scan()
    if problems:
        for path, line_no, snippet in problems[:50]:
            print("{}:{}: {}".format(path, line_no, snippet))
        print("{} lines contain hyphen or dash characters".format(len(problems)))
        sys.exit(1)
    print("Clean: no hyphen or dash characters found in the project.")


def cmd_all(opts, args):
    started = time.perf_counter()
    cmd_generate(opts, args)
    cmd_evaluate(opts, args)
    cmd_check_style(opts, args)
    print("Completed in {:.1f}s".format(sub(time.perf_counter(), started)))
    cmd_test(opts, args)


COMMANDS = {
    "generate": cmd_generate, "build": cmd_build, "ask": cmd_ask, "search": cmd_search,
    "recommend": cmd_recommend, "assess": cmd_assess, "optimise": cmd_optimise, "evaluate": cmd_evaluate,
    "export_sft": cmd_export_sft, "serve": cmd_serve, "ui": cmd_ui, "test": cmd_test,
    "check_style": cmd_check_style, "all": cmd_all,
}


def main(argv):
    if not argv or argv[0] in ("help", "h"):
        print(HELP)
        return
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print("Unknown command {}. Run python routewright.py help".format(command))
        sys.exit(2)
    opts, args = parse_kv(rest)
    COMMANDS[command](opts, args)
