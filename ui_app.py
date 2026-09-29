"""Routewright AI web interface. Start with: python routewright.py ui"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routewright_ai import optimiser  # noqa: E402
from routewright_ai.engine import RoutewrightEngine  # noqa: E402
from routewright_ai.utils import fmt_money  # noqa: E402

st.set_page_config(page_title="Routewright AI", layout="wide")

EXAMPLES = [
    "Our grocery drivers keep arriving outside the one hour slot customers booked. What should we do?",
    "Nobody is home when we deliver parcels so we keep going back twice.",
    "Drivers cannot finish their rounds before the end of their shift and we pay overtime every day.",
    "Our electric vans run out of battery before the round ends now it is winter.",
    "Vans are leaving the depot far too late every morning because loading takes forever.",
]


@st.cache_resource(show_spinner="Loading the Routewright index")
def engine():
    return RoutewrightEngine.shared()


def pct(x):
    return "{:.0%}".format(x)


def page_ask(eng):
    st.header("Diagnose a delivery problem")
    st.caption("Describe what is going wrong in your own words. Routewright diagnoses the incident, finds comparable "
               "incidents and ranks fixes by how often they genuinely restored on time delivery.")
    example = st.selectbox("Start from an example or write your own", ["Write my own"] + EXAMPLES)
    question = st.text_area("What is happening?", value="" if example == "Write my own" else example, height=110)
    opts = eng.options()
    with st.expander("Filters and generation settings"):
        c1, c2, c3 = st.columns(3)
        services = c1.multiselect("Service type", opts["service_types"])
        areas = c2.multiselect("Area type", opts["area_types"])
        fleets = c3.multiselect("Fleet", opts["fleet_types"])
        c4, c5 = st.columns(2)
        provider = c4.selectbox("Answer provider", ["extractive", "anthropic", "ollama"])
        k = c5.slider("Precedent incidents in context", 3, 15, 8)
    if not st.button("Diagnose", type="primary") or not question.strip():
        return
    filters = {c: v for c, v in (("service_type", services), ("area_type", areas), ("fleet_type", fleets)) if v}
    result = eng.ask(question, filters=filters or None, k=k, provider=provider)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Diagnosis", result["diagnosis"][0]["incident"] if result["diagnosis"] else "Unclear")
    m2.metric("Confidence", "{} ({:.2f})".format(result["confidence"]["label"], result["confidence"]["score"]))
    m3.metric("Grounding score", "{:.2f}".format(result["verification"]["grounding_score"]))
    m4.metric("Answer time", "{:.1f} ms".format(result["timings_ms"]["total"]))
    for note in result["notes"]:
        st.info(note)
    st.markdown(result["answer"])
    tab_e, tab_c, tab_v = st.tabs(["Evidence", "Precedent incidents", "Verification"])
    with tab_e:
        for block in result["recommendations"]:
            st.subheader(block["incident"])
            st.caption("Reference class: {}. Baseline full recovery {}.".format(
                block["reference_class"], pct(block["baseline_full_recovery"])))
            table = pd.DataFrame(block["all_fixes"])
            if not table.empty:
                table = table[["fix", "cases", "full_recovery_rate", "confidence_lower_bound",
                               "median_cost_avoided_gbp", "median_days_to_recover", "median_fix_cost_gbp",
                               "repeat_rate_90d"]]
                st.dataframe(table, hide_index=True, width="stretch")
                st.bar_chart(table.set_index("fix")["confidence_lower_bound"])
    with tab_c:
        for c in result["cases"]:
            with st.expander("{}  {}  ({})".format(c["incident_id"], c["operator_name"], c["recovery_status"])):
                st.write("**Incident.** " + c["incident_summary"])
                st.write("**Resolution.** " + c["resolution_narrative"])
                st.write("**Lesson.** " + c["lessons_learned"])
                st.caption("{} | {} | {} | incident cost {} | cost avoided {}".format(
                    c["service_type"], c["area_type"], c["planning_tool"], fmt_money(c["incident_cost_gbp"]),
                    fmt_money(c["cost_avoided_gbp_30d"])))
    with tab_v:
        st.json(result["verification"])
        st.json(result["timings_ms"])


def page_optimise(eng):
    st.header("Route optimiser")
    st.caption("Plan delivery routes that respect every time window, vehicle capacity and shift limit, and compare "
               "the plan with nearest neighbour and sweep territory planning on the same travel model.")
    source = st.radio("Stops", ["Generate a demo day", "Upload a CSV"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    area = c1.selectbox("Area type", list(optimiser.AREA_PRESETS), index=1)
    seconds = c3.slider("Search time in seconds", 0.2, 10.0, 1.5)
    if source == "Generate a demo day":
        promise = c2.selectbox("Delivery promise", list(optimiser.PROMISE_WINDOWS), index=1)
        stops = st.slider("Stops", 20, 400, 120)
        seed = st.number_input("Scenario seed", 0, 10000, 7)
        problem = optimiser.generate_problem(stops, area, promise, seed=int(seed))
    else:
        st.caption("Columns: stop_id, x_km, y_km, demand, window_start_min, window_end_min, service_min.")
        upload = c2.file_uploader("Stops CSV", type=["csv"])
        v1, v2 = st.columns(2)
        vehicles = v1.number_input("Vehicles", 1, 500, 10)
        capacity = v2.number_input("Vehicle capacity", 1.0, 100000.0, 80.0)
        if upload is None:
            return
        path = Path("uploaded_stops.csv")
        path.write_bytes(upload.getvalue())
        try:
            problem = optimiser.load_stops_csv(path, int(vehicles), float(capacity), area_type=area)
        finally:
            path.unlink(missing_ok=True)
    if not st.button("Optimise routes", type="primary"):
        return
    result = optimiser.compare(problem, time_budget_s=seconds)
    plan = result["plan"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("On time", "{:.1f}%".format(plan["on_time_pct"]), "{} points".format(result["on_time_gain_points"]))
    m2.metric("Distance", "{:.0f} km".format(plan["total_km"]))
    m3.metric("Vehicles used", "{} of {}".format(plan["vehicles_used"], plan["fleet_available"]))
    m4.metric("Daily cost", "GBP {:,.0f}".format(plan["cost_gbp"]))
    st.dataframe(pd.DataFrame(result["comparison"]).drop(columns=["unassigned_stop_ids", "extra_vehicles_needed",
                                                                  "search_iterations", "improvements_accepted"],
                                                         errors="ignore"), hide_index=True, width="stretch")
    points = []
    for route in plan["routes"]:
        for stop in route["sequence"]:
            idx = problem.stop_ids.index(stop["stop_id"])
            points.append({"x_km": float(problem.xy[idx][0]), "y_km": float(problem.xy[idx][1]),
                           "vehicle": "Vehicle {}".format(route["vehicle"])})
    points.append({"x_km": float(problem.depot[0]), "y_km": float(problem.depot[1]), "vehicle": "Depot"})
    st.subheader("Stops by vehicle")
    st.scatter_chart(pd.DataFrame(points), x="x_km", y="y_km", color="vehicle")
    st.subheader("Route sheets")
    for route in plan["routes"]:
        with st.expander("Vehicle {}: {} stops, {} km, back at {}".format(route["vehicle"], route["stops"], route["km"],
                                                                           route["finish"])):
            st.dataframe(pd.DataFrame(route["sequence"]), hide_index=True, width="stretch")


def page_search(eng):
    st.header("Incident explorer")
    query = st.text_input("Search the incident base", "vans leaving the depot late after slow loading")
    c1, c2 = st.columns(2)
    mode = c1.selectbox("Retrieval method", ["hybrid", "fusion", "bm25", "dense"])
    k = c2.slider("Results", 5, 50, 15)
    if not query.strip():
        return
    out = eng.search(query, k=k, mode=mode)
    frame = pd.DataFrame(out["results"])
    st.caption("Retrieved in {} ms".format(out["timings_ms"]["retrieve"]))
    st.dataframe(frame[["incident_id", "score", "service_type", "area_type", "incident_type", "fix_strategy",
                        "recovery_status", "cost_avoided_gbp_30d"]], hide_index=True, width="stretch")
    c = eng.case(st.selectbox("Open an incident", frame["incident_id"].tolist()))
    st.write("**Incident.** " + c["incident_summary"])
    st.write("**Resolution.** " + c["resolution_narrative"])
    st.write("**Lesson.** " + c["lessons_learned"])


def page_assess(eng):
    st.header("Operator risk profile")
    st.caption("Profile an operation to see its predicted incident exposure, the drivers behind it and a prepared "
               "playbook for the incidents it is most likely to face.")
    opts = eng.options()
    with st.form("profile"):
        c1, c2, c3 = st.columns(3)
        profile = {
            "service_type": c1.selectbox("Service type", opts["service_types"]),
            "area_type": c2.selectbox("Area type", opts["area_types"]),
            "fleet_type": c3.selectbox("Fleet", opts["fleet_types"]),
            "region": c1.selectbox("Region", opts["regions"]),
            "operator_size_band": c2.selectbox("Size band", opts["size_bands"], index=1),
            "planning_tool": c3.selectbox("Planning tool", opts["planning_tools"], index=1),
            "visibility_level": c1.selectbox("Visibility", opts["visibility_levels"], index=1),
            "delivery_promise": c2.selectbox("Delivery promise", opts["delivery_promises"]),
            "season": c3.selectbox("Season", opts["seasons"]),
            "daily_drops": c1.number_input("Daily drops", 1, 1_000_000, 1400),
            "avg_stops_per_route": c2.number_input("Stops per route", 1, 400, 100),
            "avg_route_km": c3.number_input("Route length in km", 1, 1000, 70),
            "avg_consignment_weight_kg": c1.number_input("Consignment weight in kg", 0.1, 5000.0, 2.5),
            "time_window_minutes": c2.number_input("Delivery window in minutes", 15, 1440, 600),
            "agency_driver_share_pct": c3.slider("Agency driver share percent", 0, 100, 15),
            "depot_count": c1.number_input("Depots", 1, 500, 2),
            "electric_share_pct": c2.slider("Electric fleet share percent", 0, 100, 10),
        }
        submitted = st.form_submit_button("Assess operation", type="primary")
    if not submitted:
        return
    out = eng.assess(profile)
    cols = st.columns(len(out["risks"]))
    for col, risk in zip(cols, out["risks"].values()):
        col.metric(risk["label"], pct(risk["probability"]), help="Base rate {}".format(pct(risk["base_rate"])))
        for d in risk["drivers"]:
            col.caption("{}: {}".format(d["factor"], d["effect"]))
    st.subheader("Prepared playbook")
    st.dataframe(pd.DataFrame(out["playbook"]), hide_index=True, width="stretch")


def page_insights(eng):
    st.header("Incident insights")
    s = eng.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents", "{:,}".format(s["incidents"]))
    c2.metric("Incident cost", fmt_money(s["total_incident_cost_gbp"]))
    c3.metric("Median detection time", "{:.1f} hours".format(s["median_detection_lag_hours"]))
    c4.metric("Repeat within 90 days", pct(s["repeat_rate_90d"]))
    a, b = st.columns(2)
    a.subheader("Incident types")
    a.bar_chart(pd.Series(s["incident_types"]))
    b.subheader("Recovery")
    b.bar_chart(pd.Series(s["recovery"]))
    st.subheader("What fixes each incident")
    opts = eng.options()
    incident = st.selectbox("Incident type", opts["incident_types"])
    service = st.selectbox("Service context", ["All"] + opts["service_types"])
    block = eng.recommend(incident, service_type=None if service == "All" else service)
    st.caption("Reference class: {}".format(block["reference_class"]))
    table = pd.DataFrame(block["all_fixes"]).set_index("fix")
    st.bar_chart(table["full_recovery_rate"])
    st.dataframe(table, width="stretch")


def main():
    eng = engine()
    st.sidebar.title("Routewright AI")
    st.sidebar.caption("Evidence grounded fixes and time window aware routing for last mile delivery.")
    pages = {"Diagnose": page_ask, "Route optimiser": page_optimise, "Incident explorer": page_search,
             "Operator risk profile": page_assess, "Incident insights": page_insights}
    page = st.sidebar.radio("Workspace", list(pages))
    st.sidebar.divider()
    st.sidebar.caption("{:,} historical incidents indexed".format(eng.index.size))
    pages[page](eng)


main()
