"""Synthetic delivery incident generator.

Generates a large, internally consistent dataset of last mile delivery
incidents with vectorised numpy operations and a causal model:

    operator profile  ==>  incident type, severity and detection speed
    incident + chosen fix + planning maturity + visibility + response speed  ==>  recovery
    recovery  ==>  on time rate, cost per drop, cost avoided, emissions, repeat risk

Recovery depends on the ground truth efficacy matrix in vocab.py, so the data
carries a real, learnable signal about which fixes work for which incidents.
Text fields are composed from the numbers so every narrative agrees with its
own row.
"""

import json
import random
from datetime import date

import numpy as np
import pandas as pd

from . import narratives
from .utils import EPS, sigmoid, sub
from .vocab import (AREA_TYPES, CODENAMES, DELIVERY_PROMISES, DETECTION_CHANNELS, EFFICACY_MAP, EFFICACY_TIERS,
                    FIX_KEYS, FIXES, FLEET_TYPES, INCIDENT_KEYS, INCIDENTS, INSTINCTIVE_FIXES, OPERATOR_SUFFIX,
                    PLANNING_TOOLS, PROMISE_WINDOW_MINUTES, RECOVERY_LEVELS, REGIONS, SEASONS, SERVICE_TYPES,
                    SEVERITIES, SHIFTS, SIZE_BANDS, TEAMS, VISIBILITY_LEVELS, efficacy_matrix)

COLUMNS = [
    "incident_id", "operator_name", "service_type", "area_type", "region", "operator_size_band", "fleet_type",
    "planning_tool", "visibility_level", "delivery_promise", "season", "daily_drops", "vehicles_in_fleet",
    "drivers_employed", "avg_stops_per_route", "avg_route_km", "avg_consignment_weight_kg", "time_window_minutes",
    "agency_driver_share_pct", "depot_count", "electric_share_pct", "detection_date", "resolution_date",
    "detection_channel", "detection_lag_hours", "incident_type", "severity", "affected_shift", "root_cause",
    "baseline_on_time_rate_pct", "incident_on_time_rate_pct", "on_time_retention_ratio",
    "first_attempt_success_pct", "failed_deliveries_per_day", "route_overrun_minutes",
    "complaints_per_1000_drops", "baseline_cost_per_drop_gbp", "incident_cost_per_drop_gbp",
    "daily_delivery_cost_gbp", "incident_cost_gbp", "fix_strategy", "team_owner", "time_to_fix_days",
    "fix_cost_gbp", "recovery_status", "post_fix_on_time_rate_pct", "on_time_recovery_ratio",
    "post_fix_cost_per_drop_gbp", "cost_avoided_gbp_30d", "days_to_recover", "return_on_fix", "co2_kg_per_drop",
    "customer_satisfaction", "repeat_incident_90d", "incident_summary", "resolution_narrative", "lessons_learned",
    "tags",
]

FIX_DAYS = {
    "dynamic_routing": 21, "slot_capacity": 10, "service_time_model": 14, "traffic_aware_planning": 18,
    "predictive_eta": 20, "safe_place_lockers": 12, "address_validation": 16, "wave_loading": 7,
    "flex_capacity": 5, "driver_retention": 45, "route_familiarity": 21, "energy_planning": 14,
    "load_securing": 10, "photo_pod": 12, "access_database": 14, "workload_balancing": 7,
    "collection_integration": 12, "control_tower": 30, "micro_hubs": 60, "customer_rebooking": 10,
    "extended_shifts": 1, "more_stops_per_route": 1,
}
INCIDENT_DROP = {
    "window_misses": 0.14, "failed_first_attempt": 0.08, "route_overrun": 0.12, "late_departures": 0.10,
    "address_quality": 0.06, "capacity_shortfall": 0.16, "driver_shortage": 0.13, "congestion": 0.11,
    "ev_range": 0.07, "damaged_goods": 0.03, "pod_disputes": 0.02, "access_restrictions": 0.05,
    "customer_comms": 0.03, "unbalanced_routes": 0.08, "missed_collections": 0.06,
}
SLOW_TO_SPOT = {"pod_disputes": 2.0, "damaged_goods": 1.8, "address_quality": 1.6, "unbalanced_routes": 2.2,
                "customer_comms": 1.5, "missed_collections": 1.4}
PREFERRED_CHANNEL = {
    "failed_first_attempt": "Customer Complaints", "customer_comms": "Customer Complaints",
    "damaged_goods": "Customer Complaints", "pod_disputes": "Customer Complaints",
    "missed_collections": "Client Escalation", "window_misses": "Client Escalation",
    "capacity_shortfall": "Client Escalation", "driver_shortage": "Driver Feedback",
    "unbalanced_routes": "Driver Feedback", "ev_range": "Driver Feedback", "access_restrictions": "Driver Feedback",
}
PREFERRED_PROMISE = {
    "Parcel Courier": "Next Day Untimed", "Ecommerce Same Day": "Same Day", "Grocery Home Delivery": "One Hour Slot",
    "Two Person Furniture Delivery": "Half Day Window", "Pharmacy and Healthcare": "Two Hour Slot",
    "Food Service Wholesale": "Two Hour Slot", "Builders Merchant": "Half Day Window",
    "Retail Store Replenishment": "Two Hour Slot", "Pallet Network Final Mile": "Half Day Window",
    "Meal Kit Subscription": "Next Day Untimed",
}
EMISSION_KG_PER_KM = {"Diesel Vans": 0.25, "Electric Vans": 0.05, "Mixed Van Fleet": 0.17, "Rigid Trucks": 0.68,
                      "Cargo Bikes and Vans": 0.06}


def _choice(rng, n_options, weights, size):
    p = np.asarray(weights, dtype=float)
    return rng.choice(n_options, size=size, p=p / p.sum())


def _gumbel(rng, shape):
    u = rng.uniform(EPS, 1.0, size=shape)
    return np.negative(np.log(np.negative(np.log(u))))


def _labels(values, idx):
    return np.array(values, dtype=object)[idx]


def generate(rows=16000, seed=42):
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    n = int(rows)

    svc_names = list(SERVICE_TYPES)
    si = _choice(rng, len(svc_names), [SERVICE_TYPES[s]["weight"] for s in svc_names], n)
    service = _labels(svc_names, si)
    sp = {k: np.array([SERVICE_TYPES[s][k] for s in svc_names], dtype=float)[si]
          for k in ("drops", "stops_per_route", "weight_kg", "cost_per_drop", "residential", "bulky")}

    area_names = list(AREA_TYPES)
    ai = _choice(rng, len(area_names), [AREA_TYPES[a]["weight"] for a in area_names], n)
    area = _labels(area_names, ai)
    ap = {k: np.array([AREA_TYPES[a][k] for a in area_names], dtype=float)[ai]
          for k in ("speed_kmh", "route_km", "congestion", "rural")}

    band_names = list(SIZE_BANDS)
    bi = _choice(rng, len(band_names), [SIZE_BANDS[b]["weight"] for b in band_names], n)
    band = _labels(band_names, bi)
    planning = np.zeros(n, dtype=int)
    visibility = np.zeros(n, dtype=int)
    for b_idx, b in enumerate(band_names):
        mask = bi == b_idx
        planning[mask] = _choice(rng, 4, SIZE_BANDS[b]["planning"], int(mask.sum()))
        visibility[mask] = _choice(rng, 3, SIZE_BANDS[b]["visibility"], int(mask.sum()))

    region = _labels(list(REGIONS), _choice(rng, len(REGIONS), list(REGIONS.values()), n))
    season = _labels(list(SEASONS), _choice(rng, len(SEASONS), list(SEASONS.values()), n))
    fleet_names = list(FLEET_TYPES)
    fleet = _labels(fleet_names, _choice(rng, len(fleet_names), list(FLEET_TYPES.values()), n))
    heavy = sp["bulky"] >= 0.8
    fleet = np.where(heavy & (rng.uniform(0.0, 1.0, n) < 0.6), "Rigid Trucks", fleet)
    fleet = np.where(heavy & (fleet == "Cargo Bikes and Vans"), "Diesel Vans", fleet)
    fleet = np.where((ap["rural"] >= 0.5) & (fleet == "Cargo Bikes and Vans"), "Diesel Vans", fleet)

    promise_names = list(DELIVERY_PROMISES)
    promise = _labels(promise_names, _choice(rng, len(promise_names), list(DELIVERY_PROMISES.values()), n))
    preferred = np.array([PREFERRED_PROMISE[s] for s in service], dtype=object)
    promise = np.where(rng.uniform(0.0, 1.0, n) < 0.6, preferred, promise)
    window = np.array([PROMISE_WINDOW_MINUTES[p] for p in promise])

    scale = np.array([SIZE_BANDS[b]["scale"] for b in band_names])[bi]
    drops = np.maximum(np.round(sp["drops"] * scale * rng.lognormal(0.0, 0.4, n)), 20).astype(int)
    density = np.where(area == "Dense Urban", 1.15, np.where(ap["rural"] >= 1.0, 0.7, 1.0))
    stops = np.maximum(np.round(sp["stops_per_route"] * density * rng.lognormal(0.0, 0.2, n)), 3).astype(int)
    vehicles = np.maximum(np.ceil(drops / stops * 1.08), 1).astype(int)
    drivers = np.maximum(np.round(vehicles * rng.uniform(1.05, 1.35, n)), 1).astype(int)
    route_km = np.round(ap["route_km"] * rng.lognormal(0.0, 0.25, n), 1)
    weight = np.round(sp["weight_kg"] * rng.lognormal(0.0, 0.3, n), 1)
    is_peak = (season == "Peak Season").astype(float)
    agency = np.round(np.clip(rng.gamma(2.0, 7.5, n) + 10.0 * is_peak, 0.0, 80.0), 1)
    depots = np.maximum(np.round(1.2 * scale ** 0.7 * rng.lognormal(0.0, 0.3, n)), 1).astype(int)
    electric = np.select(
        [fleet == "Electric Vans", fleet == "Mixed Van Fleet", fleet == "Cargo Bikes and Vans"],
        [rng.uniform(85.0, 100.0, n), rng.uniform(20.0, 60.0, n), rng.uniform(50.0, 90.0, n)],
        rng.uniform(0.0, 8.0, n))
    electric = np.round(electric, 1)

    residential = sp["residential"]
    tight = (window <= 120).astype(float)
    weak_plan = (planning <= 1).astype(float)
    no_vis = (visibility == 0).astype(float)
    parcelish = np.isin(service, ["Parcel Courier", "Ecommerce Same Day", "Pallet Network Final Mile"]).astype(float)
    is_dense = (area == "Dense Urban").astype(float)
    is_winter = (season == "Winter").astype(float)
    feats = {
        "window_misses": 1.2 * tight + 0.5 * weak_plan,
        "failed_first_attempt": 1.3 * residential + 0.6 * no_vis + 0.3 * (promise == "Next Day Untimed"),
        "route_overrun": 0.008 * stops + 0.5 * weak_plan + 0.4 * ap["rural"],
        "late_departures": 0.18 * np.log(drops) + 0.5 * is_peak,
        "address_quality": 0.9 * ap["rural"] + 0.4 * residential,
        "capacity_shortfall": 1.4 * is_peak + 0.1 * bi,
        "driver_shortage": 0.03 * agency + 0.3 * is_peak,
        "congestion": 1.4 * ap["congestion"],
        "ev_range": 0.022 * electric + 0.6 * is_winter + 0.4 * ap["rural"],
        "damaged_goods": 1.3 * sp["bulky"],
        "pod_disputes": 1.0 * parcelish + 0.5 * is_dense,
        "access_restrictions": 0.9 * is_dense + 0.8 * (fleet == "Rigid Trucks") + 0.3 * (region == "London"),
        "customer_comms": 0.8 * no_vis + 0.5 * residential,
        "unbalanced_routes": 0.6 * weak_plan + 0.15 * np.log(vehicles),
        "missed_collections": 0.8 * parcelish + 0.3 * weak_plan,
    }
    base = {"window_misses": 0.7, "failed_first_attempt": 0.35, "route_overrun": 0.85, "late_departures": 0.2,
            "address_quality": 0.9, "capacity_shortfall": 0.9, "driver_shortage": 0.95, "congestion": 0.65,
            "ev_range": 0.45, "damaged_goods": 0.9, "pod_disputes": 0.8, "access_restrictions": 0.8,
            "customer_comms": 0.9, "unbalanced_routes": 0.7, "missed_collections": 0.9}
    logits = np.stack([feats[k] + base[k] for k in INCIDENT_KEYS], axis=1)
    inc_idx = np.argmax(1.1 * logits + _gumbel(rng, logits.shape), axis=1)
    inc_key = _labels(INCIDENT_KEYS, inc_idx)

    pressure = (0.45 * np.subtract(3, planning) + 0.4 * np.subtract(2, visibility) + 0.4 * is_peak
                + 0.01 * agency + rng.normal(0.0, 0.7, n))
    sev = np.digitize(pressure, np.quantile(pressure, [0.25, 0.62, 0.88])).astype(int)

    slow = np.array([SLOW_TO_SPOT.get(k, 1.0) for k in inc_key])
    lag = np.round(np.clip(rng.exponential(np.array([96.0, 30.0, 4.0])[visibility] * slow) * (1.0 + 0.15 * sev),
                           0.5, 1000.0), 1)
    p_alert = np.array([0.03, 0.2, 0.8])[visibility]
    channel = np.empty(n, dtype=object)
    u_ch = rng.uniform(0.0, 1.0, n)
    for i in range(n):
        if u_ch[i] < p_alert[i] and lag[i] < 24:
            channel[i] = "Control Tower Alert"
        elif inc_key[i] in PREFERRED_CHANNEL and prng.random() < 0.6:
            channel[i] = PREFERRED_CHANNEL[inc_key[i]]
        else:
            channel[i] = prng.choice(DETECTION_CHANNELS[1:])
    shift = _labels(SHIFTS, _choice(rng, 4, [22, 38, 15, 25], n))

    root = np.empty(n, dtype=object)
    fix = np.empty(n, dtype=object)
    p_strong = np.clip(0.2 + 0.08 * planning + 0.06 * visibility, 0.2, 0.6)
    u_pick = rng.uniform(0.0, 1.0, n)
    for i in range(n):
        key = inc_key[i]
        causes = INCIDENTS[key]["root_causes"]
        root[i] = causes[prng.randrange(len(causes))]
        tiers = EFFICACY_MAP[key]
        if u_pick[i] < p_strong[i]:
            fix[i] = prng.choice(tiers["strong"])
        elif u_pick[i] < p_strong[i] + 0.25 and tiers["moderate"]:
            fix[i] = prng.choice(tiers["moderate"])
        elif prng.random() < 0.35:
            fix[i] = prng.choice(INSTINCTIVE_FIXES + tiers["harmful"])
        else:
            fix[i] = prng.choice(FIX_KEYS)
    team = np.array([FIXES[k]["team"] for k in fix], dtype=object)
    team = np.where(rng.uniform(0.0, 1.0, n) < 0.12, _labels(TEAMS, rng.integers(0, len(TEAMS), n)), team)

    fix_idx = np.array([FIX_KEYS.index(k) for k in fix])
    efficacy = np.array(efficacy_matrix())[inc_idx, fix_idx]
    weak_fix = efficacy <= EFFICACY_TIERS["neutral"] + EPS
    days_fix = np.array([FIX_DAYS[k] for k in fix], dtype=float)
    days_fix = np.round(np.clip(days_fix * rng.lognormal(0.0, 0.45, n) * np.where(planning == 0, 1.3, 1.0), 0.5, 240.0), 1)

    pos = 4.2 * efficacy + 0.3 * planning + 0.3 * visibility + 0.8 + rng.normal(0.0, 0.35, n)
    neg = 3.8 + 0.012 * days_fix + 0.35 * sev + 0.004 * lag
    p_full = sigmoid(np.subtract(pos, neg))
    p_part = np.subtract(1.0, p_full) * (0.35 + 0.3 * efficacy)
    u_rec = rng.uniform(0.0, 1.0, n)
    rec = np.where(u_rec < p_full, 0, np.where(u_rec < p_full + p_part, 1, 2)).astype(int)

    base_ot = np.clip(rng.normal(93.0 + 1.2 * planning + 0.8 * visibility, 1.8, n), 82.0, 99.6)
    base_ot = np.subtract(base_ot, 1.5 * tight)
    base_ot = np.round(np.clip(base_ot, 80.0, 99.6), 1)
    drop = np.clip(0.03 + 0.045 * sev + np.array([INCIDENT_DROP[k] for k in inc_key]) + rng.normal(0.0, 0.03, n),
                   0.01, 0.6)
    inc_ot = np.round(base_ot * np.subtract(1.0, drop), 1)
    retention = np.round(inc_ot / base_ot, 3)

    is_ffa = (inc_key == "failed_first_attempt").astype(float)
    ftr = rng.normal(np.subtract(95.0, 4.0 * residential), 2.0, n)
    ftr = np.where(is_ffa > 0, rng.uniform(62.0, 84.0, n), np.subtract(ftr, 1.5 * sev))
    ftr = np.round(np.clip(ftr, 55.0, 99.5), 1)
    fails = np.round(drops * np.subtract(100.0, np.minimum(ftr, inc_ot)) / 100.0).astype(int)
    overrun_bonus = np.select(
        [inc_key == "route_overrun", inc_key == "congestion", inc_key == "late_departures",
         inc_key == "unbalanced_routes", inc_key == "address_quality"],
        [rng.uniform(40.0, 130.0, n), rng.uniform(20.0, 70.0, n), rng.uniform(20.0, 75.0, n),
         rng.uniform(30.0, 90.0, n), rng.uniform(10.0, 40.0, n)], 0.0)
    overrun = np.round(5.0 + overrun_bonus + 8.0 * sev + rng.exponential(6.0, n)).astype(int)
    comms_extra = np.isin(inc_key, ["customer_comms", "pod_disputes", "damaged_goods", "failed_first_attempt"])
    complaints = np.round(2.0 + 25.0 * drop + 6.0 * comms_extra + rng.gamma(1.5, 1.0, n), 1)

    area_cost = np.where(ap["rural"] >= 1.0, 1.35, np.where(area == "Dense Urban", 1.12, 1.0))
    base_cpd = np.round(sp["cost_per_drop"] * area_cost * rng.lognormal(0.0, 0.15, n), 2)
    inc_cpd = np.round(base_cpd * (1.0 + 0.9 * drop + 0.05 * sev + 0.15 * comms_extra), 2)
    daily_cost = np.round(drops * base_cpd, 2)
    residual = np.where(rec == 0, rng.uniform(2.0, 6.0, n), np.where(rec == 1, rng.uniform(10.0, 25.0, n), 45.0))
    duration = lag / 24.0 + days_fix + residual
    extra_ratio = np.subtract(inc_cpd, base_cpd) / base_cpd
    incident_cost = np.round(daily_cost * extra_ratio * duration)

    post_ratio = np.where(rec == 0, rng.uniform(0.985, 1.025, n),
                          np.where(rec == 1, rng.uniform(0.93, 0.975, n),
                                   np.minimum(retention + rng.uniform(0.005, 0.05, n), 0.97)))
    post_ot = np.round(np.minimum(base_ot * post_ratio, 99.9), 1)
    post_ratio = np.round(post_ot / base_ot, 3)
    post_cpd = np.where(rec == 0, base_cpd * rng.uniform(0.88, 1.02, n),
                        np.where(rec == 1, base_cpd * rng.uniform(1.0, 1.08, n), inc_cpd * rng.uniform(0.95, 1.0, n)))
    post_cpd = np.round(post_cpd, 2)
    avoided = np.round(drops * 30.0 * np.maximum(np.subtract(inc_cpd, post_cpd), 0.0))
    days_rec = np.where(rec == 0, np.round(3 + rng.gamma(2.0, 2.0, n) * (1 + 0.4 * sev)),
                        np.where(rec == 1, np.round(14 + rng.gamma(2.0, 5.0, n)), 90)).astype(int)

    band_cost = np.array([0.6, 1.0, 1.8, 3.0])[bi]
    fix_cost = days_fix * 380.0 * band_cost
    fix_cost = fix_cost + np.where(fix == "extended_shifts", 0.08 * daily_cost * 14.0, 0.0)
    fix_cost = fix_cost + np.where(fix == "more_stops_per_route", 0.02 * daily_cost * 14.0, 0.0)
    fix_cost = fix_cost + np.where(fix == "flex_capacity", 0.1 * daily_cost * 10.0, 0.0)
    fix_cost = np.round(np.maximum(fix_cost, 150.0))
    roi = np.round(avoided / fix_cost, 2)

    km_per_drop = route_km / stops
    routing_gain = np.where((rec == 0) & np.isin(fix, ["dynamic_routing", "workload_balancing", "traffic_aware_planning",
                                                        "micro_hubs", "energy_planning"]), 0.88, 1.0)
    co2 = np.round(km_per_drop * np.array([EMISSION_KG_PER_KM[f] for f in fleet]) * routing_gain, 3)

    csat = 8.5 + rng.normal(0.0, 0.8, n)
    csat = np.subtract(csat, 0.7 * sev + np.array([0.0, 1.0, 2.3])[rec] + 0.6 * comms_extra)
    csat = np.clip(np.round(csat), 1, 10).astype(int)
    p_repeat = 0.08 + np.array([0.0, 0.12, 0.25])[rec] + 0.15 * weak_fix + 0.10 * (visibility == 0)
    repeat = np.where(rng.uniform(0.0, 1.0, n) < p_repeat, "Yes", "No")

    start = date(2021, 1, 1).toordinal()
    end = date(2026, 6, 30).toordinal()
    detect = start + (rng.uniform(0.0, 1.0, n) * sub(end, start)).astype(int)
    resolve = detect + np.ceil(lag / 24.0 + days_fix).astype(int)
    names = [prng.choice(CODENAMES) + " " + prng.choice(OPERATOR_SUFFIX) for _ in range(n)]

    frame = pd.DataFrame({
        "incident_id": ["INC{:06d}".format(i + 1) for i in range(n)],
        "operator_name": names, "service_type": service, "area_type": area, "region": region,
        "operator_size_band": band, "fleet_type": fleet, "planning_tool": _labels(PLANNING_TOOLS, planning),
        "visibility_level": _labels(VISIBILITY_LEVELS, visibility), "delivery_promise": promise, "season": season,
        "daily_drops": drops, "vehicles_in_fleet": vehicles, "drivers_employed": drivers,
        "avg_stops_per_route": stops, "avg_route_km": route_km, "avg_consignment_weight_kg": weight,
        "time_window_minutes": window, "agency_driver_share_pct": agency, "depot_count": depots,
        "electric_share_pct": electric,
        "detection_date": [date.fromordinal(int(d)).strftime("%Y/%m/%d") for d in detect],
        "resolution_date": [date.fromordinal(int(d)).strftime("%Y/%m/%d") for d in resolve],
        "detection_channel": channel, "detection_lag_hours": lag,
        "incident_type": [INCIDENTS[k]["name"] for k in inc_key], "severity": _labels(SEVERITIES, sev),
        "affected_shift": shift, "root_cause": root, "baseline_on_time_rate_pct": base_ot,
        "incident_on_time_rate_pct": inc_ot, "on_time_retention_ratio": retention,
        "first_attempt_success_pct": ftr, "failed_deliveries_per_day": fails, "route_overrun_minutes": overrun,
        "complaints_per_1000_drops": complaints, "baseline_cost_per_drop_gbp": base_cpd,
        "incident_cost_per_drop_gbp": inc_cpd, "daily_delivery_cost_gbp": daily_cost,
        "incident_cost_gbp": incident_cost.astype(np.int64), "fix_strategy": [FIXES[k]["name"] for k in fix],
        "team_owner": team, "time_to_fix_days": days_fix, "fix_cost_gbp": fix_cost.astype(np.int64),
        "recovery_status": _labels(RECOVERY_LEVELS, rec), "post_fix_on_time_rate_pct": post_ot,
        "on_time_recovery_ratio": post_ratio, "post_fix_cost_per_drop_gbp": post_cpd,
        "cost_avoided_gbp_30d": avoided.astype(np.int64), "days_to_recover": days_rec, "return_on_fix": roi,
        "co2_kg_per_drop": co2, "customer_satisfaction": csat, "repeat_incident_90d": repeat,
    })
    helper = pd.DataFrame({"incident_key": inc_key, "fix_key": fix, "severity_idx": sev})
    summaries, resolutions, lessons, tag_list = [], [], [], []
    for record, extra in zip(frame.to_dict("records"), helper.to_dict("records")):
        record.update(extra)
        summaries.append(narratives.incident_summary(record, prng))
        resolutions.append(narratives.resolution_narrative(record, prng))
        lessons.append(narratives.lessons_learned(record, prng))
        tag_list.append(narratives.tags(record))
    frame["incident_summary"] = summaries
    frame["resolution_narrative"] = resolutions
    frame["lessons_learned"] = lessons
    frame["tags"] = tag_list
    return frame[COLUMNS]


def ground_truth():
    return {
        "tiers": EFFICACY_TIERS,
        "incidents": {
            INCIDENTS[k]["name"]: {t: [FIXES[j]["name"] for j in EFFICACY_MAP[k][t]]
                                   for t in ("strong", "moderate", "harmful")}
            for k in INCIDENT_KEYS
        },
    }


def write_ground_truth(path):
    with open(path, "w", encoding="utf8") as fh:
        json.dump(ground_truth(), fh, indent=2)
