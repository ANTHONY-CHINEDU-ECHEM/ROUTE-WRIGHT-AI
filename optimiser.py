"""Route optimiser for last mile delivery.

Solves the capacitated vehicle routing problem with time windows (CVRPTW):
every stop has a location, a demand, a delivery window and a service time;
every vehicle has a capacity and a shift length. The goal is to serve every
stop inside its window with as few kilometres and vehicles as possible.

Method
    1. Construction with the Clarke and Wright savings algorithm, restricted
       to each stop's nearest neighbours and checked for time window, capacity
       and shift feasibility at every merge.
    2. Route elimination: when the plan needs more vehicles than the fleet
       has, stops from the smallest routes are reinserted elsewhere.
    3. Local search: two opt moves within each route and relocate moves
       between neighbouring routes, accepting only feasible improvements.
    4. Ruin and recreate until the time budget runs out: a random stop and its
       nearest neighbours are removed and greedily reinserted at their cheapest
       feasible positions, keeping the change only when the plan gets shorter
       without needing more vehicles. This escapes the local optima that pure
       local search gets stuck in.

Baselines
    Two methods planners commonly use are simulated with the same travel model
    so the gain is measured honestly: a greedy nearest neighbour plan that
    ignores time windows, and a sweep plan that cuts territories by angle
    around the depot, much like fixed postcode rounds.

Plan cost counts distance at GBP 0.45 per kilometre, driver time at GBP 16.50
per hour, a 50 percent premium on overtime, and a GBP 3.50 service penalty
for every stop served late or not at all, covering redelivery and customer
contact. The constants are module level so operators can set their own.

Coordinates are kilometres on a local grid with the origin at the south west
corner of the service area, so every value is positive. Times are minutes
from the start of the shift.
"""

import csv
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from .utils import EPS, last, safe_float, sub

AREA_PRESETS = {
    "Dense Urban": {"box_km": 12.0, "speed_kmh": 16.0, "circuity": 1.45},
    "Urban": {"box_km": 24.0, "speed_kmh": 24.0, "circuity": 1.35},
    "Suburban": {"box_km": 40.0, "speed_kmh": 32.0, "circuity": 1.30},
    "Rural": {"box_km": 80.0, "speed_kmh": 44.0, "circuity": 1.25},
    "Mixed Region": {"box_km": 55.0, "speed_kmh": 34.0, "circuity": 1.30},
}
PROMISE_WINDOWS = {"One Hour Slot": 60, "Two Hour Slot": 120, "Half Day Window": 300, "Next Day Untimed": 600,
                   "Same Day": 240}
SHIFT_START_HOUR = 7
COST_PER_KM_GBP = 0.45
DRIVER_COST_PER_HOUR_GBP = 16.5
CO2_KG_PER_KM = 0.25
OVERTIME_PREMIUM = 0.5
LATE_STOP_PENALTY_GBP = 3.5


@dataclass
class Problem:
    """A routing problem. Node 0 is the depot and stops are nodes 1 to n."""

    depot: tuple
    xy: np.ndarray
    demand: np.ndarray
    window_start: np.ndarray
    window_end: np.ndarray
    service: np.ndarray
    vehicles: int
    capacity: float
    shift_minutes: float
    speed_kmh: float
    circuity: float = 1.3
    stop_ids: list = field(default_factory=list)

    def __post_init__(self):
        n = len(self.xy)
        if not self.stop_ids:
            self.stop_ids = ["S{:04d}".format(i + 1) for i in range(n)]
        pts = np.vstack([np.asarray(self.depot, dtype=float).reshape(1, 2), np.asarray(self.xy, dtype=float)])
        diff = np.subtract(pts[:, None, :], pts[None, :, :])
        self.dist = np.sqrt((diff ** 2).sum(axis=2)) * self.circuity
        self.travel = self.dist / self.speed_kmh * 60.0
        zero = np.zeros(1)
        self._D = self.dist.tolist()
        self._T = self.travel.tolist()
        self._ws = np.concatenate([zero, self.window_start]).tolist()
        self._we = np.concatenate([zero, self.window_end]).tolist()
        self._svc = np.concatenate([zero, self.service]).tolist()
        self._dem = np.concatenate([zero, self.demand]).tolist()

    @property
    def size(self):
        return len(self.xy)


def generate_problem(n_stops=120, area_type="Urban", promise="Two Hour Slot", seed=7, vehicles=None,
                     capacity=None, shift_minutes=600.0, clusters=5):
    """Create a realistic demo problem: clustered stops, booked windows, parcel demands."""
    rng = np.random.default_rng(seed)
    preset = AREA_PRESETS[area_type]
    box = preset["box_km"]
    centre = box / 2.0
    k = max(1, int(clusters))
    cx = rng.uniform(0.15 * box, 0.85 * box, k)
    cy = rng.uniform(0.15 * box, 0.85 * box, k)
    which = rng.integers(0, k, n_stops)
    spread = 0.09 * box
    x = np.clip(cx[which] + rng.normal(0.0, spread, n_stops), 0.0, box)
    y = np.clip(cy[which] + rng.normal(0.0, spread, n_stops), 0.0, box)
    uniform_mask = rng.uniform(0.0, 1.0, n_stops) < 0.25
    x = np.where(uniform_mask, rng.uniform(0.0, box, n_stops), x)
    y = np.where(uniform_mask, rng.uniform(0.0, box, n_stops), y)
    width = float(PROMISE_WINDOWS[promise])
    usable = max(sub(shift_minutes, width + 60.0), 1.0)
    if width >= shift_minutes:
        ws = np.zeros(n_stops)
        we = np.full(n_stops, sub(shift_minutes, 30.0))
    else:
        slots = max(1, int(usable // width) + 1)
        slot = rng.integers(0, slots, n_stops)
        ws = 30.0 + slot * width
        ws = np.minimum(ws, usable)
        we = ws + width
    demand = np.round(rng.gamma(2.0, 1.5, n_stops) + 1.0, 1)
    service = np.round(rng.uniform(2.0, 6.0, n_stops), 1)
    if capacity is None:
        capacity = float(max(40.0, np.ceil(demand.sum() / max(n_stops / 18.0, 1.0))))
    if vehicles is None:
        vehicles = int(max(2, math.ceil(demand.sum() / capacity * 1.35)))
    return Problem(depot=(centre, centre), xy=np.column_stack([np.round(x, 3), np.round(y, 3)]), demand=demand,
                   window_start=np.round(ws, 1), window_end=np.round(we, 1), service=service, vehicles=int(vehicles),
                   capacity=float(capacity), shift_minutes=float(shift_minutes), speed_kmh=preset["speed_kmh"],
                   circuity=preset["circuity"])


def load_stops_csv(path, vehicles, capacity, shift_minutes=600.0, area_type="Urban", depot_x=None, depot_y=None):
    """Load stops from a CSV with columns stop_id, x_km, y_km, demand, window_start_min, window_end_min, service_min."""
    ids, xs, ys, dem, ws, we, svc = [], [], [], [], [], [], []
    with open(path, newline="", encoding="utf8") as fh:
        for row in csv.DictReader(fh):
            ids.append(row["stop_id"])
            xs.append(float(row["x_km"]))
            ys.append(float(row["y_km"]))
            dem.append(float(row.get("demand", 1) or 1))
            ws.append(float(row.get("window_start_min", 0) or 0))
            we.append(float(row.get("window_end_min", shift_minutes) or shift_minutes))
            svc.append(float(row.get("service_min", 3) or 3))
    preset = AREA_PRESETS.get(area_type, AREA_PRESETS["Urban"])
    xs_arr, ys_arr = np.asarray(xs), np.asarray(ys)
    depot = (float(depot_x) if depot_x is not None else float(np.median(xs_arr)),
             float(depot_y) if depot_y is not None else float(np.median(ys_arr)))
    return Problem(depot=depot, xy=np.column_stack([xs_arr, ys_arr]), demand=np.asarray(dem),
                   window_start=np.asarray(ws), window_end=np.asarray(we), service=np.asarray(svc),
                   vehicles=int(vehicles), capacity=float(capacity), shift_minutes=float(shift_minutes),
                   speed_kmh=preset["speed_kmh"], circuity=preset["circuity"], stop_ids=ids)


def write_stops_csv(problem, path):
    with open(path, "w", newline="", encoding="utf8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stop_id", "x_km", "y_km", "demand", "window_start_min", "window_end_min", "service_min"])
        for i in range(problem.size):
            writer.writerow([problem.stop_ids[i], problem.xy[i][0], problem.xy[i][1], problem.demand[i],
                             problem.window_start[i], problem.window_end[i], problem.service[i]])


class _Model:
    """Fast feasibility and cost checks on plain Python lists."""

    def __init__(self, p):
        self.p = p
        self.D, self.T = p._D, p._T
        self.ws, self.we, self.svc, self.dem = p._ws, p._we, p._svc, p._dem
        self.cap, self.shift = p.capacity, p.shift_minutes

    def feasible(self, route):
        T, ws, we, svc = self.T, self.ws, self.we, self.svc
        t, prev, load = 0.0, 0, 0.0
        for s in route:
            t += T[prev][s]
            if t < ws[s]:
                t = ws[s]
            if t > we[s] + EPS:
                return False
            t += svc[s]
            load += self.dem[s]
            prev = s
        if load > self.cap + EPS:
            return False
        return t + T[prev][0] <= self.shift + EPS

    def load(self, route):
        return sum(self.dem[s] for s in route)

    def distance(self, route):
        D = self.D
        total, prev = 0.0, 0
        for s in route:
            total += D[prev][s]
            prev = s
        return total + D[prev][0]


def _neighbours(problem, k):
    d = problem.dist[1:, 1:].copy()
    np.fill_diagonal(d, np.inf)
    k = int(min(k, max(problem.size, 1)))
    return np.argsort(d, axis=1)[:, :k] + 1


def _savings_routes(model, problem, k=25):
    n = problem.size
    routes, route_of, unassigned = {}, {}, []
    for s in range(1, n + 1):
        if model.feasible([s]):
            routes[s] = [s]
            route_of[s] = s
        else:
            unassigned.append(s)
    if n < 2:
        return routes, route_of, unassigned
    nb = _neighbours(problem, k)
    d0 = problem.dist[0]
    pairs_i = np.repeat(np.arange(1, n + 1), nb.shape[1])
    pairs_j = nb.ravel()
    keep = pairs_i < pairs_j
    pairs_i, pairs_j = pairs_i[keep], pairs_j[keep]
    saving = np.subtract(d0[pairs_i] + d0[pairs_j], problem.dist[pairs_i, pairs_j])
    order = np.argsort(np.negative(saving), kind="stable")
    for idx in order:
        if saving[idx] <= 0:
            break
        i, j = int(pairs_i[idx]), int(pairs_j[idx])
        if i not in route_of or j not in route_of:
            continue
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        a, b = routes[ri], routes[rj]
        if model.load(a) + model.load(b) > model.cap + EPS:
            continue
        options = []
        if last(a) == i and b[0] == j:
            options.append(a + b)
        if last(b) == j and a[0] == i:
            options.append(b + a)
        if last(a) == i and last(b) == j:
            options.append(a + list(reversed(b)))
        if a[0] == i and b[0] == j:
            options.append(list(reversed(a)) + b)
        for merged in options:
            if model.feasible(merged):
                routes[ri] = merged
                del routes[rj]
                for s in b:
                    route_of[s] = ri
                break
    return routes, route_of, unassigned


def _best_insertion(model, route, s):
    """Cheapest feasible position for stop s in route, as (added distance, new route) or None."""
    D = model.D
    if model.load(route) + model.dem[s] > model.cap + EPS:
        return None
    path = [0] + route + [0]
    best = None
    for pos in range(1, len(path)):
        a, b = path[sub(pos, 1)], path[pos]
        added = sub(D[a][s] + D[s][b], D[a][b])
        if best is not None and added >= best[0]:
            continue
        candidate = route[:sub(pos, 1)] + [s] + route[sub(pos, 1):]
        if model.feasible(candidate):
            best = (added, candidate)
    return best


def _eliminate_routes(model, routes, target):
    changed = True
    while len(routes) > target and changed:
        changed = False
        order = sorted(routes, key=lambda r: len(routes[r]))
        for victim in order:
            trial = {r: list(v) for r, v in routes.items() if r != victim}
            ok = True
            for s in routes[victim]:
                best, best_r = None, None
                for r, route in trial.items():
                    ins = _best_insertion(model, route, s)
                    if ins is not None and (best is None or ins[0] < best[0]):
                        best, best_r = ins, r
                if best is None:
                    ok = False
                    break
                trial[best_r] = best[1]
            if ok:
                routes.clear()
                routes.update(trial)
                changed = True
                break
    return routes


def _two_opt(model, route):
    D = model.D
    improved = True
    while improved:
        improved = False
        path = [0] + route + [0]
        m = len(route)
        for i in range(1, m):
            for k in range(i + 1, m + 1):
                a, b = path[sub(i, 1)], path[i]
                c, d = path[k], path[k + 1]
                delta = sub(D[a][c] + D[b][d], D[a][b] + D[c][d])
                if delta < sub(0.0, 1e3 * EPS):
                    candidate = route[:sub(i, 1)] + list(reversed(route[sub(i, 1):k])) + route[k:]
                    if model.feasible(candidate):
                        route = candidate
                        improved = True
                        break
            if improved:
                break
    return route


def _relocate_pass(model, routes, nb, deadline):
    D = model.D
    where = {s: r for r, route in routes.items() for s in route}
    improved = False
    for s in list(where):
        if time.perf_counter() > deadline:
            break
        ra = where[s]
        a = routes[ra]
        pos = a.index(s)
        prev = a[sub(pos, 1)] if pos > 0 else 0
        nxt = a[pos + 1] if pos + 1 < len(a) else 0
        gain = sub(D[prev][s] + D[s][nxt], D[prev][nxt])
        reduced = a[:pos] + a[pos + 1:]
        if reduced and not model.feasible(reduced):
            continue
        targets = {where[int(t)] for t in nb[sub(s, 1)] if int(t) in where and where[int(t)] != ra}
        best, best_r = None, None
        for rb in targets:
            ins = _best_insertion(model, routes[rb], s)
            if ins is not None and ins[0] < sub(gain, 1e3 * EPS) and (best is None or ins[0] < best[0]):
                best, best_r = ins, rb
        if best is None:
            continue
        routes[best_r] = best[1]
        where[s] = best_r
        if reduced:
            routes[ra] = reduced
        else:
            del routes[ra]
        improved = True
    return improved


def _total_km(model, routes):
    return sum(model.distance(r) for r in routes.values())


def _ruin_recreate(model, routes, nb, deadline, rng, vehicles):
    """Large neighbourhood search: remove a cluster of stops and reinsert them greedily."""
    best = {k: list(v) for k, v in routes.items()}
    best_km = _total_km(model, best)
    next_key = max(best) + 1 if best else 1
    stops = [s for route in best.values() for s in route]
    iterations = accepted = 0
    while stops and time.perf_counter() < deadline:
        iterations += 1
        cand = {k: list(v) for k, v in best.items()}
        seed_stop = rng.choice(stops)
        size = rng.randint(4, 14)
        removed = [seed_stop] + [int(t) for t in nb[sub(seed_stop, 1)][:size]]
        removed_set = set(removed)
        for k in list(cand):
            kept = [s for s in cand[k] if s not in removed_set]
            if kept:
                cand[k] = kept
            else:
                del cand[k]
        removed = [s for s in removed if s in set(stops)]
        rng.shuffle(removed)
        ok = True
        changed = set()
        for s in removed:
            choice, where = None, None
            for k, route in cand.items():
                ins = _best_insertion(model, route, s)
                if ins is not None and (choice is None or ins[0] < choice[0]):
                    choice, where = ins, k
            if choice is None:
                if len(cand) < vehicles and model.feasible([s]):
                    cand[next_key] = [s]
                    changed.add(next_key)
                    next_key += 1
                    continue
                ok = False
                break
            cand[where] = choice[1]
            changed.add(where)
        if not ok or len(cand) > max(len(best), vehicles):
            continue
        for k in changed:
            if k in cand:
                cand[k] = _two_opt(model, cand[k])
        km = _total_km(model, cand)
        if km < sub(best_km, 1e3 * EPS) or (len(cand) < len(best) and km <= best_km * 1.02):
            best, best_km = cand, km
            accepted += 1
    return best, iterations, accepted


def _simulate(problem, route):
    """Arrivals, lateness and timings for one route, waiting when early."""
    T, D = problem._T, problem._D
    ws, we, svc, dem = problem._ws, problem._we, problem._svc, problem._dem
    t, prev, load, km = 0.0, 0, 0.0, 0.0
    stops, late_minutes, late = [], 0.0, 0
    for s in route:
        t += T[prev][s]
        km += D[prev][s]
        arrive = max(t, ws[s])
        lateness = max(sub(arrive, we[s]), 0.0)
        if lateness > EPS:
            late += 1
            late_minutes += lateness
        stops.append({"stop_id": problem.stop_ids[sub(s, 1)], "arrival": _clock(arrive),
                      "window": "{} to {}".format(_clock(ws[s]), _clock(we[s])), "on_time": lateness <= EPS})
        t = arrive + svc[s]
        load += dem[s]
        prev = s
    t += T[prev][0]
    km += D[prev][0]
    return {"stops": stops, "km": km, "minutes": t, "load": load, "late": late, "late_minutes": late_minutes}


def _clock(minutes):
    total = int(round(SHIFT_START_HOUR * 60 + minutes))
    return "{:02d}:{:02d}".format((total // 60) % 24, total % 60)


def summarise(problem, routes, name, runtime_ms=0.0, unassigned=None):
    sims = [_simulate(problem, r) for r in routes if r]
    served = sum(len(r) for r in routes)
    late = sum(s["late"] for s in sims)
    km = sum(s["km"] for s in sims)
    minutes = [s["minutes"] for s in sims]
    overtime = sum(max(sub(m, problem.shift_minutes), 0.0) for m in minutes)
    driver_hours = sum(minutes) / 60.0
    unassigned = unassigned or []
    total = served + len(unassigned)
    return {
        "method": name,
        "vehicles_used": len(sims),
        "fleet_available": problem.vehicles,
        "stops_served": served,
        "unassigned_stops": len(unassigned),
        "on_time_pct": safe_float(100.0 * sub(served, late) / total if total else 0.0, 1),
        "late_stops": int(late + len(unassigned)),
        "total_late_minutes": safe_float(sum(s["late_minutes"] for s in sims), 1),
        "total_km": safe_float(km, 1),
        "longest_route_minutes": safe_float(max(minutes) if minutes else 0.0, 1),
        "overtime_minutes": safe_float(overtime, 1),
        "capacity_utilisation_pct": safe_float(
            100.0 * sum(s["load"] for s in sims) / (problem.capacity * max(len(sims), 1)), 1),
        "co2_kg": safe_float(km * CO2_KG_PER_KM, 1),
        "cost_gbp": safe_float(km * COST_PER_KM_GBP + driver_hours * DRIVER_COST_PER_HOUR_GBP
                               + overtime / 60.0 * DRIVER_COST_PER_HOUR_GBP * OVERTIME_PREMIUM
                               + (late + len(unassigned)) * LATE_STOP_PENALTY_GBP, 2),
        "runtime_ms": safe_float(runtime_ms, 1),
        "routes": [{"vehicle": i + 1, "stops": len(r), "km": safe_float(s["km"], 1),
                    "finish": _clock(s["minutes"]), "load": safe_float(s["load"], 1),
                    "late_stops": s["late"], "sequence": s["stops"]}
                   for i, (r, s) in enumerate(zip([r for r in routes if r], sims))],
    }


def nearest_neighbour_plan(problem):
    """Greedy planner baseline: nearest next stop, ignoring time windows."""
    started = time.perf_counter()
    D, T, svc, dem = problem._D, problem._T, problem._svc, problem._dem
    remaining = set(range(1, problem.size + 1))
    routes = []
    while remaining:
        route, prev, t, load = [], 0, 0.0, 0.0
        while True:
            best, best_d = None, None
            for s in remaining:
                if load + dem[s] > problem.capacity + EPS:
                    continue
                finish = t + T[prev][s] + svc[s] + T[s][0]
                if route and finish > problem.shift_minutes:
                    continue
                if best_d is None or D[prev][s] < best_d:
                    best, best_d = s, D[prev][s]
            if best is None:
                break
            t += T[prev][best] + svc[best]
            load += dem[best]
            route.append(best)
            remaining.discard(best)
            prev = best
        if not route:
            route = [remaining.pop()]
        routes.append(route)
    return summarise(problem, routes, "Nearest neighbour, windows ignored",
                     sub(time.perf_counter(), started) * 1000.0)


def sweep_plan(problem):
    """Territory planner baseline: sweep by angle around the depot, cut by capacity and shift."""
    started = time.perf_counter()
    dx, dy = problem.depot
    angle = np.arctan2(np.subtract(problem.xy[:, 1], dy), np.subtract(problem.xy[:, 0], dx))
    order = [int(i) + 1 for i in np.argsort(angle, kind="stable")]
    T, svc, dem = problem._T, problem._svc, problem._dem
    routes, route, load, t, prev = [], [], 0.0, 0.0, 0
    for s in order:
        step = T[prev][s] + svc[s]
        if route and (load + dem[s] > problem.capacity + EPS or t + step + T[s][0] > problem.shift_minutes):
            routes.append(route)
            route, load, t, prev = [], 0.0, 0.0, 0
            step = T[prev][s] + svc[s]
        route.append(s)
        load += dem[s]
        t += step
        prev = s
    if route:
        routes.append(route)
    return summarise(problem, routes, "Sweep territories", sub(time.perf_counter(), started) * 1000.0)


def optimise(problem, time_budget_s=1.5, seed=11):
    """Full Routewright optimisation: savings construction, route elimination and local search."""
    started = time.perf_counter()
    deadline = started + float(time_budget_s)
    model = _Model(problem)
    routes, _, unassigned = _savings_routes(model, problem)
    routes = _eliminate_routes(model, routes, problem.vehicles)
    nb = _neighbours(problem, 12) if problem.size > 1 else np.zeros((1, 1), dtype=int)
    rng = random.Random(seed)
    improving = True
    while improving and time.perf_counter() < deadline:
        for r in list(routes):
            routes[r] = _two_opt(model, routes[r])
        improving = _relocate_pass(model, routes, nb, deadline)
        if not improving and time.perf_counter() < deadline:
            keys = list(routes)
            rng.shuffle(keys)
            routes = {k: routes[k] for k in keys}
    routes, iterations, accepted = _ruin_recreate(model, routes, nb, deadline, rng, problem.vehicles)
    plan = [routes[r] for r in sorted(routes, key=lambda r: routes[r][0])]
    result = summarise(problem, plan, "Routewright optimiser", sub(time.perf_counter(), started) * 1000.0,
                       unassigned=unassigned)
    result["search_iterations"] = iterations
    result["improvements_accepted"] = accepted
    result["unassigned_stop_ids"] = [problem.stop_ids[sub(s, 1)] for s in unassigned]
    result["extra_vehicles_needed"] = max(sub(len(plan), problem.vehicles), 0)
    return result


def compare(problem, time_budget_s=1.5):
    """Optimise and benchmark against the two planner baselines on the same travel model."""
    best = optimise(problem, time_budget_s)
    baselines = [nearest_neighbour_plan(problem), sweep_plan(problem)]
    summary = {k: v for k, v in best.items() if k != "routes"}
    rows = [summary] + [{k: v for k, v in b.items() if k != "routes"} for b in baselines]
    ref = max(baselines, key=lambda b: b["on_time_pct"])
    return {
        "problem": {"stops": problem.size, "vehicles": problem.vehicles, "capacity": problem.capacity,
                    "shift_minutes": problem.shift_minutes, "speed_kmh": problem.speed_kmh},
        "plan": best,
        "comparison": rows,
        "on_time_gain_points": safe_float(max(sub(best["on_time_pct"], ref["on_time_pct"]), 0.0), 1),
        "km_saving_pct_vs_sweep": safe_float(
            max(sub(1.0, best["total_km"] / baselines[1]["total_km"]), 0.0) * 100.0 if baselines[1]["total_km"] else 0.0, 1),
    }
