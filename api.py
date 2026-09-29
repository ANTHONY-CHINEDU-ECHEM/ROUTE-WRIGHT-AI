"""REST API.

Start with: python routewright.py serve
Interactive documentation is served at /docs once running.

Set ROUTEWRIGHT_API_KEY to require an Authorization header of the form
"Bearer <key>" on every endpoint except health.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .engine import RoutewrightEngine
from .utils import sub

STARTED = time.time()
FilterValue = Union[str, List[str]]


@asynccontextmanager
async def lifespan(app):
    RoutewrightEngine.shared()
    yield


app = FastAPI(
    title="Routewright AI",
    version=__version__,
    description="Evidence grounded diagnosis and fix ranking for last mile delivery incidents, drawn from a base "
                "of historical incidents with verified citations, plus a time window aware route optimiser.",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("ROUTEWRIGHT_CORS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def require_key(authorization: Optional[str] = Header(default=None)):
    expected = os.environ.get("ROUTEWRIGHT_API_KEY")
    if not expected:
        return
    if authorization != "Bearer " + expected:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000,
                          examples=["Our vans keep missing the one hour slots customers booked. What should we do?"])
    filters: Optional[Dict[str, FilterValue]] = Field(
        default=None, description="Hard filters on incident metadata, for example {\"service_type\": \"Parcel Courier\"}.")
    k: int = Field(default=8, ge=1, le=25, description="Number of precedent incidents in the context.")
    provider: Optional[str] = Field(default=None, description="extractive, anthropic or ollama.")
    strict: bool = Field(default=True, description="Replace model answers that fail verification.")
    include_evidence_pack: bool = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    filters: Optional[Dict[str, FilterValue]] = None
    k: int = Field(default=10, ge=1, le=100)
    mode: str = Field(default="hybrid", pattern="^(hybrid|fusion|bm25|dense)$")


class RecommendRequest(BaseModel):
    incident: str = Field(..., examples=["Route overruns"])
    service_type: Optional[str] = None
    area_type: Optional[str] = None
    fleet_type: Optional[str] = None


class AssessRequest(BaseModel):
    service_type: str = "Parcel Courier"
    area_type: str = "Urban"
    region: str = "North West England"
    operator_size_band: str = "Regional"
    fleet_type: str = "Diesel Vans"
    planning_tool: str = "Static Route Software"
    visibility_level: str = "Daily KPI Dashboards"
    delivery_promise: str = "Next Day Untimed"
    season: str = "Standard Trading"
    daily_drops: float = Field(default=1400, gt=0)
    avg_stops_per_route: float = Field(default=110, gt=0)
    avg_route_km: float = Field(default=70, gt=0)
    avg_consignment_weight_kg: float = Field(default=2.5, gt=0)
    time_window_minutes: float = Field(default=600, gt=0)
    agency_driver_share_pct: float = Field(default=15, ge=0, le=100)
    depot_count: float = Field(default=2, ge=1)
    electric_share_pct: float = Field(default=10, ge=0, le=100)


class StopIn(BaseModel):
    stop_id: str
    x_km: float = Field(..., ge=0)
    y_km: float = Field(..., ge=0)
    demand: float = Field(default=1.0, ge=0)
    window_start_min: float = Field(default=0.0, ge=0)
    window_end_min: float = Field(default=600.0, ge=0)
    service_min: float = Field(default=3.0, ge=0)


class OptimiseRequest(BaseModel):
    stops: Optional[List[StopIn]] = Field(default=None, description="Your own stops. Leave empty for a demo problem.")
    depot_x_km: Optional[float] = Field(default=None, ge=0)
    depot_y_km: Optional[float] = Field(default=None, ge=0)
    vehicles: int = Field(default=10, ge=1, le=500)
    capacity: float = Field(default=80.0, gt=0)
    shift_minutes: float = Field(default=600.0, gt=0)
    area_type: str = "Urban"
    promise: str = "Two Hour Slot"
    demo_stops: int = Field(default=120, ge=2, le=600)
    seed: int = 7
    seconds: float = Field(default=1.5, gt=0, le=20)
    include_routes: bool = True


@app.get("/")
def root():
    return {"name": "Routewright AI", "version": __version__, "docs": "/docs"}


@app.get("/health")
def health():
    engine = RoutewrightEngine.shared()
    return {"status": "ok", "incidents": engine.index.size, "uptime_seconds": round(sub(time.time(), STARTED), 1)}


@app.get("/stats", dependencies=[Depends(require_key)])
def stats():
    return RoutewrightEngine.shared().stats()


@app.get("/options", dependencies=[Depends(require_key)])
def options():
    return RoutewrightEngine.shared().options()


@app.post("/ask", dependencies=[Depends(require_key)])
def ask(req: AskRequest):
    result = dict(RoutewrightEngine.shared().ask(req.question, filters=req.filters, k=req.k,
                                               provider=req.provider, strict=req.strict))
    if req.include_evidence_pack:
        from .answer import evidence_pack
        result["evidence_pack"] = evidence_pack(req.question, result)
    return result


@app.post("/search", dependencies=[Depends(require_key)])
def search(req: SearchRequest):
    return RoutewrightEngine.shared().search(req.query, filters=req.filters, k=req.k, mode=req.mode)


@app.post("/recommend", dependencies=[Depends(require_key)])
def recommend(req: RecommendRequest):
    try:
        return RoutewrightEngine.shared().recommend(req.incident, req.service_type, req.area_type, req.fleet_type)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'"))


@app.post("/assess", dependencies=[Depends(require_key)])
def assess(req: AssessRequest):
    return RoutewrightEngine.shared().assess(req.model_dump())


@app.post("/optimise", dependencies=[Depends(require_key)])
def optimise(req: OptimiseRequest):
    from . import optimiser
    import numpy as np
    if req.stops:
        xs = np.array([s.x_km for s in req.stops])
        ys = np.array([s.y_km for s in req.stops])
        preset = optimiser.AREA_PRESETS.get(req.area_type, optimiser.AREA_PRESETS["Urban"])
        depot = (req.depot_x_km if req.depot_x_km is not None else float(np.median(xs)),
                 req.depot_y_km if req.depot_y_km is not None else float(np.median(ys)))
        problem = optimiser.Problem(
            depot=depot, xy=np.column_stack([xs, ys]), demand=np.array([s.demand for s in req.stops]),
            window_start=np.array([s.window_start_min for s in req.stops]),
            window_end=np.array([s.window_end_min for s in req.stops]),
            service=np.array([s.service_min for s in req.stops]), vehicles=req.vehicles, capacity=req.capacity,
            shift_minutes=req.shift_minutes, speed_kmh=preset["speed_kmh"], circuity=preset["circuity"],
            stop_ids=[s.stop_id for s in req.stops])
    else:
        if req.area_type not in optimiser.AREA_PRESETS or req.promise not in optimiser.PROMISE_WINDOWS:
            raise HTTPException(status_code=422, detail="Unknown area_type or promise.")
        problem = optimiser.generate_problem(req.demo_stops, req.area_type, req.promise, seed=req.seed)
    result = optimiser.compare(problem, time_budget_s=req.seconds)
    if not req.include_routes:
        result["plan"] = {k: v for k, v in result["plan"].items() if k != "routes"}
    return result


@app.get("/incidents/{incident_id}", dependencies=[Depends(require_key)])
def incident(incident_id: str):
    found = RoutewrightEngine.shared().case(incident_id.upper())
    if not found:
        raise HTTPException(status_code=404, detail="No incident with id " + incident_id)
    return found
