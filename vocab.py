"""Domain knowledge for Routewright AI.

The single source of truth for the last mile delivery world that the synthetic
dataset simulates and that the retrieval engine understands: operator
profiles, the fifteen delivery performance incidents, the fixes operators
apply, and the ground truth efficacy matrix linking each incident to the fixes
that genuinely restore on time performance.

Outcomes in the generated data are driven by that matrix together with
planning maturity, operational visibility, speed of response and severity, so
a system that ranks fixes from evidence can be scored against a known truth.
"""

SERVICE_TYPES = {
    "Parcel Courier": {"weight": 22, "drops": 1400, "stops_per_route": 120, "weight_kg": 2.5, "cost_per_drop": 1.9, "residential": 0.85, "bulky": 0.0},
    "Ecommerce Same Day": {"weight": 10, "drops": 600, "stops_per_route": 45, "weight_kg": 3.0, "cost_per_drop": 4.8, "residential": 0.9, "bulky": 0.0},
    "Grocery Home Delivery": {"weight": 14, "drops": 900, "stops_per_route": 18, "weight_kg": 28.0, "cost_per_drop": 9.5, "residential": 0.95, "bulky": 0.3},
    "Two Person Furniture Delivery": {"weight": 7, "drops": 160, "stops_per_route": 9, "weight_kg": 70.0, "cost_per_drop": 42.0, "residential": 0.9, "bulky": 1.0},
    "Pharmacy and Healthcare": {"weight": 7, "drops": 450, "stops_per_route": 35, "weight_kg": 1.5, "cost_per_drop": 6.5, "residential": 0.6, "bulky": 0.0},
    "Food Service Wholesale": {"weight": 10, "drops": 380, "stops_per_route": 16, "weight_kg": 180.0, "cost_per_drop": 24.0, "residential": 0.05, "bulky": 0.6},
    "Builders Merchant": {"weight": 7, "drops": 140, "stops_per_route": 8, "weight_kg": 900.0, "cost_per_drop": 55.0, "residential": 0.3, "bulky": 1.0},
    "Retail Store Replenishment": {"weight": 8, "drops": 220, "stops_per_route": 10, "weight_kg": 450.0, "cost_per_drop": 38.0, "residential": 0.0, "bulky": 0.5},
    "Pallet Network Final Mile": {"weight": 8, "drops": 300, "stops_per_route": 14, "weight_kg": 380.0, "cost_per_drop": 31.0, "residential": 0.2, "bulky": 0.8},
    "Meal Kit Subscription": {"weight": 7, "drops": 700, "stops_per_route": 40, "weight_kg": 6.0, "cost_per_drop": 5.5, "residential": 1.0, "bulky": 0.0},
}

AREA_TYPES = {
    "Dense Urban": {"weight": 22, "speed_kmh": 16, "route_km": 45, "congestion": 1.0, "rural": 0.0},
    "Urban": {"weight": 28, "speed_kmh": 24, "route_km": 70, "congestion": 0.7, "rural": 0.0},
    "Suburban": {"weight": 24, "speed_kmh": 32, "route_km": 105, "congestion": 0.4, "rural": 0.2},
    "Rural": {"weight": 14, "speed_kmh": 44, "route_km": 190, "congestion": 0.1, "rural": 1.0},
    "Mixed Region": {"weight": 12, "speed_kmh": 34, "route_km": 130, "congestion": 0.5, "rural": 0.5},
}

SIZE_BANDS = {
    "Local": {"weight": 26, "scale": 0.25, "planning": (0.35, 0.40, 0.20, 0.05), "visibility": (0.60, 0.35, 0.05)},
    "Regional": {"weight": 34, "scale": 1.0, "planning": (0.12, 0.33, 0.37, 0.18), "visibility": (0.35, 0.48, 0.17)},
    "National": {"weight": 27, "scale": 4.0, "planning": (0.03, 0.12, 0.45, 0.40), "visibility": (0.15, 0.50, 0.35)},
    "Enterprise": {"weight": 13, "scale": 12.0, "planning": (0.01, 0.05, 0.34, 0.60), "visibility": (0.05, 0.40, 0.55)},
}

FLEET_TYPES = {"Diesel Vans": 34, "Electric Vans": 16, "Mixed Van Fleet": 24, "Rigid Trucks": 16, "Cargo Bikes and Vans": 10}
PLANNING_TOOLS = ["Manual Planning", "Spreadsheet Planning", "Static Route Software", "Dynamic Route Optimisation"]
VISIBILITY_LEVELS = ["Basic Reporting", "Daily KPI Dashboards", "Real Time Control Tower"]
DELIVERY_PROMISES = {"One Hour Slot": 14, "Two Hour Slot": 22, "Half Day Window": 24, "Next Day Untimed": 28, "Same Day": 12}
PROMISE_WINDOW_MINUTES = {"One Hour Slot": 60, "Two Hour Slot": 120, "Half Day Window": 300, "Next Day Untimed": 600,
                          "Same Day": 240}
REGIONS = {"London": 16, "South East England": 12, "Midlands": 12, "North West England": 11, "Scotland": 7,
           "Wales and South West": 8, "Ireland": 6, "Western Europe": 12, "North America": 10, "Australia and New Zealand": 6}
SEASONS = {"Standard Trading": 55, "Peak Season": 18, "Winter": 15, "Summer Holidays": 12}

SEVERITIES = ["Low", "Medium", "High", "Critical"]
DETECTION_CHANNELS = ["Control Tower Alert", "Weekly KPI Review", "Customer Complaints", "Client Escalation",
                      "Driver Feedback"]
RECOVERY_LEVELS = ["Full Recovery", "Partial Recovery", "Not Recovered"]
TEAMS = ["Transport Planning", "Depot Operations", "Driver Management", "Customer Service", "Fleet Engineering",
         "Technology", "Client Account Team"]
SHIFTS = ["Early Shift", "Day Shift", "Evening Shift", "All Shifts"]

CODENAMES = [
    "Harbourline", "Kestrel", "Meridian", "Ashford", "Halcyon", "Lumen", "Solway", "Pennine", "Atlas",
    "Blackwater", "Cobalt", "Driftwood", "Elmstead", "Falcon", "Granite", "Heron", "Juniper", "Kingsway",
    "Larch", "Marlow", "Nimbus", "Oakridge", "Pioneer", "Redwing", "Saltire", "Thornbury", "Upland",
    "Vantage", "Westmoor", "Yarrow", "Zenith", "Bramble", "Cedar", "Dunmore", "Evergreen", "Foxglove",
    "Greystone", "Highfield", "Keystone", "Longford", "Millbank", "Newhaven", "Orion", "Portland",
    "Riverside", "Sterling", "Tamar", "Wharfedale", "Arden", "Beacon", "Clyde", "Derwent", "Exmoor",
    "Fenwick", "Galloway", "Hadrian", "Kielder", "Lowther", "Moorcroft", "Northwick",
]
OPERATOR_SUFFIX = ["Logistics", "Delivery", "Couriers", "Distribution", "Transport", "Freight", "Express", "Carriers"]

# Incidents. Each carries the language the dataset uses to describe it
# (symptoms), independent phrasing used only by the evaluation harness (queries)
# and the thesaurus the query understanding layer listens for (keywords).
INCIDENTS = {
    "window_misses": {
        "name": "Delivery window misses",
        "root_causes": ["Routes planned without time window constraints", "Slots sold beyond route capacity",
                        "Service times underestimated at each stop", "Planned travel times ignore time of day"],
        "symptoms": [
            "on time delivery within the promised window fell to {ontime} percent against a {base} percent norm",
            "{fails} deliveries a day arrived outside their booked slot",
            "customers booked into morning slots were reached after midday on {pct} percent of routes",
        ],
        "queries": [
            "our drivers keep arriving outside the slot the customer booked",
            "we promise two hour windows but half the vans are late for them",
            "timed deliveries are consistently late in the afternoon",
        ],
        "keywords": ["time window", "time windows", "slot", "slots", "booked slot", "window", "windows", "on time",
                     "late deliveries", "arriving late", "outside the slot", "timed deliveries", "eta", "otif"],
    },
    "failed_first_attempt": {
        "name": "Failed first attempt deliveries",
        "root_causes": ["Customers not home at delivery time", "No advance notification of arrival",
                        "No safe place or neighbour option", "Deliveries attempted outside customer availability"],
        "symptoms": [
            "first attempt success fell to {ftr} percent and {fails} deliveries a day needed a second attempt",
            "carded and not home outcomes rose by {pct} percent",
            "redelivery volumes consumed {pct} percent of next day capacity",
        ],
        "queries": [
            "nobody is home when our drivers turn up so we keep going back twice",
            "we are paying for redeliveries because the first attempt fails so often",
            "too many parcels come back to the depot undelivered",
        ],
        "keywords": ["not home", "nobody home", "first attempt", "redelivery", "redeliveries", "failed delivery",
                     "failed deliveries", "undelivered", "carded", "back to the depot", "second attempt"],
    },
    "route_overrun": {
        "name": "Route overruns",
        "root_causes": ["Too many stops loaded per route", "Route lengths planned on straight line distance",
                        "Driver breaks not modelled in plans", "Routes not rebalanced after volume growth"],
        "symptoms": [
            "routes finished {overrun} minutes beyond the planned shift on average",
            "{pct} percent of routes needed overtime to complete",
            "drivers returned {fails} undelivered stops a day because shifts ran out",
        ],
        "queries": [
            "drivers cannot finish their rounds before the end of their shift",
            "routes are running hours over and we pay overtime every day",
            "vans come back with stops still on board because they ran out of time",
        ],
        "keywords": ["overrun", "overruns", "run over", "running over", "overtime", "cannot finish", "end of shift",
                     "shift", "rounds", "too many stops", "ran out of time", "hours over", "route length"],
    },
    "late_departures": {
        "name": "Late depot departures",
        "root_causes": ["Loading not sequenced by route", "Late trunk arrivals into the depot",
                        "Sortation capacity below volume", "Vehicle checks and handover too slow"],
        "symptoms": [
            "vans left the depot {overrun} minutes after planned departure",
            "{pct} percent of routes departed late every morning",
            "loading bays were congested for {hours} hours after the planned start",
        ],
        "queries": [
            "the vans are leaving the yard far too late every morning",
            "loading takes forever so drivers start their routes late",
            "our trunk arrives late into the depot and everything goes out behind schedule",
        ],
        "keywords": ["depot", "loading", "loaded", "departure", "departures", "leaving late", "yard", "trunk",
                     "sortation", "sort", "loading bay", "morning start", "leave the depot"],
    },
    "address_quality": {
        "name": "Address and geocoding errors",
        "root_causes": ["Free text addresses not validated at checkout", "Geocodes placed at postcode centroids",
                        "New build properties missing from maps", "Rural addresses without precise locations"],
        "symptoms": [
            "{pct} percent of stops were geocoded to the wrong location",
            "drivers spent {overrun} minutes a route searching for addresses",
            "{fails} deliveries a day failed because the address could not be found",
        ],
        "queries": [
            "drivers cannot find the addresses and waste time driving around",
            "the pins on the map are in the wrong place for lots of customers",
            "new housing estates are not on our maps so parcels go missing",
        ],
        "keywords": ["address", "addresses", "geocode", "geocoding", "postcode", "cannot find the address",
                     "find the address", "find addresses", "wrong location",
                     "pins", "map", "maps", "new build", "location", "what3words"],
    },
    "capacity_shortfall": {
        "name": "Vehicle capacity shortfall",
        "root_causes": ["Peak volumes above forecast", "Fleet sized for average days",
                        "Client onboarded without a capacity plan", "Vehicles off road awaiting repair"],
        "symptoms": [
            "volume exceeded fleet capacity by {pct} percent for {days} days",
            "{fails} consignments a day were rolled over to the next day",
            "vans left full with consignments still on the depot floor",
        ],
        "queries": [
            "we simply do not have enough vans for the volume this week",
            "parcels are being left behind at the depot because every vehicle is full",
            "peak has hit and we are rolling orders to tomorrow",
        ],
        "keywords": ["capacity", "not enough vans", "enough vans", "vehicles full", "volume", "volumes", "peak",
                     "rolled over", "rolling", "left behind", "fleet size", "vans are full", "backlog"],
    },
    "driver_shortage": {
        "name": "Driver shortage and absence",
        "root_causes": ["High driver turnover", "Reliance on agency drivers unfamiliar with routes",
                        "Sickness spikes without cover", "Pay below local market rates"],
        "symptoms": [
            "{pct} percent of routes were uncovered at shift start",
            "agency drivers covered {pct} percent of routes and completed fewer stops",
            "driver absence left {fails} deliveries a day unassigned",
        ],
        "queries": [
            "we cannot find enough drivers and routes are left uncovered",
            "half our drivers are agency staff who do not know the area",
            "drivers keep quitting and absence is through the roof",
        ],
        "keywords": ["driver shortage", "enough drivers", "not enough drivers", "short of drivers", "agency",
                     "absence", "sickness", "turnover", "quitting", "recruit", "recruitment", "uncovered",
                     "staff shortage", "no drivers", "cover the rounds", "temps"],
    },
    "congestion": {
        "name": "Congestion driven delays",
        "root_causes": ["Plans use free flow travel times", "Roadworks on key corridors",
                        "School run and rush hour overlap", "Routes crossing the city centre at peak times"],
        "symptoms": [
            "average stop to stop time rose {pct} percent in the city centre",
            "routes lost {overrun} minutes a day to traffic",
            "afternoon drops ran {overrun} minutes behind plan",
        ],
        "queries": [
            "traffic in the city centre is wrecking our schedules",
            "roadworks and rush hour mean every route runs late",
            "our plans assume empty roads and reality is gridlock",
        ],
        "keywords": ["traffic", "congestion", "gridlock", "roadworks", "rush hour", "city centre", "jams",
                     "school run", "road closures", "journey times"],
    },
    "ev_range": {
        "name": "Electric van range shortfall",
        "root_causes": ["Routes planned without energy constraints", "Cold weather reducing battery range",
                        "Depot charging capacity too low", "Payload heavier than the range assumption"],
        "symptoms": [
            "{pct} percent of electric vans returned with under ten percent charge",
            "{fails} routes a week were cut short to reach a charger",
            "vans were not fully charged by morning on {pct} percent of days",
        ],
        "queries": [
            "our electric vans are running out of battery before the route ends",
            "the cold weather has killed the range on the ev fleet",
            "we cannot charge all the electric vans overnight",
        ],
        "keywords": ["electric", "ev", "evs", "battery", "range", "charge", "charging", "charger", "chargers",
                     "electric vans", "zero emission"],
    },
    "damaged_goods": {
        "name": "Damaged deliveries",
        "root_causes": ["Poor load securing", "Heavy items stacked on fragile goods",
                        "Rushed handling at the doorstep", "Packaging not fit for the network"],
        "symptoms": [
            "damage claims rose to {claims} per thousand drops",
            "{pct} percent of bulky items arrived damaged",
            "customers refused {fails} deliveries a week because of visible damage",
        ],
        "queries": [
            "customers are receiving broken items and claims are soaring",
            "furniture keeps arriving scratched and dented",
            "goods are getting damaged in the back of the vans",
        ],
        "keywords": ["damage", "damaged", "broken", "damage claims", "scratched", "dented", "breakage", "breakages",
                     "fragile", "load securing", "packaging", "refused", "broken items", "arrive broken"],
    },
    "pod_disputes": {
        "name": "Proof of delivery disputes",
        "root_causes": ["No photo or geotag at delivery", "Parcels left in unsafe places",
                        "Signature capture skipped", "Doorstep theft in high risk areas"],
        "symptoms": [
            "delivered but not received claims reached {claims} per thousand drops",
            "{pct} percent of disputes could not be defended without proof",
            "missing parcel refunds rose {pct} percent in {days} days",
        ],
        "queries": [
            "customers say parcels never arrived even though they show as delivered",
            "we keep refunding missing parcels because we cannot prove delivery",
            "parcels are being stolen from doorsteps",
        ],
        "keywords": ["proof of delivery", "pod", "never arrived", "not received", "missing parcel",
                     "missing parcels", "stolen", "theft", "dispute", "disputes", "signature", "photo", "refunds"],
    },
    "access_restrictions": {
        "name": "Access and restriction breaches",
        "root_causes": ["Low emission zone rules ignored in planning", "Loading bay and time restrictions not recorded",
                        "Gated sites without access codes", "Vehicle too large for the street"],
        "symptoms": [
            "{fails} deliveries a week failed because the vehicle could not access the site",
            "penalty charge notices rose {pct} percent in a month",
            "drivers waited {overrun} minutes a day for site access",
        ],
        "queries": [
            "our lorries cannot get into the delivery sites and keep getting parking fines",
            "we are being fined for entering the low emission zone",
            "drivers get stuck at gated sites without the access code",
        ],
        "keywords": ["access", "fines", "fine", "parking", "penalty", "pcn", "low emission", "ulez", "clean air zone",
                     "gated", "access code", "restrictions", "weight limit", "loading restrictions"],
    },
    "customer_comms": {
        "name": "Customer notification failures",
        "root_causes": ["ETA messages sent from static plans", "Notification service not integrated with telematics",
                        "Delays never communicated", "Contact details missing from orders"],
        "symptoms": [
            "where is my order contacts rose to {claims} per thousand drops",
            "{pct} percent of customers received an arrival time that proved wrong",
            "call centre volumes rose {pct} percent in {days} days",
        ],
        "queries": [
            "customers keep calling to ask where their delivery is",
            "the arrival times we text people are completely wrong",
            "nobody tells the customer when a delivery is running late",
        ],
        "keywords": ["where is my order", "wismo", "notification", "notifications", "text", "sms", "tracking",
                     "arrival time", "arrival times", "calling", "calls", "call centre", "updates", "communicate",
                     "tell the customer", "tells the customer", "let customers know", "keep customers informed",
                     "nobody tells", "proactive updates", "phone us"],
    },
    "unbalanced_routes": {
        "name": "Unbalanced route allocation",
        "root_causes": ["Fixed territories that no longer match demand", "Stops allocated by postcode only",
                        "Planner habit in route allocation", "No workload measure beyond stop count"],
        "symptoms": [
            "the busiest {pct} percent of routes carried twice the workload of the lightest",
            "some drivers finished {overrun} minutes early while others ran into overtime",
            "overtime concentrated on {pct} percent of drivers",
        ],
        "queries": [
            "some drivers finish at lunchtime while others are out until nine at night",
            "our territories are fixed and the workload is completely uneven",
            "the same few routes always run late while others are half empty",
        ],
        "keywords": ["uneven", "unbalanced", "balance", "territories", "territory", "workload", "workloads",
                     "half empty", "finish early", "allocation", "fairness", "same routes", "some drivers",
                     "while others", "finish at lunchtime"],
    },
    "missed_collections": {
        "name": "Missed collections and returns",
        "root_causes": ["Collections added after routes were planned", "Returns not visible to drivers",
                        "Collection windows not recorded", "Vehicle full before collections"],
        "symptoms": [
            "{pct} percent of booked collections were missed",
            "{fails} returns a day were not picked up on the agreed date",
            "business customers escalated missed collections {claims} times a month",
        ],
        "queries": [
            "we keep missing the returns customers booked for collection",
            "pickups are forgotten because they are added after the route is planned",
            "business customers complain that we never collect on the right day",
        ],
        "keywords": ["collection", "collections", "collect", "pickup", "pickups", "pick up", "returns", "return",
                     "reverse logistics", "booked collection"],
    },
}

FIXES = {
    "dynamic_routing": {"name": "Dynamic route optimisation with time windows", "team": "Transport Planning",
                        "mechanism": "routes were replanned daily against every time window, capacity and shift limit",
                        "keywords": ["route optimisation", "optimise routes", "dynamic routing"]},
    "slot_capacity": {"name": "Slot capacity control linked to route plans", "team": "Transport Planning",
                      "mechanism": "slots were only offered while the plan could still reach them on time",
                      "keywords": ["slot capacity", "slot management"]},
    "service_time_model": {"name": "Measured service times per stop type", "team": "Transport Planning",
                           "mechanism": "planned stop durations were replaced with measured times by property and item type",
                           "keywords": ["service times", "stop times"]},
    "traffic_aware_planning": {"name": "Time dependent traffic aware planning", "team": "Transport Planning",
                               "mechanism": "plans used historical travel times by hour of day and avoided peak corridors",
                               "keywords": ["traffic aware", "time dependent"]},
    "predictive_eta": {"name": "Predictive ETA notifications", "team": "Technology",
                       "mechanism": "customers received live arrival windows updated from vehicle telematics",
                       "keywords": ["eta notifications", "live tracking"]},
    "safe_place_lockers": {"name": "Safe place, neighbour and locker options", "team": "Customer Service",
                           "mechanism": "customers chose a safe place, neighbour or locker before the first attempt",
                           "keywords": ["safe place", "lockers", "neighbour"]},
    "address_validation": {"name": "Address validation and rooftop geocoding", "team": "Technology",
                           "mechanism": "addresses were validated at order entry and pinned to the actual entrance",
                           "keywords": ["address validation", "rooftop geocoding"]},
    "wave_loading": {"name": "Route sequenced wave loading", "team": "Depot Operations",
                     "mechanism": "vans were loaded in delivery sequence in waves timed to departure slots",
                     "keywords": ["wave loading", "loading sequence"]},
    "flex_capacity": {"name": "Flexible capacity pool with vetted subcontractors", "team": "Depot Operations",
                      "mechanism": "a vetted pool of subcontracted vans was called on against a daily volume trigger",
                      "keywords": ["flexible capacity", "subcontractors"]},
    "driver_retention": {"name": "Driver retention programme and relief pool", "team": "Driver Management",
                         "mechanism": "pay, rota stability and a trained relief pool reduced turnover and absence",
                         "keywords": ["driver retention", "relief pool"]},
    "route_familiarity": {"name": "Route familiarity and consistent driver areas", "team": "Driver Management",
                          "mechanism": "drivers kept consistent areas and new drivers shadowed experienced ones",
                          "keywords": ["route familiarity", "shadowing"]},
    "energy_planning": {"name": "Energy aware routing and charging schedule", "team": "Fleet Engineering",
                        "mechanism": "routes were planned against battery range and charging was staggered overnight",
                        "keywords": ["energy aware", "charging schedule"]},
    "load_securing": {"name": "Load securing standards and handling training", "team": "Depot Operations",
                      "mechanism": "load plans, straps and doorstep handling training were made mandatory",
                      "keywords": ["load securing", "handling training"]},
    "photo_pod": {"name": "Geotagged photo proof of delivery", "team": "Technology",
                  "mechanism": "every drop captured a geotagged photo checked against the delivery location",
                  "keywords": ["photo proof", "geotagged photo"]},
    "access_database": {"name": "Site access and restriction database", "team": "Transport Planning",
                        "mechanism": "access codes, zone rules and vehicle limits were attached to every address",
                        "keywords": ["access database", "restriction database"]},
    "workload_balancing": {"name": "Workload balanced route allocation", "team": "Transport Planning",
                           "mechanism": "routes were balanced on drive time and service time rather than stop count",
                           "keywords": ["workload balancing", "rebalance routes"]},
    "collection_integration": {"name": "Collections integrated into daily route plans", "team": "Transport Planning",
                               "mechanism": "collections and returns were planned alongside deliveries with their own windows",
                               "keywords": ["integrated collections", "returns planning"]},
    "control_tower": {"name": "Real time control tower with exception alerts", "team": "Technology",
                      "mechanism": "a control tower flagged late running routes and triggered interventions within the hour",
                      "keywords": ["control tower", "exception alerts"]},
    "micro_hubs": {"name": "Urban micro hubs and cargo bike final mile", "team": "Depot Operations",
                   "mechanism": "city centre drops moved to cargo bikes working from small local hubs",
                   "keywords": ["micro hub", "cargo bikes"]},
    "customer_rebooking": {"name": "Self service rebooking before dispatch", "team": "Customer Service",
                           "mechanism": "customers confirmed or moved their slot the evening before dispatch",
                           "keywords": ["rebooking", "confirm slot"]},
    "extended_shifts": {"name": "Extended driver shifts", "team": "Driver Management",
                        "mechanism": "driver shifts were lengthened to push more stops through each day",
                        "keywords": ["longer shifts", "extended shifts"]},
    "more_stops_per_route": {"name": "Loading more stops onto each route", "team": "Transport Planning",
                             "mechanism": "stop targets per route were raised to absorb the extra volume",
                             "keywords": ["more stops", "raise stop targets"]},
}

EFFICACY_TIERS = {"strong": 0.85, "moderate": 0.55, "neutral": 0.25, "harmful": 0.08}

EFFICACY_MAP = {
    "window_misses": {"strong": ["dynamic_routing", "slot_capacity"],
                      "moderate": ["service_time_model", "traffic_aware_planning"],
                      "harmful": ["more_stops_per_route", "extended_shifts"]},
    "failed_first_attempt": {"strong": ["safe_place_lockers", "predictive_eta"],
                             "moderate": ["customer_rebooking"], "harmful": ["more_stops_per_route"]},
    "route_overrun": {"strong": ["dynamic_routing", "service_time_model"],
                      "moderate": ["workload_balancing", "traffic_aware_planning"],
                      "harmful": ["extended_shifts", "more_stops_per_route"]},
    "late_departures": {"strong": ["wave_loading"], "moderate": ["control_tower"],
                        "harmful": ["extended_shifts"]},
    "address_quality": {"strong": ["address_validation"], "moderate": ["route_familiarity"],
                        "harmful": ["more_stops_per_route"]},
    "capacity_shortfall": {"strong": ["flex_capacity"], "moderate": ["slot_capacity", "dynamic_routing"],
                           "harmful": ["more_stops_per_route", "extended_shifts"]},
    "driver_shortage": {"strong": ["driver_retention"], "moderate": ["flex_capacity", "route_familiarity"],
                        "harmful": ["extended_shifts"]},
    "congestion": {"strong": ["traffic_aware_planning", "micro_hubs"], "moderate": ["dynamic_routing"],
                   "harmful": ["more_stops_per_route"]},
    "ev_range": {"strong": ["energy_planning"], "moderate": ["dynamic_routing"],
                 "harmful": ["more_stops_per_route", "extended_shifts"]},
    "damaged_goods": {"strong": ["load_securing"], "moderate": ["wave_loading"],
                      "harmful": ["more_stops_per_route", "extended_shifts"]},
    "pod_disputes": {"strong": ["photo_pod"], "moderate": ["safe_place_lockers"],
                     "harmful": ["more_stops_per_route"]},
    "access_restrictions": {"strong": ["access_database"], "moderate": ["micro_hubs", "route_familiarity"],
                            "harmful": ["more_stops_per_route"]},
    "customer_comms": {"strong": ["predictive_eta", "control_tower"], "moderate": ["customer_rebooking"],
                       "harmful": ["extended_shifts"]},
    "unbalanced_routes": {"strong": ["workload_balancing", "dynamic_routing"], "moderate": ["service_time_model"],
                          "harmful": ["extended_shifts"]},
    "missed_collections": {"strong": ["collection_integration"], "moderate": ["dynamic_routing", "control_tower"],
                           "harmful": ["more_stops_per_route"]},
}

INCIDENT_KEYS = list(INCIDENTS.keys())
FIX_KEYS = list(FIXES.keys())
INCIDENT_NAMES = [INCIDENTS[k]["name"] for k in INCIDENT_KEYS]
FIX_NAMES = [FIXES[k]["name"] for k in FIX_KEYS]
INCIDENT_BY_NAME = {INCIDENTS[k]["name"]: k for k in INCIDENT_KEYS}
FIX_BY_NAME = {FIXES[k]["name"]: k for k in FIX_KEYS}
INSTINCTIVE_FIXES = ["extended_shifts", "more_stops_per_route"]


def efficacy_tier(incident_key, fix_key):
    tiers = EFFICACY_MAP[incident_key]
    for tier in ("strong", "moderate", "harmful"):
        if fix_key in tiers[tier]:
            return tier
    return "neutral"


def efficacy_value(incident_key, fix_key):
    return EFFICACY_TIERS[efficacy_tier(incident_key, fix_key)]


def efficacy_matrix():
    return [[efficacy_value(i, j) for j in FIX_KEYS] for i in INCIDENT_KEYS]
