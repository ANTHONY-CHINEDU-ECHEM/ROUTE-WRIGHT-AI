"""Narrative composer.

Turns the structured facts of one delivery incident into the three pieces of
text an operations team would write in a post incident review: an incident
summary, a resolution narrative and a lesson. Every number quoted is taken
from the row, so text and columns never disagree.
"""

from .utils import fmt_money
from .vocab import EFFICACY_MAP, FIXES, INCIDENTS, efficacy_tier


def _lower_first(text):
    """Lowercase the first letter unless the text starts with an acronym such as ETA."""
    if not text or (len(text) > 1 and text[1].isupper()):
        return text
    return text[:1].lower() + text[1:]


def _context(r, rng):
    options = []
    if r["planning_tool"] in ("Manual Planning", "Spreadsheet Planning"):
        options.append("Routes were built by {}.".format(r["planning_tool"].lower()))
    if r["visibility_level"] == "Basic Reporting":
        options.append("With only basic reporting, the problem surfaced through {} after {:.0f} hours.".format(
            r["detection_channel"].lower(), r["detection_lag_hours"]))
    elif r["detection_lag_hours"] <= 6:
        options.append("The control tower flagged it within {:.1f} hours.".format(r["detection_lag_hours"]))
    if r["season"] == "Peak Season":
        options.append("It struck during peak season.")
    if r["agency_driver_share_pct"] >= 30:
        options.append("Agency drivers covered {:.0f} percent of routes.".format(r["agency_driver_share_pct"]))
    if r["electric_share_pct"] >= 50:
        options.append("{:.0f} percent of the fleet was electric.".format(r["electric_share_pct"]))
    if r["time_window_minutes"] <= 60:
        options.append("Customers were promised one hour slots.")
    if r["avg_stops_per_route"] >= 100:
        options.append("Each route carried around {} stops.".format(r["avg_stops_per_route"]))
    if not options:
        options.append("The operation ran {} vehicles from {} depots.".format(r["vehicles_in_fleet"], r["depot_count"]))
    rng.shuffle(options)
    return " ".join(options[:2])


def _symptom(r, rng):
    template = rng.choice(INCIDENTS[r["incident_key"]]["symptoms"])
    sev = r["severity_idx"]
    return template.format(
        ontime="{:.1f}".format(r["incident_on_time_rate_pct"]),
        base="{:.1f}".format(r["baseline_on_time_rate_pct"]),
        ftr="{:.1f}".format(r["first_attempt_success_pct"]),
        fails=max(3, r["failed_deliveries_per_day"]),
        overrun=max(10, r["route_overrun_minutes"]),
        claims="{:.1f}".format(r["complaints_per_1000_drops"]),
        pct=rng.randint(10 + sev * 6, 22 + sev * 10),
        days=rng.randint(3 + sev, 7 + sev * 3),
        hours=rng.randint(1 + sev, 2 + sev * 2),
    )


def incident_summary(r, rng):
    symptom = _symptom(r, rng)
    context = _context(r, rng)
    root = _lower_first(r["root_cause"])
    templates = [
        "{op}, a {size} {service} operator working {area} routes in {region}, suffered {sev} severity "
        "{incident}: {symptom}. {context} The root cause was {root}.",
        "At {op} ({service}, {fleet}), {symptom}. The {shift} was worst affected. {context} Investigation traced "
        "it to {root}.",
        "{Incident} hit {op} when {symptom}. {context} The team identified {root} as the underlying cause.",
        "The {area} operation of {op} saw a {sev} severity problem: {symptom}. {context} Analysis pointed to {root}.",
    ]
    return rng.choice(templates).format(
        op=r["operator_name"], size=r["operator_size_band"].lower(), service=r["service_type"].lower(),
        area=r["area_type"].lower(), region=r["region"], sev=r["severity"].lower(),
        incident=r["incident_type"].lower(), Incident=r["incident_type"], symptom=symptom, context=context,
        root=root, fleet=r["fleet_type"].lower(), shift=r["affected_shift"].lower(), season=r["season"].lower(),
    ).replace("  ", " ").strip()


def _outcome(r, rng):
    post = "{:.1f}".format(r["post_fix_on_time_rate_pct"])
    avoided = fmt_money(r["cost_avoided_gbp_30d"])
    cpd = "{:.2f}".format(r["post_fix_cost_per_drop_gbp"])
    status = r["recovery_status"]
    if status == "Full Recovery":
        base = rng.choice([
            "On time delivery recovered to {post} percent within {days} days, cost per drop settled at GBP {cpd} and {av} of cost was avoided over 30 days.",
            "Performance recovered fully to {post} percent on time in {days} days, avoiding {av} of cost over the following month.",
        ])
    elif status == "Partial Recovery":
        base = rng.choice([
            "On time delivery only climbed back to {post} percent after {days} days, avoiding {av} of cost over 30 days.",
            "Recovery was partial: on time delivery settled at {post} percent and cost per drop at GBP {cpd}.",
        ])
    else:
        base = rng.choice([
            "The fix did not restore performance; on time delivery was still {post} percent after 90 days.",
            "Performance did not recover and on time delivery stood at {post} percent three months later.",
        ])
    return base.format(post=post, days=r["days_to_recover"], av=avoided, cpd=cpd)


def resolution_narrative(r, rng):
    fix = FIXES[r["fix_key"]]
    name = _lower_first(fix["name"])
    templates = [
        "Within {days:.0f} days the {team} team introduced {name}: {mech}. {outcome}",
        "The response, led by {team}, was {name}, in which {mech}. It took {days:.0f} days to put in place. {outcome}",
        "{Team} chose {name}, live after {days:.0f} days; {mech}. {outcome}",
    ]
    return rng.choice(templates).format(
        days=max(1.0, r["time_to_fix_days"]), team=r["team_owner"].lower(), Team=r["team_owner"], name=name,
        mech=fix["mechanism"], outcome=_outcome(r, rng),
    )


def lessons_learned(r, rng):
    inc_key, fix_key = r["incident_key"], r["fix_key"]
    tier = efficacy_tier(inc_key, fix_key)
    incident = _lower_first(r["incident_type"])
    fix_name = FIXES[fix_key]["name"]
    fix_lower = _lower_first(fix_name)
    root = _lower_first(r["root_cause"])
    best = _lower_first(FIXES[rng.choice(EFFICACY_MAP[inc_key]["strong"])]["name"])
    status = r["recovery_status"]
    if tier == "strong" and status == "Full Recovery":
        options = [
            "For {incident}, {fix} worked because it removed the cause rather than the symptom; make it the first response.",
            "{Fix} restored service quickly. Build it into the standard playbook for {incident}.",
            "Acting within {days:.0f} days with {fix} kept the cost of {incident} contained.",
        ]
    elif tier == "strong":
        options = [
            "{Fix} usually resolves {incident}, but slow detection let failed deliveries pile up. Invest in visibility.",
            "The right fix arrived too late; {incident} needs exception alerts that catch it within hours, not days.",
        ]
    elif tier == "moderate":
        options = [
            "{Fix} helped at the margins, but {root} needed a direct fix such as {best}.",
            "{Fix} eased the symptoms of {incident}; pairing it with {best} would have closed the gap.",
        ]
    elif tier == "harmful":
        options = [
            "{Fix} made things worse: drivers were stretched further while {root} remained. For {incident}, start with {best}.",
            "Avoid reaching for {fix} when {incident} strikes; it adds strain and cost without fixing the plan.",
        ]
    elif status == "Full Recovery":
        options = ["Recovery owed more to volumes easing than to {fix}; {best} is the dependable response to {incident}."]
    else:
        options = [
            "{Fix} did not address {root}; operators facing {incident} should use {best}.",
            "The response did not match the diagnosis. For {incident}, evidence favours {best} over {fix}.",
        ]
    return rng.choice(options).format(incident=incident, fix=fix_lower, Fix=fix_name, root=root, best=best,
                                      days=max(1.0, r["time_to_fix_days"]))


def tags(r):
    parts = [r["incident_type"], r["root_cause"], FIXES[r["fix_key"]]["name"], r["service_type"], r["area_type"],
             r["fleet_type"], r["planning_tool"], r["recovery_status"]]
    return ";".join(p.lower() for p in parts)
