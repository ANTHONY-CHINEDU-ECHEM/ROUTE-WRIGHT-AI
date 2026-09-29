"""Column documentation for the Routewright AI delivery incident dataset."""

import html

COLUMN_DOCS = {
    "incident_id": ("identifier", "Unique incident identifier used for citations, INC followed by six digits."),
    "operator_name": ("text", "Operator codename and trading name."),
    "service_type": ("category", "Kind of last mile operation, ten values from Parcel Courier to Builders Merchant."),
    "area_type": ("category", "Dense Urban, Urban, Suburban, Rural or Mixed Region."),
    "region": ("category", "Operating region."),
    "operator_size_band": ("category", "Local, Regional, National or Enterprise."),
    "fleet_type": ("category", "Diesel Vans, Electric Vans, Mixed Van Fleet, Rigid Trucks or Cargo Bikes and Vans."),
    "planning_tool": ("category", "Manual Planning, Spreadsheet Planning, Static Route Software or Dynamic Route Optimisation."),
    "visibility_level": ("category", "Basic Reporting, Daily KPI Dashboards or Real Time Control Tower."),
    "delivery_promise": ("category", "Customer promise such as One Hour Slot or Next Day Untimed."),
    "season": ("category", "Trading period when the incident happened."),
    "daily_drops": ("integer", "Average deliveries per day."),
    "vehicles_in_fleet": ("integer", "Vehicles in daily operation."),
    "drivers_employed": ("integer", "Drivers employed or contracted."),
    "avg_stops_per_route": ("integer", "Average stops planned per route."),
    "avg_route_km": ("float", "Average route length in kilometres."),
    "avg_consignment_weight_kg": ("float", "Average consignment weight in kilograms."),
    "time_window_minutes": ("integer", "Length of the promised delivery window in minutes."),
    "agency_driver_share_pct": ("float", "Share of routes driven by agency drivers."),
    "depot_count": ("integer", "Depots serving the operation."),
    "electric_share_pct": ("float", "Share of the fleet that is electric."),
    "detection_date": ("date", "Date the incident was detected, YYYY/MM/DD."),
    "resolution_date": ("date", "Date the fix was live, YYYY/MM/DD."),
    "detection_channel": ("category", "How the incident was noticed."),
    "detection_lag_hours": ("float", "Hours between onset and detection."),
    "incident_type": ("category", "The delivery performance incident the case is about, fifteen values."),
    "severity": ("category", "Low, Medium, High or Critical."),
    "affected_shift": ("category", "Shift most affected."),
    "root_cause": ("category", "Underlying cause identified in the review."),
    "baseline_on_time_rate_pct": ("float", "Normal on time delivery rate within the promise."),
    "incident_on_time_rate_pct": ("float", "On time delivery rate during the incident."),
    "on_time_retention_ratio": ("float", "Incident on time rate divided by baseline. Below 1 means a drop."),
    "first_attempt_success_pct": ("float", "Share of deliveries completed at the first attempt during the incident."),
    "failed_deliveries_per_day": ("integer", "Late or failed deliveries per day during the incident."),
    "route_overrun_minutes": ("integer", "Average minutes routes ran beyond plan during the incident."),
    "complaints_per_1000_drops": ("float", "Customer complaints and contacts per thousand deliveries."),
    "baseline_cost_per_drop_gbp": ("float", "Normal delivery cost per drop in pounds sterling."),
    "incident_cost_per_drop_gbp": ("float", "Delivery cost per drop during the incident."),
    "daily_delivery_cost_gbp": ("float", "Normal total delivery cost per day."),
    "incident_cost_gbp": ("integer", "Extra cost caused by the incident while it lasted."),
    "fix_strategy": ("category", "The fix the team applied, twenty two values."),
    "team_owner": ("category", "Team that owned the fix."),
    "time_to_fix_days": ("float", "Days from detection until the fix was live."),
    "fix_cost_gbp": ("integer", "Cost of the fix including labour, overtime and extra capacity."),
    "recovery_status": ("category", "Full Recovery, Partial Recovery or Not Recovered."),
    "post_fix_on_time_rate_pct": ("float", "On time delivery rate after the fix settled."),
    "on_time_recovery_ratio": ("float", "Post fix on time rate divided by baseline."),
    "post_fix_cost_per_drop_gbp": ("float", "Cost per drop after the fix settled."),
    "cost_avoided_gbp_30d": ("integer", "Cost avoided over 30 days compared with incident running costs."),
    "days_to_recover": ("integer", "Days until performance stabilised, 90 means no recovery within 90 days."),
    "return_on_fix": ("float", "Cost avoided over 30 days divided by fix cost."),
    "co2_kg_per_drop": ("float", "Estimated tailpipe CO2 per delivery after the fix, in kilograms."),
    "customer_satisfaction": ("integer", "Customer satisfaction during the period, 1 to 10."),
    "repeat_incident_90d": ("category", "Yes if the same incident recurred within 90 days."),
    "incident_summary": ("text", "Narrative description of the incident with its symptoms and context."),
    "resolution_narrative": ("text", "Narrative of the fix and its measured result."),
    "lessons_learned": ("text", "Retrospective lesson for future incidents."),
    "tags": ("text", "Semicolon separated keywords for filtering and search."),
}


def data_dictionary_markdown(frame):
    rows = []
    for col in frame.columns:
        kind, desc = COLUMN_DOCS.get(col, ("unknown", ""))
        example = str(frame[col].iloc[0])
        if len(example) > 70:
            example = example[:67] + "..."
        rows.append("<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            col, kind, html.escape(desc), html.escape(example)))
    header = ("# Data dictionary\n\n"
              "The dataset holds {:,} last mile delivery incidents and {} columns. Every narrative is composed "
              "from the numbers in its own row, so text and structured fields never disagree.\n\n").format(
        len(frame), len(frame.columns))
    return header + ("<table>\n<tr><th>Column</th><th>Type</th><th>Description</th><th>Example</th></tr>\n"
                     + "\n".join(rows) + "\n</table>\n")
