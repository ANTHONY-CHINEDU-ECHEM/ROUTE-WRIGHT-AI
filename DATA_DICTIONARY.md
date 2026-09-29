# Data dictionary

The dataset holds 16,000 last mile delivery incidents and 58 columns. Every narrative is composed from the numbers in its own row, so text and structured fields never disagree.

<table>
<tr><th>Column</th><th>Type</th><th>Description</th><th>Example</th></tr>
<tr><td><code>incident_id</code></td><td>identifier</td><td>Unique incident identifier used for citations, INC followed by six digits.</td><td>INC000001</td></tr>
<tr><td><code>operator_name</code></td><td>text</td><td>Operator codename and trading name.</td><td>Driftwood Transport</td></tr>
<tr><td><code>service_type</code></td><td>category</td><td>Kind of last mile operation, ten values from Parcel Courier to Builders Merchant.</td><td>Retail Store Replenishment</td></tr>
<tr><td><code>area_type</code></td><td>category</td><td>Dense Urban, Urban, Suburban, Rural or Mixed Region.</td><td>Suburban</td></tr>
<tr><td><code>region</code></td><td>category</td><td>Operating region.</td><td>Wales and South West</td></tr>
<tr><td><code>operator_size_band</code></td><td>category</td><td>Local, Regional, National or Enterprise.</td><td>Local</td></tr>
<tr><td><code>fleet_type</code></td><td>category</td><td>Diesel Vans, Electric Vans, Mixed Van Fleet, Rigid Trucks or Cargo Bikes and Vans.</td><td>Electric Vans</td></tr>
<tr><td><code>planning_tool</code></td><td>category</td><td>Manual Planning, Spreadsheet Planning, Static Route Software or Dynamic Route Optimisation.</td><td>Static Route Software</td></tr>
<tr><td><code>visibility_level</code></td><td>category</td><td>Basic Reporting, Daily KPI Dashboards or Real Time Control Tower.</td><td>Basic Reporting</td></tr>
<tr><td><code>delivery_promise</code></td><td>category</td><td>Customer promise such as One Hour Slot or Next Day Untimed.</td><td>Two Hour Slot</td></tr>
<tr><td><code>season</code></td><td>category</td><td>Trading period when the incident happened.</td><td>Standard Trading</td></tr>
<tr><td><code>daily_drops</code></td><td>integer</td><td>Average deliveries per day.</td><td>29</td></tr>
<tr><td><code>vehicles_in_fleet</code></td><td>integer</td><td>Vehicles in daily operation.</td><td>3</td></tr>
<tr><td><code>drivers_employed</code></td><td>integer</td><td>Drivers employed or contracted.</td><td>4</td></tr>
<tr><td><code>avg_stops_per_route</code></td><td>integer</td><td>Average stops planned per route.</td><td>12</td></tr>
<tr><td><code>avg_route_km</code></td><td>float</td><td>Average route length in kilometres.</td><td>123.5</td></tr>
<tr><td><code>avg_consignment_weight_kg</code></td><td>float</td><td>Average consignment weight in kilograms.</td><td>282.4</td></tr>
<tr><td><code>time_window_minutes</code></td><td>integer</td><td>Length of the promised delivery window in minutes.</td><td>120</td></tr>
<tr><td><code>agency_driver_share_pct</code></td><td>float</td><td>Share of routes driven by agency drivers.</td><td>13.7</td></tr>
<tr><td><code>depot_count</code></td><td>integer</td><td>Depots serving the operation.</td><td>1</td></tr>
<tr><td><code>electric_share_pct</code></td><td>float</td><td>Share of the fleet that is electric.</td><td>92.2</td></tr>
<tr><td><code>detection_date</code></td><td>date</td><td>Date the incident was detected, YYYY/MM/DD.</td><td>2024/10/25</td></tr>
<tr><td><code>resolution_date</code></td><td>date</td><td>Date the fix was live, YYYY/MM/DD.</td><td>2024/11/11</td></tr>
<tr><td><code>detection_channel</code></td><td>category</td><td>How the incident was noticed.</td><td>Weekly KPI Review</td></tr>
<tr><td><code>detection_lag_hours</code></td><td>float</td><td>Hours between onset and detection.</td><td>115.7</td></tr>
<tr><td><code>incident_type</code></td><td>category</td><td>The delivery performance incident the case is about, fifteen values.</td><td>Customer notification failures</td></tr>
<tr><td><code>severity</code></td><td>category</td><td>Low, Medium, High or Critical.</td><td>Medium</td></tr>
<tr><td><code>affected_shift</code></td><td>category</td><td>Shift most affected.</td><td>Early Shift</td></tr>
<tr><td><code>root_cause</code></td><td>category</td><td>Underlying cause identified in the review.</td><td>Contact details missing from orders</td></tr>
<tr><td><code>baseline_on_time_rate_pct</code></td><td>float</td><td>Normal on time delivery rate within the promise.</td><td>90.6</td></tr>
<tr><td><code>incident_on_time_rate_pct</code></td><td>float</td><td>On time delivery rate during the incident.</td><td>82.4</td></tr>
<tr><td><code>on_time_retention_ratio</code></td><td>float</td><td>Incident on time rate divided by baseline. Below 1 means a drop.</td><td>0.909</td></tr>
<tr><td><code>first_attempt_success_pct</code></td><td>float</td><td>Share of deliveries completed at the first attempt during the incident.</td><td>93.1</td></tr>
<tr><td><code>failed_deliveries_per_day</code></td><td>integer</td><td>Late or failed deliveries per day during the incident.</td><td>5</td></tr>
<tr><td><code>route_overrun_minutes</code></td><td>integer</td><td>Average minutes routes ran beyond plan during the incident.</td><td>15</td></tr>
<tr><td><code>complaints_per_1000_drops</code></td><td>float</td><td>Customer complaints and contacts per thousand deliveries.</td><td>10.4</td></tr>
<tr><td><code>baseline_cost_per_drop_gbp</code></td><td>float</td><td>Normal delivery cost per drop in pounds sterling.</td><td>34.14</td></tr>
<tr><td><code>incident_cost_per_drop_gbp</code></td><td>float</td><td>Delivery cost per drop during the incident.</td><td>43.76</td></tr>
<tr><td><code>daily_delivery_cost_gbp</code></td><td>float</td><td>Normal total delivery cost per day.</td><td>990.06</td></tr>
<tr><td><code>incident_cost_gbp</code></td><td>integer</td><td>Extra cost caused by the incident while it lasted.</td><td>5960</td></tr>
<tr><td><code>fix_strategy</code></td><td>category</td><td>The fix the team applied, twenty two values.</td><td>Self service rebooking before dispatch</td></tr>
<tr><td><code>team_owner</code></td><td>category</td><td>Team that owned the fix.</td><td>Customer Service</td></tr>
<tr><td><code>time_to_fix_days</code></td><td>float</td><td>Days from detection until the fix was live.</td><td>11.7</td></tr>
<tr><td><code>fix_cost_gbp</code></td><td>integer</td><td>Cost of the fix including labour, overtime and extra capacity.</td><td>2668</td></tr>
<tr><td><code>recovery_status</code></td><td>category</td><td>Full Recovery, Partial Recovery or Not Recovered.</td><td>Full Recovery</td></tr>
<tr><td><code>post_fix_on_time_rate_pct</code></td><td>float</td><td>On time delivery rate after the fix settled.</td><td>91.8</td></tr>
<tr><td><code>on_time_recovery_ratio</code></td><td>float</td><td>Post fix on time rate divided by baseline.</td><td>1.013</td></tr>
<tr><td><code>post_fix_cost_per_drop_gbp</code></td><td>float</td><td>Cost per drop after the fix settled.</td><td>31.94</td></tr>
<tr><td><code>cost_avoided_gbp_30d</code></td><td>integer</td><td>Cost avoided over 30 days compared with incident running costs.</td><td>10283</td></tr>
<tr><td><code>days_to_recover</code></td><td>integer</td><td>Days until performance stabilised, 90 means no recovery within 90 days.</td><td>14</td></tr>
<tr><td><code>return_on_fix</code></td><td>float</td><td>Cost avoided over 30 days divided by fix cost.</td><td>3.85</td></tr>
<tr><td><code>co2_kg_per_drop</code></td><td>float</td><td>Estimated tailpipe CO2 per delivery after the fix, in kilograms.</td><td>0.515</td></tr>
<tr><td><code>customer_satisfaction</code></td><td>integer</td><td>Customer satisfaction during the period, 1 to 10.</td><td>8</td></tr>
<tr><td><code>repeat_incident_90d</code></td><td>category</td><td>Yes if the same incident recurred within 90 days.</td><td>No</td></tr>
<tr><td><code>incident_summary</code></td><td>text</td><td>Narrative description of the incident with its symptoms and context.</td><td>Customer notification failures hit Driftwood Transport when where i...</td></tr>
<tr><td><code>resolution_narrative</code></td><td>text</td><td>Narrative of the fix and its measured result.</td><td>Within 12 days the customer service team introduced self service re...</td></tr>
<tr><td><code>lessons_learned</code></td><td>text</td><td>Retrospective lesson for future incidents.</td><td>Self service rebooking before dispatch helped at the margins, but c...</td></tr>
<tr><td><code>tags</code></td><td>text</td><td>Semicolon separated keywords for filtering and search.</td><td>customer notification failures;contact details missing from orders;...</td></tr>
</table>
