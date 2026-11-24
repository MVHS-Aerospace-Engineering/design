"""
Mission requirements (README) and the notional concept. These are NOT
engineering decisions - change them only if the requirements change.
"""
from parameters import P

P.define("endurance_min", 30.0, "min", "REQUIREMENT",
         "Remain airborne >= 30 min with no payload (flown entirely as loiter; "
         "takeoff/climb/descent time NOT credited -> conservative)")
P.define("takeoff_roll_max_m", 80.0, "m", "REQUIREMENT",
         "SCENARIO: takeoff ground roll < 80 m on grass/turf (team: ground roll only)")
P.define("landing_roll_max_m", 100.0, "m", "REQUIREMENT",
         "SCENARIO: landing ground roll < 100 m on grass/turf, no brakes (team: ground roll only)")
P.define("span_max_m", 1.5, "m", "REQUIREMENT", "Wingspan <= 1.5 m (team limit)")
P.define("payload_kg", 0.0, "kg", "REQUIREMENT", "No payload for the endurance mission")
P.define("n_motors", 2, "-", "REQUIREMENT",
         "Notional concept: twin wing-mounted props, fixed tricycle gear, high wing")
