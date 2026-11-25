# Conceptual Design Phase

You will find a visual diagram of this process in this folder (which I honestly think is better; it also took me like an hour to make on google slides).

## Reminder of Requirements

- The aircraft shall remain airborne for a minimum of 45 minutes with no payload (raised from 30 min, Nov 2026).
- The aircraft shall take off within 75 m on the paved strip and within 80 m on soft grass, and land (with wheel brakes) within the same distances (field survey, Nov 2026; was 250 ft).
- The aircraft shall support full autonomous and remote-controlled operation.

## Notional Aircraft Concept

Fixed wing, dual wing-mounted prop aircraft with fixed landing gear (think DHC-6 Twin Otter)

## Critical Inputs

### Propulsion (Critical Input #1)

The propulsion model shows how thrust changes with altitude and speed as well as how much energy it takes to produce the desired amount of thrust.

#### Variables
1) Available Shaft Power vs. Airspeed (normally thrust lapse, but we aren't using fuel)
2) Energy Draw vs. Time (normally fuel flow, but, again, fuel isn't being used)

### Aerodynamics (Critical Input #2)

We use a drag polar and lift curve *(google what they are; I'm too lazy to explain)* to determine the aircraft's aerodynamic behavior in a compact way.

#### Variables
1) Drag Polar
2) Lift Curve

### Weight

Historical weight regressions (like Raymer's) are fit to full-size aircraft, so they don't work well for a 1.5 m 3D-printed RC twin. Instead, we build the weight up from the bottom: datasheet masses for every part, structure from material density × area, and the wing spar and landing gear sized from their loads. A 10% growth allowance covers glue, paint, and connectors.

#### Variables
1) Component datasheet masses (`data/*.csv`)
2) Material densities × areas, plus load-sized spar and landing gear

## Energy-based Constraint Analysis

In short, we graph everything above (thrust-to-weight ratio, wing loading, etc) to find the optimal point for our aircraft. The thing is, we get ratios from doing this, not actual sizes, so we need to perform mission-analysis later.

At the end of this, we will get a final number for motor size, wing area, and weight.

## How the Code Works

Run `.venv/bin/python conceptual-design/conceptual.py` from `design/`. It searches every combination of wing airfoil, wing construction (`wing_constructions` in `parameters.py`; currently printed LW-PLA only), motor, prop, and battery in the catalogs. For each one, it finds the aircraft with the lowest **mass + 30 g per m/s of stall speed** that meets every requirement (mass alone always shrinks the wing to the fastest stall speed any constraint allows; the 30 g/(m/s) weight is a team decision in `parameters.py`, and the stall-speed trade study in the report shows what other weights would give). It writes:

- `outputs/design_point.json`: the hand-off to preliminary design (geometry, parts, aero, masses, margins)
- `outputs/constraint_diagram.png`: power loading vs. wing loading with every requirement line
- `outputs/dimensions.txt` + `outputs/airfoils/*.dat`: every dimension for CAD (mm/deg) and the exact airfoil sections
- `outputs/performance.txt`: what the aircraft does - speeds (stall, V_mc, loiter, V_y, glide, max), endurance, climb/glide, field lengths, roll rate and turns, stability and handling, one-motor-out, power system, watch-outs, and what the model does not compute. Regenerate with `.venv/bin/python conceptual-design/print_performance.py`
- `outputs/parts_list.txt` / `.csv`: bill of materials - every purchased part, stock size (spar, gear strap, nose wire, wire), laminate/foam and printed-part masses, with sources; the masses add up to the takeoff mass. Regenerate with `.venv/bin/python conceptual-design/print_parts.py`

### The sizing loop (one design)

```
 guess W_TO ─► geometry ─► aerodynamics ─► powertrain ─► performance ─► mission BMF ─► weights ─► new W_TO
     ▲                                                                                             │
     └──────────────────────────── repeat until W_TO changes < 0.1 g ◄─────────────────────────────┘
```

Inside the same loop, and converged together with the mass: the fin (coupled engine-out trim, `lateral.py`), the stabilizer (scissor plot, `controls.py`), the CG band (as-built CG scatter, `weights.cg_budget`), the wing station along the fuselage (puts the nominal CG mid-band), and the trimmed stall speed (tail download at the forward CG).

| File | Job |
|---|---|
| `requirements.py` | README requirements (45 min; takeoff/braked landing ≤ 75 m pavement, ≤ 80 m soft grass; ≤1.5 m span) |
| `parameters.py` | **every** input that isn't calculated, each tagged with its source (REQUIREMENT / STANDARD / RAYMER / DATASHEET / DESIGN CHOICE / SITE / ESTIMATE) |
| `atmosphere.py` | ISA density and viscosity (hot day by default) |
| `catalog.py` | loads `data/`: airfoil polars, APC prop tables, motors, batteries, ESCs, avionics, materials |
| `geometry.py` | wing/tail/fuselage/nacelle/gear layout; fuselage sized by the battery + avionics it carries |
| `aerodynamics.py` | **critical input #2**: drag polar and lift curve. The wing uses a Prandtl lifting-line solution (real chord, washout, root-to-tip airfoil loft, deflected flaps), low-Re airfoil polars for profile drag, and the Raymer component build-up for everything else |
| `controls.py` | control surfaces sized from requirements: ailerons (roll helix pb/2V >= 0.07), flaps or flaperons (landing CLmax), elevator (landing trim + takeoff, forward CG), neutral point (incl. fuselage), tail incidence, elevator trim drag, crow ailerons. The **stabilizer comes from a scissor plot**: the CG band (from `weights.cg_budget`) hangs from the aft limit (static margin ≥ 5%) and the elevator must control its forward end. Uses NeuralFoil on the actual deflected airfoils, trusted only where its confidence is high |
| `lateral.py` | **coupled** roll/yaw trim: side force, rolling and yawing moment solved together for sideslip, bank, aileron and rudder. Engine-out at 1.2 V_s (zero-sideslip bank ≤ 5°, stopped dead prop, live-prop slipstream) sizes the fin (never below the Raymer tail-volume value, kept for the unmodelled Dutch-roll/spiral damping); a 0.2 V_SO crosswind (14 CFR 23.233) checks rudder and ailerons |
| `propulsion.py` | **critical input #1**: the BMF equations from the Electric Propulsion Fundamentals sheet, plus a battery → ESC → motor → prop model (power available vs. airspeed, energy draw) |
| `weights.py` | bottom-up weights, carbon spar sized for maneuver + gust loads (14 CFR 23) |
| `mission.py` | takeoff roll, climb, loiter, glide, and the segment-by-segment BMF |
| `constraints.py` | constraint diagram |
| `sizing.py` | the loop above + the search: screen → SLSQP optimizer for each control-surface configuration → snap to real batteries → twist (tip airfoil + washout) refinement → twist trade study |
| `conceptual.py` | runs everything and prints the report |
| `data/build_airfoil_polars.py` | regenerates `data/airfoils/` with NeuralFoil (only if you change airfoils) |

### What is still an estimate

The report ends with **INPUTS TO MEASURE / VERIFY**. Replacing those numbers with measurements is the most useful thing the team can do next. The big ones are:

1. **Avionics power.** The Pi 5 + FPV draw about 12 W, which is about 40% of the battery energy. Measure it on the bench with the real software running.
2. **Grass rolling and braking friction.** Pull the plane, or a weighted cart, across the soft-grass strip with a luggage scale: μ = pull force / weight. Do it with **two wheel sizes** (e.g. 2 in and 3 in): that gives both `mu_grass_ref` and the soil exponent `bekker_n` the wheel model uses. Repeat with the brakes on for `mu_brake_grass`.
3. **Printed part densities.** Weigh a printed LW-PLA test panel and a foam/glass wing section.
4. **ESC efficiency and battery resistance.** Use a thrust stand and the charger's IR readout.

Change a value in `parameters.py` or `data/*.csv` and re-run; the whole design updates.

- `conceptual.py`: full search (~10 min on all cores). Use it when parts, catalogs, or requirements change.
- `conceptual.py --evaluate`: re-checks the saved design point in seconds without searching. Use it after replacing an estimate with a measurement, to see whether the current design still meets every requirement.

The full report is saved to `outputs/report.txt`.

**For the CAD team:** `outputs/dimensions.txt` lists every dimension in mm and degrees (datum at the nose tip): wing, spar, control-surface hinge lines, tails, fuselage stations, nacelles, props, landing gear + wheels/brakes, bay contents, the CG band to balance to, and a control-throw table. The exact airfoil sections are in `outputs/airfoils/*.dat` (Selig format). Regenerate with `.venv/bin/python conceptual-design/print_dimensions.py`.

**Every final parameter in one place:** `outputs/design_parameters.txt`. It lists the design variables, selected parts with datasheet values, all geometry and layout positions, aerodynamics, in-flight thrust, mission BMF, mass breakdown, constraint margins, and every input ratio/coefficient with its source. Regenerate it any time with `.venv/bin/python conceptual-design/print_design.py`.
