"""
Performance and handling summary of the conceptual design point - what the
aircraft does, in the terms a pilot / autopilot tuner / ops planner uses.

    .venv/bin/python conceptual-design/print_performance.py            (prints)
    .venv/bin/python conceptual-design/print_performance.py -o out.txt  (also saves)

Reads outputs/design_point.json; conceptual.py also writes
outputs/performance.txt. Everything is the sizing model's prediction at the
design takeoff mass (hot day, ISA+15 K, field elevation per parameters.py);
a few simple flight-mechanics relations are applied here and named where used.
What the model does NOT compute is listed at the end.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "outputs", "design_point.json")
G0 = 9.80665


def render(j):
    P = {p["name"]: p["value"] for p in j.get("inputs", [])}
    a, pf, bm = j["aero"], j["performance"], j["mission_bmf"]
    ct, st = j["controls"], j["controls"]["stability"]
    lat, ld = j.get("lateral") or {}, j.get("landing") or {}
    loads = j.get("loads") or {}
    g = j["geometry"]
    m = j["takeoff_mass_kg"]
    W = m * G0
    b = g["wing"]["b"]
    Vs, Vsl = a["V_stall_ms"], pf["stall_speed_landing_ms"]
    lo, cl, gl = a["loiter"], a["climb"], a["best_glide"]
    eo, xw, dv = lat.get("engine_out") or {}, lat.get("crosswind") or {}, lat.get("derivatives") or {}
    ap = ld.get("approach") or {}
    ail = ct["ailerons"]
    L = []
    w = L.append

    def head(t):
        w("")
        w("=" * 86)
        w(f" {t}")
        w("=" * 86)

    def kv(k, v, unit="", note=""):
        if isinstance(v, float):
            v = f"{v:.3g}" if abs(v) < 0.1 else f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}"
        w(f"  {k:46s} {str(v):>10s} {unit:7s} {note}")

    def kmh(v):
        return f"({v * 3.6:.0f} km/h)"

    w("AIRCRAFT PERFORMANCE AND HANDLING")
    w(f"Source: conceptual-design/outputs/design_point.json   (generated {j.get('generated', '?')})")
    s = j["selection"]
    w(f"Takeoff mass {m:.3f} kg, span {b:.2f} m, wing area {g['wing']['S']:.4f} m^2, "
      f"wing loading {W / g['wing']['S'] / G0 * 10:.0f} g/dm^2")
    w(f"{s['wing_airfoil']} wing ({s['control_config']}), 2 x {s['motor']} + {s['prop']}, {s['battery']}")
    w(f"Conditions: ISA{P.get('isa_temp_offset_K', 0):+.0f} K at {P.get('field_elevation_m', 0):.0f} m "
      f"(hot day - performance is a little better on a standard day). Model predictions - verify in flight.")

    # ---------------------------------------------------------------- speeds
    head("1. SPEEDS")
    kv("stall, clean (trimmed at forward CG)", Vs, "m/s", kmh(Vs))
    kv("stall, landing configuration", Vsl, "m/s", kmh(Vsl) + (" flaperons drooped"
                                                                 if ail.get("is_flaperon") else ""))
    for ph in (30, 45, 60):
        kv(f"stall in a level {ph}-deg bank (V_s / sqrt(cos))", Vs / math.sqrt(math.cos(math.radians(ph))), "m/s")
    kv("liftoff (1.1 V_s)", P.get("v_liftoff_ratio", 1.1) * Vs, "m/s")
    kv("approach (1.3 V_s landing)", ap.get("V", 1.3 * Vsl), "m/s")
    kv("touchdown (1.15 V_s landing)", ld.get("V_touchdown_ms", 1.15 * Vsl), "m/s")
    kv("minimum control speed, one motor out (V_mc)", j["engine_out"].get("V_mc", 1.2 * Vs), "m/s",
       "= 1.2 V_s (14 CFR 23.149)")
    kv("best endurance (loiter)", lo["V"], "m/s", kmh(lo["V"]))
    kv("best rate of climb (V_y)", cl["V"], "m/s")
    kv("best glide (motors off)", gl["V"], "m/s")
    kv("maximum level speed (at the throttle cap)", a["V_max_ms"], "m/s", kmh(a["V_max_ms"]))
    if loads:
        n = loads["n_limit"]
        kv("maneuvering speed V_A = V_s sqrt(n_limit)", Vs * math.sqrt(n), "m/s",
           "full, abrupt control only below this (14 CFR 23.335)")
        kv("design cruise speed V_C (structure sized here)", loads["V_C_ms"], "m/s")

    # ---------------------------------------------------------------- endurance
    head("2. ENDURANCE, RANGE, CLIMB, GLIDE")
    kv("loiter time available (after takeoff + climb)", bm["loiter_max_min"], "min",
       f"requirement {P.get('endurance_min', 30):.0f} min")
    kv("total airborne time", bm["airborne_max_min"], "min", "takeoff + climb + loiter + glide down")
    rng = lo["V"] * bm["loiter_max_min"] * 60 / 1000
    kv("still-air distance flown at loiter speed", rng, "km", "(best-range speed is a bit faster; not computed)")
    kv("loiter power drawn from the battery", lo["P_chem"], "W", f"incl. avionics; {lo['I_batt']:.2f} A")
    kv("loiter lift-to-drag ratio", lo["LD"], "")
    kv("climb rate (at the throttle cap)", cl["roc"], "m/s", f"{cl['roc'] * 196.85:.0f} ft/min")
    kv("climb gradient (best)", cl["grad_max"] * 100, "%")
    kv(f"time to climb to {P.get('climb_altitude_m', 120):.0f} m", bm["t_climb_s"], "s")
    kv("best glide ratio (motors off)", gl["LD"], ": 1")
    kv("sink rate at best glide", gl["V"] / gl["LD"], "m/s")
    kv(f"glide time from {P.get('climb_altitude_m', 120):.0f} m", gl["t"], "s",
       f"glide distance {P.get('climb_altitude_m', 120) * gl['LD']:.0f} m")

    # ---------------------------------------------------------------- field
    head("3. TAKEOFF AND LANDING")
    kv("takeoff ground roll, soft grass", pf["takeoff_roll_m"], "m",
       f"limit {P.get('takeoff_roll_max_grass_m', 80):.0f} m")
    kv("takeoff ground roll, pavement", pf.get("takeoff_roll_pavement_m"), "m",
       f"limit {P.get('takeoff_roll_max_pavement_m', 75):.0f} m")
    kv("landing roll, soft grass (braked)", pf["landing_roll_m"], "m",
       f"limit {P.get('landing_roll_max_grass_m', 80):.0f} m, incl. {P.get('brake_delay_s', 1):.0f} s free roll")
    kv("landing roll, pavement (braked)", pf.get("landing_roll_pavement_m"), "m",
       f"limit {P.get('landing_roll_max_pavement_m', 75):.0f} m")
    if ld.get("crow"):
        kv("  ... without crow (ailerons up)", ld.get("roll_grass_without_crow_m"), "m")
    kv("takeoff technique", st.get("takeoff_technique", ""), "",
       f"nose-up ground stance {st.get('ground_stance_deg', 0):.1f} deg" if st.get("takeoff_technique") == "stance" else "")
    if ap:
        kv("approach glide angle, motors idle", ap["glide_deg"], "deg", "landing configuration, trimmed")
        kv("go-around (balked landing) climb gradient", ap["balked_gradient"] * 100, "%",
           f"minimum {P.get('balked_landing_gradient_min', 0.033) * 100:.1f}% (14 CFR 23.77)")
    if xw:
        kv("crosswind demonstrated (0.2 V_SO)", xw["V_crosswind"], "m/s",
           f"rudder {abs(xw['rudder_deg']):.0f} of {xw['rudder_usable_deg']:.0f} deg trusted, "
           f"aileron {abs(xw['aileron_frac']):.0%}, bank {xw['phi_deg']:+.1f} deg")
        if xw.get("ratio", 0) > 0:
            kv("  rough crosswind limit (rudder runs out)", xw["V_crosswind"] / xw["ratio"], "m/s",
               "linear scaling of the trim - estimate only")

    # ---------------------------------------------------------------- roll / turn
    head("4. ROLL AND TURNING")
    pb = ail["roll_helix_pb_2V"]
    kv("roll helix pb/2V, full aileron", pb, "",
       f"requirement {ail['requirement']} (Gilruth, NACA 755 - 'satisfactory' roll power)")
    for name, V in (("approach", ap.get("V", 1.3 * Vsl)), ("loiter", lo["V"]), ("max level", a["V_max_ms"])):
        p = pb * 2 * V / b
        kv(f"steady roll rate at {name} speed ({V:.1f} m/s)", math.degrees(p), "deg/s",
           f"0 -> 30 deg bank in ~{30 / math.degrees(p):.1f} s (roll-mode lag ignored)")
    kv("aileron throw", ail["throw_deg"], "deg", "(flaperons: about the drooped position when landing)"
       if ail.get("is_flaperon") else "")
    V = lo["V"]
    for ph in (30, 45):
        n = 1 / math.cos(math.radians(ph))
        R = V ** 2 / (G0 * math.tan(math.radians(ph)))
        margin = V / (Vs * math.sqrt(n))
        kv(f"level turn at loiter speed, {ph}-deg bank", R, "m radius",
           f"{2 * math.pi * R / V:.0f} s per 360, load factor {n:.2f}, {margin:.2f}x its stall speed "
           f"(needs more power than loiter)")
    ph_max = math.degrees(math.acos(min(1.0, (Vs / V) ** 2)))
    kv("steepest level bank at loiter speed before stall", ph_max, "deg",
       "add speed (or power) before banking harder")
    if loads:
        kv("limit load factor (structure)", loads["n_limit"], "g", f"ultimate {loads['n_ultimate']:.2f} g")

    # ---------------------------------------------------------------- stability
    head("5. STABILITY AND HANDLING")
    sm = st.get("static_margin_range", [None, None])
    kv("neutral point", st["neutral_point_frac_mac"] * 100, "% MAC", "incl. the fuselage")
    kv("CG range (balance here)", f"{st['cg_forward_frac_mac'] * 100:.1f}-{st['cg_aft_frac_mac'] * 100:.1f}",
       "% MAC", f"x = {j['layout']['cg']['x_forward_limit'] * 1000:.0f}-{j['layout']['cg']['x_aft_limit'] * 1000:.0f}"
                f" mm from the nose tip")
    kv("static margin (pitch stability)", f"{sm[0] * 100:.1f}-{sm[1] * 100:.1f}", "% MAC",
       "aft CG = least stable, most responsive in pitch")
    kv("elevator: trusted up-throw / authority", ct["elevator"]["throw_deg"], "deg",
       f"{ct['elevator']['authority_ratio']:.2f}x what landing trim + takeoff need at the forward CG")
    if dv:
        kv("weathercock stability Cn_beta", dv["cn_beta"], "/rad",
           "> 0: nose turns into the relative wind (fuselage alone {:+.3f})".format(dv["cn_beta_fus"]))
        kv("effective dihedral Cl_beta", dv["cl_beta"], "/rad",
           "< 0: a sideslip rolls the wing level (high wing + fin; no geometric dihedral)")
        kv("adverse yaw / roll from full aileron", dv["cn_aileron_full"] / dv["cl_aileron_full"], "",
           "Cn/Cl - small; coordinate turns with a little rudder")
    kv("where the stall starts (fraction of semispan)", pf.get("stall_onset_semispan_clean"), "",
       "clean; mid-span, not at the tip - roll control remains at the stall")
    kv("  ... landing configuration", pf.get("stall_onset_semispan_landing"), "")
    kv("washout (tip twist)", j["selection"].get("wing_washout_deg", 0.0), "deg")

    # ---------------------------------------------------------------- engine out
    if eo:
        head("6. ONE MOTOR OUT")
        kv("control speed V_mc", eo["V"], "m/s", "live motor at full (capped) thrust, dead prop stopped")
        kv("bank toward the working motor", -eo["phi_deg"], "deg", "zero-sideslip technique (max 5 deg allowed)")
        kv("rudder needed / trusted", f"{abs(eo['rudder_deg']):.1f} / {eo['rudder_usable_deg']:.1f}", "deg")
        kv("aileron needed", abs(eo["aileron_frac"]) * 100, "% of throw",
           f"limit {P.get('trim_aileron_max_frac', 0.75) * 100:.0f}%")
        kv("yawing moment to hold", eo["yaw_moment"], "N.m",
           f"{eo['T_one']:.2f} N thrust + {eo['D_dead_prop']:.2f} N stopped-prop drag")
        w("  Below V_mc, or if the rudder runs out, reduce the working motor's throttle (the FC can do this")
        w("  automatically): with less thrust there is less yaw to hold, and with both motors at idle there is none.")

    # ---------------------------------------------------------------- power
    head("7. POWER SYSTEM")
    kv("throttle cap (set in the FC, both motors)", j["throttle_cap"], "", "max ESC duty in every mode")
    th = [r for r in j.get("thrust_in_flight", []) if r["point"].startswith("static")]
    if th:
        kv("static thrust / weight (at the cap)", th[0]["thrust_per_weight"], "",
           f"{th[0]['thrust_avail_N']:.2f} N")
    kv("max motor current", pf["max_motor_current_A"], "A")
    kv("max battery current", pf["max_battery_current_A"], "A")

    # ---------------------------------------------------------------- warnings
    head("8. WATCH-OUTS")
    notes = []
    tr = ap.get("trim") or {}
    if tr and not tr.get("trusted", True):
        notes.append(f"Approach trim at the FORWARD CG needs {tr['elevator_deg']:+.1f} deg of elevator at a tail angle "
                     f"of {tr['alpha_tail_deg']:+.1f} deg - outside the range NeuralFoil trusts (tail near its "
                     "negative stall). Balance toward the aft half of the CG range, and check elevator authority "
                     "on the first approaches.")
    for k, v in (j.get("constraint_margins") or {}).items():
        if 0 <= v < 0.02 and k not in ("converged", "esc_available", "prop_data_range", "brake_fits", "nose_gear_fits", "stall_speed"):
            notes.append(f"'{k}' is at its limit (margin {v * 100:.1f}%) - it sized the design; "
                         "a heavier build or measured inputs may push it over.")
    ph_lo = math.degrees(math.acos(min(1.0, (Vs / lo["V"]) ** 2)))
    if ph_lo < 50:
        notes.append(f"At loiter speed ({lo['V']:.1f} m/s, only {lo['V'] / Vs:.2f} x the stall speed) the wing stalls "
                     f"in a level turn steeper than ~{ph_lo:.0f} deg of bank - limit the autopilot's bank angle in "
                     "loiter or add speed for turns.")
    if dv and dv.get("cn_beta", 1) < 0.06:
        notes.append(f"Weathercock stability is modest (Cn_beta {dv['cn_beta']:.3f}/rad): expect the nose to "
                     "wander in gusts; this is why the fin is never allowed below the Raymer tail-volume size.")
    for i, s_ in enumerate(notes, 1):
        w(f"  {i}. {s_}")
    if not notes:
        w("  none")

    head("9. NOT COMPUTED BY THIS MODEL (check in flight / preliminary design)")
    for s_ in ("Dynamic modes: short-period, phugoid, Dutch roll, spiral, roll-mode time constant",
               "Control forces / servo torque (hinge moments) and gust response",
               "Best-range speed, turn performance with power, ceiling",
               "Stall departure behaviour beyond where the stall starts; spin characteristics",
               "Propwash effects on the tail and ground effect on the tail during the flare"):
        w(f"  - {s_}")
    return "\n".join(L)


def write(json_path=DEFAULT, out_path=None):
    with open(json_path) as f:
        text = render(json.load(f))
    if out_path:
        with open(out_path, "w") as f:
            f.write(text + "\n")
    return text


if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[args.index("-o") + 1] if "-o" in args else None
    print(write(DEFAULT, out))
